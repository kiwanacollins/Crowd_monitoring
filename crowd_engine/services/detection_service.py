"""
Real-time Detection Service
============================
Manages per-camera background threads, each running YOLOv8n + ByteTrack to:
  1. Capture frames from the video source (file, RTSP, webcam)
  2. Track every person across frames with a persistent ID
  3. Detect when a person crosses a configurable counting line (IN / OUT)
  4. Annotate frames and store the latest JPEG for MJPEG streaming
  5. Record crossing events and expose live stats / metrics

Usage
-----
    svc = DetectionService()
    svc.start_camera("cam-01", source="crowd.mp4", label="Main Entrance",
                     line_cfg={"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5})

    # In a FastAPI streaming endpoint:
    frame_bytes = svc.get_latest_frame("cam-01")

    # Counts
    stats = svc.get_stats("cam-01")  # CameraStats dataclass
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from crowd_engine.infra.logger import get_logger

log = get_logger(__name__)

# ── Colour palette (BGR) ───────────────────────────────────────────────────
_C_BOX  = (50,  205,  50)   # lime green — bounding boxes
_C_LINE = (0,   255, 255)   # yellow     — counting line
_C_IN   = (50,  220,  50)   # green      — "IN" label
_C_OUT  = (50,   50, 220)   # red        — "OUT" label
_C_TEXT = (255, 255, 255)   # white      — HUD text

_JPEG_QUALITY      = 80     # slightly higher quality for smoother stream appearance
_MAX_EVENTS        = 2000   # in-memory event buffer size
_HOG_DETECT_INTERVAL = 4   # run expensive HOG every N frames in fallback mode
_HOG_MAX_WIDTH     = 640   # downscale fallback detection input for speed
_WEBCAM_WIDTH      = 640
_WEBCAM_HEIGHT     = 480
_RENDER_FPS        = 25.0  # target FPS for the MJPEG render thread
_YOLO_IMGSZ        = 416   # inference input size — 35% faster than 640 on CPU
_YOLO_CONF         = 0.30  # confidence threshold (lower = higher recall)
_YOLO_IOU          = 0.45  # NMS IOU threshold
_FRAME_SKIP_MAX    = 8     # max frames to skip per cycle when catching up to real-time


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class CountingLine:
    """Normalised coordinates (0–1) for a virtual counting line."""
    nx1: float
    ny1: float
    nx2: float
    ny2: float

    def to_pixels(self, w: int, h: int) -> Tuple[int, int, int, int]:
        return int(self.nx1 * w), int(self.ny1 * h), int(self.nx2 * w), int(self.ny2 * h)


@dataclass
class CameraStats:
    camera_id: str
    label: str
    count_in: int = 0
    count_out: int = 0
    current_count: int = 0
    fps: float = 0.0
    running: bool = False
    session_start: str = ""
    source: str = ""


# ── Camera worker ──────────────────────────────────────────────────────────

class _CameraWorker:
    """Background thread that processes one camera source."""

    def __init__(
        self,
        camera_id: str,
        source: Any,
        label: str,
        line: Optional[CountingLine],
        on_event: Optional[Callable[[str, int, str], None]] = None,
    ) -> None:
        self.camera_id   = camera_id
        self.source      = source
        self.label       = label
        self.line        = line
        self._on_event   = on_event

        self.count_in    = 0
        self.count_out   = 0
        self.current_count = 0
        self.fps         = 0.0
        self.session_start = datetime.now(timezone.utc).isoformat()

        self._prev_centroids: Dict[int, Tuple[int, int]] = {}
        self._counted_ids: set = set()

        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock  = threading.Lock()

        self._running = False
        self._thread  = threading.Thread(target=self._run, daemon=True, name=f"cam-{camera_id}")

    # ── Public ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def reset_counts(self) -> None:
        self.count_in  = 0
        self.count_out = 0
        self._counted_ids.clear()
        self._prev_centroids.clear()

    def get_latest_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def stats(self) -> CameraStats:
        return CameraStats(
            camera_id=self.camera_id,
            label=self.label,
            count_in=self.count_in,
            count_out=self.count_out,
            current_count=self.current_count,
            fps=round(self.fps, 1),
            running=self._running,
            session_start=self.session_start,
            source=str(self.source),
        )

    # ── Main loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        model = None
        hog = None
        hog_frame_idx = 0
        hog_last_boxes: List[Tuple[int, int, int, int, float]] = []
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            log.info("Camera %s using YOLOv8 detector", self.camera_id)
        except Exception as exc:
            log.warning(
                "Camera %s falling back to OpenCV HOG detector (YOLO unavailable: %s)",
                self.camera_id, exc,
            )
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Resolve source: "0" -> int for webcam
        src: Any = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        is_live = isinstance(src, int)

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error("Cannot open source: %s", src)
            self._running = False
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if is_live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _WEBCAM_HEIGHT)

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps < 1 or src_fps > 120:
            src_fps = 25.0

        log.info("Camera %s started — %s (source %.1f FPS)", self.camera_id, src, src_fps)

        # macOS permission warm-up for live cameras
        if is_live:
            fail_count = 0
            for _ in range(5):
                ret, test_frame = cap.read()
                if not ret or test_frame is None or np.mean(test_frame) < 2.0:
                    fail_count += 1
                time.sleep(0.1)
            if fail_count >= 4:
                log.warning(
                    "Camera %s: no usable frames during warm-up — "
                    "macOS: System Settings → Privacy & Security → Camera → enable Terminal.",
                    self.camera_id,
                )

        # ── Shared state: inference writes, render reads ────────────────────
        # Latest raw frame stored by inference loop, read by render thread
        _raw:  List[Optional[np.ndarray]] = [None]
        _raw_lock = threading.Lock()
        # Latest tracked objects: [{x1, y1, x2, y2, tid, conf}]
        _objs: List[list] = [[]]
        _objs_lock = threading.Lock()

        # ── Render thread: smooth MJPEG at up to 25 FPS ────────────────────
        _render_interval = 1.0 / min(_RENDER_FPS, src_fps)

        def render_loop() -> None:
            while self._running:
                t0 = time.monotonic()

                with _raw_lock:
                    raw = _raw[0]
                if raw is None:
                    time.sleep(0.02)
                    continue

                frame = raw.copy()
                h, w = frame.shape[:2]

                # Overlay last known detections (temporal persistence between YOLO runs)
                with _objs_lock:
                    objs = list(_objs[0])
                for obj in objs:
                    x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), _C_BOX, 2)
                    cv2.putText(
                        frame, f"#{obj['tid']} {obj['conf']:.2f}",
                        (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_BOX, 2,
                    )

                if self.line:
                    lx1, ly1, lx2, ly2 = self.line.to_pixels(w, h)
                    cv2.line(frame, (lx1, ly1), (lx2, ly2), _C_LINE, 2)
                    cv2.putText(frame, "IN",  (lx1 + 4,  ly1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_IN,  2)
                    cv2.putText(frame, "OUT", (lx2 - 40, ly2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_OUT, 2)

                _draw_hud(frame, w, self.label, self.count_in,
                          self.count_out, self.current_count, self.fps)

                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
                )
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = jpeg.tobytes()

                sleep_t = _render_interval - (time.monotonic() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        render_thread = threading.Thread(
            target=render_loop, daemon=True, name=f"rnd-{self.camera_id}"
        )
        render_thread.start()

        # ── Inference loop ──────────────────────────────────────────────────
        frame_count = 0
        fps_t       = time.monotonic()
        frame_idx   = 0
        session_t   = time.monotonic()

        while self._running:
            # For video files: skip stale frames so playback stays real-time.
            if not is_live:
                elapsed_wall = time.monotonic() - session_t
                expected_idx = int(elapsed_wall * src_fps)
                skip = min(expected_idx - frame_idx, _FRAME_SKIP_MAX)
                for _ in range(max(0, skip - 1)):
                    if not cap.grab():
                        break
                    frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                session_t = time.monotonic()
                time.sleep(0.03)
                continue

            frame_idx += 1

            # Share raw frame with render thread
            with _raw_lock:
                _raw[0] = frame

            h, w = frame.shape[:2]
            curr_centroids: Dict[int, Tuple[int, int]] = {}
            self.current_count = 0
            new_objs: list = []

            # ── YOLOv8 + ByteTrack ─────────────────────────────────────────
            if model is not None:
                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=_YOLO_CONF,
                    iou=_YOLO_IOU,
                    imgsz=_YOLO_IMGSZ,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )
                boxes = results[0].boxes
                if boxes is not None and boxes.id is not None:
                    self.current_count = len(boxes)
                    for xyxy, tid, conf in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.id.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                    ):
                        track_id = int(tid)
                        x1, y1, x2, y2 = map(int, xyxy)
                        curr_centroids[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)
                        new_objs.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "tid": track_id, "conf": float(conf),
                        })

            # ── OpenCV HOG fallback ────────────────────────────────────────
            elif hog is not None:
                hog_frame_idx += 1
                if hog_frame_idx == 1 or hog_frame_idx % _HOG_DETECT_INTERVAL == 0:
                    det_scale = 1.0
                    det_frame = frame
                    if w > _HOG_MAX_WIDTH:
                        det_scale = _HOG_MAX_WIDTH / float(w)
                        det_h = max(1, int(h * det_scale))
                        det_frame = cv2.resize(frame, (_HOG_MAX_WIDTH, det_h),
                                               interpolation=cv2.INTER_LINEAR)
                    rects, weights = hog.detectMultiScale(
                        det_frame, winStride=(8, 8), padding=(8, 8), scale=1.08,
                    )
                    hog_last_boxes = []
                    inv = 1.0 / det_scale
                    for i, (x, y, bw, bh) in enumerate(rects):
                        conf = float(weights[i]) if len(weights) > i else 0.0
                        hog_last_boxes.append((
                            int(x * inv), int(y * inv),
                            int((x + bw) * inv), int((y + bh) * inv), conf,
                        ))
                self.current_count = len(hog_last_boxes)
                for i, (x1, y1, x2, y2, conf) in enumerate(hog_last_boxes):
                    new_objs.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "tid": i, "conf": conf,
                    })
                    curr_centroids[i] = ((x1 + x2) // 2, (y1 + y2) // 2)

            with _objs_lock:
                _objs[0] = new_objs

            # ── Counting line ──────────────────────────────────────────────
            if self.line:
                lx1, ly1, lx2, ly2 = self.line.to_pixels(w, h)
                for tid, centroid in curr_centroids.items():
                    if tid not in self._prev_centroids or tid in self._counted_ids:
                        continue
                    direction = _check_line_cross(
                        self._prev_centroids[tid], centroid,
                        lx1, ly1, lx2, ly2,
                    )
                    if direction == "in":
                        self.count_in += 1
                        self._counted_ids.add(tid)
                        if self._on_event:
                            self._on_event(self.camera_id, tid, "in")
                    elif direction == "out":
                        self.count_out += 1
                        self._counted_ids.add(tid)
                        if self._on_event:
                            self._on_event(self.camera_id, tid, "out")

            self._prev_centroids = curr_centroids

            # ── FPS rolling average (tracks inference throughput) ──────────
            frame_count += 1
            elapsed = time.monotonic() - fps_t
            if elapsed >= 2.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_t = time.monotonic()

        cap.release()
        log.info("Camera %s stopped", self.camera_id)


# ── Geometry helpers ───────────────────────────────────────────────────────

def _check_line_cross(
    prev: Tuple[int, int],
    curr: Tuple[int, int],
    lx1: int, ly1: int,
    lx2: int, ly2: int,
) -> Optional[str]:
    """
    Detect whether the movement prev→curr crosses the counting line.
    Returns 'in', 'out', or None.
    Uses cross-product side test.
    """
    dlx = lx2 - lx1
    dly = ly2 - ly1

    def side(px: int, py: int) -> float:
        return float(dlx * (py - ly1) - dly * (px - lx1))

    s1 = side(*prev)
    s2 = side(*curr)

    if s1 == 0.0 or s2 == 0.0:
        return None
    if (s1 > 0) == (s2 > 0):
        return None  # same side — no crossing

    return "in" if s1 > 0 else "out"


def _draw_hud(
    frame: np.ndarray,
    w: int,
    label: str,
    count_in: int,
    count_out: int,
    current: int,
    fps: float,
) -> None:
    """Draw a semi-transparent HUD bar at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, label, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _C_TEXT, 2)

    stats_txt = (
        f"IN: {count_in}   OUT: {count_out}   "
        f"NOW: {current}   {fps:.1f} FPS"
    )
    cv2.putText(frame, stats_txt, (int(w * 0.32), 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _C_TEXT, 2)


# ── Service singleton ──────────────────────────────────────────────────────

class DetectionService:
    """
    Manages all camera workers.  Designed to be instantiated once and shared
    across the FastAPI application.
    """

    def __init__(self) -> None:
        self._workers: Dict[str, _CameraWorker] = {}
        self._lock    = threading.Lock()
        self._events: List[dict] = []

    def start_camera(
        self,
        camera_id: str,
        source: Any,
        label: str = "",
        line_cfg: Optional[dict] = None,
    ) -> None:
        """Start detection on a camera.  No-op if already running."""
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker and worker._running:
                log.info("Camera %s already running", camera_id)
                return

            line: Optional[CountingLine] = None
            if line_cfg:
                line = CountingLine(
                    nx1=float(line_cfg.get("x1", 0.0)),
                    ny1=float(line_cfg.get("y1", 0.5)),
                    nx2=float(line_cfg.get("x2", 1.0)),
                    ny2=float(line_cfg.get("y2", 0.5)),
                )

            worker = _CameraWorker(
                camera_id=camera_id,
                source=source,
                label=label or camera_id,
                line=line,
                on_event=self._record_event,
            )
            self._workers[camera_id] = worker
            worker.start()
            log.info("Started camera %s (%s)", camera_id, source)

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            worker.stop()
            return True
        return False

    def reset_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            worker.reset_counts()
            return True
        return False

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            worker = self._workers.get(camera_id)
        return worker.get_latest_frame() if worker else None

    def get_stats(self, camera_id: str) -> Optional[CameraStats]:
        with self._lock:
            worker = self._workers.get(camera_id)
        return worker.stats() if worker else None

    def list_cameras(self) -> List[CameraStats]:
        with self._lock:
            workers = list(self._workers.values())
        return [w.stats() for w in workers]

    def get_events(self, camera_id: Optional[str] = None, limit: int = 200) -> List[dict]:
        events = self._events[-_MAX_EVENTS:]
        if camera_id:
            events = [e for e in events if e["camera_id"] == camera_id]
        return events[-limit:]

    def get_metrics(self) -> dict:
        stats = self.list_cameras()
        total_in  = sum(s.count_in  for s in stats)
        total_out = sum(s.count_out for s in stats)
        avg_fps   = (sum(s.fps for s in stats) / len(stats)) if stats else 0.0
        return {
            "cameras_active": sum(1 for s in stats if s.running),
            "cameras_total":  len(stats),
            "total_in":       total_in,
            "total_out":      total_out,
            "net_occupancy":  total_in - total_out,
            "avg_fps":        round(avg_fps, 1),
            "events_logged":  len(self._events),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _record_event(self, camera_id: str, track_id: int, direction: str) -> None:
        event = {
            "camera_id": camera_id,
            "track_id":  track_id,
            "direction": direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._events.append(event)
        if len(self._events) > _MAX_EVENTS:
            self._events = self._events[-_MAX_EVENTS:]
