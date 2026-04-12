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

_JPEG_QUALITY = 75          # balance quality vs. stream bandwidth
_MAX_EVENTS   = 2000        # in-memory event buffer size
_HOG_DETECT_INTERVAL = 4    # run expensive HOG every N frames in fallback mode
_HOG_MAX_WIDTH = 640        # downscale fallback detection input for speed
_WEBCAM_WIDTH = 640
_WEBCAM_HEIGHT = 480


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
            # Python 3.13 often lacks torch wheels; keep stream alive with HOG fallback.
            log.warning(
                "Camera %s falling back to OpenCV HOG detector (YOLO unavailable: %s)",
                self.camera_id,
                exc,
            )
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Resolve source: "0" -> int for webcam
        src: Any = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error("Cannot open source: %s", src)
            self._running = False
            return

        # Keep capture latency low and reduce webcam decode work.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _WEBCAM_HEIGHT)

        log.info("Camera %s started — %s", self.camera_id, src)

        # macOS permission warm-up: try a few frames before processing
        # Permission denied = ret=False OR all-black frames
        if isinstance(src, int):
            import numpy as np
            fail_count = 0
            for _ in range(5):
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    fail_count += 1
                elif np.mean(test_frame) < 2.0:
                    fail_count += 1
                time.sleep(0.1)
            if fail_count >= 4:
                log.warning(
                    "Camera %s: no usable frames during warm-up (permission denied?) — "
                    "macOS: System Settings → Privacy & Security → Camera → enable Terminal.",
                    self.camera_id,
                )

        frame_count = 0
        fps_t = time.monotonic()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # Loop video files back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]

            curr_centroids: Dict[int, Tuple[int, int]] = {}
            self.current_count = 0

            # ── Detector path: YOLOv8 (preferred) or OpenCV HOG fallback ──
            if model is not None:
                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],        # person only
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
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        curr_centroids[track_id] = (cx, cy)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), _C_BOX, 2)
                        label_txt = f"#{track_id} {conf:.2f}"
                        cv2.putText(
                            frame, label_txt, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_BOX, 2,
                        )
            elif hog is not None:
                hog_frame_idx += 1
                if hog_frame_idx == 1 or hog_frame_idx % _HOG_DETECT_INTERVAL == 0:
                    det_scale = 1.0
                    det_frame = frame
                    if w > _HOG_MAX_WIDTH:
                        det_scale = _HOG_MAX_WIDTH / float(w)
                        det_h = max(1, int(h * det_scale))
                        det_frame = cv2.resize(
                            frame, (_HOG_MAX_WIDTH, det_h), interpolation=cv2.INTER_LINEAR
                        )

                    rects, weights = hog.detectMultiScale(
                        det_frame,
                        winStride=(8, 8),
                        padding=(8, 8),
                        scale=1.08,
                    )

                    hog_last_boxes = []
                    inv = 1.0 / det_scale
                    for i, (x, y, bw, bh) in enumerate(rects):
                        conf = float(weights[i]) if len(weights) > i else 0.0
                        x1 = int(x * inv)
                        y1 = int(y * inv)
                        x2 = int((x + bw) * inv)
                        y2 = int((y + bh) * inv)
                        hog_last_boxes.append((x1, y1, x2, y2, conf))

                self.current_count = len(hog_last_boxes)
                for i, (x1, y1, x2, y2, conf) in enumerate(hog_last_boxes):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), _C_BOX, 2)
                    cv2.putText(
                        frame,
                        f"H{i} {conf:.2f}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        _C_BOX,
                        2,
                    )

                cv2.putText(
                    frame,
                    "HOG fallback active",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    _C_TEXT,
                    2,
                )

            # ── Counting line logic ──────────────────────────────────────
            if self.line:
                lx1, ly1, lx2, ly2 = self.line.to_pixels(w, h)
                cv2.line(frame, (lx1, ly1), (lx2, ly2), _C_LINE, 2)
                # Label ends of line
                cv2.putText(frame, "IN",  (lx1 + 4, ly1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_IN,  2)
                cv2.putText(frame, "OUT", (lx2 - 40, ly2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_OUT, 2)

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

            # ── HUD overlay ──────────────────────────────────────────────
            _draw_hud(frame, w, self.label, self.count_in, self.count_out,
                      self.current_count, self.fps)

            # ── Store JPEG ───────────────────────────────────────────────
            ok, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = jpeg.tobytes()

            # ── FPS rolling average ──────────────────────────────────────
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
