"""
Real-time Detection Service
============================
Manages per-camera background threads.  Detection pipeline (priority order):

  1. PP-PicoDet-M 320 (ONNX Runtime) + BoxMot ByteTrack  ← preferred (fast CPU)
  2. YOLOv8s / YOLOv8n (Ultralytics) + built-in ByteTrack ← YOLO fallback
  3. OpenCV HOG people detector                            ← last resort

Counting improvements over naive approach
-----------------------------------------
* Foot-anchor  — uses bottom-centre of bbox (more stable than body centre
                 for horizontal lines; crosses line before torso does).
* Dwell-confirm — a crossing is only counted after the person stays on the
                  new side for _CROSS_CONFIRM_FRAMES consecutive frames.
                  Eliminates false counts from bbox jitter near the line.
* Stateful per-track side dict — replaces the old "counted_ids" set so a
                  person who re-crosses the line is counted again correctly.
* ROI crop     — when a counting line is configured, PicoDet only processes
                  a band ±_ROI_MARGIN around the line (big speed gain on CPU).

Usage
-----
    svc = DetectionService()
    svc.start_camera("cam-01", source="crowd.mp4", label="Main Entrance",
                     line_cfg={"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5})

    frame_bytes = svc.get_latest_frame("cam-01")   # MJPEG frame
    stats       = svc.get_stats("cam-01")           # CameraStats dataclass
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

_JPEG_QUALITY      = 80     # MJPEG stream quality
_MAX_EVENTS        = 2000   # in-memory event buffer size
_HOG_DETECT_INTERVAL = 4   # run HOG every N frames (fallback only)
_HOG_MAX_WIDTH     = 640   # downscale HOG input for speed
_WEBCAM_WIDTH      = 640
_WEBCAM_HEIGHT     = 480
_RENDER_FPS        = 25.0  # target FPS for the MJPEG render thread
_FRAME_SKIP_MAX    = 8     # max frames to skip per cycle when catching up

# ── Counting / tracking parameters ────────────────────────────────────────
_CROSS_CONFIRM_FRAMES = 3   # consecutive frames on new side to confirm a crossing
_ROI_MARGIN           = 0.35  # half-height of ROI band around counting line (±35% of h)

# ── PicoDet-M inference parameters ────────────────────────────────────────
_PICODET_CONF    = 0.35
_PICODET_NMS_IOU = 0.45

# ── MobileNetSSD parameters ────────────────────────────────────────────────
_SSD_CONF_THRESH = 0.40   # minimum confidence for person detections

# ── YOLO fallback parameters ───────────────────────────────────────────────
_YOLO_IMGSZ = 416
_YOLO_CONF  = 0.25
_YOLO_IOU   = 0.45
_YOLO_MODELS = ["yolov8s.pt", "yolov8n.pt"]   # best-first cascade


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
        self._cross_states:   Dict[int, dict]           = {}
        # cross_states per track: {"side": bool, "candidate": Optional[bool], "dwell": int}

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
        self._cross_states.clear()
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
        # ── Detector / tracker initialisation (done after src_fps is known) ─
        detector   = None   # PicoDetM or MobileNetSSD DNN net
        det_type   = None   # "picodet" | "ssd"
        tracker_bt = None   # ByteTrack-Lite instance
        model      = None   # YOLO fallback (unused on Py 3.13)
        hog        = None   # HOG absolute last resort
        hog_frame_idx  = 0
        hog_last_boxes: List[Tuple[int, int, int, int, float]] = []

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

        # ── Model / tracker init — done here so src_fps is available ─────────

        # 1. Try PicoDet-M ONNX + ByteTrack-Lite (fastest CPU, no torch needed)
        try:
            from crowd_engine.infra.picodet import PicoDetM
            from crowd_engine.infra.bytetrack_lite import ByteTrack as LiteByteTrack
            _det = PicoDetM()
            if _det.load():
                detector   = _det
                det_type   = "picodet"
                tracker_bt = LiteByteTrack(
                    track_thresh=0.45, track_buffer=30,
                    match_thresh=0.8,  frame_rate=int(src_fps),
                )
                log.info("Camera %s → PicoDet-M + ByteTrack-Lite ✔", self.camera_id)
            else:
                log.warning("Camera %s: PicoDet-M not available — trying MobileNetSSD", self.camera_id)
        except Exception as exc:
            log.warning("Camera %s: PicoDet unavailable (%s) — trying MobileNetSSD", self.camera_id, exc)

        # 2. MobileNetSSD DNN + ByteTrack-Lite (model files ship with the project)
        if detector is None:
            try:
                import os as _os
                from crowd_engine.infra.bytetrack_lite import ByteTrack as LiteByteTrack
                _prototxt = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
                    _os.path.abspath(__file__)))), "MobileNetSSD_deploy.prototxt")
                _caffemodel = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
                    _os.path.abspath(__file__)))), "MobileNetSSD_deploy.caffemodel")
                if _os.path.isfile(_prototxt) and _os.path.isfile(_caffemodel):
                    _net = cv2.dnn.readNetFromCaffe(_prototxt, _caffemodel)
                    detector   = _net
                    det_type   = "ssd"
                    tracker_bt = LiteByteTrack(
                        track_thresh=0.40, track_buffer=25,
                        match_thresh=0.75, frame_rate=int(src_fps),
                    )
                    log.info("Camera %s → MobileNetSSD + ByteTrack-Lite ✔", self.camera_id)
                else:
                    log.warning("Camera %s: MobileNetSSD model files missing — trying YOLO", self.camera_id)
            except Exception as exc:
                log.warning("Camera %s: MobileNetSSD failed (%s) — trying YOLO", self.camera_id, exc)

        # 3. YOLO fallback (requires torch; unavailable on Python 3.13)
        if detector is None:
            try:
                from ultralytics import YOLO
                for _m in _YOLO_MODELS:
                    try:
                        model = YOLO(_m)
                        log.info("Camera %s → %s (YOLO fallback)", self.camera_id, _m)
                        break
                    except Exception:
                        pass
            except Exception as exc:
                log.warning("Camera %s: YOLO unavailable (%s) — using HOG", self.camera_id, exc)

        # 4. HOG absolute last resort
        if detector is None and model is None:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            log.info("Camera %s → OpenCV HOG (last resort)", self.camera_id)

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

        # ── Shared state (3 threads: capture → buffer ← render + inference) ─
        # Captured frame — written by capture thread at source FPS
        _cap_frame:  List[Optional[np.ndarray]] = [None]
        _cap_lock    = threading.Lock()
        _cap_seq:    List[int] = [0]
        _video_looped: List[bool] = [False]   # capture signals video restart

        # Tracked objects from last YOLO/HOG run
        _objs: List[list] = [[]]
        _objs_lock = threading.Lock()

        # ── Capture thread: reads source at native FPS ──────────────────────
        _cap_interval = 1.0 / src_fps

        def capture_loop() -> None:
            frame_idx   = 0
            session_t   = time.monotonic()

            while self._running:
                t0 = time.monotonic()

                # Video files: skip stale frames to keep wall-clock pace
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
                    with _cap_lock:
                        _video_looped[0] = True
                    time.sleep(0.03)
                    continue

                frame_idx += 1
                with _cap_lock:
                    _cap_frame[0] = frame
                    _cap_seq[0]  += 1

                # Pace video files to source FPS; webcams self-pace
                if not is_live:
                    sleep_t = _cap_interval - (time.monotonic() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

        threading.Thread(
            target=capture_loop, daemon=True, name=f"cap-{self.camera_id}"
        ).start()

        # ── Render thread: smooth MJPEG at source FPS ───────────────────────
        _render_interval = 1.0 / min(_RENDER_FPS, src_fps)

        def render_loop() -> None:
            last_seq = -1
            while self._running:
                t0 = time.monotonic()

                with _cap_lock:
                    raw = _cap_frame[0]
                    seq = _cap_seq[0]
                if raw is None or seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq

                frame = raw.copy()
                h, w = frame.shape[:2]

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

        threading.Thread(
            target=render_loop, daemon=True, name=f"rnd-{self.camera_id}"
        ).start()

        # ── Inference loop (this thread) ────────────────────────────────────
        frame_count  = 0
        fps_t        = time.monotonic()
        last_inf_seq = -1
        _tracker_persist = True   # False for first frame after video loop

        while self._running:
            # Grab latest captured frame; skip if unchanged since last run
            with _cap_lock:
                frame = _cap_frame[0]
                seq   = _cap_seq[0]
                looped = _video_looped[0]
                if looped:
                    _video_looped[0] = False
            if frame is None or seq == last_inf_seq:
                time.sleep(0.005)
                continue
            last_inf_seq = seq

            # Video looped → reset tracker + counting state
            if looped and not is_live:
                _tracker_persist = False
                self._prev_centroids.clear()
                self._cross_states.clear()
                if tracker_bt is not None:
                    from crowd_engine.infra.bytetrack_lite import ByteTrack as LiteByteTrack
                    _thresh = 0.45 if det_type == "picodet" else 0.40
                    tracker_bt = LiteByteTrack(
                        track_thresh=_thresh, track_buffer=30,
                        match_thresh=0.8,  frame_rate=int(src_fps),
                    )

            h, w = frame.shape[:2]
            curr_centroids: Dict[int, Tuple[int, int]] = {}
            new_objs: list = []
            new_count = 0

            # ── ROI crop (PicoDet only; speeds up inference by ~4×) ────────
            roi_frame = frame
            y_offset  = 0
            if det_type == "picodet" and self.line:
                line_ny   = (self.line.ny1 + self.line.ny2) / 2.0
                margin_px = int(_ROI_MARGIN * h)
                roi_y0 = max(0, int(line_ny * h) - margin_px)
                roi_y1 = min(h, int(line_ny * h) + margin_px)
                if roi_y1 - roi_y0 >= 64:          # sanity: at least 64 px tall
                    roi_frame = frame[roi_y0:roi_y1, :]
                    y_offset  = roi_y0

            # ── PicoDet-M + ByteTrack-Lite ─────────────────────────────────
            if det_type == "picodet" and detector is not None and tracker_bt is not None:
                dets = detector.detect(roi_frame,
                                       conf=_PICODET_CONF,
                                       nms_iou=_PICODET_NMS_IOU)
                # Restore full-frame y coordinates from ROI space
                for d in dets:
                    d["y1"] += y_offset
                    d["y2"] += y_offset

                if dets:
                    dets_np = np.array(
                        [[d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], 0.0]
                         for d in dets],
                        dtype=np.float32,
                    )
                else:
                    dets_np = np.empty((0, 6), dtype=np.float32)

                tracks = tracker_bt.update(dets_np, frame)
                new_count = len(tracks)

                for t in tracks:
                    x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                    tid    = int(t[4])
                    conf_t = float(t[5]) if len(t) > 5 else 1.0
                    cx     = (x1 + x2) // 2
                    foot_y = y2
                    curr_centroids[tid] = (cx, foot_y)
                    new_objs.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "tid": tid, "conf": conf_t,
                    })

            # ── MobileNetSSD DNN + ByteTrack-Lite ─────────────────────────
            elif det_type == "ssd" and detector is not None and tracker_bt is not None:
                _SSD_PERSON = 15   # MobileNetSSD COCO class index for "person"
                _SSD_SIZE   = 300
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (_SSD_SIZE, _SSD_SIZE)),
                    0.007843, (_SSD_SIZE, _SSD_SIZE), 127.5,
                )
                detector.setInput(blob)
                detections = detector.forward()   # [1, 1, N, 7]
                dets_list = []
                for k in range(detections.shape[2]):
                    conf_k = float(detections[0, 0, k, 2])
                    cls_k  = int(detections[0, 0, k, 1])
                    if conf_k < _SSD_CONF_THRESH or cls_k != _SSD_PERSON:
                        continue
                    x1k = int(detections[0, 0, k, 3] * w)
                    y1k = int(detections[0, 0, k, 4] * h)
                    x2k = int(detections[0, 0, k, 5] * w)
                    y2k = int(detections[0, 0, k, 6] * h)
                    dets_list.append([x1k, y1k, x2k, y2k, conf_k, 0.0])

                dets_np = (np.array(dets_list, dtype=np.float32)
                           if dets_list else np.empty((0, 6), dtype=np.float32))
                tracks = tracker_bt.update(dets_np, frame)
                new_count = len(tracks)

                for t in tracks:
                    x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                    tid    = int(t[4])
                    conf_t = float(t[5]) if len(t) > 5 else 1.0
                    cx     = (x1 + x2) // 2
                    foot_y = y2
                    curr_centroids[tid] = (cx, foot_y)
                    new_objs.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "tid": tid, "conf": conf_t,
                    })

            # ── YOLO + built-in ByteTrack ──────────────────────────────────
            elif model is not None:
                results = model.track(
                    frame,
                    persist=_tracker_persist,
                    classes=[0],
                    conf=_YOLO_CONF,
                    iou=_YOLO_IOU,
                    imgsz=_YOLO_IMGSZ,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )
                _tracker_persist = True
                boxes = results[0].boxes
                if boxes is not None and boxes.id is not None:
                    new_count = len(boxes)
                    for xyxy, tid, conf_t in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.id.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                    ):
                        track_id = int(tid)
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx = (x1 + x2) // 2
                        curr_centroids[track_id] = (cx, y2)   # foot anchor
                        new_objs.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "tid": track_id, "conf": float(conf_t),
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
                        c = float(weights[i]) if len(weights) > i else 0.0
                        hog_last_boxes.append((
                            int(x * inv), int(y * inv),
                            int((x + bw) * inv), int((y + bh) * inv), c,
                        ))
                new_count = len(hog_last_boxes)
                for i, (x1, y1, x2, y2, conf_t) in enumerate(hog_last_boxes):
                    new_objs.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "tid": i, "conf": conf_t,
                    })
                    curr_centroids[i] = ((x1 + x2) // 2, y2)  # foot anchor

            self.current_count = new_count

            with _objs_lock:
                _objs[0] = new_objs

            # ── Dwell-confirmed counting line ──────────────────────────────
            if self.line:
                lx1, ly1, lx2, ly2 = self.line.to_pixels(w, h)
                for tid, foot in curr_centroids.items():
                    direction = _update_crossing_state(
                        self._cross_states, tid, foot,
                        lx1, ly1, lx2, ly2,
                    )
                    if direction == "in":
                        self.count_in += 1
                        if self._on_event:
                            self._on_event(self.camera_id, tid, "in")
                    elif direction == "out":
                        self.count_out += 1
                        if self._on_event:
                            self._on_event(self.camera_id, tid, "out")

                # Purge states for tracks that have disappeared
                for gone in set(self._cross_states) - set(curr_centroids):
                    del self._cross_states[gone]

            # ── FPS rolling average ────────────────────────────────────────
            frame_count += 1
            elapsed = time.monotonic() - fps_t
            if elapsed >= 2.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_t = time.monotonic()

        cap.release()
        log.info("Camera %s stopped", self.camera_id)


# ── Geometry helpers ───────────────────────────────────────────────────────

def _line_side(
    px: int, py: int,
    lx1: int, ly1: int,
    lx2: int, ly2: int,
) -> bool:
    """
    Cross-product side test.
    Returns True if (px,py) is on the positive side of the directed line.
    """
    return float((lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)) > 0.0


def _update_crossing_state(
    states: Dict,
    tid: int,
    foot: Tuple[int, int],          # (cx, y2) — foot-anchor point
    lx1: int, ly1: int,
    lx2: int, ly2: int,
) -> Optional[str]:
    """
    Dwell-confirmed, stateful line crossing detector.

    A crossing is only confirmed after the person's foot stays on the *new*
    side for _CROSS_CONFIRM_FRAMES consecutive frames — eliminates false
    counts from bounding-box jitter near the line.

    Returns 'in', 'out', or None.
    Modifies `states` in place.
    """
    px, py      = foot
    curr_side   = _line_side(px, py, lx1, ly1, lx2, ly2)

    if tid not in states:
        # First detection: record side, no crossing yet
        states[tid] = {"side": curr_side, "candidate": None, "dwell": 0}
        return None

    st = states[tid]

    if curr_side == st["side"]:
        # Still on the confirmed side — cancel any pending crossing
        st["candidate"] = None
        st["dwell"]     = 0
        return None

    # On the *opposite* side from confirmed
    if st["candidate"] is None or st["candidate"] != curr_side:
        # New candidate side — start dwell counter
        st["candidate"] = curr_side
        st["dwell"]     = 1
        return None

    st["dwell"] += 1

    if st["dwell"] >= _CROSS_CONFIRM_FRAMES:
        # Crossing confirmed — update state and fire event
        old_side        = st["side"]
        st["side"]      = curr_side
        st["candidate"] = None
        st["dwell"]     = 0
        # Direction: "in" if was on positive side (moving toward negative side)
        return "in" if old_side else "out"

    return None


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
