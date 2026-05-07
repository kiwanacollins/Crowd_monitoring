"""
YOLOv8 Provider — primary people detector.

Uses Ultralytics YOLOv8n for single-frame estimation, compatible with the
existing CrowdCountProvider interface and fallback orchestrator.

For continuous video tracking (frame-by-frame with persistent IDs) use
DetectionService instead — it wraps this model with ByteTrack.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from crowd_engine.domain.entities import CameraInput, CrowdEstimate
from crowd_engine.infra.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "yolov8n.pt"  # nano — fast, ~6 MB, auto-downloads on first run


class YOLOv8Provider:
    """Crowd estimator using Ultralytics YOLOv8n (COCO person class)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._model = None
        self._available = False
        self._last_latency_ms: float = -1.0
        self._init_model()

    # ── CrowdCountProvider contract ────────────────────────────────────────

    def name(self) -> str:
        return "yolov8"

    def estimate_crowd(self, camera_input: CameraInput) -> CrowdEstimate:
        if not self._available:
            return CrowdEstimate.error_result(
                self.name(), camera_input, "YOLOv8 model not loaded"
            )
        try:
            import cv2

            t0 = time.monotonic()
            source = camera_input.source

            if isinstance(source, np.ndarray):
                frame = source
            else:
                src = int(source) if isinstance(source, str) and source.isdigit() else source
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    return CrowdEstimate.error_result(
                        self.name(), camera_input, f"Cannot open: {source}"
                    )
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return CrowdEstimate.error_result(
                        self.name(), camera_input, "Failed to grab frame"
                    )

            results = self._model(frame, classes=[0], verbose=False)
            boxes = results[0].boxes
            count = len(boxes) if boxes is not None else 0
            confidence = float(boxes.conf.mean().item()) if count > 0 else 0.0

            self._last_latency_ms = (time.monotonic() - t0) * 1000
            log.info(
                "YOLOv8 estimate",
                extra={"count": count, "latency_ms": round(self._last_latency_ms, 1)},
            )
            return CrowdEstimate(
                count=count,
                confidence=min(confidence, 1.0),
                timestamp=datetime.now(timezone.utc),
                source=self.name(),
                camera_id=camera_input.camera_id,
                latitude=camera_input.latitude,
                longitude=camera_input.longitude,
                metadata={
                    "model": self._model_name,
                    "latency_ms": round(self._last_latency_ms, 1),
                },
            )
        except Exception as exc:
            log.exception("YOLOv8 provider error", exc_info=exc)
            return CrowdEstimate.error_result(self.name(), camera_input, str(exc))

    def health(self) -> dict:
        return {
            "status": "ok" if self._available else "unavailable",
            "latency_ms": self._last_latency_ms,
            "details": f"{self._model_name} loaded" if self._available else "Model not loaded",
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)
            # Warm-up pass to avoid cold-start latency on first real frame
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, classes=[0], verbose=False)
            self._available = True
            log.info("YOLOv8 model loaded: %s", self._model_name)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None) or "dependency"
            if missing == "torch":
                log.warning("torch not installed — YOLOv8 provider disabled")
            elif missing == "ultralytics":
                log.warning("ultralytics not installed — YOLOv8 provider disabled")
            else:
                log.warning("Missing %s — YOLOv8 provider disabled", missing)
        except ImportError:
            # Fallback for non-module import errors
            log.warning("ultralytics dependencies missing — YOLOv8 provider disabled")
        except Exception as exc:
            log.warning("Failed to load YOLOv8 model: %s", exc)
