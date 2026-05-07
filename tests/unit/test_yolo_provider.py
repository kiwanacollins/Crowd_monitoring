"""
Unit tests for the YOLOv8 provider.

The YOLO model and torch are mocked so these tests run without GPU or
downloading weights.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crowd_engine.domain.entities import CameraInput


def _make_camera(**kwargs) -> CameraInput:
    defaults = dict(source="test.mp4", latitude=0.3156, longitude=32.5816)
    defaults.update(kwargs)
    return CameraInput(**defaults)


def _make_mock_results(num_people: int, classes: list[int] | None = None):
    """Build a fake Ultralytics Results-like object."""
    boxes = MagicMock()
    if num_people > 0:
        # conf as a tensor-like mock
        conf_mock = MagicMock()
        conf_mock.mean.return_value.item.return_value = 0.85
        boxes.conf = conf_mock
        boxes.__len__ = lambda self: num_people
    else:
        conf_mock = MagicMock()
        conf_mock.mean.return_value.item.return_value = 0.0
        boxes.conf = conf_mock
        boxes.__len__ = lambda self: 0
    result = MagicMock()
    result.boxes = boxes
    return [result]


def _make_provider():
    """Return a YOLOv8Provider with YOLO model mocked so no weights download."""
    # Ultralytics transitively imports torch; on some Python versions (e.g. 3.13)
    # torch wheels may be unavailable. To keep these unit tests lightweight and
    # deterministic, inject a fake `ultralytics` module.
    mock_model = MagicMock()
    mock_model.return_value = _make_mock_results(0)  # warm-up call should not raise

    mock_yolo_ctor = MagicMock(return_value=mock_model)
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = mock_yolo_ctor

    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        from crowd_engine.providers.yolo_provider import YOLOv8Provider

        provider = YOLOv8Provider(model_name="yolov8n.pt")

    return provider, mock_model


class TestYOLOv8Provider:
    """Tests for YOLOv8Provider."""

    def test_name(self):
        provider, _ = _make_provider()
        assert provider.name() == "yolov8"

    def test_health_ok(self):
        provider, _ = _make_provider()
        h = provider.health()
        assert h["status"] == "ok"
        assert "details" in h

    def test_estimate_counts_persons(self):
        provider, mock_model = _make_provider()
        mock_model.return_value = _make_mock_results(num_people=5)

        cam = _make_camera()
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap_cls.return_value = mock_cap
            result = provider.estimate_crowd(cam)

        assert result.count == 5
        assert result.source == "yolov8"

    def test_estimate_returns_zero_for_empty_frame(self):
        provider, mock_model = _make_provider()
        mock_model.return_value = _make_mock_results(num_people=0)

        cam = _make_camera()
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap_cls.return_value = mock_cap
            result = provider.estimate_crowd(cam)

        assert result.count == 0

    def test_estimate_returns_error_on_unopenable_source(self):
        provider, _ = _make_provider()
        cam = _make_camera(source="rtsp://missing/stream")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap
            result = provider.estimate_crowd(cam)

        # Should return an error CrowdEstimate, not raise
        assert result.count == 0

    def test_estimate_accepts_numpy_frame_directly(self):
        """Provider should skip VideoCapture when given a raw numpy frame."""
        provider, mock_model = _make_provider()
        mock_model.return_value = _make_mock_results(num_people=3)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam = _make_camera(source=frame)
        result = provider.estimate_crowd(cam)
        assert result.count == 3

    def test_health_unavailable_when_model_not_loaded(self):
        """If _available is False, health should report unavailable."""
        provider, _ = _make_provider()
        provider._available = False
        h = provider.health()
        assert h["status"] == "unavailable"
