"""
Unit tests for DetectionService core logic.

Camera workers and the YOLO model are mocked so these tests run without
any video files, GPU, or network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── _check_line_cross geometry ─────────────────────────────────────────────

class TestCheckLineCross:
    """Tests for the _check_line_cross helper function."""

    def _cross(self, prev, curr, lx1=0, ly1=240, lx2=640, ly2=240):
        from crowd_engine.services.detection_service import _check_line_cross
        return _check_line_cross(prev, curr, lx1, ly1, lx2, ly2)

    def test_top_to_bottom_is_out(self):
        # prev is above the horizontal line (y<240), curr is below (y>240)
        result = self._cross(prev=(320, 100), curr=(320, 350))
        assert result == "out"

    def test_bottom_to_top_is_in(self):
        result = self._cross(prev=(320, 350), curr=(320, 100))
        assert result == "in"

    def test_no_crossing_same_side(self):
        # Both points above the line
        result = self._cross(prev=(320, 100), curr=(320, 200))
        assert result is None

    def test_vertical_line_crossing(self):
        # Vertical line at x=320
        from crowd_engine.services.detection_service import _check_line_cross
        result = _check_line_cross(
            prev=(100, 240), curr=(500, 240),
            lx1=320, ly1=0, lx2=320, ly2=480
        )
        assert result in ("in", "out")

    def test_point_on_line_returns_none(self):
        # prev exactly on the line → side == 0
        result = self._cross(prev=(320, 240), curr=(320, 350))
        assert result is None


# ── CameraStats ────────────────────────────────────────────────────────────

class TestCameraStats:
    def test_initial_counts_are_zero(self):
        from crowd_engine.services.detection_service import CameraStats
        stats = CameraStats(camera_id="cam1", label="Test", source="test.mp4")
        assert stats.count_in == 0
        assert stats.count_out == 0
        assert stats.current_count == 0


# ── _CameraWorker reset ────────────────────────────────────────────────────

class TestCameraWorkerReset:
    def test_reset_clears_counts(self):
        from crowd_engine.services.detection_service import _CameraWorker
        worker = _CameraWorker(
            camera_id="cam1", source="test.mp4",
            label="Test", line=None, on_event=None,
        )
        worker.count_in = 10
        worker.count_out = 5
        worker._counted_ids = {1, 2, 3}
        worker.reset_counts()
        assert worker.count_in == 0
        assert worker.count_out == 0
        assert len(worker._counted_ids) == 0


# ── DetectionService (no real cameras) ────────────────────────────────────

class TestDetectionService:
    """Smoke tests for DetectionService lifecycle without real hardware."""

    def _make_service(self):
        """Instantiate DetectionService without starting any threads."""
        from crowd_engine.services.detection_service import DetectionService
        import threading
        svc = DetectionService.__new__(DetectionService)
        svc._workers = {}
        svc._lock = threading.Lock()
        svc._events = []
        return svc

    def test_list_cameras_empty(self):
        svc = self._make_service()
        stats = svc.list_cameras()
        assert isinstance(stats, list)
        assert len(stats) == 0

    def test_get_stats_unknown_camera(self):
        svc = self._make_service()
        result = svc.get_stats("nonexistent")
        assert result is None

    def test_reset_unknown_camera_returns_false(self):
        svc = self._make_service()
        assert svc.reset_camera("nonexistent") is False

    def test_metrics_aggregation(self):
        """get_metrics should aggregate counts across all workers."""
        from crowd_engine.services.detection_service import DetectionService, _CameraWorker

        svc = self._make_service()

        # Build two fake workers with known counts
        w1 = _CameraWorker("a", "a.mp4", "A", None)
        w1.count_in = 10
        w1.count_out = 3
        w1.fps = 25.0
        w1._running = True

        w2 = _CameraWorker("b", "b.mp4", "B", None)
        w2.count_in = 5
        w2.count_out = 2
        w2.fps = 30.0
        w2._running = True

        svc._workers = {"a": w1, "b": w2}

        metrics = svc.get_metrics()
        assert metrics["total_in"] == 15
        assert metrics["total_out"] == 5
        assert metrics["net_occupancy"] == 10
        assert metrics["cameras_total"] == 2
        assert metrics["cameras_active"] == 2
