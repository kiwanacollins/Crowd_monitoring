"""
Unit tests for DetectionService core logic.

Camera workers and the YOLO model are mocked so these tests run without
any video files, GPU, or network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── _update_crossing_state geometry ───────────────────────────────────────

class TestCheckLineCross:
    """Tests for the dwell-confirmed crossing logic (replaces _check_line_cross)."""

    def _cross_sequence(self, positions, lx1=0, ly1=240, lx2=640, ly2=240):
        """Walk a track through a sequence of foot positions and collect events."""
        from crowd_engine.services.detection_service import _update_crossing_state
        states: dict = {}
        events = []
        for i, (px, py) in enumerate(positions):
            r = _update_crossing_state(states, 1, (px, py), lx1, ly1, lx2, ly2)
            if r:
                events.append(r)
        return events

    def test_top_to_bottom_is_out(self):
        # Walk from above (y=100) to clearly below (y=350) over several frames
        # Dwell requires 3 consecutive frames on new side before confirming
        positions = [(320, 100), (320, 150), (320, 200),
                     (320, 250), (320, 280), (320, 350)]
        events = self._cross_sequence(positions)
        assert "out" in events
        assert "in" not in events

    def test_bottom_to_top_is_in(self):
        positions = [(320, 350), (320, 300), (320, 260),
                     (320, 220), (320, 180), (320, 100)]
        events = self._cross_sequence(positions)
        assert "in" in events
        assert "out" not in events

    def test_no_crossing_same_side(self):
        # Both points above the line — no crossing
        positions = [(320, 100), (320, 150), (320, 200)]
        events = self._cross_sequence(positions)
        assert events == []

    def test_vertical_line_crossing(self):
        # Vertical line at x=320; walk left→right
        from crowd_engine.services.detection_service import _update_crossing_state
        states: dict = {}
        events = []
        for x in [100, 150, 200, 250, 320, 380, 420, 460]:
            r = _update_crossing_state(states, 1, (x, 240), 320, 0, 320, 480)
            if r:
                events.append(r)
        assert len(events) == 1
        assert events[0] in ("in", "out")

    def test_jitter_does_not_double_count(self):
        # Oscillate right at the line — should count at most once per side
        positions = (
            [(320, 100)] * 5         # start firmly above
            + [(320, 250), (320, 230), (320, 250), (320, 230), (320, 250)]  # jitter
            + [(320, 260), (320, 270), (320, 280)]   # settle below
        )
        events = self._cross_sequence(positions)
        assert events.count("out") <= 1    # no double-counting


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
        worker._cross_states = {1: {"side": True, "candidate": None, "dwell": 0}}
        worker.reset_counts()
        assert worker.count_in == 0
        assert worker.count_out == 0
        assert len(worker._cross_states) == 0


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
