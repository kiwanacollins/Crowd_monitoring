"""
ByteTrack Lite — Minimal multi-object tracker
==============================================
Pure NumPy + SciPy implementation of the ByteTrack two-stage IOU tracker.
No torch, no torch-dependent packages required.

Algorithm (simplified ByteTrack):
  1. Predict each active track's next position with a constant-velocity
     Kalman filter (8-dim state: cx,cy,ar,h,vx,vy,var,vh).
  2. HIGH-score detections (conf ≥ track_thresh) are matched to active
     tracks first (IOU cost, Hungarian algorithm).
  3. Remaining HIGH-score detections try to match LOST tracks.
  4. LOW-score detections only match confirmed tracks (guards against
     mis-firing on background).
  5. Unmatched high-conf detections start new TENTATIVE tracks.
  6. Tracks inactive for more than track_buffer frames are DELETED.

Output (per call to update()):
  np.ndarray [M, 8]  columns: x1, y1, x2, y2, track_id, conf, class_id, 0
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Kalman Filter (constant velocity, XYAH state) ─────────────────────────

class _KF:
    """
    Minimal Kalman filter for bounding-box tracking.
    State:  [cx, cy, a, h, vcx, vcy, va, vh]
    Measurement: [cx, cy, a, h]
    """

    _F = np.eye(8, dtype=np.float64)
    _F[:4, 4:] = np.eye(4)               # transition: add velocity to pos

    _H = np.eye(4, 8, dtype=np.float64)  # measurement: observe first 4 dims

    def __init__(self) -> None:
        self.x  = np.zeros((8, 1), np.float64)   # state
        self.P  = np.eye(8, dtype=np.float64) * 10.0  # covariance
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> None:
        """Set initial state from xyah measurement [cx,cy,a,h]."""
        self.x[:4, 0] = measurement
        self.x[4:, 0] = 0.0
        h = measurement[3]
        std = [
            2 * self._std_weight_pos * h,
            2 * self._std_weight_pos * h,
            1e-2,
            2 * self._std_weight_pos * h,
            10 * self._std_weight_vel * h,
            10 * self._std_weight_vel * h,
            1e-5,
            10 * self._std_weight_vel * h,
        ]
        self.P = np.diag(np.array(std, np.float64) ** 2)

    def predict(self) -> np.ndarray:
        """Kalman predict step. Returns predicted measurement [cx,cy,a,h]."""
        h = abs(float(self.x[3, 0]))
        std = [
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            1e-2,
            self._std_weight_pos * h,
            self._std_weight_vel * h,
            self._std_weight_vel * h,
            1e-5,
            self._std_weight_vel * h,
        ]
        Q = np.diag(np.array(std, np.float64) ** 2)
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + Q
        return self._H @ self.x

    def update(self, measurement: np.ndarray) -> None:
        """Kalman update step given xyah measurement."""
        h = abs(float(self.x[3, 0]))
        std = [
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            1e-1,
            self._std_weight_pos * h,
        ]
        R = np.diag(np.array(std, np.float64) ** 2)
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        z = measurement.reshape(4, 1)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(8) - K @ self._H) @ self.P

    @property
    def xyah(self) -> np.ndarray:
        return self._H @ self.x


# ── Track lifecycle ────────────────────────────────────────────────────────

class _TrackState:
    New      = 0   # just initialised, not yet confirmed
    Tracked  = 1
    Lost     = 2
    Removed  = 3


class _Track:
    _id_counter = 1

    def __init__(self, xyxy: np.ndarray, conf: float, cls: float) -> None:
        self.id         = _Track._id_counter
        _Track._id_counter += 1
        self.conf       = conf
        self.cls        = cls
        self.state      = _TrackState.New
        self.hits       = 1
        self.age        = 0          # frames since last detection
        self.kf         = _KF()
        self.kf.initiate(_xyxy2xyah(xyxy))
        self._xyxy      = xyxy.copy()

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    def predict(self) -> None:
        xyah = self.kf.predict().flatten()
        self._xyxy = _xyah2xyxy(xyah)
        self.age += 1

    def update(self, xyxy: np.ndarray, conf: float) -> None:
        self.kf.update(_xyxy2xyah(xyxy))
        self._xyxy = xyxy.copy()
        self.conf  = conf
        self.hits += 1
        self.age   = 0
        self.state = _TrackState.Tracked


# ── Coordinate helpers ─────────────────────────────────────────────────────

def _xyxy2xyah(b: np.ndarray) -> np.ndarray:
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    h  = b[3] - b[1]
    a  = (b[2] - b[0]) / max(1.0, h)
    return np.array([cx, cy, a, h], np.float64)


def _xyah2xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = float(m[0]), float(m[1]), float(m[2]), float(m[3])
    w  = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def _iou_matrix(bb: np.ndarray, bb2: np.ndarray) -> np.ndarray:
    """IOU matrix [len(bb), len(bb2)]."""
    if len(bb) == 0 or len(bb2) == 0:
        return np.zeros((len(bb), len(bb2)), np.float32)
    x1 = np.maximum(bb[:, 0:1],  bb2[:, 0])
    y1 = np.maximum(bb[:, 1:2],  bb2[:, 1])
    x2 = np.minimum(bb[:, 2:3],  bb2[:, 2])
    y2 = np.minimum(bb[:, 3:4],  bb2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (bb[:, 2] - bb[:, 0])  * (bb[:, 3] - bb[:, 1])
    area2 = (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def _hungarian(cost: np.ndarray, thresh: float
               ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian assignment on a cost matrix (lower = better match).
    Returns (matched_pairs, unmatched_rows, unmatched_cols).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(cost)
    matched    = [(int(r), int(c)) for r, c in zip(ri, ci) if cost[r, c] <= thresh]
    matched_r  = {r for r, _ in matched}
    matched_c  = {c for _, c in matched}
    unmatched_r = [i for i in range(cost.shape[0]) if i not in matched_r]
    unmatched_c = [i for i in range(cost.shape[1]) if i not in matched_c]
    return matched, unmatched_r, unmatched_c


# ── Public ByteTrack class ─────────────────────────────────────────────────

class ByteTrack:
    """
    Two-stage ByteTrack multi-object tracker.

    Parameters
    ----------
    track_thresh : float
        Minimum confidence to start a new track.
    track_buffer : int
        Frames a lost track is kept before deletion.
    match_thresh : float
        Minimum IOU (as 1-IOU cost) for a valid match.
    frame_rate : int
        Source FPS (used to scale track_buffer).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int   = 30,
        match_thresh: float = 0.8,
        frame_rate:   int   = 25,
    ) -> None:
        self.track_thresh = track_thresh
        self.max_age      = max(1, int(track_buffer * frame_rate / 30))
        self.match_thresh = match_thresh   # 1-IOU threshold

        self._tracked: List[_Track] = []
        self._lost:    List[_Track] = []
        self._frame_id = 0

    def reset(self) -> None:
        self._tracked.clear()
        self._lost.clear()
        self._frame_id = 0
        _Track._id_counter = 1          # reset IDs on video loop

    def update(
        self,
        dets: np.ndarray,        # [N, 6]  x1 y1 x2 y2 conf class_id
        frame: np.ndarray,       # BGR frame (unused — kept for API compat)
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Returns
        -------
        np.ndarray  [M, 8]
            Columns: x1, y1, x2, y2, track_id, conf, class_id, 0
        """
        self._frame_id += 1

        # ── Predict all tracks ─────────────────────────────────────────
        for t in self._tracked + self._lost:
            t.predict()

        # ── Split dets by confidence tier ─────────────────────────────
        if len(dets) == 0:
            high_dets  = np.empty((0, 6), np.float32)
            low_dets   = np.empty((0, 6), np.float32)
        else:
            mask       = dets[:, 4] >= self.track_thresh
            high_dets  = dets[mask]
            low_dets   = dets[~mask]

        # ── Stage 1: match high-conf dets to active tracks ─────────────
        active_tracks = [t for t in self._tracked
                         if t.state == _TrackState.Tracked]

        matches1, unm_trk1, unm_det1 = self._associate(
            active_tracks, high_dets, 1.0 - self.match_thresh
        )
        for ti, di in matches1:
            active_tracks[ti].update(high_dets[di, :4], high_dets[di, 4])

        # ── Stage 2: match remaining high-conf dets to lost tracks ─────
        r_high    = high_dets[unm_det1]
        r_trks    = [active_tracks[i] for i in unm_trk1] + self._lost

        matches2, unm_trk2, unm_det2 = self._associate(
            r_trks, r_high, 0.5           # looser IOU for lost tracks
        )
        for ti, di in matches2:
            r_trks[ti].update(r_high[di, :4], r_high[di, 4])
            r_trks[ti].state = _TrackState.Tracked
            if r_trks[ti] in self._lost:
                self._lost.remove(r_trks[ti])
                self._tracked.append(r_trks[ti])

        # ── Stage 3: match low-conf dets to still-unmatched active tracks
        still_unm = [r_trks[i] for i in unm_trk2
                     if r_trks[i].state == _TrackState.Tracked]
        matches3, unm_trk3, _ = self._associate(still_unm, low_dets, 0.5)
        for ti, di in matches3:
            still_unm[ti].update(low_dets[di, :4], low_dets[di, 4])

        # ── Mark unmatched active tracks as Lost ──────────────────────
        for i in unm_trk3:
            t = still_unm[i]
            t.state = _TrackState.Lost
            if t in self._tracked:
                self._tracked.remove(t)
            self._lost.append(t)

        # ── Spawn new tracks for unmatched high-conf dets ─────────────
        for di in unm_det2:
            d  = r_high[di]
            nt = _Track(d[:4], float(d[4]), float(d[5]))
            nt.state = _TrackState.Tracked
            self._tracked.append(nt)

        # ── Remove stale lost tracks ──────────────────────────────────
        self._lost = [t for t in self._lost if t.age <= self.max_age]

        # ── Collect output ─────────────────────────────────────────────
        out: List[np.ndarray] = []
        for t in self._tracked:
            if t.state == _TrackState.Tracked and t.hits >= 1:
                b = t.xyxy
                out.append(np.array(
                    [b[0], b[1], b[2], b[3], t.id, t.conf, t.cls, 0.0],
                    dtype=np.float32,
                ))

        return np.array(out, np.float32) if out else np.empty((0, 8), np.float32)

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _associate(
        tracks: List[_Track],
        dets:   np.ndarray,
        cost_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        trk_boxes = np.array([t.xyxy for t in tracks], np.float32)
        det_boxes  = dets[:, :4].astype(np.float32)
        iou = _iou_matrix(trk_boxes, det_boxes)
        cost = 1.0 - iou
        return _hungarian(cost, cost_thresh)
