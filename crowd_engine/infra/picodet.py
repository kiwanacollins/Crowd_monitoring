"""
PicoDet-M 320 ONNX Inference
=============================
Lightweight real-time person detector optimised for CPU.

Model: PP-PicoDet-M 320×320 (COCO, no-postprocess variant)
~3M parameters, ~25–60 FPS on modern CPU cores.

The "no_postprocess" ONNX exports raw feature maps for all 4 FPN levels:
  outputs[0..3]  — classification logits  [1, H*W, 80]
  outputs[4..7]  — DFL box distributions  [1, H*W, 4*(reg_max+1)]

Post-processing implemented here in pure NumPy:
  1. Sigmoid scores
  2. DFL integral (softmax → weighted sum) to get (l,t,r,b) distances
  3. Anchor grid decode → (x1,y1,x2,y2)
  4. Scale back to original frame resolution
  5. Greedy NMS per class
"""
from __future__ import annotations

import os
import urllib.request
from typing import Dict, List, Optional

import cv2
import numpy as np

from crowd_engine.infra.logger import get_logger

log = get_logger(__name__)

# ── Model constants ────────────────────────────────────────────────────────
_MODEL_FILENAME = "picodet_m_320_coco_no_postprocess.onnx"
_MODEL_URL = (
    "https://bj.bcebos.com/paddlehub/fastdeploy/"
    + _MODEL_FILENAME
)
_INPUT_SIZE = 320
_STRIDES    = [8, 16, 32, 64]
_REG_MAX    = 7                  # DFL regression max (8 bins: 0..7)

# ImageNet normalisation (RGB order)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _make_anchor_grids() -> List[np.ndarray]:
    """
    Precompute (cx, cy) centre-point grids for each FPN stride level.
    For 320 input: 40×40, 20×20, 10×10, 5×5 grids.
    """
    grids = []
    for stride in _STRIDES:
        n = _INPUT_SIZE // stride           # grid side length
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cx = (xs.flatten() + 0.5).astype(np.float32)   # cell centre x
        cy = (ys.flatten() + 0.5).astype(np.float32)   # cell centre y
        grids.append(np.stack([cx, cy], axis=1))        # [n², 2]
    return grids


_ANCHOR_GRIDS: List[np.ndarray] = _make_anchor_grids()


class PicoDetM:
    """
    PP-PicoDet-M 320 ONNX wrapper.

    Usage::

        det = PicoDetM()
        if det.load():
            boxes = det.detect(frame)
    """

    def __init__(self, model_path: str = _MODEL_FILENAME) -> None:
        self._path     = model_path
        self._session  = None          # onnxruntime.InferenceSession
        self._inp_name = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Download the ONNX model (if missing) and load it.  Returns True on success."""
        try:
            import onnxruntime as ort
        except ImportError:
            log.warning("onnxruntime not installed — PicoDet-M unavailable")
            return False

        if not os.path.exists(self._path):
            log.info("Downloading PicoDet-M → %s  (≈9 MB) …", self._path)
            try:
                urllib.request.urlretrieve(_MODEL_URL, self._path)
                size_kb = os.path.getsize(self._path) // 1024
                log.info("PicoDet-M downloaded (%d KB)", size_kb)
            except Exception as exc:
                log.error("PicoDet-M download failed: %s", exc)
                if os.path.exists(self._path):
                    os.remove(self._path)  # remove partial file
                return False

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2           # 2 threads per camera is optimal on CPU
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                self._path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._inp_name = self._session.get_inputs()[0].name
            log.info("PicoDet-M loaded ✔  (%s)", self._path)
            return True
        except Exception as exc:
            log.error("PicoDet-M session init failed: %s", exc)
            return False

    # ── Inference ─────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        nms_iou: float = 0.45,
        target_class: int = 0,          # 0 = "person" in COCO
    ) -> List[Dict]:
        """
        Run PicoDet-M on a BGR frame.

        Returns a list of detection dicts::

            [{"x1": int, "y1": int, "x2": int, "y2": int, "conf": float}, ...]

        Coordinates are in original frame pixels.
        """
        if self._session is None:
            return []

        orig_h, orig_w = frame.shape[:2]

        # ── Pre-process ─────────────────────────────────────────────────
        img = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        inp = np.transpose(img, (2, 0, 1))[None]   # NCHW [1,3,320,320]

        # ── Inference ───────────────────────────────────────────────────
        outs = self._session.run(None, {self._inp_name: inp})
        # outs[0..3] = score maps at 4 strides
        # outs[4..7] = box distribution maps at 4 strides

        # ── Decode all FPN levels ────────────────────────────────────────
        all_boxes:  List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        for lvl, stride in enumerate(_STRIDES):
            score_logits = outs[lvl][0]       # [N, 80]  raw logits
            box_dists    = outs[4 + lvl][0]   # [N, 32]  DFL distributions
            anchors      = _ANCHOR_GRIDS[lvl] # [N,  2]  (cx, cy)

            # Sigmoid per cell for target class only (avoids processing 80 classes)
            person_scores = _sigmoid(score_logits[:, target_class])  # [N]
            mask = person_scores > conf
            if not mask.any():
                continue

            boxes = _dfl_decode(box_dists[mask], anchors[mask], stride)
            all_boxes.append(boxes)
            all_scores.append(person_scores[mask])

        if not all_boxes:
            return []

        boxes  = np.concatenate(all_boxes,  axis=0)   # [M, 4]
        scores = np.concatenate(all_scores, axis=0)   # [M]

        # ── Scale to original resolution ─────────────────────────────────
        sx = orig_w / _INPUT_SIZE
        sy = orig_h / _INPUT_SIZE
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * sx).clip(0, orig_w)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * sy).clip(0, orig_h)

        keep = _nms(boxes, scores, nms_iou)
        return [
            {
                "x1": int(boxes[k, 0]), "y1": int(boxes[k, 1]),
                "x2": int(boxes[k, 2]), "y2": int(boxes[k, 3]),
                "conf": float(scores[k]),
            }
            for k in keep
        ]


# ── Pure-NumPy helpers (module-level for clarity) ─────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _dfl_decode(
    box_dists: np.ndarray,    # [N, 32]
    anchors:   np.ndarray,    # [N,  2]  (cx, cy) in feature-map cell units
    stride:    int,
) -> np.ndarray:
    """
    Distribution Focal Loss decode.
    Converts raw DFL distributions → (x1,y1,x2,y2) in model-input pixels.
    """
    project = np.arange(_REG_MAX + 1, dtype=np.float32)   # [0,1,…,7]
    dist = box_dists.reshape(-1, 4, _REG_MAX + 1)          # [N, 4, 8]
    prob = _softmax(dist)                                   # [N, 4, 8]
    ltrb = (prob * project).sum(axis=-1)                   # [N, 4]  (l,t,r,b)

    cx = anchors[:, 0] * stride    # pixel centre x
    cy = anchors[:, 1] * stride    # pixel centre y
    x1 = cx - ltrb[:, 0] * stride
    y1 = cy - ltrb[:, 1] * stride
    x2 = cx + ltrb[:, 2] * stride
    y2 = cy + ltrb[:, 3] * stride
    return np.stack([x1, y1, x2, y2], axis=1)   # [N, 4]


def _nms(
    boxes:  np.ndarray,   # [N, 4]  x1y1x2y2
    scores: np.ndarray,   # [N]
    iou_thr: float,
) -> List[int]:
    """Greedy NMS — returns kept indices sorted by score descending."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou < iou_thr]
    return keep
