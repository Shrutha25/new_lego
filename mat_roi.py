"""Shared mat ROI (region of interest) loading.

The ROI is a polygon (list of pixel points), not a rectangle - a fixed
camera looking at a mat from an angle usually sees it as a trapezoid, not
an axis-aligned box. crop_to_roi() masks everything outside the polygon to
0, then crops to the polygon's bounding box, so training images and live
inference alike never see background clutter (desk, cables, chairs) or
camera framing drift outside the mat's actual outline.

Run select_mat_roi.py once to calibrate it; re-run it if the camera or mat
genuinely moves.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import config

ROI_OUTLINE_COLOR = (0, 255, 0)

ROI_PATH = Path(config.DATA_DIR) / "mat_roi.json"


def load_mat_roi(path=ROI_PATH):
    """List of (x, y) polygon vertices in pixel coords of the calibrated frame, or None if uncalibrated."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return [tuple(p) for p in d["points"]]


def save_mat_roi(points, path=ROI_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"points": [[int(x), int(y)] for x, y in points]}, f, indent=2)


def _bounding_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def draw_roi_outline(image, roi, label="Place the car inside this outline"):
    """Draws the calibrated mat outline in place on `image` (BGR, modified directly)
    so a trainee can see exactly where the classifier expects the build to sit -
    same purpose as the crop, just visualized instead of applied."""
    if roi is None:
        return image
    cv2.polylines(image, [np.array(roi, dtype=np.int32)],
                  isClosed=True, color=ROI_OUTLINE_COLOR, thickness=2)
    if label:
        x0, y0, _, _ = _bounding_box(roi)
        cv2.putText(image, label, (x0, max(0, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROI_OUTLINE_COLOR, 1, cv2.LINE_AA)
    return image


def crop_to_roi(image, roi):
    """image: HxW or HxWxC array. roi: list of (x, y) polygon vertices, or None (no-op)."""
    if roi is None:
        return image
    x0, y0, x1, y1 = _bounding_box(roi)
    cropped = image[y0:y1, x0:x1].copy()

    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    shifted = np.array([[x - x0, y - y0] for x, y in roi], dtype=np.int32)
    cv2.fillPoly(mask, [shifted], 255)
    cropped[mask == 0] = 0

    return cropped
