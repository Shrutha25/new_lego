"""Interactive mat ROI calibration (polygon).

Grabs a live frame and lets you click points tracing the mat's actual
outline - useful since a fixed camera looking at a mat from an angle
usually sees it as a trapezoid or irregular shape, not an axis-aligned
rectangle. Saves the polygon so train.py and inference_loop.py both mask
out everything outside it before doing anything else - background clutter
and camera framing drift outside the mat can no longer affect a prediction.

Run this once before your first training run, and again any time the
camera or mat is physically moved.

Controls:
    left click  - add a point (trace the mat's edge in order)
    u           - undo last point
    ENTER       - finish and save (need at least 3 points)
    ESC / q     - cancel without saving

Usage:
    python select_mat_roi.py
"""
import numpy as np
import cv2

import config
from capture import build_pipeline, get_aligned_frames
from mat_roi import save_mat_roi

points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def draw(base_image):
    display = base_image.copy()
    if len(points) > 1:
        cv2.polylines(display, [np.array(points, dtype=np.int32)],
                      isClosed=len(points) > 2, color=(0, 255, 0), thickness=2)
    for p in points:
        cv2.circle(display, p, 4, (0, 0, 255), -1)
    cv2.putText(display, "click: add point  u: undo  ENTER: save  ESC/q: cancel", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return display


def main():
    pipeline, align, depth_scale = build_pipeline()
    print("Warming up camera...")
    for _ in range(config.CAMERA_WARMUP_FRAMES):
        get_aligned_frames(pipeline, align)

    color_image, _ = get_aligned_frames(pipeline, align)
    pipeline.stop()

    if color_image is None:
        raise SystemExit("Could not get a frame from the camera.")

    window = "select mat ROI - click points around the mat outline"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("Click points tracing the mat's outline, in order around the edge.")
    print("u = undo last point, ENTER = finish (need >= 3 points), ESC/q = cancel")

    while True:
        cv2.imshow(window, draw(color_image))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled - nothing saved.")
        if key == ord("u") and points:
            points.pop()
        if key in (13, 10):  # ENTER
            if len(points) >= 3:
                break
            print("Need at least 3 points before finishing.")

    cv2.destroyAllWindows()
    save_mat_roi(points)
    print(f"Saved mat ROI ({len(points)} points) -> {config.DATA_DIR}/mat_roi.json")
    print("Re-run build_manifest.py --contact-sheets to eyeball old captures if you want to "
          "sanity-check the mat stayed within this outline across your existing sessions.")


if __name__ == "__main__":
    main()
