"""Depth-based hand/occlusion/presence gating (Section 7).

Cheap pre-filter that runs on raw depth data only, no model needed. Skips a
frame if something (typically a hand) sits closer to the camera than the
calibrated build area, if the scene's depth doesn't resemble the build area
at all (e.g. the camera got bumped or repositioned), or if nothing at all
has been placed on the mat yet - either way, the frame isn't fed to the
classifier, which was never trained to handle input that isn't even a view
of the build area, and shouldn't be trusted to decide "is anything even
there" purely from its own (noisy) tag confidences.

Run this file directly to calibrate the expected build-area depth range
(and a per-pixel empty-mat baseline, for the presence check) against the
real camera mount before using it live:

    python occlusion_filter.py --frames 30
"""
import json
from pathlib import Path

import numpy as np

import config
from mat_roi import crop_to_roi

CALIBRATION_PATH = Path(config.DATA_DIR) / "build_depth_range.json"
BASELINE_PATH = Path(config.DATA_DIR) / "build_depth_baseline.npy"

# How much closer than the calibrated empty-mat baseline a pixel needs to
# read, at minimum, to count as "something is physically sitting there" -
# comfortably above normal depth-sensor noise (a few mm) so a flat, empty
# mat never trips this on its own.
PRESENCE_MARGIN_MM = 15.0
# Fraction of the (valid) ROI that needs to read as "elevated" before we
# call it "something's on the mat." Deliberately tiny - this only needs to
# catch "literally nothing here yet," not identify *what* is there (that's
# the classifier's job) or how much of the build is done.
PRESENCE_MIN_FRACTION = 0.01


def frame_skip_reason(depth_image_mm, roi=None, build_min_mm=None, build_max_mm=None, margin_mm=None,
                       baseline_depth_mm=None, presence_margin_mm=None, presence_min_fraction=None):
    """None if this frame looks like a normal, non-empty view of the calibrated build area,
    otherwise a short reason it should be skipped (not fed to the classifier).

    Checks, in order: something closer than build_min_mm is treated as a hand
    blocking the view (the original occlusion check); the scene's *typical*
    depth landing well outside [build_min_mm, build_max_mm] entirely means
    the camera probably isn't even aimed at the build area (e.g. bumped or
    repositioned) - a hand-only check can't catch this, since a wrongly-aimed
    camera isn't "occluded", it's just looking at something else, and
    blindly classifying it produces meaningless output; and, if a per-pixel
    empty-mat baseline was calibrated, whether *nothing* on the mat reads
    meaningfully closer than that baseline - i.e. the mat is still bare, so
    there's no point asking the classifier what's on it (Issue #7,
    2026-09-10: a currently-empty mat could still occasionally get "chassis"
    confirmed purely from classifier noise; this catches that independently
    of the model, since it never even reaches the classifier).
    """
    build_min_mm = config.BUILD_DEPTH_MIN_MM if build_min_mm is None else build_min_mm
    build_max_mm = config.BUILD_DEPTH_MAX_MM if build_max_mm is None else build_max_mm
    margin_mm = config.HAND_OCCLUSION_MARGIN_MM if margin_mm is None else margin_mm
    presence_margin_mm = PRESENCE_MARGIN_MM if presence_margin_mm is None else presence_margin_mm
    presence_min_fraction = PRESENCE_MIN_FRACTION if presence_min_fraction is None else presence_min_fraction

    region = crop_to_roi(depth_image_mm, roi)
    valid = region[(region > 0) & (region < config.MAX_VALID_DEPTH_MM)]
    if valid.size == 0:
        return "no depth data in view"

    nearest = float(valid.min())
    if nearest < (build_min_mm - margin_mm):
        return "hand in view - waiting..."

    median = float(np.median(valid))
    if median < (build_min_mm - margin_mm) or median > (build_max_mm + margin_mm):
        return "camera doesn't look aimed at the build area"

    if baseline_depth_mm is not None:
        if object_present_fraction(depth_image_mm, baseline_depth_mm, roi, presence_margin_mm) < presence_min_fraction:
            return "nothing on the mat yet"

    return None


def is_occluded(depth_image_mm, roi=None, build_min_mm=None, build_max_mm=None, margin_mm=None):
    return frame_skip_reason(depth_image_mm, roi, build_min_mm, build_max_mm, margin_mm) is not None


def object_present_fraction(depth_image_mm, baseline_depth_mm, roi=None, presence_margin_mm=PRESENCE_MARGIN_MM):
    """Fraction of the (masked) ROI where the current frame reads meaningfully
    closer than the calibrated empty-mat baseline at that same pixel - i.e.
    something is physically sitting there that wasn't there during
    calibration. Compares per-pixel (not just an overall min/median), so an
    object placed anywhere in the ROI is caught, not just one near the mat's
    closest edge."""
    current = crop_to_roi(depth_image_mm, roi)
    baseline = crop_to_roi(baseline_depth_mm, roi)
    valid = ((current > 0) & (current < config.MAX_VALID_DEPTH_MM)
             & (baseline > 0) & (baseline < config.MAX_VALID_DEPTH_MM))
    if not np.any(valid):
        return 0.0
    elevated = valid & ((baseline - current) > presence_margin_mm)
    return float(np.count_nonzero(elevated)) / float(np.count_nonzero(valid))


def calibrate_build_depth_range(depth_frames_mm, roi=None, percentile_low=5, percentile_high=95):
    """depth_frames_mm: depth images (mm) from a clean, hand-free view of the empty build area."""
    all_valid = []
    for frame in depth_frames_mm:
        region = crop_to_roi(frame, roi)
        all_valid.append(region[(region > 0) & (region < config.MAX_VALID_DEPTH_MM)])
    stacked = np.concatenate(all_valid)
    return float(np.percentile(stacked, percentile_low)), float(np.percentile(stacked, percentile_high))


def calibrate_baseline_depth(depth_frames_mm):
    """Per-pixel median depth (mm) across calibration frames, full frame (uncropped) -
    cropped to whatever ROI is active at check time, so it stays valid if the
    ROI is later re-calibrated without re-running this capture."""
    stacked = np.stack(depth_frames_mm, axis=0)
    # 0 means "no return" and values >= MAX_VALID_DEPTH_MM are the sensor's
    # out-of-range sentinel, not real depth - exclude both from the median
    # rather than letting them drag the baseline toward 0 or toward tens of
    # meters.
    stacked = np.where((stacked > 0) & (stacked < config.MAX_VALID_DEPTH_MM), stacked, np.nan)
    with np.errstate(all="ignore"):
        baseline = np.nanmedian(stacked, axis=0)
    return np.nan_to_num(baseline, nan=0.0).astype(np.float32)


def load_calibration(path=CALIBRATION_PATH, baseline_path=BASELINE_PATH):
    """Returns (build_min_mm, build_max_mm, baseline_depth_mm). baseline_depth_mm
    is None if calibration hasn't been (re-)run since the presence check was added -
    the presence check is then simply skipped, same as before this feature existed."""
    if not Path(path).exists():
        return config.BUILD_DEPTH_MIN_MM, config.BUILD_DEPTH_MAX_MM, None
    with open(path) as f:
        data = json.load(f)
    baseline = np.load(baseline_path) if Path(baseline_path).exists() else None
    return data["build_min_mm"], data["build_max_mm"], baseline


def save_calibration(build_min_mm, build_max_mm, baseline_depth_mm, path=CALIBRATION_PATH, baseline_path=BASELINE_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"build_min_mm": build_min_mm, "build_max_mm": build_max_mm}, f, indent=2)
    np.save(baseline_path, baseline_depth_mm)


if __name__ == "__main__":
    import argparse

    from capture import build_pipeline, get_aligned_frames
    from mat_roi import load_mat_roi

    parser = argparse.ArgumentParser(
        description="Calibrate the expected build-area depth range from a clean, hand-free view."
    )
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    mat_roi = load_mat_roi()
    if mat_roi is None:
        print("No mat ROI calibrated (run select_mat_roi.py first for a tighter, more accurate "
              "reading) - measuring depth over the full frame instead.")

    pipeline, align, depth_scale = build_pipeline()
    print(f"Point the camera at the empty build area (no hands) - capturing {args.frames} frames...")
    depth_frames_mm = []
    try:
        while len(depth_frames_mm) < args.frames:
            _, depth_image = get_aligned_frames(pipeline, align)
            if depth_image is None:
                continue
            depth_frames_mm.append(depth_image.astype(np.float32) * depth_scale * 1000.0)
    finally:
        pipeline.stop()

    build_min_mm, build_max_mm = calibrate_build_depth_range(depth_frames_mm, roi=mat_roi)
    baseline_depth_mm = calibrate_baseline_depth(depth_frames_mm)
    save_calibration(build_min_mm, build_max_mm, baseline_depth_mm)
    print(f"Calibrated build depth range: {build_min_mm:.0f}mm - {build_max_mm:.0f}mm -> saved to {CALIBRATION_PATH}")
    print(f"Calibrated empty-mat depth baseline -> saved to {BASELINE_PATH}")
