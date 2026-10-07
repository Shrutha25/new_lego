"""Shared constants for the LEGO assembly guidance system.

Single source of truth for the tag list, capture/model/matcher tunables so
every script (capture, training, inference) agrees on them.
"""

TAGS = ["chassis", "wf_right", "wr_right", "wf_left", "wr_left", "windshield", "roof"]

TAG_LABELS = {
    "chassis": "chassis",
    "wf_right": "front-right wheel",
    "wr_right": "rear-right wheel",
    "wf_left": "front-left wheel",
    "wr_left": "rear-left wheel",
    "windshield": "windshield",
    "roof": "roof",
}

# Wheels grouped by the side of the car they belong to (Section 4) - used by
# the state matcher to know which side is "in view" for partial-mismatch checks.
SIDE_GROUPS = {
    "right": ["wf_right", "wr_right"],
    "left": ["wf_left", "wr_left"],
}

DATA_DIR = "data"
MODELS_DIR = "models"
STATES_PATH = "states.json"

# --- Web viewer (Section 11) ---
WEB_DIR = "web"
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_JPEG_QUALITY = 80  # quality for the MJPEG video_feed stream, not the training data
# Shown once every step is confirmed, instead of repeating the last step's
# model - a separate model of the fully assembled car.
FINAL_MODEL = "assets/models/Car.gltf"
PRODUCT_NAME = "LEGO Race Car"

# --- Capture (Section 5.1) ---
BURST_SIZE = 10
BURST_DURATION_S = 2.0  # ~5 fps across the burst
BURST_GAP_S = 10.0
COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS = 848, 480, 30
DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS = 848, 480, 30
CAMERA_WARMUP_FRAMES = 30  # discarded after pipeline.start() while auto-exposure settles

# --- Model input (Section 6) ---
IMAGE_SIZE = 224
DEPTH_CLIP_MM = 2000.0  # depth values beyond this are clamped before normalizing to [0, 1]

# --- Depth occlusion gating (Section 7) ---
# Defaults only - calibrate.py / occlusion_filter.py's calibration routine
# overrides these with a measured range for the actual camera mount.
BUILD_DEPTH_MIN_MM = 300.0
BUILD_DEPTH_MAX_MM = 900.0
# RealSense reports 0 for "no return" (already excluded everywhere as
# invalid) but some firmware/driver combos also report the raw uint16
# sentinel max (65535) for out-of-range pixels, which - after the normal
# depth_scale * 1000 conversion to mm - looks like a real ~65535mm (65m)
# depth reading unless explicitly excluded too. Left unfiltered, a handful
# of such pixels can drag calibration's build_max_mm and empty-mat baseline
# out to multiple meters, since a straight `> 0` check treats them as valid.
# No real tabletop build area is anywhere near this far, so anything beyond
# it is unambiguously sensor noise, not scene content.
MAX_VALID_DEPTH_MM = 5000.0
HAND_OCCLUSION_MARGIN_MM = 150.0

# --- Temporal smoothing / state matcher (Section 8) ---
# Reverted to 0.7 (Issue #9, 2026-09-11): briefly lowered to 0.5 to help
# under-confident-but-attached tags confirm, but that made it just as easy
# for tags that genuinely AREN'T attached to spuriously cross the bar too -
# confirmed live: wf_left/wr_left locked in as confirmed (one-way, per
# Issue #8) from a transient false reading while the trainee hadn't touched
# the left wheels yet. A single global threshold can't serve both cases at
# once; see TAG_CONFIDENCE_THRESHOLDS below for the per-tag alternative.
CONFIDENCE_THRESHOLD = 0.7
# A tag confirms once it reads above its threshold in at least this
# fraction of the last CONFIRMATION_WINDOW_FRAMES frames - a sliding-window
# majority vote, not a strict consecutive-frame run. Tolerates normal
# frame-to-frame noise (so a genuinely-attached part isn't blocked from ever
# confirming by one stray low-confidence frame) while still requiring
# sustained evidence, so a brief false-positive streak isn't enough to lock
# in a wrong confirmation for the rest of the session.
CONFIRMATION_WINDOW_FRAMES = 15
CONFIRMATION_MIN_FRACTION = 0.8

# Per-tag override for CONFIDENCE_THRESHOLD, checked before falling back to
# the global default above - lets one specific under-confident tag get a
# lower bar without loosening the threshold for every tag (which is what
# caused the wf_left/wr_left regression above). Empty for now: a live
# reading of windshield=0.056 with the windshield actually attached is too
# far below any reasonable threshold to fix this way at all - see the
# windshield note in README.md's tuning section before adding an entry here.
TAG_CONFIDENCE_THRESHOLDS = {}

# Un-confirm was removed in Issue #8 (2026-09-11) because the classifier
# was unreliable enough that it caused more harm than good - genuinely
# attached wheels reading absent for long enough to un-confirm, sending the
# trainee back to redo a finished step. Reinstated for testing (Issue #15,
# 2026-09-11) now that the chassis-implies-wheels and right-implies-left
# shortcuts have been retrained out (Issues #13/#14) - worth re-checking
# whether the classifier is now reliable enough for this to be safe again,
# since one-way confirmation can never reflect a part genuinely being
# removed mid-session. If wheels start falsely un-confirming again, that's
# the signal to revert to one-way (set state_matcher.py's _maybe_unconfirm
# call back to a no-op, or raise these further) rather than tuning blindly.
UNCONFIRM_WINDOW_FRAMES = 60
UNCONFIRM_MAX_FRACTION = 0.25

# Frames of sustained, depth-confirmed "nothing on the mat" before an
# in-progress session auto-resets (Issue #12, 2026-09-11). One-way
# confirmation (above) means nothing clears itself between repeated test
# runs except an explicit restart - in practice that meant every retest
# during a demo-prep session inherited whatever got wrongly confirmed
# (however briefly and spuriously) in the *previous* attempt, which looks
# identical to "jumped straight to step 4" even when the live confidence
# readings are, at that moment, all correctly low. Clearing the mat between
# attempts is already the trainee's natural gesture, not a new step to
# remember - this just makes the system actually notice it, using the same
# depth signal from Issue #11, not the classifier. Deliberately several
# times longer than CONFIRMATION_WINDOW_FRAMES so a brief hand-over-the-mat
# occlusion can never masquerade as a deliberate clear-and-restart.
EMPTY_RESET_FRAMES = 90
