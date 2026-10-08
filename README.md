# LEGO Assembly Guidance System

Guides a trainee through a LEGO car assembly using a fixed RealSense D455 camera: recognizes which components are attached (in any order), catches skipped-ahead steps, catches partial completion (e.g. one wheel of a side missing), and ignores hands / held-but-unattached parts.

Full design rationale lives in `lego_assembly_system_plan.md`. This file is just the run-book: what to type, in what order.

## 1. Install dependencies

```
python -m pip install -r requirements.txt
```

Installs `pyrealsense2`, `opencv-python`, `numpy`, `torch`, `torchvision`, `pillow`.

## 2. Files

| File | Purpose |
|---|---|
| `config.py` | Shared constants — tags, capture timing, thresholds. Tune here, not in the scripts. |
| `states.json` | The assembly step table the matcher walks. |
| `capture.py` | RealSense burst capture for training data. |
| `build_manifest.py` | Turns captured folders into a labeled `manifest.csv`. |
| `train.py` | Trains the RGB-D multi-label classifier. |
| `occlusion_filter.py` | Depth-based hand-occlusion gate, plus a calibration routine. |
| `state_matcher.py` | Rule-based state tracker (persistent, order-independent). |
| `inference_loop.py` | Live loop: camera → occlusion filter → classifier → state matcher → on-screen guidance. |

## 3. Calibrate the occlusion gate

Point the camera at the **empty build area** (no hands, no parts) and run:

```
python occlusion_filter.py --frames 30
```

This measures the expected depth range of the build surface (`data/build_depth_range.json`) and a per-pixel empty-mat depth baseline (`data/build_depth_baseline.npy`), used to gate frames where nothing has been placed on the mat yet - independent of the classifier, so a currently-empty mat can't get a tag confirmed purely from model noise. `inference_loop.py` loads both automatically; if it's never run, the depth-range check falls back to the defaults in `config.py` and the "nothing on the mat yet" check is skipped entirely (with a console warning). Re-run this if you move the camera or the build surface - **and re-run it once now** even if you calibrated before this check existed, since the baseline file didn't get saved until this feature was added.

Optional `--roi X0 Y0 X1 Y1` restricts the measurement to a pixel region instead of the full frame.

## 4. Capture training data

One physical assembly state per session — don't change the build mid-capture. Each session runs bursts of frames with a pause between them so you can reposition/vary lighting/vary hand position:

```
python capture.py --state <folder_name> --bursts 22
```

- `--bursts` — how many bursts to shoot (plan target: ~20-25 per state → ~200-250 images).
- `--burst-size`, `--burst-duration`, `--gap` — override the defaults from `config.py` (10 frames/burst, ~2s/burst, 10s gap) if needed.
- Press `q` any time to stop early.

`<folder_name>` must be one of the states `build_manifest.py` knows how to label (see `FOLDER_TAGS` in that file). Run these in order, matching the live workflow (right side first, then left):

```
python capture.py --state s0_empty              --bursts 20   # background only, no chassis
python capture.py --state s1_chassis             --bursts 22   # chassis alone
python capture.py --state s2a_partial_right       --bursts 20   # chassis + front-right wheel only
python capture.py --state s2a_partial_right_rear  --bursts 20   # chassis + rear-right wheel only
python capture.py --state s2a_right_complete      --bursts 22   # chassis + both right wheels
python capture.py --state s2b_partial_left        --bursts 20   # chassis + front-left wheel only
python capture.py --state s2b_partial_left_rear   --bursts 20   # chassis + rear-left wheel only
python capture.py --state s2b_left_complete       --bursts 22   # chassis + both left wheels
python capture.py --state s3_windshield           --bursts 22   # + windshield
python capture.py --state s4_roof                 --bursts 22   # + roof

# Negative captures - part visible in hand, near but NOT attached (critical, don't skip):
python capture.py --state neg_wheel_held          --bursts 20
python capture.py --state neg_windshield_held     --bursts 20
python capture.py --state neg_roof_held           --bursts 20

# Negative captures - camera pointed at something that isn't a lego part at
# all (critical, don't skip - without this "chassis" fires on almost anything
# unfamiliar, e.g. a ceiling or a t-shirt print, since it's otherwise only
# ever seen "empty mat" or "a hand" as chassis=0):
python capture.py --state neg_random_object       --bursts 25
```

For `neg_random_object`, place the object **on the mat, inside the calibrated ROI**, at the same framing/distance as a normal build - not just pointed elsewhere in the room. The classifier only ever sees the region `crop_to_roi()` crops to (see `mat_roi.py`); the ROI already excludes background clutter outside the mat, so it's not the source of this bug - the model has just never seen "something is sitting in the mat area, but it isn't the chassis" as a negative example, only "the mat is empty" or "a bare hand is over it." Vary the object every few bursts - t-shirts/fabric, books, mugs, other toys - so the model learns "on the mat" isn't the same as "chassis," rather than learning to recognize one specific decoy object.

After each session, skim the saved frames in `data/<folder_name>/` and delete any where the relevant part is cut off out of frame — a cut-off part could get mislabeled as absent.

## 5. Build the labeled manifest

```
python build_manifest.py --data-dir data --out manifest.csv
```

Walks every folder under `data/`, looks up its tag vector from the folder name, and writes `manifest.csv` (image paths + 7 binary tag columns). Any folder it doesn't recognize is skipped with a warning — add it to `FOLDER_TAGS` in `build_manifest.py` if that happens.

## 6. Check the labels before training

Labels are inherited purely from folder name (step 5) — nothing catches a mistyped `--state` during capture on its own, and a mislabeled session will train silently wrong. Verify before spending time training:

```
python build_manifest.py --data-dir data --out manifest.csv --verify --contact-sheets data/_contact_sheets
```

This prints:
- **Per-tag positive counts** — a tag with 0 positives (or a suspiciously low/high percentage) usually means a capture session went into the wrong folder.
- **Per-folder image counts** — sanity-check against how many bursts you actually shot.
- **File-integrity check** — flags unreadable images and empty/all-zero depth files.

And writes one thumbnail-grid PNG per folder to `data/_contact_sheets/` — an unreadable frame shows up as a red tile, so you can eyeball a whole session at once for wrong-folder captures or parts cut off out of frame (the manual check the plan calls for in Section 5.3), instead of scrolling through hundreds of individual files.

Fix any flagged folders (recapture, delete bad frames, or fix `FOLDER_TAGS`) and re-run `build_manifest.py` before moving on.

## 7. Train the classifier

```
python train.py --manifest manifest.csv --epochs 30
```

Trains a ResNet18 widened to 4 input channels (RGB + depth) with a 7-way sigmoid head. Saves the best checkpoint (by validation loss) to `models/rgbd_classifier.pt`.

Useful flags:
- `--batch-size`, `--lr`, `--val-split` — training hyperparameters.
- `--no-pretrained` — train from scratch instead of ImageNet-initialized weights.
- `--out <path>` — checkpoint destination.

GPU is used automatically if available.

## 8. Run live guidance

```
python inference_loop.py
```

Starts the web UI and opens it in your browser automatically (`http://localhost:5000`). There is no OpenCV window by default - the browser page is the interface, and its RESTART button resets the tracked assembly. Press `Ctrl+C` in the terminal to quit. `--checkpoint <path>` overrides the default `models/rgbd_classifier.pt`.

Flags:
- `--window` - also show the debug OpenCV window (`q` quit, `r` reset).
- `--no-browser` - don't auto-open the browser.
- `--roi X0 Y0 X1 Y1` - restrict the occlusion check to a sub-region of the frame.
- `--no-web` - skip the web server (requires `--window`).

## 9. Web viewer

`inference_loop.py` starts a local Flask server (`web_server.py`) in the same process, so the browser page reflects the exact same live guidance as the OpenCV window - no separate process, no polling a file on disk.

Open `http://localhost:5000` (or `http://<this-pc's-LAN-IP>:5000` from another device on the same network) while `inference_loop.py` is running. The page shows:
- The live camera feed (`/video_feed`, an MJPEG stream of the same annotated frame the OpenCV window shows).
- The current step's 3D model, loaded via [`<model-viewer>`](https://modelviewer.dev/) from `web/assets/models/`.
- The live guidance message/instructions, confirmed-tags chips, and a step progress list - all pulled from `/api/status`, polled every 500ms.

**Two things to drop in before this looks/feels finished:**
- **glTF models** — one `.glb` per step, named per `web/assets/models/README.md` (e.g. `step_1.glb`, `step_2a.glb`). Until a step's file is there, the model panel shows a placeholder message instead of a blank viewer.
- **Figma wireframes** — export each screen as PNG/JPG into `web/design/`, then ask Claude to restyle `web/index.html` / `web/style.css` to match. The current layout is a functional placeholder, not the final design.

## 10. Tuning

In `config.py`:
- `CONFIDENCE_THRESHOLD` (default 0.5) — probability a tag must exceed to count as "on" in a given frame. Lowered from an earlier 0.7 because some tags (windshield, front wheels) can read well under a strict threshold even when genuinely attached, depending on angle/lighting.
- `CONFIRMATION_WINDOW_FRAMES` / `CONFIRMATION_MIN_FRACTION` (default 15 frames / 0.6) — a tag confirms once it reads above threshold in at least this fraction of the last N frames (a sliding-window majority vote, not a strict consecutive run). Tolerates normal frame noise so a genuinely-attached part isn't blocked from confirming by one stray dip, while still requiring sustained evidence so a brief false-positive streak (lighting flicker, shadow) can't lock in a wrong confirmation for the rest of the session.

Confirmation is deliberately one-way: once a tag confirms, it stays confirmed for the rest of the session (there is no un-confirm tunable any more). An earlier version un-confirmed a tag after sustained absent readings, meant to catch a part genuinely being removed mid-session - in practice the classifier's per-tag confidence isn't reliable enough for that to be safe, and a wheel that was never touched could still read absent for long enough to send the trainee back to redo a finished step. See `state_matcher.py`'s module docstring for the full reasoning. A full restart (`r` / the RESTART button / the web UI's restart button) is the only way to clear a confirmed tag.
- `BUILD_DEPTH_MIN_MM` / `HAND_OCCLUSION_MARGIN_MM` — occlusion gate defaults, overridden by calibration once you've run step 3.

## 11. Validation checklist

- Walk through assembly slowly, confirm each state transition fires and partial states withhold advancement.
- Deliberately skip ahead (attach several parts without pausing) — the matcher should jump straight to the furthest completed step.
- Hold a part near the chassis without attaching it — should not trigger a false positive.
- Rotate the car between the right and left views mid-session — confirmed tags from the side rotated out of frame should stay confirmed, not reset.
