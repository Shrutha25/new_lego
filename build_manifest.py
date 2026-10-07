"""Builds an image -> tag-vector manifest from capture folder names (Section 5.4).

Labels are inherited entirely from the folder a session was captured into -
there's no manual per-image tagging. FOLDER_TAGS below mirrors the capture
plan in Section 5.2 (positive states) and 5.3 (held-part negatives). Add a
row there for any new folder capture.py is pointed at.

Usage:
    python build_manifest.py --data-dir data --out manifest.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import config

ALL_TAGS = config.TAGS


def tags(*active):
    vec = {t: 0 for t in ALL_TAGS}
    for t in active:
        vec[t] = 1
    return vec


FOLDER_TAGS = {
    "s0_empty": tags(),
    "s1_chassis": tags("chassis"),
    # More "chassis alone, no wheels at all" captures (Issue #13,
    # 2026-09-11) - same physical state and label as s1_chassis, just a
    # separate folder so it's easy to tell how many of these exist. Chassis
    # co-occurs with all four wheels in ~3150 training images (every state
    # from partial-right onward) versus only ~640 where chassis is present
    # without wheels (s1_chassis + the wheel held/near negatives) - that
    # ~83/17 skew is what let the model learn "chassis clearly visible" as
    # a shortcut for "wheels are probably present too", confirmed live:
    # chassis read 0.99 with wheels at 0.95-0.99 confidence while zero
    # wheels were anywhere in frame. More of this exact state is the direct
    # fix for that specific imbalance.
    "s1_chassis_extra": tags("chassis"),
    "s2a_partial_right": tags("chassis", "wf_right"),
    # The _rear variants exist because the original plan only ever captured
    # "front wheel alone" as the partial-right/left state - the model had
    # never seen "rear wheel alone" and would guess wrong on that physical
    # configuration (Issue: rear wheel reported missing while attached).
    "s2a_partial_right_rear": tags("chassis", "wr_right"),
    "s2a_right_complete": tags("chassis", "wf_right", "wr_right"),
    # More "both right wheels attached, left genuinely absent" captures
    # (Issue #14, 2026-09-11) - same state/label as s2a_right_complete, same
    # fix pattern as s1_chassis_extra above. Right-only co-occurs with
    # left-also-present in 1867 images (windshield/roof states, plus their
    # held/near negatives) versus only 450 where right is present without
    # left (~19%/81%) - confirmed live: showing only the right wheels made
    # the left wheel step get skipped entirely, the model treating "wheels
    # confidently visible" as a shortcut for "probably all four", same
    # mechanism as the chassis case just fixed.
    "s2a_right_complete_extra": tags("chassis", "wf_right", "wr_right"),
    "s2b_partial_left": tags("chassis", "wf_left"),
    "s2b_partial_left_rear": tags("chassis", "wr_left"),
    "s2b_left_complete": tags("chassis", "wf_left", "wr_left"),
    "s3_windshield": tags("chassis", "wf_right", "wr_right", "wf_left", "wr_left", "windshield"),
    # Windshield attached directly to the bare chassis, wheels skipped
    # entirely (Issue #17, 2026-09-11) - same gap as roof (Issue #16):
    # windshield=1 only ever co-occurs with all four wheels=1 in the
    # current data (s3_windshield itself, plus neg_roof_held/near which
    # also require all four wheels), so 100% of positive windshield
    # examples also have every wheel present. Pre-emptive fix, captured
    # before live-testing actually caught it happening, same pattern as
    # s4_roof_only.
    "s3_windshield_only": tags("chassis", "windshield"),
    "s4_roof": tags("chassis", "wf_right", "wr_right", "wf_left", "wr_left", "windshield", "roof"),
    # Roof attached directly to the bare chassis, wheels/windshield skipped
    # entirely (Issue #16, 2026-09-11) - the most extreme version of the
    # co-occurrence shortcut found today: roof only ever appears in
    # s4_roof, which is the final cumulative state, so chassis+all four
    # wheels+windshield+roof are together in 100% of its training images -
    # zero contrastive examples existed anywhere. Confirmed live: showing
    # just chassis+roof+one wheel read every single tag at 0.985+,
    # including windshield and both untouched left wheels. Same fix
    # pattern as s1_chassis_extra/s2a_right_complete_extra, just needed
    # confirming the physical build allows this combination first.
    "s4_roof_only": tags("chassis", "roof"),
    # Held-but-not-attached negatives (Section 5.3): the part is visible in
    # frame but its tag stays 0, so the model learns "attached" not "visible".
    "neg_wheel_held": tags("chassis"),
    "neg_windshield_held": tags("chassis", "wf_right", "wr_right", "wf_left", "wr_left"),
    "neg_roof_held": tags("chassis", "wf_right", "wr_right", "wf_left", "wr_left", "windshield"),
    # Resting-on-mat-but-not-attached negatives (Issue #4, 2026-09-04): same
    # idea as the _held folders above, but the unattached part sits on the
    # mat next to the chassis instead of in a hand - the existing _held
    # captures never covered this presentation.
    "neg_wheel_near": tags("chassis"),
    "neg_windshield_near": tags("chassis", "wf_right", "wr_right", "wf_left", "wr_left"),
    "neg_roof_near": tags("chassis", "wf_right", "wr_right", "wf_left", "wr_left", "windshield"),
    # Bare hand/arm/watch over an empty mat, no chassis at all (Issue #1/#4).
    "neg_hand_only": tags(),
    # Anything that isn't a lego part, placed ON THE MAT within the
    # calibrated ROI - a t-shirt, a book, random household objects (Issue
    # #5, 2026-09-08). The mat ROI (mat_roi.py) only restricts *where* the
    # model looks, not *what* it's allowed to conclude is there - before
    # this, "chassis" was 1 in every folder except s0_empty/neg_hand_only
    # (86% of the whole manifest), so the model had never seen "something's
    # in the mat area but it isn't the chassis," only "the mat is empty" or
    # "a hand is over it," and defaulted to its majority-class guess
    # (chassis=1) on anything else - including a t-shirt sitting right where
    # the chassis normally would be, well inside the ROI.
    "neg_random_object": tags(),
}


def find_pairs(folder: Path):
    """Yield (rgb_path, depth_path) for every rgb/depth pair capture.py saved."""
    for rgb_path in sorted(folder.glob("*_rgb.png")):
        depth_path = rgb_path.with_name(rgb_path.name.replace("_rgb.png", "_depth.npy"))
        if depth_path.exists():
            yield rgb_path, depth_path


def build(data_dir: Path, out_path: Path):
    rows = []
    skipped_folders = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if folder.name not in FOLDER_TAGS:
            skipped_folders.append(folder.name)
            continue
        vec = FOLDER_TAGS[folder.name]
        for rgb_path, depth_path in find_pairs(folder):
            row = {"rgb_path": rgb_path.as_posix(), "depth_path": depth_path.as_posix(), "folder": folder.name}
            row.update(vec)
            rows.append(row)

    if skipped_folders:
        print("WARNING: no tag mapping for these folders, skipped:")
        for f in skipped_folders:
            print(f"  - {f}  (add it to FOLDER_TAGS in build_manifest.py)")

    if not rows:
        raise SystemExit("No labeled images found - check --data-dir and FOLDER_TAGS mappings.")

    fieldnames = ["rgb_path", "depth_path", "folder"] + ALL_TAGS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_folders = len(set(r["folder"] for r in rows))
    print(f"Wrote {len(rows)} labeled images from {n_folders} folders to {out_path}")
    return rows


def verify(rows, contact_sheet_dir=None):
    """Sanity-check the manifest before it's fed to train.py.

    Auto-labeling from folder name means a single mistyped --state during
    capture silently mislabels a whole session - this catches that kind of
    mistake via class balance, file integrity, and (optionally) a visual
    contact sheet per folder instead of letting it poison training silently.
    """
    total = len(rows)

    print("\n--- Per-tag positive counts ---")
    for t in ALL_TAGS:
        n_pos = sum(int(r[t]) for r in rows)
        pct = 100 * n_pos / total if total else 0
        if n_pos == 0:
            flag = "  <-- check this: no positive examples"
        elif n_pos == total:
            flag = "  <-- check this: no negative examples (model can't learn 'absent' for this tag)"
        else:
            flag = ""
        print(f"  {t:12s}: {n_pos:4d} / {total} ({pct:4.0f}%){flag}")

    print("\n--- Per-folder image counts ---")
    for folder, n in sorted(Counter(r["folder"] for r in rows).items()):
        print(f"  {folder:24s}: {n}")

    print("\n--- Checking files are readable ---")
    bad = []
    for row in rows:
        try:
            with Image.open(row["rgb_path"]) as img:
                img.verify()
        except Exception as e:
            bad.append((row["rgb_path"], str(e)))
        try:
            depth = np.load(row["depth_path"])
            if depth.size == 0 or float(depth.max()) == 0.0:
                bad.append((row["depth_path"], "empty or all-zero depth"))
        except Exception as e:
            bad.append((row["depth_path"], str(e)))
    if bad:
        print(f"  {len(bad)} problem file(s):")
        for path, err in bad[:20]:
            print(f"    {path}: {err}")
    else:
        print("  all files OK")

    if contact_sheet_dir:
        _write_contact_sheets(rows, Path(contact_sheet_dir))


def _write_contact_sheets(rows, out_dir, thumb=96, cols=8):
    """One thumbnail-grid PNG per folder, so a whole session can be eyeballed
    at a glance for wrong-folder captures or parts cut off out of frame."""
    by_folder = defaultdict(list)
    for row in rows:
        by_folder[row["folder"]].append(row["rgb_path"])

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Writing contact sheets to {out_dir} ---")
    for folder, paths in sorted(by_folder.items()):
        n_rows = (len(paths) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, n_rows * thumb), (30, 30, 30))
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    thumb_img = im.convert("RGB").resize((thumb, thumb))
            except Exception:
                thumb_img = Image.new("RGB", (thumb, thumb), (200, 30, 30))  # unreadable file -> red tile
            x, y = (i % cols) * thumb, (i // cols) * thumb
            sheet.paste(thumb_img, (x, y))
        out_path = out_dir / f"{folder}.png"
        sheet.save(out_path)
        print(f"  {out_path}  ({len(paths)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--verify", action="store_true",
                         help="Print class-balance and file-integrity sanity checks after building the manifest")
    parser.add_argument("--contact-sheets", default=None, metavar="DIR",
                         help="Write one thumbnail-grid PNG per folder to DIR for visual label spot-checking")
    args = parser.parse_args()
    rows = build(Path(args.data_dir), Path(args.out))
    if args.verify or args.contact_sheets:
        verify(rows, contact_sheet_dir=args.contact_sheets)


if __name__ == "__main__":
    main()
