"""RealSense burst capture for training data (Section 5.1).

Captures bursts of aligned RGB + depth frames into a folder named after the
assembly state being shown to the camera right now (see Section 5.2/5.3 for
the folder names build_manifest.py expects). One state per session - don't
change the physical assembly mid-session.

Usage:
    python capture.py --state s1_chassis --bursts 22
    python capture.py --state neg_wheel_held --bursts 20
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

import config


def build_pipeline():
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, config.COLOR_WIDTH, config.COLOR_HEIGHT, rs.format.bgr8, config.COLOR_FPS)
    rs_config.enable_stream(rs.stream.depth, config.DEPTH_WIDTH, config.DEPTH_HEIGHT, rs.format.z16, config.DEPTH_FPS)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return pipeline, align, depth_scale


def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())  # uint16, raw sensor units
    return color_image, depth_image


def next_burst_index(out_dir: Path) -> int:
    """Lowest unused burst index in out_dir, so re-running a capture session
    appends new bursts instead of overwriting existing burstNNN_fNN files."""
    used = [int(m.group(1)) for p in out_dir.glob("burst*_f*_rgb.png")
            for m in [re.match(r"burst(\d+)_f\d+_rgb\.png", p.name)] if m]
    return max(used) + 1 if used else 0


def run_session(pipeline, align, depth_scale, out_dir, bursts, burst_size, burst_duration_s, gap_s):
    interval = burst_duration_s / burst_size
    window = "capture preview (press q to stop)"
    start = next_burst_index(out_dir)

    for offset in range(bursts):
        b = start + offset
        print(f"\nBurst {offset + 1}/{bursts} (index {b}) - vary angle/distance/lighting/hand position now.")
        saved = 0
        for i in range(burst_size):
            t0 = time.time()
            color_image, depth_image = get_aligned_frames(pipeline, align)
            if color_image is None:
                continue
            depth_mm = (depth_image.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

            stem = f"burst{b:03d}_f{i:02d}"
            cv2.imwrite(str(out_dir / f"{stem}_rgb.png"), color_image)
            np.save(out_dir / f"{stem}_depth.npy", depth_mm)

            preview = color_image.copy()
            cv2.putText(preview, f"REC burst {offset + 1}/{bursts} frame {i + 1}/{burst_size}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
            saved += 1
            time.sleep(max(0.0, interval - (time.time() - t0)))
        print(f"  saved {saved}/{burst_size} frames")

        if offset < bursts - 1:
            gap_end = time.time() + gap_s
            while time.time() < gap_end:
                color_image, _ = get_aligned_frames(pipeline, align)
                if color_image is not None:
                    remaining = gap_end - time.time()
                    preview = color_image.copy()
                    cv2.putText(preview, f"next burst in {remaining:0.1f}s - reposition now", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return


def main():
    parser = argparse.ArgumentParser(description="Capture RGB-D bursts for one labeled assembly state.")
    parser.add_argument("--state", required=True, help="Folder name for this state, e.g. s1_chassis")
    parser.add_argument("--bursts", type=int, default=22, help="Bursts to capture (Section 5.1: ~20-25 per state)")
    parser.add_argument("--burst-size", type=int, default=config.BURST_SIZE)
    parser.add_argument("--burst-duration", type=float, default=config.BURST_DURATION_S)
    parser.add_argument("--gap", type=float, default=config.BURST_GAP_S)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    args = parser.parse_args()

    out_dir = Path(args.data_dir) / args.state
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline, align, depth_scale = build_pipeline()
    print(f"Capturing into {out_dir} - {args.bursts} bursts x {args.burst_size} frames, {args.gap}s gap between bursts.")
    print("Press q at any time to stop early.")
    try:
        run_session(pipeline, align, depth_scale, out_dir, args.bursts, args.burst_size, args.burst_duration, args.gap)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
