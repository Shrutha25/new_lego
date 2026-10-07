"""Live guidance loop (Section 9).

capture frame -> occlusion filter -> classifier -> state matcher -> guidance overlay.

Usage:
    python inference_loop.py --checkpoint models/rgbd_classifier.pt
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config
import web_server
from capture import build_pipeline, get_aligned_frames
from mat_roi import ROI_PATH, crop_to_roi, draw_roi_outline, load_mat_roi
from occlusion_filter import CALIBRATION_PATH, PRESENCE_MIN_FRACTION, frame_skip_reason, load_calibration, object_present_fraction
from state_matcher import StateMatcher
from train import build_model

NORMALIZE_RGB = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def wrap_text(text, font, scale, thickness, max_width):
    """Greedy word-wrap so instruction text never runs off the edge of the frame."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def preprocess(color_image_bgr, depth_mm, roi=None, image_size=config.IMAGE_SIZE):
    rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
    rgb = crop_to_roi(rgb, roi)
    rgb = cv2.resize(rgb, (image_size, image_size))
    depth = crop_to_roi(depth_mm.astype(np.float32), roi)
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    depth = np.clip(depth, 0, config.DEPTH_CLIP_MM) / config.DEPTH_CLIP_MM

    rgb_t = NORMALIZE_RGB(TF.to_tensor(rgb))
    depth_t = torch.from_numpy(depth).unsqueeze(0)
    x = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0)  # 1, 4, H, W
    return x


RESTART_BUTTON_SIZE = (110, 40)
RESTART_BUTTON_MARGIN = 10


def restart_button_rect(frame_width):
    w, h = RESTART_BUTTON_SIZE
    x1 = frame_width - RESTART_BUTTON_MARGIN
    y0 = RESTART_BUTTON_MARGIN
    return (x1 - w, y0, x1, y0 + h)


def draw_restart_button(display, rect):
    x0, y0, x1, y1 = rect
    cv2.rectangle(display, (x0, y0), (x1, y1), (50, 50, 50), -1)
    cv2.rectangle(display, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.putText(display, "RESTART", (x0 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def make_restart_click_handler(rect, reset_flag):
    x0, y0, x1, y1 = rect

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and x0 <= x <= x1 and y0 <= y <= y1:
            reset_flag["requested"] = True

    return on_mouse


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    tags = ckpt.get("tags", config.TAGS)
    model = build_model(len(tags))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(Path(config.MODELS_DIR) / "rgbd_classifier.pt"))
    parser.add_argument("--states", default=config.STATES_PATH)
    parser.add_argument("--no-web", action="store_true", help="Don't start the local web viewer server")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tags = load_model(args.checkpoint, device)
    matcher = StateMatcher(states_path=args.states, tags=tags)
    build_min_mm, build_max_mm, baseline_depth_mm = load_calibration()
    mat_roi = load_mat_roi()  # same polygon used for both occlusion check and classifier input
    if mat_roi is None:
        print("No mat ROI calibrated (run select_mat_roi.py) - classifying full frames.")
    if baseline_depth_mm is None:
        print("No empty-mat depth baseline calibrated (re-run occlusion_filter.py --frames 30) - "
              "skipping the 'nothing on the mat yet' safety check.")
    elif ROI_PATH.exists() and CALIBRATION_PATH.exists() and ROI_PATH.stat().st_mtime > CALIBRATION_PATH.stat().st_mtime:
        # Bit for bit the same bug that caused Issue #9 (2026-09-11): the
        # depth baseline is keyed to whatever ROI was active when it was
        # captured, and re-running select_mat_roi.py without recalibrating
        # depth afterward leaves the presence check silently comparing
        # against a mismatched crop region - it can misjudge "empty" on a
        # genuinely empty mat instead of catching it. Loud and hard to miss
        # on purpose - this exact staleness bug already cost real debugging
        # time once.
        print("=" * 70)
        print("WARNING: mat_roi.json is newer than the depth calibration.")
        print("The 'nothing on the mat yet' safety check may be stale and")
        print("unreliable until you re-run:")
        print("    python occlusion_filter.py --frames 30")
        print("(point the camera at the empty mat first)")
        print("=" * 70)

    live_state = None
    if not args.no_web:
        live_state = web_server.LiveState()
        web_server.start_server_thread(live_state)
        print(f"Web viewer: http://localhost:{config.WEB_PORT}")

    pipeline, align, depth_scale = build_pipeline()
    window = "LEGO assembly guidance (q: quit, r or RESTART button: reset session)"
    button_rect = restart_button_rect(config.COLOR_WIDTH)
    reset_flag = {"requested": False}
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, make_restart_click_handler(button_rect, reset_flag))
    try:
        # RealSense auto-exposure/white-balance (and sometimes early depth
        # frames) need a moment to settle after pipeline.start(). Without
        # this, a handful of dark/garbage startup frames can get scored,
        # and since confirmed tags never un-confirm, a bad frame here can
        # falsely lock in "complete" for the rest of the session.
        print("Warming up camera...")
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            get_aligned_frames(pipeline, align)

        session_started_at = time.time()
        total_frames = 0
        mismatch_frames = 0

        while True:
            color_image, depth_image = get_aligned_frames(pipeline, align)
            if color_image is None:
                continue
            depth_mm = depth_image.astype(np.float32) * depth_scale * 1000.0

            display = color_image.copy()
            draw_roi_outline(display, mat_roi)
            # The web UI has its own instructions/restart button elsewhere on
            # the page now, so the frame streamed to it should be clean - no
            # guidance text or RESTART button baked into the pixels, just the
            # camera image (still with the ROI outline, minus its text label,
            # since that's a useful placement guide and not "text" clutter
            # the same way the guidance overlay is).
            web_frame = color_image.copy()
            draw_roi_outline(web_frame, mat_roi, label=None)
            skip_reason = frame_skip_reason(depth_mm, roi=mat_roi, build_min_mm=build_min_mm, build_max_mm=build_max_mm,
                                             baseline_depth_mm=baseline_depth_mm)
            if skip_reason:
                cv2.putText(display, skip_reason, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if live_state is not None:
                    live_state.set_status({**live_state.get_status(), "message": skip_reason, "instructions": None})
            else:
                x = preprocess(color_image, depth_mm, roi=mat_roi).to(device)
                with torch.no_grad():
                    probs = torch.sigmoid(model(x))[0].cpu().tolist()
                tag_probs = dict(zip(tags, probs))

                # Checked again here, independently of frame_skip_reason's
                # skip decision above - see StateMatcher.update()'s
                # docstring for why relying solely on the caller skipping
                # frames upstream isn't quite enough on its own.
                physical_presence = None
                if baseline_depth_mm is not None:
                    physical_presence = object_present_fraction(
                        depth_mm, baseline_depth_mm, roi=mat_roi
                    ) >= PRESENCE_MIN_FRACTION

                guidance = matcher.update(tag_probs, physical_presence=physical_presence)

                total_frames += 1
                if guidance.mismatch:
                    mismatch_frames += 1

                if live_state is not None:
                    model_path = guidance.next_step["model"] if guidance.next_step else config.FINAL_MODEL
                    live_state.set_status({
                        "message": guidance.message,
                        "instructions": guidance.instructions,
                        "mismatch": guidance.mismatch,
                        "model": model_path,
                        "next_step": guidance.next_step,
                        "completed_step": guidance.completed_step,
                        "confirmed_tags": [t for t in tags if guidance.session_state[t]],
                        "tag_probs": {t: round(p, 3) for t, p in tag_probs.items()},
                        "tag_labels": config.TAG_LABELS,
                        "states": matcher.states,
                        "product_name": config.PRODUCT_NAME,
                        "session_started_at": session_started_at,
                        "total_frames": total_frames,
                        "mismatch_frames": mismatch_frames,
                    })

                color = (0, 165, 255) if guidance.mismatch else (0, 200, 0)
                font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                max_text_width = display.shape[1] - 20
                y = 30
                for line in wrap_text(guidance.message, font, scale, thickness, max_text_width):
                    cv2.putText(display, line, (10, y), font, scale, color, thickness)
                    y += 34

                if guidance.instructions:
                    for line in wrap_text(guidance.instructions, font, 0.6, 1, max_text_width):
                        cv2.putText(display, line, (10, y), font, 0.6, (255, 255, 255), 1)
                        y += 22

                y += 10
                # Only list confirmed tags - the heading is driven purely by
                # session_state, so showing every tag's raw per-frame
                # probability here (including ones near/below threshold)
                # made the list look like it contradicted the heading.
                stable_tags = [t for t in tags if guidance.session_state[t]]
                if stable_tags:
                    for t in stable_tags:
                        cv2.putText(display, f"[X] {t}: {tag_probs[t]:.2f}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y += 20
                else:
                    cv2.putText(display, "(nothing confirmed yet)", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            draw_restart_button(display, button_rect)
            if live_state is not None:
                live_state.set_frame(web_frame)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            web_reset_requested = live_state is not None and live_state.consume_reset_request()
            if key == ord("q"):
                break
            elif key == ord("r") or reset_flag["requested"] or web_reset_requested:
                matcher.reset()
                reset_flag["requested"] = False
                session_started_at = time.time()
                total_frames = 0
                mismatch_frames = 0
                print("session state reset")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
