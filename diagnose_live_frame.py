"""One-off diagnostic: capture a single live frame, save it exactly like
capture.py would, and report what the live camera pipeline actually looks
like (depth scale, value ranges) plus the model's raw output on it - so a
live frame can be directly compared against training-set images.

Point the camera at a genuinely EMPTY mat (nothing on it, no hands) before
running this.

Usage:
    python diagnose_live_frame.py --checkpoint models/rgbd_classifier.pt
"""
import argparse

import cv2
import numpy as np
import torch

import config
from capture import build_pipeline, get_aligned_frames
from inference_loop import load_model, preprocess
from mat_roi import load_mat_roi

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="models/rgbd_classifier.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tags = load_model(args.checkpoint, device)
roi = load_mat_roi()

pipeline, align, depth_scale = build_pipeline()
print(f"Live depth_scale: {depth_scale}")
print("Warming up camera...")
for _ in range(config.CAMERA_WARMUP_FRAMES):
    get_aligned_frames(pipeline, align)

color_image, depth_image = get_aligned_frames(pipeline, align)
pipeline.stop()

if color_image is None:
    raise SystemExit("Could not get a frame from the camera.")

depth_mm = depth_image.astype(np.float32) * depth_scale * 1000.0

cv2.imwrite("data/_live_diagnostic_rgb.png", color_image)
np.save("data/_live_diagnostic_depth.npy", depth_mm.astype(np.uint16))
print("Saved live frame -> data/_live_diagnostic_rgb.png / _depth.npy")

valid_depth = depth_mm[depth_mm > 0]
print(f"depth_mm stats: min={valid_depth.min():.1f} max={valid_depth.max():.1f} "
      f"median={np.median(valid_depth):.1f} valid_pixels={valid_depth.size}/{depth_mm.size}")
print(f"color_image dtype={color_image.dtype} shape={color_image.shape} "
      f"min={color_image.min()} max={color_image.max()} mean={color_image.mean():.1f}")

x = preprocess(color_image, depth_mm, roi=roi).to(device)
with torch.no_grad():
    probs = torch.sigmoid(model(x))[0].cpu().tolist()

print("\nLive model output:")
for t, p in zip(tags, probs):
    print(f"  {t:12s} {p:.4f}")
