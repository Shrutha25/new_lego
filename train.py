"""RGB-D 4-channel multi-label classifier training (Section 6).

Backbone is a ResNet18 with its first conv layer widened to 4 input channels
(RGB + aligned depth), the extra channel initialized from the mean of the
pretrained RGB weights. The head is 7 independent sigmoid outputs (one BCE
loss per tag, no softmax) since multiple parts can be attached at once.

Usage:
    python train.py --manifest manifest.csv --epochs 30
"""
import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode

import config
from mat_roi import crop_to_roi, load_mat_roi

NORMALIZE_RGB = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class JointRGBDAugment:
    """Applies the same random rotation/zoom to RGB and depth together (Section 5.5).

    Color jitter only applies to RGB - it has no physical meaning on depth.
    Horizontal flip is intentionally never used: it would swap left/right
    and corrupt the wf_left/wr_left vs wf_right/wr_right labels.
    """

    def __init__(self, rotation_deg=12, scale_range=(0.85, 1.05), jitter=0.2):
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.color_jitter = T.ColorJitter(brightness=jitter, contrast=jitter)

    def __call__(self, rgb_img, depth_img):
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        scale = random.uniform(*self.scale_range)
        rgb_img = TF.affine(rgb_img, angle=angle, translate=(0, 0), scale=scale, shear=0,
                             interpolation=InterpolationMode.BILINEAR)
        depth_img = TF.affine(depth_img, angle=angle, translate=(0, 0), scale=scale, shear=0,
                               interpolation=InterpolationMode.NEAREST)
        rgb_img = self.color_jitter(rgb_img)
        return rgb_img, depth_img


class RGBDDataset(Dataset):
    def __init__(self, manifest_path, tag_names, image_size=config.IMAGE_SIZE, train=True):
        with open(manifest_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.tags = tag_names
        self.image_size = image_size
        self.augment = JointRGBDAugment() if train else None
        self.roi = load_mat_roi()  # (x0, y0, x1, y1) or None - see mat_roi.py

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        try:
            rgb = Image.open(row["rgb_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Can't read {row['rgb_path']} - likely a corrupted capture. "
                f"Run: python build_manifest.py --verify to find bad files, delete the "
                f"rgb/depth pair, then rebuild the manifest."
            ) from e
        rgb_arr = crop_to_roi(np.array(rgb), self.roi)  # roi is a polygon, not a rectangle - can't use PIL's .crop()
        rgb = Image.fromarray(rgb_arr).resize((self.image_size, self.image_size))

        try:
            depth_mm = np.load(row["depth_path"]).astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Can't read {row['depth_path']} - likely a corrupted capture. "
                f"Run: python build_manifest.py --verify to find bad files, delete the "
                f"rgb/depth pair, then rebuild the manifest."
            ) from e
        depth_mm = crop_to_roi(depth_mm, self.roi)
        depth_mm = cv2.resize(depth_mm, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        depth_norm = np.clip(depth_mm, 0, config.DEPTH_CLIP_MM) / config.DEPTH_CLIP_MM
        depth_img = Image.fromarray(depth_norm, mode="F")

        if self.augment:
            rgb, depth_img = self.augment(rgb, depth_img)

        rgb_t = NORMALIZE_RGB(TF.to_tensor(rgb))
        depth_t = TF.to_tensor(depth_img)  # already float in [0, 1]

        x = torch.cat([rgb_t, depth_t], dim=0)  # 4, H, W
        y = torch.tensor([float(row[t]) for t in self.tags], dtype=torch.float32)
        return x, y


def build_model(num_tags, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)

    old_conv = net.conv1
    new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
    net.conv1 = new_conv
    net.fc = nn.Linear(net.fc.in_features, num_tags)
    return net


def compute_pos_weight(train_subset, tag_names):
    """Per-tag BCEWithLogitsLoss pos_weight = n_negative / n_positive on the training split.

    The cumulative labeling scheme means later steps (e.g. roof) show up in
    far fewer folders than early ones (e.g. the wheels), so an unweighted
    loss lets the model minimize error by just predicting those tags absent
    most of the time. Weighting false negatives on the rare tags more
    heavily counteracts that without needing a perfectly balanced capture set.
    """
    base = train_subset.dataset
    rows = [base.rows[i] for i in train_subset.indices]
    weights = []
    for t in tag_names:
        n_pos = sum(int(r[t]) for r in rows)
        n_neg = len(rows) - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"WARNING: '{t}' has {n_pos} positive / {n_neg} negative examples in the training split - "
                  f"no loss weighting can teach the model to tell this tag apart. Capture more data for it.")
        # Clamp both counts to >=1: with zero negatives, n_neg/n_pos would be
        # exactly 0 and BCEWithLogitsLoss's pos_weight=0 zeroes the gradient
        # for that tag's *only* present class outright, leaving it untrained.
        weights.append(max(n_neg, 1) / max(n_pos, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_preds.append((torch.sigmoid(logits) > 0.5).float().cpu())
        all_targets.append(y.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    per_tag_acc = (preds == targets).float().mean(dim=0)
    return total_loss / len(loader.dataset), per_tag_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--out", default=str(Path(config.MODELS_DIR) / "rgbd_classifier.pt"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true",
                         help="Disable per-tag class-imbalance loss weighting (on by default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4,
                         help="DataLoader worker processes - 0 loads data on the main thread, "
                              "serialized with GPU compute; >0 loads the next batch in the "
                              "background while the GPU works on the current one.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.manifest, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Split by burst, not by individual frame: frames within a burst are
    # captured ~0.2s apart with only slight motion (Section 5.1), so a
    # per-frame shuffle leaks near-duplicate twins across train/val - the
    # model then "generalizes" to validation frames it has effectively
    # already memorized, producing inflated validation accuracy that
    # doesn't reflect real held-out performance.
    def burst_key(row):
        m = re.search(r"burst(\d+)_f\d+", Path(row["rgb_path"]).name)
        return (row["folder"], m.group(1) if m else Path(row["rgb_path"]).name)

    groups = {}
    for idx, row in enumerate(all_rows):
        groups.setdefault(burst_key(row), []).append(idx)

    group_keys = list(groups.keys())
    random.Random(args.seed).shuffle(group_keys)
    n_val_groups = max(1, int(len(group_keys) * args.val_split))
    val_groups, train_groups = group_keys[:n_val_groups], group_keys[n_val_groups:]
    val_idx = [i for g in val_groups for i in groups[g]]
    train_idx = [i for g in train_groups for i in groups[g]]
    print(f"split by burst: {len(train_groups)} train bursts / {len(val_groups)} val bursts")

    train_ds = Subset(RGBDDataset(args.manifest, config.TAGS, train=True), train_idx)
    val_ds = Subset(RGBDDataset(args.manifest, config.TAGS, train=False), val_idx)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    model = build_model(len(config.TAGS), pretrained=not args.no_pretrained).to(device)

    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = compute_pos_weight(train_ds, config.TAGS).to(device)
        print("pos_weight per tag: " + " ".join(f"{t}:{w:.2f}" for t, w in zip(config.TAGS, pos_weight.tolist())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        acc_str = " ".join(f"{t}:{a:.2f}" for t, a in zip(config.TAGS, val_acc.tolist()))
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc/tag= {acc_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "tags": config.TAGS}, out_path)
            print(f"  saved new best checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
