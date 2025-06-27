"""
Skeleton‑based training script (debug‑friendly version)

Run example:
```bash
cd ~/truthful  # important so relative paths work!
PYTHONPATH=. python ai_detector/scripts/train_skeleton.py \
  --ai_dir   ai_detector/data/ai \
  --real_dir ai_detector/data/real \
  --epochs   10 \
  --batch_size 32 \
  --backbone resnet50
```
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision.transforms import Compose
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from ai_detector.models.skeleton_model import create_skeleton_model, SkeletonLoss
from ai_detector.models.advanced_model import get_advanced_transforms

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
class FrameDataset(Dataset):
    """Treats a folder full of frames as a sequence (num_frames per item)."""

    def __init__(self, image_paths: List[Path], label: int, transform, num_frames: int):
        self.image_paths = sorted(image_paths)
        self.label = label
        self.transform = transform
        self.num_frames = num_frames
        if len(self.image_paths) < num_frames:
            raise ValueError(f"Not enough frames ({len(self.image_paths)}) for num_frames={num_frames}")

    def __len__(self):
        return len(self.image_paths) // self.num_frames

    def __getitem__(self, idx):
        start = idx * self.num_frames
        paths = self.image_paths[start : start + self.num_frames]
        frames = [self.transform(default_loader(str(p))) for p in paths]
        return torch.stack(frames), torch.tensor(self.label, dtype=torch.float32)


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def build_dataloader(ai_dir: Path,
                      real_dir: Path,
                      num_frames: int,
                      batch_size: int,
                      max_per_class: int,
                      num_workers: int):
    transform = get_advanced_transforms('train')

    ai_imgs   = list_images(ai_dir)
    real_imgs = list_images(real_dir)

    print(f"[INFO] Found {len(ai_imgs):,} AI frames  in {ai_dir}")
    print(f"[INFO] Found {len(real_imgs):,} REAL frames in {real_dir}")

    if len(ai_imgs) == 0 or len(real_imgs) == 0:
        raise SystemExit("One of the classes is empty – check your paths!")

    ai_ds   = FrameDataset(ai_imgs,   1, transform, num_frames)
    real_ds = FrameDataset(real_imgs, 0, transform, num_frames)

    if max_per_class:
        ai_ds   = Subset(ai_ds,   range(min(len(ai_ds),   max_per_class)))
        real_ds = Subset(real_ds, range(min(len(real_ds), max_per_class)))

    full_ds = ConcatDataset([ai_ds, real_ds])
    return DataLoader(full_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Skeleton model trainer (debug‑friendly)")
    parser.add_argument('--ai_dir',   required=True, type=Path)
    parser.add_argument('--real_dir', required=True, type=Path)
    parser.add_argument('--epochs',   type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--backbone',   type=str, choices=['resnet50','efficientnet_b3','convnext_tiny'], default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--max_per_class', type=int, default=None)
    parser.add_argument('--save_path', type=Path, default=Path('results/skeleton_model.pt'))
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Data & model
    # ---------------------------------------------------------------------
    dataloader = build_dataloader(args.ai_dir, args.real_dir, args.num_frames,
                                  args.batch_size, args.max_per_class, args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_skeleton_model(backbone=args.backbone,
                                  num_frames=args.num_frames,
                                  freeze_backbone=args.freeze_backbone,
                                  enable_multitask=False).to(device)

    criterion = SkeletonLoss(classification_weight=1.0, skeleton_weight=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for frames, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}: mean loss = {running_loss / len(dataloader):.4f}")

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path.resolve()}")


if __name__ == "__main__":
    main()
