"""
Updated training script for the oprtimized AI model.

Usage:
    python train_skeleton.py --data_dir data --epochs 50 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
from tqdm import tqdm

from ai_detector.models.skeleton_model import create_skeleton_model, SkeletonLoss
from ai_detector.models.advanced_model import get_advanced_transforms

# --- Simple dataset to load frames as video clips ---
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label, transform, num_frames):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.num_frames = num_frames

        self.frame_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.frame_paths) // self.num_frames

    def __getitem__(self, idx):
        start = idx * self.num_frames
        paths = self.frame_paths[start:start + self.num_frames]

        images = [self.transform(default_loader(p)) for p in paths]
        frames = torch.stack(images)  # (num_frames, 3, H, W)
        return frames, self.label


def build_dataloader(data_dir, num_frames, batch_size, max_samples_per_class, num_workers):
    transform = get_advanced_transforms('train')
    ai_dataset = FrameDataset(os.path.join(data_dir, 'ai'), 1, transform, num_frames)
    real_dataset = FrameDataset(os.path.join(data_dir, 'real'), 0, transform, num_frames)

    if max_samples_per_class:
        ai_dataset = torch.utils.data.Subset(ai_dataset, range(min(len(ai_dataset), max_samples_per_class)))
        real_dataset = torch.utils.data.Subset(real_dataset, range(min(len(real_dataset), max_samples_per_class)))

    full_dataset = torch.utils.data.ConcatDataset([ai_dataset, real_dataset])
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_skeleton_model(
        backbone=args.backbone,
        num_frames=args.num_frames,
        freeze_backbone=args.freeze_backbone,
        enable_multitask=False
    ).to(device)

    dataloader = build_dataloader(
        args.data_dir, args.num_frames, args.batch_size, args.max_samples_per_class, args.num_workers
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = SkeletonLoss(
        classification_weight=1.0,
        skeleton_weight=0.2
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for frames, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            frames = frames.to(device)  # (B, T, C, H, W)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            results = model(frames)
            loss_dict = criterion(results, labels)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--max_samples_per_class', type=int, default=90000)
    parser.add_argument('--save_path', type=str, default='skeleton_model.pt')

    args = parser.parse_args()
    train(args)
