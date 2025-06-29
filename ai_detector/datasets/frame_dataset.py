"""
Enhanced frame dataset with advanced temporal sampling strategies.

This module provides:
1. Multiple frame sampling strategies (sequential, uniform, rand_stride)
2. FrameMix augmentation for improved generalization
3. Support for validation split and metrics calculation
"""

import os
import random
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from torchvision.datasets.folder import default_loader
from torchvision import transforms

class FrameDataset(Dataset):
    """
    Enhanced dataset for video frames with advanced sampling strategies.
    
    Supports:
    - Sequential sampling (original implementation)
    - Uniform sampling across the entire video
    - Random stride sampling with minimum time separation
    - FrameMix augmentation
    """
    
    def __init__(
        self, 
        image_paths: List[Path], 
        label: int, 
        transform: Callable,
        num_frames: int,
        frame_sampling: str = "sequential",
        min_stride_secs: float = 0.5,
        fps: float = 30.0,
        framemix_enabled: bool = False,
        framemix_prob: float = 0.25,
        framemix_partner: Optional['FrameDataset'] = None
    ):
        """
        Initialize the frame dataset.
        
        Args:
            image_paths: List of paths to frame images
            label: Class label (0=real, 1=ai)
            transform: Image transforms to apply
            num_frames: Number of frames to sample per clip
            frame_sampling: Sampling strategy ('sequential', 'uniform', 'rand_stride')
            min_stride_secs: Minimum time separation between frames (for rand_stride)
            fps: Frames per second (for time calculations)
            framemix_enabled: Whether to enable FrameMix augmentation
            framemix_prob: Probability of applying FrameMix
            framemix_partner: Partner dataset for FrameMix (same label)
        """
        self.image_paths = sorted(image_paths)
        self.label = label
        self.transform = transform
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.min_stride_frames = max(1, int(min_stride_secs * fps))
        self.framemix_enabled = framemix_enabled
        self.framemix_prob = framemix_prob
        self.framemix_partner = framemix_partner
        
        # Group frames by video ID (assuming format: videoID_frame_XX.jpg)
        self.video_frames = self._group_frames_by_video()
        
        # Validate we have enough frames
        if not self.video_frames:
            raise ValueError(f"No valid videos found with at least {num_frames} frames")
    
    def _group_frames_by_video(self) -> Dict[str, List[Path]]:
        """Group frames by video ID."""
        video_frames = {}
        
        for path in self.image_paths:
            # Extract video ID from filename (assumes format: videoID_frame_XX.jpg)
            filename = path.stem
            parts = filename.split('_frame_')
            
            if len(parts) != 2:
                continue  # Skip files that don't match the expected format
            
            video_id = parts[0]
            
            if video_id not in video_frames:
                video_frames[video_id] = []
            
            video_frames[video_id].append(path)
        
        # Sort frames within each video and filter videos with too few frames
        result = {}
        for video_id, frames in video_frames.items():
            if len(frames) >= self.num_frames:
                # Sort frames by frame number
                sorted_frames = sorted(frames, key=lambda p: int(p.stem.split('_frame_')[1]))
                result[video_id] = sorted_frames
        
        return result
    
    def __len__(self) -> int:
        """Return the number of video clips in the dataset."""
        return len(self.video_frames)
    
    def _sample_sequential_frames(self, video_frames: List[Path]) -> List[Path]:
        """Sample frames sequentially from the beginning."""
        return video_frames[:self.num_frames]
    
    def _sample_uniform_frames(self, video_frames: List[Path]) -> List[Path]:
        """Sample frames uniformly across the video."""
        if len(video_frames) <= self.num_frames:
            return video_frames
        
        indices = np.linspace(0, len(video_frames) - 1, self.num_frames, dtype=int)
        return [video_frames[i] for i in indices]
    
    def _sample_rand_stride_frames(self, video_frames: List[Path]) -> List[Path]:
        """Sample frames with random stride but minimum separation."""
        if len(video_frames) <= self.num_frames:
            return video_frames
        
        # Calculate maximum possible stride
        max_stride = (len(video_frames) - 1) // (self.num_frames - 1)
        
        if max_stride <= self.min_stride_frames:
            # If we can't achieve minimum stride, fall back to uniform sampling
            return self._sample_uniform_frames(video_frames)
        
        # Choose a random stride between min and max
        stride = random.randint(self.min_stride_frames, max_stride)
        
        # Choose a random starting point
        max_start = len(video_frames) - stride * (self.num_frames - 1) - 1
        start = random.randint(0, max_start)
        
        # Sample frames with the chosen stride
        indices = [start + i * stride for i in range(self.num_frames)]
        return [video_frames[i] for i in indices]
    
    def _apply_framemix(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply FrameMix augmentation."""
        if not self.framemix_enabled or random.random() > self.framemix_prob or self.framemix_partner is None:
            return frames
        
        # Determine how many frames to replace (ceil of half)
        num_replace = math.ceil(self.num_frames / 2)
        
        # Get a random sample from the partner dataset
        partner_idx = random.randint(0, len(self.framemix_partner) - 1)
        partner_frames, _ = self.framemix_partner[partner_idx]
        
        # Choose random indices to replace
        replace_indices = random.sample(range(self.num_frames), num_replace)
        
        # Create mixed frames
        mixed_frames = frames.clone()
        for i, idx in enumerate(replace_indices):
            partner_idx = random.randint(0, self.num_frames - 1)
            mixed_frames[idx] = partner_frames[partner_idx]
        
        return mixed_frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a video clip with the specified sampling strategy."""
        # Get video ID and frames
        video_id = list(self.video_frames.keys())[idx]
        video_frames = self.video_frames[video_id]
        
        # Sample frames based on strategy
        if self.frame_sampling == "uniform":
            sampled_paths = self._sample_uniform_frames(video_frames)
        elif self.frame_sampling == "rand_stride":
            sampled_paths = self._sample_rand_stride_frames(video_frames)
        else:  # sequential (default)
            sampled_paths = self._sample_sequential_frames(video_frames)
        
        # Load and transform frames
        frames = torch.stack([
            self.transform(default_loader(str(path))) 
            for path in sampled_paths
        ])
        
        # Apply FrameMix if enabled
        if self.framemix_enabled and self.framemix_partner is not None:
            frames = self._apply_framemix(frames)
        
        return frames, torch.tensor(self.label, dtype=torch.float32)

def list_images(root: Path) -> List[Path]:
    """List all image files in a directory recursively."""
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def build_datasets(
    ai_dir: Path,
    real_dir: Path,
    num_frames: int,
    frame_sampling: str = "rand_stride",
    min_stride_secs: float = 0.5,
    fps: float = 30.0,
    framemix_prob: float = 0.25,
    val_split: float = 0.2,
    max_per_class: Optional[int] = None,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Build training and validation datasets with advanced sampling.
    
    Args:
        ai_dir: Directory containing AI-generated frames
        real_dir: Directory containing real frames
        num_frames: Number of frames per clip
        frame_sampling: Frame sampling strategy
        min_stride_secs: Minimum time separation between frames
        fps: Frames per second
        framemix_prob: Probability of applying FrameMix
        val_split: Validation split ratio
        max_per_class: Maximum samples per class
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Get transforms
    from ai_detector.models.advanced_model import get_advanced_transforms
    train_transform = get_advanced_transforms('train')
    val_transform = get_advanced_transforms('val')
    
    # List image paths
    ai_imgs = list_images(ai_dir)
    real_imgs = list_images(real_dir)
    
    print(f"[INFO] Found {len(ai_imgs):,} AI frames in {ai_dir}")
    print(f"[INFO] Found {len(real_imgs):,} REAL frames in {real_dir}")
    
    if len(ai_imgs) == 0 or len(real_imgs) == 0:
        raise ValueError("One of the classes is empty – check your paths!")
    
    # Create datasets without FrameMix first
    ai_ds = FrameDataset(
        ai_imgs, 1, train_transform, num_frames, 
        frame_sampling, min_stride_secs, fps, 
        framemix_enabled=False
    )
    
    real_ds = FrameDataset(
        real_imgs, 0, train_transform, num_frames, 
        frame_sampling, min_stride_secs, fps, 
        framemix_enabled=False
    )
    
    # Apply max_per_class if specified
    if max_per_class:
        if len(ai_ds) > max_per_class:
            indices = list(range(len(ai_ds)))
            random.shuffle(indices)
            ai_ds = Subset(ai_ds, indices[:max_per_class])
        
        if len(real_ds) > max_per_class:
            indices = list(range(len(real_ds)))
            random.shuffle(indices)
            real_ds = Subset(real_ds, indices[:max_per_class])
    
    # Split into train and validation sets
    ai_train_size = int((1 - val_split) * len(ai_ds))
    ai_val_size = len(ai_ds) - ai_train_size
    
    real_train_size = int((1 - val_split) * len(real_ds))
    real_val_size = len(real_ds) - real_train_size
    
    ai_train_ds, ai_val_ds = random_split(
        ai_ds, [ai_train_size, ai_val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    real_train_ds, real_val_ds = random_split(
        real_ds, [real_train_size, real_val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create validation dataset (no FrameMix)
    val_ds = ConcatDataset([ai_val_ds, real_val_ds])
    
    # Now create training datasets with FrameMix
    if framemix_prob > 0:
        # Create new datasets with FrameMix enabled
        ai_train_ds_with_mix = FrameDataset(
            ai_imgs, 1, train_transform, num_frames, 
            frame_sampling, min_stride_secs, fps, 
            framemix_enabled=True, framemix_prob=framemix_prob
        )
        
        real_train_ds_with_mix = FrameDataset(
            real_imgs, 0, train_transform, num_frames, 
            frame_sampling, min_stride_secs, fps, 
            framemix_enabled=True, framemix_prob=framemix_prob
        )
        
        # Set each dataset as the other's FrameMix partner
        ai_train_ds_with_mix.framemix_partner = ai_train_ds_with_mix
        real_train_ds_with_mix.framemix_partner = real_train_ds_with_mix
        
        # Apply the same train/val split
        ai_train_ds = Subset(ai_train_ds_with_mix, range(ai_train_size))
        real_train_ds = Subset(real_train_ds_with_mix, range(real_train_size))
    
    # Combine train datasets
    train_ds = ConcatDataset([ai_train_ds, real_train_ds])
    
    print(f"[INFO] Train dataset: {len(train_ds)} samples")
    print(f"[INFO] Validation dataset: {len(val_ds)} samples")
    
    return train_ds, val_ds

def build_dataloaders(
    ai_dir: Path,
    real_dir: Path,
    num_frames: int,
    batch_size: int,
    frame_sampling: str = "rand_stride",
    min_stride_secs: float = 0.5,
    fps: float = 30.0,
    framemix_prob: float = 0.25,
    val_split: float = 0.2,
    max_per_class: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation dataloaders with advanced sampling.
    
    Args:
        ai_dir: Directory containing AI-generated frames
        real_dir: Directory containing real frames
        num_frames: Number of frames per clip
        batch_size: Batch size
        frame_sampling: Frame sampling strategy
        min_stride_secs: Minimum time separation between frames
        fps: Frames per second
        framemix_prob: Probability of applying FrameMix
        val_split: Validation split ratio
        max_per_class: Maximum samples per class
        num_workers: Number of worker processes
        seed: Random seed
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_ds, val_ds = build_datasets(
        ai_dir, real_dir, num_frames, 
        frame_sampling, min_stride_secs, fps, 
        framemix_prob, val_split, max_per_class, seed
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader