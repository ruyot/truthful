"""
Advanced Video Dataset for Multi-Frame AI Detection

This module implements a sophisticated dataset loader that:
- Samples multiple frames per video
- Ensures balanced batches
- Supports cross-validation splits
- Handles video-level data augmentation
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import logging
from typing import Tuple, List, Dict, Optional, Union
import random
import json
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class VideoFrameDataset(Dataset):
    """
    Advanced dataset for video-based AI detection with multi-frame sampling.
    
    Expected directory structure:
    data/
    ├── ai/
    │   ├── video1_frame_01.jpg
    │   ├── video1_frame_02.jpg
    │   ├── video1_frame_03.jpg
    │   ├── video2_frame_01.jpg
    │   └── ...
    └── real/
        ├── video1_frame_01.jpg
        ├── video1_frame_02.jpg
        └── ...
    """
    
    def __init__(
        self, 
        data_dir: str, 
        num_frames: int = 5,
        transform=None, 
        video_ids: Optional[List[str]] = None,
        max_videos_per_class: Optional[int] = None,
        frame_sampling_strategy: str = 'uniform'
    ):
        """
        Initialize the video frame dataset.
        
        Args:
            data_dir: Root directory containing 'ai' and 'real' subdirectories
            num_frames: Number of frames to sample per video
            transform: Image transforms to apply
            video_ids: Specific video IDs to include (for train/val splits)
            max_videos_per_class: Limit videos per class
            frame_sampling_strategy: 'uniform', 'random', or 'temporal'
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.frame_sampling_strategy = frame_sampling_strategy
        
        # Group frames by video
        self.video_data = self._group_frames_by_video()
        
        # Filter by video IDs if provided
        if video_ids:
            self.video_data = {vid: data for vid, data in self.video_data.items() if vid in video_ids}
        
        # Limit videos per class if specified
        if max_videos_per_class:
            self.video_data = self._limit_videos_per_class(max_videos_per_class)
        
        # Create video list for indexing
        self.video_list = list(self.video_data.keys())
        
        # Calculate class distribution
        self.class_distribution = self._calculate_class_distribution()
        
        logger.info(f"Loaded VideoFrameDataset:")
        logger.info(f"  Total videos: {len(self.video_list)}")
        logger.info(f"  AI videos: {self.class_distribution['ai']}")
        logger.info(f"  Real videos: {self.class_distribution['real']}")
        logger.info(f"  Frames per video: {num_frames}")
        logger.info(f"  Sampling strategy: {frame_sampling_strategy}")
    
    def _group_frames_by_video(self) -> Dict[str, Dict]:
        """Group frame files by video ID."""
        video_data = {}
        
        for class_name in ['ai', 'real']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            for frame_file in class_dir.glob("*.jpg"):
                # Extract video ID from filename (assumes format: videoID_frame_XX.jpg)
                parts = frame_file.stem.split('_frame_')
                if len(parts) != 2:
                    logger.warning(f"Unexpected filename format: {frame_file}")
                    continue
                
                video_id = parts[0]
                frame_num = int(parts[1])
                
                if video_id not in video_data:
                    video_data[video_id] = {
                        'class': class_name,
                        'label': 1 if class_name == 'ai' else 0,
                        'frames': []
                    }
                
                video_data[video_id]['frames'].append({
                    'path': frame_file,
                    'frame_num': frame_num
                })
        
        # Sort frames by frame number for each video
        for video_id in video_data:
            video_data[video_id]['frames'].sort(key=lambda x: x['frame_num'])
        
        return video_data
    
    def _limit_videos_per_class(self, max_videos: int) -> Dict[str, Dict]:
        """Limit the number of videos per class."""
        ai_videos = {vid: data for vid, data in self.video_data.items() if data['class'] == 'ai'}
        real_videos = {vid: data for vid, data in self.video_data.items() if data['class'] == 'real'}
        
        # Randomly sample videos
        if len(ai_videos) > max_videos:
            ai_video_ids = random.sample(list(ai_videos.keys()), max_videos)
            ai_videos = {vid: ai_videos[vid] for vid in ai_video_ids}
        
        if len(real_videos) > max_videos:
            real_video_ids = random.sample(list(real_videos.keys()), max_videos)
            real_videos = {vid: real_videos[vid] for vid in real_video_ids}
        
        return {**ai_videos, **real_videos}
    
    def _calculate_class_distribution(self) -> Dict[str, int]:
        """Calculate class distribution."""
        distribution = {'ai': 0, 'real': 0}
        for video_data in self.video_data.values():
            distribution[video_data['class']] += 1
        return distribution
    
    def _sample_frames(self, frames: List[Dict], num_frames: int) -> List[Dict]:
        """Sample frames from a video based on the sampling strategy."""
        if len(frames) <= num_frames:
            # If we have fewer frames than needed, repeat some frames
            sampled = frames.copy()
            while len(sampled) < num_frames:
                sampled.extend(frames[:num_frames - len(sampled)])
            return sampled[:num_frames]
        
        if self.frame_sampling_strategy == 'uniform':
            # Uniformly sample frames across the video
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            return [frames[i] for i in indices]
        
        elif self.frame_sampling_strategy == 'random':
            # Randomly sample frames
            return random.sample(frames, num_frames)
        
        elif self.frame_sampling_strategy == 'temporal':
            # Sample frames with temporal consistency (consecutive frames)
            start_idx = random.randint(0, max(0, len(frames) - num_frames))
            return frames[start_idx:start_idx + num_frames]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.frame_sampling_strategy}")
    
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a video sample.
        
        Returns:
            frames: Tensor of shape (num_frames, 3, H, W)
            label: 0 for real, 1 for AI
            video_id: Video identifier
        """
        video_id = self.video_list[idx]
        video_data = self.video_data[video_id]
        
        # Sample frames
        sampled_frames = self._sample_frames(video_data['frames'], self.num_frames)
        
        # Load and transform frames
        frame_tensors = []
        for frame_info in sampled_frames:
            try:
                # Load image
                image = Image.open(frame_info['path']).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                frame_tensors.append(image)
                
            except Exception as e:
                logger.warning(f"Error loading frame {frame_info['path']}: {e}")
                # Create a black frame as fallback
                if self.transform:
                    black_frame = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
                else:
                    black_frame = torch.zeros(3, 224, 224)
                frame_tensors.append(black_frame)
        
        # Stack frames
        frames = torch.stack(frame_tensors)  # (num_frames, 3, H, W)
        
        return frames, video_data['label'], video_id
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        total_videos = len(self.video_list)
        ai_count = self.class_distribution['ai']
        real_count = self.class_distribution['real']
        
        # Calculate weights inversely proportional to class frequency
        ai_weight = total_videos / (2 * ai_count) if ai_count > 0 else 1.0
        real_weight = total_videos / (2 * real_count) if real_count > 0 else 1.0
        
        return torch.tensor([real_weight, ai_weight], dtype=torch.float32)
    
    def get_video_level_split(self, val_split: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str]]:
        """
        Create video-level train/validation split to prevent data leakage.
        
        Args:
            val_split: Fraction of videos for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_video_ids, val_video_ids)
        """
        ai_videos = [vid for vid, data in self.video_data.items() if data['class'] == 'ai']
        real_videos = [vid for vid, data in self.video_data.items() if data['class'] == 'real']
        
        # Split each class separately to maintain balance
        ai_train, ai_val = train_test_split(
            ai_videos, test_size=val_split, random_state=random_state
        )
        real_train, real_val = train_test_split(
            real_videos, test_size=val_split, random_state=random_state
        )
        
        train_videos = ai_train + real_train
        val_videos = ai_val + real_val
        
        logger.info(f"Video-level split:")
        logger.info(f"  Train: {len(train_videos)} videos ({len(ai_train)} AI, {len(real_train)} real)")
        logger.info(f"  Val: {len(val_videos)} videos ({len(ai_val)} AI, {len(real_val)} real)")
        
        return train_videos, val_videos

def create_balanced_dataloader(
    dataset: VideoFrameDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a balanced dataloader using weighted sampling.
    
    Args:
        dataset: VideoFrameDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (ignored if using weighted sampling)
        
    Returns:
        DataLoader with balanced sampling
    """
    # Calculate sample weights for balanced sampling
    sample_weights = []
    for video_id in dataset.video_list:
        video_data = dataset.video_data[video_id]
        if video_data['class'] == 'ai':
            weight = 1.0 / dataset.class_distribution['ai']
        else:
            weight = 1.0 / dataset.class_distribution['real']
        sample_weights.append(weight)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Ensure consistent batch sizes
    )

def create_advanced_dataloaders(
    data_dir: str,
    num_frames: int = 5,
    batch_size: int = 16,
    val_split: float = 0.2,
    train_transform=None,
    val_transform=None,
    max_videos_per_class: Optional[int] = None,
    num_workers: int = 4,
    frame_sampling_strategy: str = 'uniform',
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create advanced train and validation dataloaders with video-level splits.
    
    Args:
        data_dir: Root data directory
        num_frames: Number of frames per video
        batch_size: Batch size
        val_split: Validation split ratio
        train_transform: Training transforms
        val_transform: Validation transforms
        max_videos_per_class: Max videos per class
        num_workers: Number of worker processes
        frame_sampling_strategy: Frame sampling strategy
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    # Create full dataset to get video splits
    full_dataset = VideoFrameDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        transform=None,  # No transform for splitting
        max_videos_per_class=max_videos_per_class,
        frame_sampling_strategy=frame_sampling_strategy
    )
    
    # Get video-level split
    train_video_ids, val_video_ids = full_dataset.get_video_level_split(
        val_split=val_split, 
        random_state=random_state
    )
    
    # Create separate datasets with transforms
    train_dataset = VideoFrameDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        transform=train_transform,
        video_ids=train_video_ids,
        frame_sampling_strategy=frame_sampling_strategy
    )
    
    val_dataset = VideoFrameDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        transform=val_transform,
        video_ids=val_video_ids,
        frame_sampling_strategy='uniform'  # Use uniform sampling for validation
    )
    
    # Create balanced dataloaders
    train_loader = create_balanced_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Dataset info
    dataset_info = {
        'total_videos': len(full_dataset.video_list),
        'train_videos': len(train_video_ids),
        'val_videos': len(val_video_ids),
        'train_ai': len([vid for vid in train_video_ids if full_dataset.video_data[vid]['class'] == 'ai']),
        'train_real': len([vid for vid in train_video_ids if full_dataset.video_data[vid]['class'] == 'real']),
        'val_ai': len([vid for vid in val_video_ids if full_dataset.video_data[vid]['class'] == 'ai']),
        'val_real': len([vid for vid in val_video_ids if full_dataset.video_data[vid]['class'] == 'real']),
        'frames_per_video': num_frames
    }
    
    logger.info("Created advanced dataloaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Dataset info: {dataset_info}")
    
    return train_loader, val_loader, dataset_info

if __name__ == "__main__":
    # Test the dataset
    from ai_detector.models.advanced_model import get_advanced_transforms
    
    data_dir = "data"
    
    # Create transforms
    train_transform = get_advanced_transforms('train')
    val_transform = get_advanced_transforms('val')
    
    # Create dataloaders
    train_loader, val_loader, info = create_advanced_dataloaders(
        data_dir=data_dir,
        num_frames=5,
        batch_size=4,
        train_transform=train_transform,
        val_transform=val_transform,
        max_videos_per_class=10  # Small test
    )
    
    print("Dataset Info:", info)
    
    # Test a batch
    for batch_idx, (frames, labels, video_ids) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Labels: {labels}")
        print(f"  Video IDs: {video_ids}")
        if batch_idx >= 2:
            break