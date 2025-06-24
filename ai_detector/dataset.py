"""
Dataset utilities for AI detector training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
from typing import Tuple, List, Optional, Union
import random
from typing import Any

logger = logging.getLogger(__name__)

class AIDetectorDataset(Dataset):
    """
    Dataset for AI vs Real image classification.
    
    Expected directory structure:
    data/
    ├── ai/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    └── real/
        ├── image1.jpg
        ├── image2.png
        └── ...
    """
    
    def __init__(self, data_dir: str, transform: Optional[Any] = None, max_samples_per_class: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing 'ai' and 'real' subdirectories
            transform: Image transforms to apply
            max_samples_per_class: Limit samples per class (for balanced training)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load AI images (label = 1)
        ai_dir = os.path.join(data_dir, 'ai')
        if os.path.exists(ai_dir):
            ai_images = self._load_images_from_dir(ai_dir, max_samples_per_class)
            self.samples.extend(ai_images)
            self.labels.extend([1] * len(ai_images))
            logger.info(f"Loaded {len(ai_images)} AI images")
        
        # Load Real images (label = 0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            real_images = self._load_images_from_dir(real_dir, max_samples_per_class)
            self.samples.extend(real_images)
            self.labels.extend([0] * len(real_images))
            logger.info(f"Loaded {len(real_images)} real images")
        
        # Shuffle the dataset
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        
        logger.info(f"Total dataset size: {len(self.samples)} images")
        logger.info(f"Class distribution - Real: {self.labels.count(0)}, AI: {self.labels.count(1)}")
    
    def _load_images_from_dir(self, directory: str, max_samples: Optional[int] = None) -> List[str]:
        """Load image paths from directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                images.append(os.path.join(directory, filename))
        
        # Limit samples if specified
        if max_samples and len(images) > max_samples:
            images = random.sample(images, max_samples)
        
        return images
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Convert PIL Image to tensor if no transform
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            
            return image, label
        
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a random other sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

def create_dataloaders(
    data_dir: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    val_split: float = 0.2,
    max_samples_per_class: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Root data directory
        train_transform: Training transforms
        val_transform: Validation transforms
        batch_size: Batch size
        val_split: Validation split ratio
        max_samples_per_class: Max samples per class
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = AIDetectorDataset(
        data_dir=data_dir,
        transform=None,  # We'll apply transforms later
        max_samples_per_class=max_samples_per_class
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))
    
    # Create separate datasets with different transforms
    train_dataset = AIDetectorDataset(data_dir, train_transform, max_samples_per_class)
    val_dataset = AIDetectorDataset(data_dir, val_transform, max_samples_per_class)
    
    # Use subset of indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    return train_loader, val_loader