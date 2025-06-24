"""
Training Pipeline for Skeleton-Based AI Video Classifier

This module implements training for the skeleton-based detector with:
1. Combined DFD + VidProM dataset training
2. Skeleton computation and storage
3. Multi-task learning with prompt embeddings
4. Distance-based matching evaluation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
from pathlib import Path

# Import our modules
from ai_detector.models.skeleton_model import (
    SkeletonBasedDetector, SkeletonLoss, create_skeleton_model
)
from ai_detector.models.advanced_model import get_advanced_transforms
from ai_detector.datasets.vidprom_dataset import CombinedVideoDataset
from ai_detector.datasets.video_dataset import create_balanced_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkeletonTrainer:
    """
    Trainer for skeleton-based AI video detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize skeleton trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.setup_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f'logs/skeleton_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Initialize model
        self.model = self.create_model()
        
        # Initialize dataset and dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Initialize loss function and optimizer
        self.criterion = self.create_loss_function()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Training state
        self.best_val_auc = 0.0
        self.best_skeleton_auc = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'skeleton_auc': [], 'fused_auc': []
        }
        
        logger.info(f"Skeleton trainer initialized on {self.device}")
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = ['weights', 'logs', 'plots', 'skeletons', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_model(self) -> SkeletonBasedDetector:
        """Create skeleton-based model."""
        model = create_skeleton_model(**self.config['model'])
        return model.to(self.device)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        train_transform = get_advanced_transforms('train')
        val_transform = get_advanced_transforms('val')
        
        # Create combined dataset
        full_dataset = CombinedVideoDataset(
            dfd_dir=self.config['data']['dfd_dir'],
            vidprom_dir=self.config['data']['vidprom_dir'],
            num_frames=self.config['model']['num_frames'],
            transform=None,  # Will apply transforms separately
            max_videos_per_source=self.config['data'].get('max_videos_per_source')
        )
        
        # Split into train/val
        total_videos = len(full_dataset)
        val_size = int(self.config['data']['val_split'] * total_videos)
        train_size = total_videos - val_size
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_videos))
        
        # Create separate datasets with transforms
        train_dataset = CombinedVideoDataset(
            dfd_dir=self.config['data']['dfd_dir'],
            vidprom_dir=self.config['data']['vidprom_dir'],
            num_frames=self.config['model']['num_frames'],
            transform=train_transform,
            max_videos_per_source=self.config['data'].get('max_videos_per_source')
        )
        
        val_dataset = CombinedVideoDataset(
            dfd_dir=self.config['data']['dfd_dir'],
            vidprom_dir=self.config['data']['vidprom_dir'],
            num_frames=self.config['model']['num_frames'],
            transform=val_transform,
            max_videos_per_source=self.config['data'].get('max_videos_per_source')
        )
        
        # Create subset datasets
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created dataloaders:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_loss_function(self) -> SkeletonLoss:
        """Create skeleton loss function."""
        return SkeletonLoss(
            classification_weight=self.config['training']['loss']['classification_weight'],
            prompt_weight=self.config['training']['loss']['prompt_weight'],
            skeleton_weight=self.config['training']['loss']['skeleton_weight']
        )
    
    def create_optimizer(self):
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def create_scheduler(self):
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=1e-6
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_losses = {
            'total': 0.0, 'classification': 0.0, 
            'prompt': 0.0, 'skeleton': 0.0
        }
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (frames, labels, video_ids) in enumerate(pbar):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Generate dummy prompts for multi-task learning
            prompts = None
            if self.model.enable_multitask:
                prompts = torch.randint(0, 1000, (len(labels), 20)).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            results = self.model(frames, prompts)
            
            # Compute losses
            losses = self.criterion(results, labels)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Statistics
            for key in running_losses:
                running_losses[key] += losses[key].item()
            
            preds = torch.sigmoid(results['logits']) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses["total"].item():.4f}',
                'Cls': f'{losses["classification"].item():.4f}',
                'Skel': f'{losses["skeleton"].item():.4f}'
            })
        
        # Calculate epoch metrics
        epoch_metrics = {}
        for key in running_losses:
            epoch_metrics[f'train_{key}_loss'] = running_losses[key] / len(self.train_loader)
        
        epoch_metrics['train_acc'] = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Log metrics
        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        logger.info(f'Train Epoch {epoch+1}: '
                   f'Loss={epoch_metrics["train_total_loss"]:.4f}, '
                   f'Acc={epoch_metrics["train_acc"]:.4f}')
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        running_losses = {
            'total': 0.0, 'classification': 0.0, 
            'prompt': 0.0, 'skeleton': 0.0
        }
        all_preds = []
        all_labels = []
        all_probs = []
        all_embeddings = []
        
        with torch.no_grad():
            for frames, labels, video_ids in tqdm(self.val_loader, desc='Validating'):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Generate dummy prompts
                prompts = None
                if self.model.enable_multitask:
                    prompts = torch.randint(0, 1000, (len(labels), 20)).to(self.device)
                
                # Forward pass
                results = self.model(frames, prompts)
                
                # Compute losses
                losses = self.criterion(results, labels)
                
                # Statistics
                for key in running_losses:
                    running_losses[key] += losses[key].item()
                
                probs = torch.sigmoid(results['logits'])
                preds = probs > 0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_embeddings.append(results['embeddings'].cpu().numpy())
        
        # Calculate epoch metrics
        epoch_metrics = {}
        for key in running_losses:
            epoch_metrics[f'val_{key}_loss'] = running_losses[key] / len(self.val_loader)
        
        epoch_metrics['val_acc'] = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate AUC if possible
        if len(set(all_labels)) > 1:
            from sklearn.metrics import roc_auc_score
            epoch_metrics['val_auc'] = roc_auc_score(all_labels, all_probs)
        else:
            epoch_metrics['val_auc'] = 0.0
        
        # Store embeddings for skeleton computation
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels_np = np.array(all_labels)
        
        # Separate AI and Real embeddings
        ai_embeddings = all_embeddings[all_labels_np == 1]
        real_embeddings = all_embeddings[all_labels_np == 0]
        
        # Compute skeletons if we have both classes
        skeleton_auc = 0.0
        if len(ai_embeddings) > 0 and len(real_embeddings) > 0:
            self.model.compute_skeletons(ai_embeddings, real_embeddings)
            
            # Test skeleton prediction
            test_embeddings = torch.tensor(all_embeddings, device=self.device)
            skeleton_results = self.model.skeleton_predict(test_embeddings)
            skeleton_probs = skeleton_results['skeleton_probs'].cpu().numpy()
            
            if len(set(all_labels)) > 1:
                skeleton_auc = roc_auc_score(all_labels, skeleton_probs)
            
            epoch_metrics['skeleton_auc'] = skeleton_auc
        
        # Log metrics
        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        logger.info(f'Val Epoch {epoch+1}: '
                   f'Loss={epoch_metrics["val_total_loss"]:.4f}, '
                   f'Acc={epoch_metrics["val_acc"]:.4f}, '
                   f'AUC={epoch_metrics["val_auc"]:.4f}, '
                   f'Skeleton AUC={skeleton_auc:.4f}')
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'weights/latest_skeleton_checkpoint.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'weights/best_skeleton_model.pt')
            # Also save skeletons
            self.model.save_skeletons('skeletons')
            logger.info(f'Saved best skeleton model with AUC={metrics.get("val_auc", 0):.4f}')
    
    def train(self):
        """Main training loop."""
        logger.info("Starting skeleton-based training...")
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['train_total_loss'])
            self.training_history['val_loss'].append(val_metrics['val_total_loss'])
            self.training_history['train_acc'].append(train_metrics['train_acc'])
            self.training_history['val_acc'].append(val_metrics['val_acc'])
            self.training_history['skeleton_auc'].append(val_metrics.get('skeleton_auc', 0))
            
            # Check for best model
            current_auc = val_metrics.get('val_auc', 0)
            is_best = current_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = current_auc
                self.best_skeleton_auc = val_metrics.get('skeleton_auc', 0)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        # Save final training history
        with open('results/skeleton_training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Skeleton training completed!")
        logger.info(f"Best validation AUC: {self.best_val_auc:.4f}")
        logger.info(f"Best skeleton AUC: {self.best_skeleton_auc:.4f}")
        
        self.writer.close()

def create_skeleton_config(
    dfd_dir: str = "data",
    vidprom_dir: str = "data",
    backbone: str = 'efficientnet_b3',
    epochs: int = 100,
    batch_size: int = 16,
    enable_multitask: bool = False
) -> Dict[str, Any]:
    """
    Create skeleton training configuration.
    
    Args:
        dfd_dir: DFD dataset directory
        vidprom_dir: VidProM dataset directory
        backbone: CNN backbone
        epochs: Training epochs
        batch_size: Batch size
        enable_multitask: Enable multi-task learning
        
    Returns:
        Configuration dictionary
    """
    config = {
        'model': {
            'backbone': backbone,
            'num_frames': 5,
            'freeze_backbone': False,
            'enable_multitask': enable_multitask
        },
        'data': {
            'dfd_dir': dfd_dir,
            'vidprom_dir': vidprom_dir,
            'val_split': 0.2,
            'num_workers': 4,
            'max_videos_per_source': None
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'loss': {
                'classification_weight': 1.0,
                'prompt_weight': 0.1 if enable_multitask else 0.0,
                'skeleton_weight': 0.1
            }
        }
    }
    
    return config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Skeleton-Based AI Video Classifier')
    
    # Data arguments
    parser.add_argument('--dfd_dir', type=str, default='data', help='DFD dataset directory')
    parser.add_argument('--vidprom_dir', type=str, default='data', help='VidProM dataset directory')
    parser.add_argument('--max_videos', type=int, default=None, help='Max videos per source')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                       choices=['resnet50', 'efficientnet_b3', 'convnext_tiny'],
                       help='CNN backbone')
    parser.add_argument('--enable_multitask', action='store_true',
                       help='Enable multi-task prompt prediction')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_skeleton_config(
        dfd_dir=args.dfd_dir,
        vidprom_dir=args.vidprom_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        enable_multitask=args.enable_multitask
    )
    
    # Override max videos if specified
    if args.max_videos:
        config['data']['max_videos_per_source'] = args.max_videos
    
    # Override learning rate
    config['training']['lr'] = args.lr
    
    # Save configuration
    with open('results/skeleton_training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create and run trainer
    trainer = SkeletonTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()