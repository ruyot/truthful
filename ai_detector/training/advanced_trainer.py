"""
Advanced Training Pipeline for AI Video Classifier

This module implements a robust training pipeline with:
- Focal Loss for handling class imbalance
- Comprehensive metrics tracking
- Cross-validation support
- Advanced optimization strategies
- Automatic model checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from ai_detector.models.advanced_model import (
    AdvancedAIDetector, FocalLoss, create_advanced_model, 
    get_advanced_transforms, MODEL_CONFIGS
)
from ai_detector.datasets.video_dataset import create_advanced_dataloaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """
    Advanced trainer for AI video detection with comprehensive features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.setup_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f'logs/advanced_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Initialize model
        self.model = self.create_model()
        
        # Initialize dataloaders
        self.train_loader, self.val_loader, self.dataset_info = self.create_dataloaders()
        
        # Initialize loss function and optimizer
        self.criterion = self.create_loss_function()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Training state
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.patience_counter = 0
        self.best_loss = float('inf')
        
        logger.info(f"Advanced trainer initialized on {self.device}")
        logger.info(f"Model: {self.config['model']['backbone']}")
        logger.info(f"Dataset: {self.dataset_info}")
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = ['weights', 'logs', 'plots', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_model(self) -> AdvancedAIDetector:
        """Create and initialize the model."""
        model = create_advanced_model(**self.config['model'])
        model = model.to(self.device)
        
        # Initialize weights for new layers
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
        model.aggregator.apply(init_weights)
        
        return model
    
    def create_dataloaders(self):
        """Create train and validation dataloaders."""
        train_transform = get_advanced_transforms('train', self.config['data']['image_size'])
        val_transform = get_advanced_transforms('val', self.config['data']['image_size'])
        
        return create_advanced_dataloaders(
            data_dir=self.config['data']['data_dir'],
            num_frames=self.config['model']['num_frames'],
            batch_size=self.config['training']['batch_size'],
            val_split=self.config['data']['val_split'],
            train_transform=train_transform,
            val_transform=val_transform,
            max_videos_per_class=self.config['data'].get('max_videos_per_class'),
            num_workers=self.config['data']['num_workers'],
            frame_sampling_strategy=self.config['data']['frame_sampling_strategy'],
            random_state=self.config['data']['random_state']
        )
    
    def create_loss_function(self):
        """Create the loss function."""
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'focal':
            return FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        elif loss_config['type'] == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_config['type']}")
    
    def create_optimizer(self):
        """Create the optimizer with different learning rates for backbone and head."""
        opt_config = self.config['training']['optimizer']
        
        # Different learning rates for backbone and aggregator
        if self.config['model']['freeze_backbone']:
            params = self.model.aggregator.parameters()
        else:
            params = [
                {
                    'params': self.model.backbone.parameters(), 
                    'lr': opt_config['lr'] * opt_config['backbone_lr_ratio']
                },
                {
                    'params': self.model.aggregator.parameters(), 
                    'lr': opt_config['lr']
                }
            ]
        
        if opt_config['type'] == 'adam':
            return optim.Adam(
                params, 
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'adamw':
            return optim.AdamW(
                params, 
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'sgd':
            return optim.SGD(
                params, 
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    def create_scheduler(self):
        """Create the learning rate scheduler."""
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config['min_lr']
            )
        elif sched_config['type'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=sched_config['patience'],
                factor=sched_config['factor'],
                verbose=True
            )
        elif sched_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (frames, labels, video_ids) in enumerate(pbar):
            frames = frames.to(self.device)  # (batch_size, num_frames, 3, H, W)
            labels = labels.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip_norm'):
                clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, zero_division='warn')
        epoch_recall = recall_score(all_labels, all_preds, zero_division='warn')
        epoch_f1 = f1_score(all_labels, all_preds, zero_division='warn')
        
        # Calculate AUC if we have both classes
        if len(set(all_labels)) > 1:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        else:
            epoch_auc = 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1': epoch_f1,
            'auc': epoch_auc
        }
        
        # Log epoch metrics
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'Train/{metric_name.capitalize()}', value, epoch)
        
        logger.info(f'Train Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, '
                   f'F1={epoch_f1:.4f}, AUC={epoch_auc:.4f}')
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for frames, labels, video_ids in tqdm(self.val_loader, desc='Validating'):
                frames = frames.to(self.device)
                labels = labels.float().to(self.device)
                
                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, zero_division='warn')
        epoch_recall = recall_score(all_labels, all_preds, zero_division='warn')
        epoch_f1 = f1_score(all_labels, all_preds, zero_division='warn')
        
        if len(set(all_labels)) > 1:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        else:
            epoch_auc = 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1': epoch_f1,
            'auc': epoch_auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        # Log epoch metrics
        for metric_name, value in metrics.items():
            if metric_name not in ['predictions', 'labels', 'probabilities']:
                self.writer.add_scalar(f'Val/{metric_name.capitalize()}', value, epoch)
        
        logger.info(f'Val Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, '
                   f'F1={epoch_f1:.4f}, AUC={epoch_auc:.4f}')
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'model_config': self.config['model'],
            'training_config': self.config['training'],
            'dataset_info': self.dataset_info,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'weights/latest_advanced_checkpoint.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'weights/best_advanced_model.pt')
            logger.info(f'Saved best model with AUC={metrics["auc"]:.4f}, F1={metrics["f1"]:.4f}')
    
    def plot_confusion_matrix(self, labels: Any, preds: Any, epoch: int):
        """Plot and save confusion matrix."""
        # Convert to integers for confusion matrix
        labels_int = [int(x) for x in labels]
        preds_int = [int(x) for x in preds]
        
        cm = confusion_matrix(labels_int, preds_int)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(f'plots/confusion_matrix_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['train_acc'], label='Train')
        axes[0, 1].plot(self.training_history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        axes[1, 0].plot(self.training_history['train_f1'], label='Train')
        axes[1, 0].plot(self.training_history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC curves
        axes[1, 1].plot(self.training_history['train_auc'], label='Train')
        axes[1, 1].plot(self.training_history['val_auc'], label='Validation')
        axes[1, 1].set_title('AUC Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        logger.info("Starting advanced training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['train_auc'].append(train_metrics['auc'])
            self.training_history['val_auc'].append(val_metrics['auc'])
            self.training_history['learning_rates'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_val_f1 = val_metrics['f1']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Plot confusion matrix every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_confusion_matrix(
                    val_metrics['labels'], 
                    val_metrics['predictions'], 
                    epoch
                )
            
            # Early stopping
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['training']['patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Final plots and results
        self.plot_training_curves()
        
        # Save final training history
        with open('results/training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training completed!")
        logger.info(f"Best validation AUC: {self.best_val_auc:.4f}")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        
        self.writer.close()

def create_training_config(
    model_config: str = 'balanced',
    data_dir: str = 'data',
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3
) -> Dict[str, Any]:
    """
    Create a training configuration.
    
    Args:
        model_config: Model configuration ('fast', 'balanced', 'accurate')
        data_dir: Data directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Training configuration dictionary
    """
    config = {
        'model': MODEL_CONFIGS[model_config].copy(),
        'data': {
            'data_dir': data_dir,
            'val_split': 0.2,
            'image_size': 224,
            'num_workers': 4,
            'frame_sampling_strategy': 'uniform',
            'random_state': 42,
            'max_videos_per_class': None
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': 15,
            'grad_clip_norm': 1.0,
            'loss': {
                'type': 'focal',
                'alpha': 1.0,
                'gamma': 2.0
            },
            'optimizer': {
                'type': 'adamw',
                'lr': learning_rate,
                'weight_decay': 1e-4,
                'backbone_lr_ratio': 0.1
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6
            }
        }
    }
    
    return config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Advanced AI Video Classifier')
    
    # Model arguments
    parser.add_argument('--model_config', type=str, default='balanced',
                       choices=['fast', 'balanced', 'accurate'],
                       help='Model configuration preset')
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet50', 'efficientnet_b3', 'convnext_tiny'],
                       help='Override backbone architecture')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_videos', type=int, default=None, help='Max videos per class')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'bce'],
                       help='Loss function')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_training_config(
        model_config=args.model_config,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Override backbone if specified
    if args.backbone:
        config['model']['backbone'] = args.backbone
    
    # Override max videos if specified
    if args.max_videos:
        config['data']['max_videos_per_class'] = args.max_videos
    
    # Override loss function if specified
    config['training']['loss']['type'] = args.loss
    
    # Save configuration
    with open('results/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create and run trainer
    trainer = AdvancedTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()