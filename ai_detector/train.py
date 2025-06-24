"""
Training script for the AI detector model.

Usage:
    python train.py --data_dir data --epochs 50 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import logging
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import create_model, get_transforms
from dataset import create_dataloaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    """Training class for the AI detector model."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs('weights', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f'logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Create model
        self.model = create_model(
            freeze_backbone=args.freeze_backbone,
            dropout_rate=args.dropout_rate,
            hidden_dim=args.hidden_dim
        ).to(self.device)
        
        # Create dataloaders
        train_transform = get_transforms('train')
        val_transform = get_transforms('val')
        
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=args.data_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=args.batch_size,
            val_split=args.val_split,
            max_samples_per_class=args.max_samples_per_class,
            num_workers=args.num_workers
        )
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        
        if args.freeze_backbone:
            # Only optimize classifier parameters
            params = self.model.classifier.parameters()
        else:
            # Optimize all parameters with different learning rates
            params = [
                {'params': self.model.backbone.parameters(), 'lr': args.learning_rate * 0.1},
                {'params': self.model.classifier.parameters(), 'lr': args.learning_rate}
            ]
        
        self.optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
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
        
        self.train_losses.append(epoch_loss)
        
        # Log epoch metrics
        self.writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
        self.writer.add_scalar('Train/EpochAccuracy', epoch_acc, epoch)
        
        logger.info(f'Train Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}')
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.float().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = (outputs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, zero_division='warn')
        val_recall = recall_score(all_labels, all_preds, zero_division='warn')
        val_f1 = f1_score(all_labels, all_preds, zero_division='warn')
        val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        
        # Log metrics
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
        self.writer.add_scalar('Val/Precision', val_precision, epoch)
        self.writer.add_scalar('Val/Recall', val_recall, epoch)
        self.writer.add_scalar('Val/F1', val_f1, epoch)
        self.writer.add_scalar('Val/AUC', val_auc, epoch)
        
        logger.info(f'Val Epoch {epoch+1}: Loss={val_loss:.4f}, Acc={val_acc:.4f}, '
                   f'Precision={val_precision:.4f}, Recall={val_recall:.4f}, '
                   f'F1={val_f1:.4f}, AUC={val_auc:.4f}')
        
        return val_loss, val_acc, {
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_config': {
                'freeze_backbone': self.args.freeze_backbone,
                'dropout_rate': self.args.dropout_rate,
                'hidden_dim': self.args.hidden_dim
            },
            'args': vars(self.args)
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'weights/latest_checkpoint.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'weights/ai_detector.pt')
            logger.info(f'Saved best model with val_acc={val_acc:.4f}')
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, float(val_acc), bool(is_best))
            
            # Early stopping
            if epoch > 10 and val_loss > self.best_val_loss * 1.1:
                logger.info("Early stopping triggered")
                break
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        
        with open('weights/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train AI Detector Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--max_samples_per_class', type=int, default=None, help='Max samples per class')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()