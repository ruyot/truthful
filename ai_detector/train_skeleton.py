"""
Enhanced Skeleton-based training script with advanced features:

- Multiple frame sampling strategies (sequential, uniform, rand_stride)
- FrameMix augmentation for improved generalization
- TensorBoard logging for comprehensive metrics
- Early stopping on fused AUC
- Intermediate checkpoint saving
- Temporal CNN head option

Run example:
```bash
cd ~/truthful  # important so relative paths work!
PYTHONPATH=. python ai_detector/train_skeleton.py \
  --ai_dir   ai_detector/data/ai \
  --real_dir ai_detector/data/real \
  --epochs   40 \
  --batch_size 8 \
  --num_frames 7 \
  --backbone convnext_tiny \
  --frame_sampling rand_stride \
  --min_stride_secs 0.7 \
  --ckpt_freq 5 \
  --temporal_head temporal_cnn
```
"""

import argparse
import os
import math
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from ai_detector.models.skeleton_model import create_skeleton_model, SkeletonLoss
from ai_detector.models.advanced_model import get_advanced_transforms
from ai_detector.datasets.frame_dataset import build_dataloaders

def validate(
    model: nn.Module, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """
    Validate the model and compute metrics.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch (for logging)
        writer: TensorBoard writer
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Validating"):
            frames, labels = frames.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            
            # Compute loss
            loss_dict = criterion(outputs, labels)
            val_loss += loss_dict['total'].item()
            
            # Store predictions and labels
            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeddings.append(outputs['embeddings'].cpu().numpy())
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Classification accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Separate AI and Real embeddings
    ai_mask = all_labels == 1
    real_mask = all_labels == 0
    
    ai_embeddings = all_embeddings[ai_mask]
    real_embeddings = all_embeddings[real_mask]
    
    # Compute skeletons
    if len(ai_embeddings) > 0 and len(real_embeddings) > 0:
        model.compute_skeletons(ai_embeddings, real_embeddings)
        
        # Get skeleton predictions
        embeddings_tensor = torch.tensor(all_embeddings, device=device)
        skeleton_results = model.skeleton_predict(embeddings_tensor)
        skeleton_probs = skeleton_results['skeleton_probs'].cpu().numpy()
        
        # Calculate skeleton AUC
        skeleton_auc = roc_auc_score(all_labels, skeleton_probs)
        
        # Get fused predictions
        fused_results = model.fused_predict(frames, fusion_weight=0.5)
        fused_probs = fused_results['fused_probs'].cpu().numpy()
        
        # Calculate fused AUC
        fused_auc = roc_auc_score(all_labels, fused_probs)
    else:
        skeleton_auc = 0.0
        fused_auc = 0.0
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # Log metrics
    metrics = {
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': accuracy,
        'val_auc': auc,
        'val_skeleton_auc': skeleton_auc,
        'val_fused_auc': fused_auc
    }
    
    if writer:
        for name, value in metrics.items():
            writer.add_scalar(f'Validation/{name}', value, epoch)
    
    return metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    save_path: Path,
    is_best: bool = False,
    is_intermediate: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Validation metrics
        args: Training arguments
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
        is_intermediate: Whether this is an intermediate checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args),
        'config': {
            'model': {
                'backbone': args.backbone,
                'num_frames': args.num_frames,
                'freeze_backbone': args.freeze_backbone,
                'enable_multitask': args.enable_multitask,
                'temporal_head': args.temporal_head
            },
            'training': {
                'frame_sampling': args.frame_sampling,
                'min_stride_secs': args.min_stride_secs,
                'framemix_prob': args.framemix_prob
            }
        }
    }
    
    # Determine save path
    if is_best:
        save_file = save_path.parent / "best_skeleton_model.pt"
    elif is_intermediate:
        save_file = save_path.parent / f"skeleton_model_epoch_{epoch}.pt"
    else:
        save_file = save_path
    
    # Save checkpoint
    torch.save(checkpoint, save_file)
    print(f"Checkpoint saved to {save_file}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced skeleton model trainer")
    
    # Data arguments
    parser.add_argument('--ai_dir', required=True, type=Path, help="Directory with AI frames")
    parser.add_argument('--real_dir', required=True, type=Path, help="Directory with real frames")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument('--backbone', type=str, 
                        choices=['resnet50', 'efficientnet_b3', 'convnext_tiny'], 
                        default='resnet50', help="Backbone architecture")
    parser.add_argument('--num_frames', type=int, default=5, help="Frames per clip")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze backbone weights")
    parser.add_argument('--enable_multitask', action='store_true', help="Enable multi-task learning")
    parser.add_argument('--temporal_head', type=str, choices=['attention', 'temporal_cnn'], 
                        default='attention', help="Temporal aggregation method")
    
    # Frame sampling arguments
    parser.add_argument('--frame_sampling', type=str, 
                        choices=['sequential', 'uniform', 'rand_stride'], 
                        default='rand_stride', help="Frame sampling strategy")
    parser.add_argument('--min_stride_secs', type=float, default=0.5, 
                        help="Minimum time separation between frames (seconds)")
    parser.add_argument('--fps', type=float, default=30.0, 
                        help="Frames per second (for time calculations)")
    parser.add_argument('--framemix_prob', type=float, default=0.25, 
                        help="Probability of applying FrameMix augmentation")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes")
    parser.add_argument('--max_per_class', type=int, default=None, help="Max samples per class")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument('--save_path', type=Path, default=Path('results/skeleton_model.pt'), 
                        help="Path to save final model")
    parser.add_argument('--ckpt_freq', type=int, default=5, 
                        help="Save checkpoint every N epochs")
    parser.add_argument('--patience', type=int, default=10, 
                        help="Early stopping patience (epochs)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create TensorBoard writer
    log_dir = Path("logs") / f"skeleton_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    
    # Log hyperparameters
    writer.add_text("Hyperparameters", str(vars(args)))
    
    # ---------------------------------------------------------------------
    # Data & model
    # ---------------------------------------------------------------------
    print(f"Building dataloaders with {args.frame_sampling} sampling strategy...")
    train_loader, val_loader = build_dataloaders(
        ai_dir=args.ai_dir,
        real_dir=args.real_dir,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        frame_sampling=args.frame_sampling,
        min_stride_secs=args.min_stride_secs,
        fps=args.fps,
        framemix_prob=args.framemix_prob,
        val_split=args.val_split,
        max_per_class=args.max_per_class,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Creating model with {args.backbone} backbone and {args.temporal_head} temporal head...")
    model = create_skeleton_model(
        backbone=args.backbone,
        num_frames=args.num_frames,
        freeze_backbone=args.freeze_backbone,
        enable_multitask=args.enable_multitask
    ).to(device)
    
    # Set temporal head for base detector
    model.base_detector.temporal_head = args.temporal_head
    
    # Recreate aggregator with specified temporal head
    feature_dim = model.base_detector.feature_dim
    if args.temporal_head == 'temporal_cnn':
        model.base_detector.aggregator = TemporalCNNAggregator(feature_dim, args.num_frames).to(device)
    else:
        model.base_detector.aggregator = AttentionAggregator(feature_dim, args.num_frames).to(device)
    
    # Create loss function and optimizer
    criterion = SkeletonLoss(
        classification_weight=1.0, 
        prompt_weight=0.1 if args.enable_multitask else 0.0,
        skeleton_weight=0.2
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    print(f"Starting training for {args.epochs} epochs...")
    
    best_fused_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            frames, labels = frames.to(device), labels.to(device)
            
            # Check for multi-task learning
            if args.enable_multitask:
                # Generate dummy prompts for training
                # In a real scenario, you would use actual prompts
                prompts = torch.randint(0, 1000, (labels.size(0), 20), device=device)
                outputs = model(frames, prompts)
            else:
                outputs = model(frames)
            
            # Compute loss
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total']
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > 0.5).float()
            
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_loss = running_loss / len(train_loader)
        
        # Log training metrics
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer)
        
        print(f"Epoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}, " +
              f"Accuracy: {val_metrics['val_accuracy']:.4f}, " +
              f"AUC: {val_metrics['val_auc']:.4f}, " +
              f"Skeleton AUC: {val_metrics['val_skeleton_auc']:.4f}, " +
              f"Fused AUC: {val_metrics['val_fused_auc']:.4f}")
        
        # Check for best model
        is_best = val_metrics['val_fused_auc'] > best_fused_auc
        if is_best:
            best_fused_auc = val_metrics['val_fused_auc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_checkpoint(model, optimizer, epoch, val_metrics, args, args.save_path, is_best=True)
        else:
            patience_counter += 1
        
        # Save intermediate checkpoint
        if epoch % args.ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, args, args.save_path, is_intermediate=True)
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, args, args.save_path)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs. Best epoch: {best_epoch}")
            break
    
    # ---------------------------------------------------------------------
    # Final steps
    # ---------------------------------------------------------------------
    print(f"Training completed. Best fused AUC: {best_fused_auc:.4f} at epoch {best_epoch}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()