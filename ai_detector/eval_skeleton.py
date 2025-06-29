"""
Evaluation script for skeleton-based AI detector.

This script loads a trained skeleton model and evaluates it on a directory of videos,
providing comprehensive metrics including accuracy, AUC, and structural matching scores.

Usage:
    python ai_detector/eval_skeleton.py --model_path results/best_skeleton_model.pt \
        --ai_dir data/test/ai --real_dir data/test/real
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ai_detector.models.skeleton_model import create_skeleton_model
from ai_detector.models.advanced_model import get_advanced_transforms
from ai_detector.datasets.frame_dataset import build_dataloaders, list_images

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained skeleton model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
    else:
        # Try to extract from args
        args = checkpoint.get('args', {})
        model_config = {
            'backbone': args.get('backbone', 'resnet50'),
            'num_frames': args.get('num_frames', 5),
            'freeze_backbone': args.get('freeze_backbone', False),
            'enable_multitask': args.get('enable_multitask', False)
        }
    
    # Create model
    model = create_skeleton_model(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from {model_path}")
    print(f"Model config: {model_config}")
    
    return model

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    fusion_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for evaluation
        device: Device to use
        fusion_weight: Weight for skeleton vs base prediction fusion
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_labels = []
    all_base_preds = []
    all_base_probs = []
    all_skeleton_probs = []
    all_fused_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Evaluating"):
            frames, labels = frames.to(device), labels.to(device)
            
            # Get base predictions
            outputs = model(frames)
            base_probs = torch.sigmoid(outputs['logits'])
            base_preds = (base_probs > 0.5).float()
            
            # Store embeddings
            embeddings = outputs['embeddings']
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_base_preds.extend(base_preds.cpu().numpy())
            all_base_probs.extend(base_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_base_preds = np.array(all_base_preds)
    all_base_probs = np.array(all_base_probs)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
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
        all_skeleton_probs = skeleton_results['skeleton_probs'].cpu().numpy()
        
        # Compute fused predictions
        all_fused_probs = (1 - fusion_weight) * all_base_probs + fusion_weight * all_skeleton_probs
    else:
        all_skeleton_probs = np.zeros_like(all_base_probs)
        all_fused_probs = all_base_probs
    
    # Calculate metrics
    base_accuracy = accuracy_score(all_labels, all_base_preds)
    
    skeleton_preds = (all_skeleton_probs > 0.5).astype(float)
    skeleton_accuracy = accuracy_score(all_labels, skeleton_preds)
    
    fused_preds = (all_fused_probs > 0.5).astype(float)
    fused_accuracy = accuracy_score(all_labels, fused_preds)
    
    # Calculate AUC if we have both classes
    if len(np.unique(all_labels)) > 1:
        base_auc = roc_auc_score(all_labels, all_base_probs)
        skeleton_auc = roc_auc_score(all_labels, all_skeleton_probs)
        fused_auc = roc_auc_score(all_labels, all_fused_probs)
    else:
        base_auc = 0.0
        skeleton_auc = 0.0
        fused_auc = 0.0
    
    # Calculate confusion matrices
    base_cm = confusion_matrix(all_labels, all_base_preds)
    skeleton_cm = confusion_matrix(all_labels, skeleton_preds)
    fused_cm = confusion_matrix(all_labels, fused_preds)
    
    # Return metrics
    return {
        'base_accuracy': base_accuracy,
        'skeleton_accuracy': skeleton_accuracy,
        'fused_accuracy': fused_accuracy,
        'base_auc': base_auc,
        'skeleton_auc': skeleton_auc,
        'fused_auc': fused_auc,
        'base_cm': base_cm,
        'skeleton_cm': skeleton_cm,
        'fused_cm': fused_cm,
        'all_labels': all_labels,
        'all_base_probs': all_base_probs,
        'all_skeleton_probs': all_skeleton_probs,
        'all_fused_probs': all_fused_probs
    }

def plot_confusion_matrices(metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot confusion matrices for base, skeleton, and fused predictions.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot base confusion matrix
    cm = metrics['base_cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
    axes[0].set_title(f"Base Model\nAccuracy: {metrics['base_accuracy']:.4f}, AUC: {metrics['base_auc']:.4f}")
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot skeleton confusion matrix
    cm = metrics['skeleton_cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
    axes[1].set_title(f"Skeleton Model\nAccuracy: {metrics['skeleton_accuracy']:.4f}, AUC: {metrics['skeleton_auc']:.4f}")
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    # Plot fused confusion matrix
    cm = metrics['fused_cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
    axes[2].set_title(f"Fused Model\nAccuracy: {metrics['fused_accuracy']:.4f}, AUC: {metrics['fused_auc']:.4f}")
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot ROC curves for base, skeleton, and fused predictions.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Get data
    y_true = metrics['all_labels']
    
    # Plot base ROC curve
    fpr, tpr, _ = roc_curve(y_true, metrics['all_base_probs'])
    plt.plot(fpr, tpr, label=f"Base (AUC = {metrics['base_auc']:.4f})")
    
    # Plot skeleton ROC curve
    fpr, tpr, _ = roc_curve(y_true, metrics['all_skeleton_probs'])
    plt.plot(fpr, tpr, label=f"Skeleton (AUC = {metrics['skeleton_auc']:.4f})")
    
    # Plot fused ROC curve
    fpr, tpr, _ = roc_curve(y_true, metrics['all_fused_probs'])
    plt.plot(fpr, tpr, label=f"Fused (AUC = {metrics['fused_auc']:.4f})")
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate skeleton-based AI detector")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help="Path to model checkpoint")
    
    # Data arguments
    parser.add_argument('--ai_dir', type=Path, required=True, help="Directory with AI frames")
    parser.add_argument('--real_dir', type=Path, required=True, help="Directory with real frames")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes")
    
    # Evaluation arguments
    parser.add_argument('--fusion_weight', type=float, default=0.5, 
                        help="Weight for skeleton vs base prediction fusion")
    parser.add_argument('--output_dir', type=Path, default=Path('evaluation_results'),
                        help="Directory to save evaluation results")
    parser.add_argument('--frame_sampling', type=str, 
                        choices=['sequential', 'uniform', 'rand_stride'], 
                        default='uniform', help="Frame sampling strategy for evaluation")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Extract model configuration
    num_frames = model.base_detector.num_frames
    
    # Build dataloader
    print(f"Building evaluation dataloader with {args.frame_sampling} sampling...")
    eval_loader, _ = build_dataloaders(
        ai_dir=args.ai_dir,
        real_dir=args.real_dir,
        num_frames=num_frames,
        batch_size=args.batch_size,
        frame_sampling=args.frame_sampling,
        framemix_prob=0.0,  # No FrameMix for evaluation
        val_split=0.0,      # Use all data for evaluation
        num_workers=args.num_workers
    )
    
    # Evaluate model
    print(f"Evaluating model with fusion weight {args.fusion_weight}...")
    metrics = evaluate_model(model, eval_loader, device, args.fusion_weight)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Base Model:     Accuracy = {metrics['base_accuracy']:.4f}, AUC = {metrics['base_auc']:.4f}")
    print(f"Skeleton Model: Accuracy = {metrics['skeleton_accuracy']:.4f}, AUC = {metrics['skeleton_auc']:.4f}")
    print(f"Fused Model:    Accuracy = {metrics['fused_accuracy']:.4f}, AUC = {metrics['fused_auc']:.4f}")
    print("="*50)
    
    # Plot results
    print(f"Saving plots to {args.output_dir}...")
    plot_confusion_matrices(metrics, args.output_dir)
    plot_roc_curves(metrics, args.output_dir)
    
    # Save metrics
    import json
    metrics_to_save = {
        'base_accuracy': float(metrics['base_accuracy']),
        'skeleton_accuracy': float(metrics['skeleton_accuracy']),
        'fused_accuracy': float(metrics['fused_accuracy']),
        'base_auc': float(metrics['base_auc']),
        'skeleton_auc': float(metrics['skeleton_auc']),
        'fused_auc': float(metrics['fused_auc']),
        'fusion_weight': args.fusion_weight,
        'frame_sampling': args.frame_sampling,
        'model_path': args.model_path,
        'evaluation_time': datetime.now().isoformat()
    }
    
    with open(args.output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Evaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    from datetime import datetime
    main()