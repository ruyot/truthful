"""
Advanced Evaluation Pipeline for AI Video Classifier

This module provides comprehensive evaluation capabilities including:
- Cross-validation on video-level splits
- Detailed performance metrics
- Confusion matrices and ROC curves
- Per-class and per-backbone analysis
- Robustness testing across different video sources
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
from pathlib import Path
import pandas as pd

# Import our modules
from ai_detector.models.advanced_model import load_advanced_model, get_advanced_transforms
from ai_detector.datasets.video_dataset import VideoFrameDataset, create_advanced_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEvaluator:
    """
    Comprehensive evaluator for advanced AI video detection models.
    """
    
    def __init__(self, model_path: str, data_dir: str, device: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Data directory
            device: Device to use for evaluation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir
        
        # Load model
        self.model = load_advanced_model(model_path, self.device)
        self.model.eval()
        
        # Load model configuration
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint.get('model_config', {})
        self.dataset_info = checkpoint.get('dataset_info', {})
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model config: {self.model_config}")
        logger.info(f"Device: {self.device}")
    
    def create_test_dataloader(self, video_ids: Optional[List[str]] = None, batch_size: int = 16):
        """Create test dataloader."""
        test_transform = get_advanced_transforms('val')
        
        test_dataset = VideoFrameDataset(
            data_dir=self.data_dir,
            num_frames=self.model_config.get('num_frames', 5),
            transform=test_transform,
            video_ids=video_ids,
            frame_sampling_strategy='uniform'
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        return test_loader, test_dataset
    
    def predict_all(self, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Get predictions for all samples."""
        all_preds = []
        all_probs = []
        all_labels = []
        all_video_ids = []
        
        with torch.no_grad():
            for frames, labels, video_ids in tqdm(test_loader, desc="Predicting"):
                frames = frames.to(self.device)
                
                # Get logits and probabilities
                logits = self.model(frames)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_video_ids.extend(video_ids)
        
        return (
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs),
            all_video_ids
        )
    
    def calculate_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_probs: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division='warn')
        metrics['recall'] = recall_score(y_true, y_pred, zero_division='warn')
        metrics['f1'] = f1_score(y_true, y_pred, zero_division='warn')
        metrics['specificity'] = precision_score(1 - y_true, 1 - y_pred, zero_division='warn')
        
        # AUC metrics
        if len(set(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
            
            # Calculate AUC-PR
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
            metrics['auc_pr'] = np.trapz(precision_curve, recall_curve)
        else:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Class-specific metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # True/False positive/negative rates
            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
            metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            
            # Positive/Negative predictive values
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics.get('tpr', 0) + metrics.get('tnr', 0)) / 2
        
        return metrics
    
    def analyze_by_confidence(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
        """Analyze predictions by confidence level."""
        confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        analysis = {}
        
        for low, high in confidence_bins:
            bin_name = f"{low:.1f}-{high:.1f}"
            
            # AI predictions (prob >= 0.5)
            ai_mask = (y_probs >= low) & (y_probs < high) & (y_probs >= 0.5)
            if ai_mask.sum() > 0:
                ai_acc = accuracy_score(y_true[ai_mask], (y_probs[ai_mask] >= 0.5).astype(int))
                analysis[f"ai_{bin_name}"] = {
                    'count': int(ai_mask.sum()),
                    'accuracy': float(ai_acc)
                }
            
            # Real predictions (prob < 0.5)
            real_probs = 1 - y_probs
            real_mask = (real_probs >= low) & (real_probs < high) & (y_probs < 0.5)
            if real_mask.sum() > 0:
                real_acc = accuracy_score(y_true[real_mask], (y_probs[real_mask] >= 0.5).astype(int))
                analysis[f"real_{bin_name}"] = {
                    'count': int(real_mask.sum()),
                    'accuracy': float(real_acc)
                }
        
        return analysis
    
    def analyze_by_video_source(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_probs: np.ndarray,
        video_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze performance by video source/type."""
        # Extract source information from video IDs
        source_analysis = {}
        
        # Group by AI generation method (if identifiable from video ID)
        ai_sources = ['sora', 'veo', 'runway', 'pika', 'midjourney', 'stable', 'deepfake']
        real_sources = ['camera', 'phone', 'youtube', 'original']
        
        for source in ai_sources + real_sources:
            source_mask = np.array([source.lower() in vid.lower() for vid in video_ids])
            
            if source_mask.sum() > 0:
                source_metrics = self.calculate_comprehensive_metrics(
                    y_true[source_mask], 
                    y_pred[source_mask], 
                    y_probs[source_mask]
                )
                source_analysis[source] = {
                    'count': int(source_mask.sum()),
                    'metrics': source_metrics
                }
        
        return source_analysis
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
        """Plot detailed confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'], ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'], ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curve."""
        if len(set(y_true)) <= 1:
            logger.warning("Cannot plot ROC curve: only one class present")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add threshold annotations for key points
        key_indices = [np.argmax(tpr - fpr), np.argmin(np.abs(thresholds - 0.5))]
        for idx in key_indices:
            if idx < len(thresholds):
                plt.annotate(f'Threshold: {thresholds[idx]:.2f}', 
                           xy=(fpr[idx], tpr[idx]), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_probs: np.ndarray, save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        if len(set(y_true)) <= 1:
            logger.warning("Cannot plot PR curve: only one class present")
            return
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        auc_pr = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random Classifier ({baseline:.3f})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_probs: np.ndarray, save_path: Optional[str] = None):
        """Plot calibration curve."""
        if len(set(y_true)) <= 1:
            logger.warning("Cannot plot calibration curve: only one class present")
            return
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_probs, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", linewidth=2)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_comprehensive(self, save_plots: bool = True, output_dir: str = 'evaluation_results'):
        """Run comprehensive evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Starting comprehensive evaluation...")
        
        # Create test dataloader
        test_loader, test_dataset = self.create_test_dataloader()
        
        # Get predictions
        y_true, y_pred, y_probs, video_ids = self.predict_all(test_loader)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_probs)
        
        # Confidence analysis
        confidence_analysis = self.analyze_by_confidence(y_true, y_probs)
        
        # Source analysis
        source_analysis = self.analyze_by_video_source(y_true, y_pred, y_probs, video_ids)
        
        # Print results
        self._print_evaluation_results(metrics, confidence_analysis, source_analysis)
        
        # Create plots
        if save_plots:
            self.plot_confusion_matrix(y_true, y_pred, f'{output_dir}/confusion_matrix.png')
            self.plot_roc_curve(y_true, y_probs, f'{output_dir}/roc_curve.png')
            self.plot_precision_recall_curve(y_true, y_probs, f'{output_dir}/pr_curve.png')
            self.plot_calibration_curve(y_true, y_probs, f'{output_dir}/calibration_curve.png')
        
        # Save detailed results
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_config': self.model_config,
            'dataset_info': {
                'total_videos': len(video_ids),
                'ai_videos': int(np.sum(y_true)),
                'real_videos': int(len(y_true) - np.sum(y_true))
            },
            'metrics': metrics,
            'confidence_analysis': confidence_analysis,
            'source_analysis': source_analysis,
            'classification_report': classification_report(y_true, y_pred, target_names=['Real', 'AI'], output_dict=True)
        }
        
        with open(f'{output_dir}/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions for further analysis
        predictions_df = pd.DataFrame({
            'video_id': video_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'probability': y_probs
        })
        predictions_df.to_csv(f'{output_dir}/predictions.csv', index=False)
        
        logger.info(f"Evaluation complete! Results saved to {output_dir}/")
        
        return results
    
    def _print_evaluation_results(self, metrics: Dict, confidence_analysis: Dict, source_analysis: Dict):
        """Print formatted evaluation results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        # Main metrics
        print(f"Overall Performance:")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 Score:         {metrics['f1']:.4f}")
        print(f"  Specificity:      {metrics['specificity']:.4f}")
        print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
        print(f"  AUC-ROC:          {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:           {metrics['auc_pr']:.4f}")
        
        # Detailed metrics
        print(f"\nDetailed Metrics:")
        print(f"  True Positive Rate:  {metrics.get('tpr', 0):.4f}")
        print(f"  True Negative Rate:  {metrics.get('tnr', 0):.4f}")
        print(f"  False Positive Rate: {metrics.get('fpr', 0):.4f}")
        print(f"  False Negative Rate: {metrics.get('fnr', 0):.4f}")
        print(f"  Positive Pred Value: {metrics.get('ppv', 0):.4f}")
        print(f"  Negative Pred Value: {metrics.get('npv', 0):.4f}")
        
        # Confidence analysis
        if confidence_analysis:
            print(f"\nConfidence Analysis:")
            for bin_name, data in confidence_analysis.items():
                print(f"  {bin_name}: {data['count']} samples, Accuracy: {data['accuracy']:.3f}")
        
        # Source analysis
        if source_analysis:
            print(f"\nSource Analysis:")
            for source, data in source_analysis.items():
                print(f"  {source.capitalize()}: {data['count']} videos")
                print(f"    Accuracy: {data['metrics']['accuracy']:.3f}")
                print(f"    F1 Score: {data['metrics']['f1']:.3f}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Advanced AI Video Classifier')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AdvancedEvaluator(args.model_path, args.data_dir)
    
    # Run evaluation
    results = evaluator.evaluate_comprehensive(
        save_plots=args.save_plots,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}/")
    print(f"Final Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Final F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Final AUC-ROC: {results['metrics']['auc_roc']:.4f}")

if __name__ == '__main__':
    main()