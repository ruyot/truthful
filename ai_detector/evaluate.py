"""
Evaluation script for the AI detector model.

Usage:
    python evaluate.py --model_path weights/ai_detector.pt --data_dir data
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.calibration import calibration_curve
import logging
from tqdm import tqdm
import os
from typing import Optional

from model import load_model, get_transforms
from dataset import AIDetectorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model_path: str, data_dir: str, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            data_dir: Data directory
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        # Load test dataset
        test_transform = get_transforms('val')
        self.dataset = AIDetectorDataset(data_dir, test_transform)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        logger.info(f"Loaded model and dataset with {len(self.dataset)} samples")
    
    def predict_all(self):
        """Get predictions for all samples."""
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="Predicting"):
                images = images.to(self.device)
                
                # Get probabilities
                probs = self.model(images)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate comprehensive metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division='warn'),
            'recall': recall_score(y_true, y_pred, zero_division='warn'),
            'f1': f1_score(y_true, y_pred, zero_division='warn'),
            'auc': roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0.0
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs, save_path: Optional[str] = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true, y_probs, save_path: Optional[str] = None):
        """Plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_probs, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_predictions_by_confidence(self, y_true, y_probs):
        """Analyze predictions by confidence level."""
        confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        
        print("\nPrediction Analysis by Confidence Level:")
        print("-" * 50)
        
        for low, high in confidence_bins:
            # For AI predictions (prob > 0.5)
            ai_mask = (y_probs >= low) & (y_probs < high)
            if ai_mask.sum() > 0:
                ai_acc = accuracy_score(y_true[ai_mask], (y_probs[ai_mask] > 0.5).astype(int))
                print(f"AI predictions [{low:.1f}-{high:.1f}): {ai_mask.sum()} samples, Accuracy: {ai_acc:.3f}")
            
            # For Real predictions (prob < 0.5)
            real_probs = 1 - y_probs
            real_mask = (real_probs >= low) & (real_probs < high) & (y_probs < 0.5)
            if real_mask.sum() > 0:
                real_acc = accuracy_score(y_true[real_mask], (y_probs[real_mask] > 0.5).astype(int))
                print(f"Real predictions [{low:.1f}-{high:.1f}): {real_mask.sum()} samples, Accuracy: {real_acc:.3f}")
    
    def evaluate(self, save_plots: bool = True):
        """Run comprehensive evaluation."""
        logger.info("Starting evaluation...")
        
        # Get predictions
        y_true, y_pred, y_probs = self.predict_all()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_probs)
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Real', 'AI']))
        
        # Analyze by confidence
        self.analyze_predictions_by_confidence(y_true, y_probs)
        
        # Create plots
        if save_plots:
            os.makedirs('evaluation_plots', exist_ok=True)
            
            self.plot_confusion_matrix(y_true, y_pred, 'evaluation_plots/confusion_matrix.png')
            self.plot_roc_curve(y_true, y_probs, 'evaluation_plots/roc_curve.png')
            self.plot_calibration_curve(y_true, y_probs, 'evaluation_plots/calibration_curve.png')
            
            logger.info("Plots saved to evaluation_plots/")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate AI Detector Model')
    parser.add_argument('--model_path', type=str, default='weights/ai_detector.pt', 
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_plots', action='store_true', help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, args.data_dir)
    metrics = evaluator.evaluate(args.save_plots)
    
    print(f"\nEvaluation complete! Final accuracy: {metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()