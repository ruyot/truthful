"""
Skeleton-Based Inference for AI Video Detection

This module provides production-ready inference using the skeleton-based detector
with distance matching and fusion capabilities.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, List, Tuple, Dict, Optional, Any, Sequence
import logging
from pathlib import Path
import os

from ai_detector.models.skeleton_model import SkeletonBasedDetector, create_skeleton_model
from ai_detector.models.advanced_model import get_advanced_transforms

logger = logging.getLogger(__name__)

class SkeletonBasedDetectorInference:
    """
    Production inference engine for skeleton-based AI video detection.
    """
    
    def __init__(
        self,
        model_path: str,
        skeleton_dir: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        fusion_weight: float = 0.5,
        distance_method: str = 'mahalanobis'
    ):
        """
        Initialize skeleton-based inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            skeleton_dir: Directory containing skeleton files
            device: Device for inference
            threshold: Classification threshold
            fusion_weight: Weight for skeleton vs base prediction fusion
            distance_method: Distance method for skeleton matching
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.fusion_weight = fusion_weight
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Set distance method
        self.model.skeleton_distance_method = distance_method
        
        # Load skeletons
        if skeleton_dir is None:
            skeleton_dir = str(Path(model_path).parent / "skeletons")
        
        if Path(skeleton_dir).exists():
            self.model.load_skeletons(skeleton_dir)
            self.skeleton_available = True
        else:
            logger.warning(f"Skeleton directory not found: {skeleton_dir}")
            self.skeleton_available = False
        
        # Get transforms
        self.transform = get_advanced_transforms('val')
        
        logger.info(f"Skeleton-based detector loaded on {self.device}")
        logger.info(f"Skeleton available: {self.skeleton_available}")
        logger.info(f"Fusion weight: {fusion_weight}")
        logger.info(f"Distance method: {distance_method}")
    
    def _load_model(self, model_path: str) -> SkeletonBasedDetector:
        """Load skeleton-based model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        model_config = checkpoint.get('config', {}).get('model', {})
        
        # Create model
        model = create_skeleton_model(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_frames(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """Preprocess frames for inference."""
        num_frames = self.model.base_detector.num_frames
        
        # Ensure we have the right number of frames
        if len(frames) < num_frames:
            while len(frames) < num_frames:
                frames = list(frames) + list(frames[:num_frames - len(frames)])
        elif len(frames) > num_frames:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Convert and transform frames
        frame_tensors = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            
            frame_tensor = self.transform(frame)
            frame_tensors.append(frame_tensor)
        
        frames_tensor = torch.stack(frame_tensors).unsqueeze(0)
        return frames_tensor.to(self.device)
    
    def predict(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> Dict[str, Any]:
        """
        Predict using skeleton-based detector with fusion.
        
        Args:
            frames: List of video frames
            
        Returns:
            Comprehensive prediction results
        """
        with torch.no_grad():
            # Preprocess frames
            frames_tensor = self.preprocess_frames(frames)
            
            if self.skeleton_available:
                # Use fused prediction
                results = self.model.fused_predict(frames_tensor, self.fusion_weight)
                
                prediction = int(results['fused_probs'].item() > self.threshold)
                probability = results['fused_probs'].item()
                confidence = probability if prediction == 1 else (1 - probability)
                
                # Additional skeleton information
                structural_similarity = {
                    'ai_distance': float(results['ai_distances'].item()),
                    'real_distance': float(results['real_distances'].item()),
                    'skeleton_probability': float(results['skeleton_probs'].item()),
                    'base_probability': float(results['base_probs'].item()),
                    'distance_method': self.model.skeleton_distance_method
                }
                
                # Calculate structural match score
                ai_dist = results['ai_distances'].item()
                real_dist = results['real_distances'].item()
                total_dist = ai_dist + real_dist
                
                if total_dist > 0:
                    structural_match_score = (1 - ai_dist / total_dist) * 100
                else:
                    structural_match_score = 50.0
                
            else:
                # Use base prediction only
                base_results = self.model(frames_tensor)
                base_prob = torch.sigmoid(base_results['logits']).item()
                
                prediction = int(base_prob > self.threshold)
                probability = base_prob
                confidence = probability if prediction == 1 else (1 - probability)
                
                structural_similarity = {
                    'ai_distance': None,
                    'real_distance': None,
                    'skeleton_probability': None,
                    'base_probability': probability,
                    'distance_method': None
                }
                
                structural_match_score = None
            
            return {
                'prediction': prediction,  # 0=real, 1=ai
                'probability': probability,  # Overall probability of being AI
                'confidence': confidence,  # Confidence in prediction
                'label': 'AI' if prediction == 1 else 'Real',
                'method': 'skeleton_fusion' if self.skeleton_available else 'base_only',
                'structural_similarity': structural_similarity,
                'structural_match_score': structural_match_score,
                'fusion_weight': self.fusion_weight,
                'num_frames_processed': len(frames)
            }
    
    def predict_video(
        self,
        video_path: str,
        max_frames: int = 30,
        sampling_strategy: str = 'uniform'
    ) -> Dict[str, Any]:
        """
        Predict on a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
            sampling_strategy: Frame sampling strategy
            
        Returns:
            Prediction results
        """
        try:
            # Extract frames
            frames = self._extract_video_frames(video_path, max_frames, sampling_strategy)
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
            
            # Get prediction
            result = self.predict(frames)
            result['total_frames_extracted'] = len(frames)
            result['video_path'] = video_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting video {video_path}: {e}")
            return {
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'label': 'Error',
                'error': str(e),
                'video_path': video_path
            }
    
    def _extract_video_frames(
        self,
        video_path: str,
        max_frames: int,
        sampling_strategy: str
    ) -> List[np.ndarray]:
        """Extract frames from video file."""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            # Determine frame indices
            if sampling_strategy == 'uniform':
                if total_frames <= max_frames:
                    frame_indices = list(range(total_frames))
                else:
                    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            elif sampling_strategy == 'random':
                frame_indices = np.random.choice(
                    total_frames,
                    size=min(max_frames, total_frames),
                    replace=False
                )
                frame_indices = sorted(frame_indices)
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise
        
        return frames
    
    def analyze_structural_similarity(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> Dict[str, Any]:
        """
        Detailed structural similarity analysis.
        
        Args:
            frames: List of video frames
            
        Returns:
            Detailed structural analysis
        """
        if not self.skeleton_available:
            return {'error': 'Skeletons not available for structural analysis'}
        
        with torch.no_grad():
            frames_tensor = self.preprocess_frames(frames)
            
            # Get embeddings
            results = self.model(frames_tensor)
            embeddings = results['embeddings']
            
            # Get skeleton predictions
            skeleton_results = self.model.skeleton_predict(embeddings)
            
            # Detailed distance analysis
            embedding_np = embeddings.cpu().numpy()[0]
            
            # Test different distance methods
            distance_methods = ['euclidean', 'cosine', 'mahalanobis', 'knn']
            distance_analysis = {}
            
            for method in distance_methods:
                if self.model.ai_skeleton is not None and self.model.real_skeleton is not None:
                    ai_dist = self.model.ai_skeleton.distance_to_embedding(embedding_np, method)
                    real_dist = self.model.real_skeleton.distance_to_embedding(embedding_np, method)
                    
                    distance_analysis[method] = {
                        'ai_distance': float(ai_dist),
                        'real_distance': float(real_dist),
                        'ratio': float(ai_dist / real_dist) if real_dist > 0 else float('inf')
                    }
                else:
                    distance_analysis[method] = {
                        'ai_distance': None,
                        'real_distance': None,
                        'ratio': None
                    }
            
            # Skeleton statistics
            skeleton_stats = {
                'ai_skeleton': {
                    'n_samples': self.model.ai_skeleton.n_samples if self.model.ai_skeleton is not None else 0,
                    'mean_norm': float(np.linalg.norm(self.model.ai_skeleton.mean)) if self.model.ai_skeleton is not None else 0.0,
                    'std_mean': float(np.mean(self.model.ai_skeleton.std)) if self.model.ai_skeleton is not None else 0.0
                },
                'real_skeleton': {
                    'n_samples': self.model.real_skeleton.n_samples if self.model.real_skeleton is not None else 0,
                    'mean_norm': float(np.linalg.norm(self.model.real_skeleton.mean)) if self.model.real_skeleton is not None else 0.0,
                    'std_mean': float(np.mean(self.model.real_skeleton.std)) if self.model.real_skeleton is not None else 0.0
                }
            }
            
            return {
                'distance_analysis': distance_analysis,
                'skeleton_stats': skeleton_stats,
                'embedding_norm': float(np.linalg.norm(embedding_np)),
                'primary_method': self.model.skeleton_distance_method,
                'skeleton_probability': float(skeleton_results['skeleton_probs'].item())
            }

def integrate_skeleton_with_existing_pipeline(
    frame: np.ndarray,
    frame_time: float = 0.0,
    model_path: str = 'weights/best_skeleton_model.pt'
) -> Dict[str, Any]:
    """
    Integration function for existing video analysis pipeline.
    
    Args:
        frame: Video frame as numpy array
        frame_time: Timestamp of the frame
        model_path: Path to skeleton model
        
    Returns:
        Detection results in existing pipeline format
    """
    try:
        # Initialize detector (consider making this global for efficiency)
        detector = SkeletonBasedDetectorInference(model_path)
        
        # Create frame list
        frames = [frame] * detector.model.base_detector.num_frames
        
        # Get prediction
        result = detector.predict(frames)
        
        # Format for existing pipeline
        response = {
            "likelihood": result['probability'] * 100,
            "confidence": result['confidence'] * 100,
            "details": {
                "model_prediction": result['label'],
                "raw_probability": result['probability'],
                "frame_time": frame_time,
                "method": "skeleton_based_detector",
                "fusion_weight": result['fusion_weight'],
                "structural_match_score": result.get('structural_match_score'),
                "distance_method": result['structural_similarity'].get('distance_method')
            }
        }
        
        # Add structural similarity info if available
        if result.get('structural_match_score') is not None:
            response["details"]["structural_similarity_to_ai"] = result['structural_match_score']
        
        return response
    
    except Exception as e:
        logger.error(f"Skeleton detector inference failed: {e}")
        # Fallback to existing method
        return {
            "likelihood": 50.0,
            "confidence": 60.0,
            "details": {
                "error": str(e),
                "method": "fallback"
            }
        }

# Example usage
if __name__ == "__main__":
    model_path = "weights/best_skeleton_model.pt"
    
    if os.path.exists(model_path):
        detector = SkeletonBasedDetectorInference(model_path)
        
        # Test with dummy frames
        dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
        result = detector.predict(dummy_frames)
        
        print("Skeleton prediction test:")
        print(f"  Prediction: {result['label']}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Structural match score: {result.get('structural_match_score')}")
        
        # Test structural analysis
        if detector.skeleton_available:
            structural_analysis = detector.analyze_structural_similarity(dummy_frames)
            print("\nStructural analysis:")
            print(f"  Distance methods: {list(structural_analysis['distance_analysis'].keys())}")
            print(f"  Embedding norm: {structural_analysis['embedding_norm']:.3f}")
    else:
        print(f"Model not found at {model_path}")
        print("Train a skeleton model first using: python ai_detector/scripts/train_skeleton.py")