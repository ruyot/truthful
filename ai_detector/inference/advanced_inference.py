"""
Advanced Inference Pipeline for AI Video Detection

This module provides production-ready inference capabilities for the advanced
AI video classifier with multi-frame processing and robust prediction aggregation.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, List, Tuple, Dict, Optional, Any, Sequence
import logging
from pathlib import Path
import tempfile
import os

from ai_detector.models.advanced_model import load_advanced_model, get_advanced_transforms

logger = logging.getLogger(__name__)

class AdvancedAIDetectorInference:
    """
    Advanced inference engine for AI video detection with multi-frame processing.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None, 
        threshold: float = 0.5,
        num_frames: Optional[int] = None
    ):
        """
        Initialize the advanced inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference
            threshold: Classification threshold
            num_frames: Number of frames to process (auto-detected from model if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model and configuration
        self.model = load_advanced_model(model_path, self.device)
        self.model.eval()
        
        # Load model configuration
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint.get('model_config', {})
        
        # Set number of frames
        self.num_frames = num_frames or self.model_config.get('num_frames', 5)
        
        # Get transforms
        self.transform = get_advanced_transforms('val')
        
        logger.info(f"Advanced AI detector loaded on {self.device}")
        logger.info(f"Model: {self.model_config.get('backbone', 'unknown')}")
        logger.info(f"Frames per video: {self.num_frames}")
    
    def preprocess_frames(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Preprocess a list of frames for inference.
        
        Args:
            frames: List of frames (numpy arrays or PIL Images)
            
        Returns:
            Preprocessed tensor of shape (1, num_frames, 3, H, W)
        """
        # Ensure we have the right number of frames
        if len(frames) < self.num_frames:
            # Repeat frames if we have too few
            while len(frames) < self.num_frames:
                frames = list(frames) + list(frames[:self.num_frames - len(frames)])
        elif len(frames) > self.num_frames:
            # Sample frames uniformly if we have too many
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Convert and transform frames
        frame_tensors = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            
            # Apply transforms
            frame_tensor = self.transform(frame)
            frame_tensors.append(frame_tensor)
        
        # Stack frames and add batch dimension
        frames_tensor = torch.stack(frame_tensors).unsqueeze(0)  # (1, num_frames, 3, H, W)
        return frames_tensor.to(self.device)
    
    def predict_frames(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> Dict[str, Any]:
        """
        Predict on a list of frames.
        
        Args:
            frames: List of frames
            
        Returns:
            Dictionary with prediction results
        """
        with torch.no_grad():
            # Preprocess frames
            frames_tensor = self.preprocess_frames(frames)
            
            # Get prediction
            logits = self.model(frames_tensor)
            prob = torch.sigmoid(logits).item()
            prediction = int(prob > self.threshold)
            confidence = prob if prediction == 1 else (1 - prob)
            
            return {
                'prediction': prediction,  # 0=real, 1=ai
                'probability': prob,       # Probability of being AI
                'confidence': confidence,  # Confidence in prediction
                'label': 'AI' if prediction == 1 else 'Real',
                'num_frames_processed': len(frames)
            }
    
    def extract_video_frames(
        self, 
        video_path: str, 
        max_frames: int = 30,
        sampling_strategy: str = 'uniform'
    ) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            sampling_strategy: 'uniform', 'random', or 'keyframes'
            
        Returns:
            List of frame arrays
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            # Determine frame indices to extract
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
            else:  # keyframes or fallback to uniform
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
            frames = self.extract_video_frames(video_path, max_frames, sampling_strategy)
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
            
            # Get prediction
            result = self.predict_frames(frames)
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
    
    def predict_batch_videos(
        self, 
        video_paths: List[str],
        max_frames: int = 30,
        sampling_strategy: str = 'uniform'
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of videos.
        
        Args:
            video_paths: List of video file paths
            max_frames: Maximum frames per video
            sampling_strategy: Frame sampling strategy
            
        Returns:
            List of prediction results
        """
        results = []
        
        for video_path in video_paths:
            result = self.predict_video(video_path, max_frames, sampling_strategy)
            results.append(result)
        
        return results
    
    def analyze_video_temporal(
        self, 
        video_path: str, 
        segment_length: int = 10,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyze video with temporal segmentation for detailed analysis.
        
        Args:
            video_path: Path to video file
            segment_length: Number of frames per segment
            overlap: Overlap ratio between segments
            
        Returns:
            Temporal analysis results
        """
        try:
            # Extract all frames
            all_frames = self.extract_video_frames(video_path, max_frames=1000)
            
            if len(all_frames) < self.num_frames:
                # If video is too short, analyze as single segment
                return self.predict_frames(all_frames)
            
            # Create overlapping segments
            step_size = int(segment_length * (1 - overlap))
            segments = []
            
            for start_idx in range(0, len(all_frames) - self.num_frames + 1, step_size):
                end_idx = min(start_idx + segment_length, len(all_frames))
                segment_frames = all_frames[start_idx:end_idx]
                
                if len(segment_frames) >= self.num_frames:
                    segments.append(segment_frames)
            
            # Analyze each segment
            segment_results = []
            for i, segment in enumerate(segments):
                result = self.predict_frames(segment)
                result['segment_id'] = i
                result['start_frame'] = i * step_size
                segment_results.append(result)
            
            # Aggregate results
            if segment_results:
                probabilities = [r['probability'] for r in segment_results]
                overall_probability = np.mean(probabilities)
                overall_prediction = int(overall_probability > self.threshold)
                overall_confidence = overall_probability if overall_prediction == 1 else (1 - overall_probability)
                
                # Calculate consistency
                predictions = [r['prediction'] for r in segment_results]
                consistency = np.mean(np.array(predictions) == overall_prediction)
                
                return {
                    'prediction': overall_prediction,
                    'probability': overall_probability,
                    'confidence': overall_confidence,
                    'label': 'AI' if overall_prediction == 1 else 'Real',
                    'temporal_analysis': {
                        'total_segments': len(segment_results),
                        'segment_results': segment_results,
                        'consistency': consistency,
                        'probability_std': np.std(probabilities),
                        'ai_segments': sum(predictions),
                        'real_segments': len(predictions) - sum(predictions)
                    },
                    'video_path': video_path
                }
            else:
                raise ValueError("No valid segments could be created")
                
        except Exception as e:
            logger.error(f"Error in temporal analysis of {video_path}: {e}")
            return {
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'label': 'Error',
                'error': str(e),
                'video_path': video_path
            }

def integrate_with_existing_pipeline(frame: np.ndarray, frame_time: float = 0.0) -> Dict[str, Any]:
    """
    Integration function for existing video analysis pipeline.
    
    This function can replace the existing detect_frame_ai_likelihood_enhanced
    function in the main backend.
    
    Args:
        frame: Video frame as numpy array
        frame_time: Timestamp of the frame
        
    Returns:
        Detection results in the same format as existing pipeline
    """
    try:
        # Initialize detector (you might want to make this a global variable)
        detector = AdvancedAIDetectorInference('weights/best_advanced_model.pt')
        
        # Create a list with just this frame (repeated to meet num_frames requirement)
        frames = [frame] * detector.num_frames
        
        # Get prediction
        result = detector.predict_frames(frames)
        
        # Format for existing pipeline
        return {
            "likelihood": result['probability'] * 100,  # Convert to percentage
            "confidence": result['confidence'] * 100,   # Convert to percentage
            "details": {
                "model_prediction": result['label'],
                "raw_probability": result['probability'],
                "frame_time": frame_time,
                "method": "advanced_ai_detector",
                "backbone": detector.model_config.get('backbone', 'unknown'),
                "num_frames": detector.num_frames
            }
        }
    
    except Exception as e:
        logger.error(f"Advanced AI detector inference failed: {e}")
        # Fallback to existing method
        return {
            "likelihood": 50.0,
            "confidence": 60.0,
            "details": {
                "error": str(e),
                "method": "fallback"
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the advanced inference
    model_path = "weights/best_advanced_model.pt"
    
    if os.path.exists(model_path):
        detector = AdvancedAIDetectorInference(model_path)
        
        # Test with dummy frames
        dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
        result = detector.predict_frames(dummy_frames)
        
        print("Test prediction:", result)
        
        # Test video prediction if video exists
        test_video = "test_video.mp4"
        if os.path.exists(test_video):
            video_result = detector.predict_video(test_video)
            print("Video prediction:", video_result)
            
            # Test temporal analysis
            temporal_result = detector.analyze_video_temporal(test_video)
            print("Temporal analysis:", temporal_result)
    else:
        print(f"Model not found at {model_path}")
        print("Train a model first using: python ai_detector/scripts/train_advanced.py")