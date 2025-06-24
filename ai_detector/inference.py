"""
Inference utilities for the AI detector model.
Integration with existing video analysis pipeline.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, List, Tuple, Optional, Sequence, Dict
import logging

from model import load_model, get_transforms

logger = logging.getLogger(__name__)

class AIDetectorInference:
    """Inference class for AI detection on images and video frames."""
    
    def __init__(self, model_path: str, device: Optional[str] = None, threshold: float = 0.5):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            device: Device to use
            threshold: Classification threshold
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms('val')
        
        logger.info(f"AI detector loaded on {self.device}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor):
            # Convert to tensor if transform didn't return one
            from torchvision import transforms
            tensor = transforms.ToTensor()(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict_single(self, image: Union[np.ndarray, Image.Image]) -> dict:
        """
        Predict on a single image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        with torch.no_grad():
            # Preprocess
            tensor = self.preprocess_image(image)
            
            # Predict
            prob = self.model(tensor).item()
            prediction = int(prob > self.threshold)
            confidence = prob if prediction == 1 else (1 - prob)
            
            return {
                'prediction': prediction,  # 0=real, 1=ai
                'probability': prob,       # Probability of being AI
                'confidence': confidence,  # Confidence in prediction
                'label': 'AI' if prediction == 1 else 'Real'
            }
    
    def predict_batch(self, images: Sequence[Union[np.ndarray, Image.Image]]) -> List[Dict]:
        """
        Predict on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        if not images:
            return []
        
        with torch.no_grad():
            # Preprocess all images
            tensors = [self.preprocess_image(img) for img in images]
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Predict
            probs = self.model(batch_tensor).cpu().numpy()
            
            # Format results
            results = []
            for prob in probs:
                prediction = int(prob > self.threshold)
                confidence = prob if prediction == 1 else (1 - prob)
                
                results.append({
                    'prediction': prediction,
                    'probability': float(prob),
                    'confidence': float(confidence),
                    'label': 'AI' if prediction == 1 else 'Real'
                })
            
            return results
    
    def analyze_video_frames(self, frames: Sequence[np.ndarray]) -> dict:
        """
        Analyze video frames for AI detection.
        
        Args:
            frames: List of video frames (numpy arrays)
            
        Returns:
            Analysis results
        """
        if not frames:
            return {'overall_likelihood': 0.0, 'frame_results': []}
        
        # Predict on all frames
        frame_results = self.predict_batch(frames)
        
        # Calculate overall statistics
        probabilities = [r['probability'] for r in frame_results]
        overall_likelihood = np.mean(probabilities) * 100  # Convert to percentage
        
        # Count AI vs Real predictions
        ai_count = sum(1 for r in frame_results if r['prediction'] == 1)
        real_count = len(frame_results) - ai_count
        
        # Calculate confidence statistics
        confidences = [r['confidence'] for r in frame_results]
        avg_confidence = np.mean(confidences)
        
        return {
            'overall_likelihood': float(overall_likelihood),
            'frame_results': frame_results,
            'statistics': {
                'total_frames': len(frames),
                'ai_frames': ai_count,
                'real_frames': real_count,
                'ai_percentage': (ai_count / len(frames)) * 100,
                'average_confidence': float(avg_confidence),
                'min_probability': float(np.min(probabilities)),
                'max_probability': float(np.max(probabilities)),
                'std_probability': float(np.std(probabilities))
            }
        }

def integrate_with_existing_pipeline(frame: np.ndarray, frame_time: float = 0.0) -> dict:
    """
    Integration function for existing video analysis pipeline.
    
    This function can replace or supplement the existing detect_frame_ai_likelihood_enhanced
    function in the main backend.
    
    Args:
        frame: Video frame as numpy array
        frame_time: Timestamp of the frame
        
    Returns:
        Detection results in the same format as existing pipeline
    """
    try:
        # Initialize detector (you might want to make this a global variable)
        detector = AIDetectorInference('weights/ai_detector.pt')
        
        # Get prediction
        result = detector.predict_single(frame)
        
        # Format for existing pipeline
        return {
            "likelihood": result['probability'] * 100,  # Convert to percentage
            "confidence": result['confidence'] * 100,   # Convert to percentage
            "details": {
                "model_prediction": result['label'],
                "raw_probability": result['probability'],
                "frame_time": frame_time,
                "method": "trained_ai_detector"
            }
        }
    
    except Exception as e:
        logger.error(f"AI detector inference failed: {e}")
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
    # Test the inference
    detector = AIDetectorInference('weights/ai_detector.pt')
    
    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = detector.predict_single(dummy_image)
    
    print("Test prediction:", result)