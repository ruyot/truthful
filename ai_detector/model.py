"""
CLIP-inspired AI Detector Model

A lightweight binary classifier for detecting AI-generated images using a pretrained
vision encoder (ResNet18) with a custom classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetectorModel(nn.Module):
    """
    CLIP-inspired AI detector using pretrained ResNet18 backbone.
    
    Architecture:
    - Pretrained ResNet18 (frozen or fine-tuned)
    - Custom classification head with dropout
    - Binary output (AI vs Real)
    """
    
    def __init__(
        self, 
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256
    ):
        """
        Initialize the AI detector model.
        
        Args:
            freeze_backbone: Whether to freeze the ResNet18 backbone
            dropout_rate: Dropout rate for regularization
            hidden_dim: Hidden dimension for the classification head
        """
        super(AIDetectorModel, self).__init__()
        
        # Load pretrained ResNet18 and remove the final classifier
        self.backbone = models.resnet18(pretrained=True)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # type: ignore # Remove the classifier
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen - only training classification head")
        else:
            logger.info("Backbone unfrozen - fine-tuning entire model")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),  # Binary classification
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize the classification head weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Probability of being AI-generated (0-1)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify using custom head
        output = self.classifier(features)
        
        return output.squeeze()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (useful for analysis).
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature tensor from backbone
        """
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities for [real, ai] classes
        """
        self.eval()
        with torch.no_grad():
            ai_prob = self.forward(x)
            real_prob = 1 - ai_prob
            return torch.stack([real_prob, ai_prob], dim=1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0=real, 1=ai)
        """
        probs = self.forward(x)
        return (probs > threshold).long()

def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        mode: 'train' or 'val'
        
    Returns:
        Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_model(
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
    hidden_dim: int = 256
) -> AIDetectorModel:
    """
    Factory function to create the AI detector model.
    
    Args:
        freeze_backbone: Whether to freeze the backbone
        dropout_rate: Dropout rate
        hidden_dim: Hidden dimension
        
    Returns:
        Initialized model
    """
    model = AIDetectorModel(
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim
    )
    
    logger.info(f"Created AI detector model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    return model

def load_model(checkpoint_path: str, device: str = 'cpu') -> AIDetectorModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same config as saved
    model_config = checkpoint.get('model_config', {})
    model = create_model(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

if __name__ == "__main__":
    # Test the model
    model = create_model()
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Sample predictions: {output}")
    
    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")