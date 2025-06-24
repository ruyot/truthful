"""
Advanced AI Video Classifier with Deep CNN Backbone

This module implements an improved AI vs Real video classifier using:
- ResNet-50/EfficientNet-B3/ConvNeXt-Tiny backbones
- Multi-frame aggregation strategy
- Focal Loss for handling class imbalance
- Advanced data augmentation
"""

import torch  
import torch.nn as nn 
import torch.nn.functional as F  
import torchvision.models as models 
from torchvision import transforms  
import timm  
from typing import Tuple, Optional, List, Dict, Any
import logging
import numpy as np  

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance and hard examples.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiFrameAggregator(nn.Module):
    """
    Aggregates predictions from multiple frames using attention mechanism.
    """
    
    def __init__(self, feature_dim: int, num_frames: int = 5):
        super(MultiFrameAggregator, self).__init__()
        self.num_frames = num_frames
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch_size, num_frames, feature_dim)
        Returns:
            predictions: (batch_size, 1)
        """
        # Calculate attention weights
        attention_weights = self.attention(frame_features)  # (batch_size, num_frames, 1)
        
        # Apply attention to aggregate features
        aggregated_features = torch.sum(frame_features * attention_weights, dim=1)  # (batch_size, feature_dim)
        
        # Generate final prediction
        prediction = self.classifier(aggregated_features)
        
        return prediction

class AdvancedAIDetector(nn.Module):
    """
    Advanced AI detector with deep CNN backbone and multi-frame aggregation.
    
    Supports multiple backbone architectures:
    - ResNet-50
    - EfficientNet-B3
    - ConvNeXt-Tiny
    """
    
    def __init__(
        self, 
        backbone: str = 'resnet50',
        num_frames: int = 5,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3
    ):
        super(AdvancedAIDetector, self).__init__()
        
        self.backbone_name = backbone
        self.num_frames = num_frames
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore
        elif backbone == 'efficientnet_b3':
            self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # type: ignore
        elif backbone == 'convnext_tiny':
            self.backbone = timm.create_model('convnext_tiny', pretrained=True)
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  # type: ignore
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Backbone {backbone} frozen")
        
        # Multi-frame aggregator
        self.aggregator = MultiFrameAggregator(feature_dim, num_frames)
        
        # Store feature dimension for external access
        self.feature_dim = feature_dim
        
        logger.info(f"Initialized AdvancedAIDetector with {backbone} backbone")
        logger.info(f"Feature dimension: {feature_dim}")
        logger.info(f"Number of frames: {num_frames}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for backbone processing
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using backbone
        features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to separate frames
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        # Aggregate frame features and classify
        logits = self.aggregator(features)
        
        return logits.squeeze(-1)  # (batch_size,)
    
    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from individual frames.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Features of shape (batch_size, feature_dim)
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
            logits = self.forward(x)
            ai_prob = torch.sigmoid(logits)
            real_prob = 1 - ai_prob
            return torch.stack([real_prob, ai_prob], dim=1)

def get_advanced_transforms(mode: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get advanced image transforms with aggressive but realistic augmentations.
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            # Geometric augmentations
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomResizedCrop(
                image_size, 
                scale=(0.7, 1.0), 
                ratio=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            
            # Color augmentations
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            
            # Convert to tensor first for blur
            transforms.ToTensor(),
            
            # Gaussian blur (applied after ToTensor)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            
            # Normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            
            # Additional noise augmentation
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_advanced_model(
    backbone: str = 'resnet50',
    num_frames: int = 5,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3
) -> AdvancedAIDetector:
    """
    Factory function to create the advanced AI detector model.
    
    Args:
        backbone: Backbone architecture ('resnet50', 'efficientnet_b3', 'convnext_tiny')
        num_frames: Number of frames to process per video
        freeze_backbone: Whether to freeze the backbone
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Initialized model
    """
    model = AdvancedAIDetector(
        backbone=backbone,
        num_frames=num_frames,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created advanced AI detector model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Backbone: {backbone}")
    logger.info(f"  Multi-frame aggregation: {num_frames} frames")
    
    return model

def load_advanced_model(checkpoint_path: str, device: str = 'cpu') -> AdvancedAIDetector:
    """
    Load a trained advanced model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same config as saved
    model_config = checkpoint.get('model_config', {})
    model = create_advanced_model(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Loaded advanced model from {checkpoint_path}")
    return model

# Model configurations for different use cases
MODEL_CONFIGS = {
    'fast': {
        'backbone': 'resnet50',
        'num_frames': 3,
        'freeze_backbone': True,
        'dropout_rate': 0.2
    },
    'balanced': {
        'backbone': 'efficientnet_b3',
        'num_frames': 5,
        'freeze_backbone': False,
        'dropout_rate': 0.3
    },
    'accurate': {
        'backbone': 'convnext_tiny',
        'num_frames': 7,
        'freeze_backbone': False,
        'dropout_rate': 0.4
    }
}

if __name__ == "__main__":
    # Test the advanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different configurations
    for config_name, config in MODEL_CONFIGS.items():
        print(f"\nTesting {config_name} configuration:")
        model = create_advanced_model(**config).to(device)
        
        # Test forward pass
        batch_size = 2
        num_frames = config['num_frames']
        dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            probs = model.predict_proba(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Sample output: {output}")
        print(f"  Sample probabilities: {probs}")