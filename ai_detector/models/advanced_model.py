"""
Advanced AI Video Classifier
* Adds _extract_features() helper so ConvNeXt‑Tiny produces a pooled vector
* Fixes previous CUDA‑OOM / shape mismatch when using ConvNeXt‑Tiny
* ResNet‑50 and EfficientNet‑B3 paths stay exactly the same.
"""
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torchvision import transforms

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class MultiFrameAggregator(nn.Module):
    def __init__(self, feature_dim: int, num_frames: int = 5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(),
            nn.Linear(feature_dim // 2, 1), nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        w   = self.attention(feat)
        agg = torch.sum(feat * w, dim=1)
        return self.classifier(agg)


class AdvancedAIDetector(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_frames: int = 5,
                 freeze_backbone: bool = False, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone_name, self.num_frames = backbone, num_frames

        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b3":
            self.backbone = timm.create_model("efficientnet_b3", pretrained=True)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "convnext_tiny":
            self.backbone = timm.create_model("convnext_tiny", pretrained=True)
            self.feature_dim = self.backbone.num_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("[Advanced] backbone frozen")

        self.aggregator = MultiFrameAggregator(self.feature_dim, num_frames)
        logger.info(f"[Advanced] init backbone={backbone} feat_dim={self.feature_dim} frames={num_frames}")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name.startswith("convnext"):
            feat = self.backbone.forward_features(x)
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        f = self._extract_features(x)
        f = f.view(B, T, self.feature_dim)
        logits = self.aggregator(f)
        return logits.squeeze(-1)

    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_features(x)


def get_advanced_transforms(mode: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get advanced image transforms for training or validation.
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size (default: 224)
        
    Returns:
        Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),  # Slightly larger for cropping
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val/test mode
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])