"""
Skeleton-Based AI Video Classifier

This module implements a skeleton-based classification approach that:
1. Trains a deep CNN backbone on combined DFD + VidProM data
2. Computes class-level embeddings (skeletons) for AI and Real videos
3. Uses distance-based matching for improved generalization
4. Supports multi-task learning with prompt embedding prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
from pathlib import Path

from ai_detector.models.advanced_model import AdvancedAIDetector, MultiFrameAggregator

logger = logging.getLogger(__name__)

class SkeletonEmbedding:
    """
    Class-level embedding (skeleton) for AI or Real videos.
    Stores mean, variance, and covariance information.
    """
    
    def __init__(self, embeddings: np.ndarray, label: str):
        """
        Initialize skeleton from a collection of embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, feature_dim)
            label: 'AI' or 'Real'
        """
        self.label = label
        self.n_samples = len(embeddings)
        self.feature_dim = embeddings.shape[1]
        
        # Compute statistics
        self.mean = np.mean(embeddings, axis=0)
        self.std = np.std(embeddings, axis=0)
        self.cov = np.cov(embeddings.T)
        
        # Compute robust statistics
        self.median = np.median(embeddings, axis=0)
        self.mad = np.median(np.abs(embeddings - self.median), axis=0)  # Median Absolute Deviation
        
        # Store raw embeddings for distance calculations
        self.embeddings = embeddings.copy()
        
        logger.info(f"Created {label} skeleton from {self.n_samples} embeddings")
        logger.info(f"  Feature dim: {self.feature_dim}")
        logger.info(f"  Mean norm: {np.linalg.norm(self.mean):.3f}")
        logger.info(f"  Std mean: {np.mean(self.std):.3f}")
    
    def distance_to_embedding(self, embedding: np.ndarray, method: str = 'mahalanobis') -> float:
        """
        Calculate distance from embedding to this skeleton.
        
        Args:
            embedding: Single embedding vector
            method: Distance method ('euclidean', 'cosine', 'mahalanobis', 'knn')
            
        Returns:
            Distance value (lower = more similar)
        """
        if method == 'euclidean':
            return float(np.linalg.norm(embedding - self.mean))
        
        elif method == 'cosine':
            return float(1 - np.dot(embedding, self.mean) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.mean)
            ))
        
        elif method == 'mahalanobis':
            try:
                diff = embedding - self.mean
                inv_cov = np.linalg.pinv(self.cov + 1e-6 * np.eye(self.feature_dim))
                return float(np.sqrt(diff.T @ inv_cov @ diff))
            except:
                # Fallback to euclidean if covariance is singular
                return float(np.linalg.norm(embedding - self.mean))
        
        elif method == 'knn':
            # Distance to k nearest neighbors in skeleton
            k = min(5, len(self.embeddings))
            distances = np.linalg.norm(self.embeddings - embedding, axis=1)
            return float(np.mean(np.sort(distances)[:k]))
        
        else:
            raise ValueError(f"Unknown distance method: {method}")
    
    def save(self, filepath: str):
        """Save skeleton to file."""
        data = {
            'label': self.label,
            'n_samples': self.n_samples,
            'feature_dim': self.feature_dim,
            'mean': self.mean,
            'std': self.std,
            'cov': self.cov,
            'median': self.median,
            'mad': self.mad,
            'embeddings': self.embeddings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {self.label} skeleton to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load skeleton from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance and restore data
        skeleton = cls.__new__(cls)
        for key, value in data.items():
            setattr(skeleton, key, value)
        
        logger.info(f"Loaded {skeleton.label} skeleton from {filepath}")
        return skeleton

class PromptEncoder(nn.Module):
    """
    Simple prompt encoder for multi-task learning.
    Encodes text prompts into embedding space.
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        super(PromptEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode prompt tokens to embedding.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Prompt embeddings of shape (batch_size, embed_dim)
        """
        # Embed tokens
        embedded = self.embedding(token_ids)  # (batch_size, seq_len, embed_dim)
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # Use last hidden state
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch_size, hidden_dim*2)
        
        # Project to embedding space
        prompt_embedding = self.projection(final_hidden)  # (batch_size, embed_dim)
        
        return prompt_embedding

class SkeletonBasedDetector(nn.Module):
    """
    Skeleton-based AI detector with distance matching and optional multi-task learning.
    """
    
    def __init__(
        self,
        backbone: str = 'efficientnet_b3',
        num_frames: int = 5,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        enable_multitask: bool = False,
        prompt_embed_dim: int = 256
    ):
        super(SkeletonBasedDetector, self).__init__()
        
        # Base detector
        self.base_detector = AdvancedAIDetector(
            backbone=backbone,
            num_frames=num_frames,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
        
        # Feature dimension
        self.feature_dim = self.base_detector.feature_dim
        
        # Multi-task learning components
        self.enable_multitask = enable_multitask
        if enable_multitask:
            self.prompt_encoder = PromptEncoder(embed_dim=prompt_embed_dim)
            self.prompt_predictor = nn.Sequential(
                nn.Linear(self.feature_dim, prompt_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(prompt_embed_dim * 2, prompt_embed_dim)
            )
        
        # Skeleton storage
        self.ai_skeleton = None
        self.real_skeleton = None
        self.skeleton_distance_method = 'mahalanobis'
        
        logger.info(f"Initialized SkeletonBasedDetector:")
        logger.info(f"  Backbone: {backbone}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        logger.info(f"  Multi-task: {enable_multitask}")
    
    def forward(self, x: torch.Tensor, prompts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through skeleton-based detector.
        
        Args:
            x: Input frames of shape (batch_size, num_frames, 3, H, W)
            prompts: Optional prompt token IDs for multi-task learning
            
        Returns:
            Dictionary with predictions and embeddings
        """
        # Get base predictions and features
        logits = self.base_detector(x)
        
        # Extract frame features for skeleton matching
        batch_size, num_frames = x.shape[:2]
        x_flat = x.view(batch_size * num_frames, *x.shape[2:])
        frame_features = self.base_detector.backbone(x_flat)
        frame_features = frame_features.view(batch_size, num_frames, self.feature_dim)
        
        # Aggregate features for skeleton matching
        video_embeddings = torch.mean(frame_features, dim=1)  # (batch_size, feature_dim)
        
        results = {
            'logits': logits,
            'embeddings': video_embeddings
        }
        
        # Multi-task prompt prediction
        if self.enable_multitask and prompts is not None:
            prompt_embeddings = self.prompt_encoder(prompts)
            predicted_prompts = self.prompt_predictor(video_embeddings)
            
            results['prompt_embeddings'] = prompt_embeddings
            results['predicted_prompts'] = predicted_prompts
        
        return results
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract video embeddings for skeleton computation."""
        with torch.no_grad():
            results = self.forward(x)
            return results['embeddings']
    
    def compute_skeletons(
        self, 
        ai_embeddings: np.ndarray, 
        real_embeddings: np.ndarray
    ):
        """
        Compute and store class skeletons.
        
        Args:
            ai_embeddings: AI video embeddings of shape (n_ai, feature_dim)
            real_embeddings: Real video embeddings of shape (n_real, feature_dim)
        """
        self.ai_skeleton = SkeletonEmbedding(ai_embeddings, 'AI')
        self.real_skeleton = SkeletonEmbedding(real_embeddings, 'Real')
        
        logger.info("Computed class skeletons:")
        logger.info(f"  AI skeleton: {self.ai_skeleton.n_samples} samples")
        logger.info(f"  Real skeleton: {self.real_skeleton.n_samples} samples")
    
    def skeleton_predict(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict using skeleton distance matching.
        
        Args:
            embeddings: Video embeddings of shape (batch_size, feature_dim)
            
        Returns:
            Dictionary with skeleton-based predictions
        """
        if self.ai_skeleton is None or self.real_skeleton is None:
            raise ValueError("Skeletons not computed. Call compute_skeletons() first.")
        
        embeddings_np = embeddings.cpu().numpy()
        batch_size = embeddings_np.shape[0]
        
        ai_distances = []
        real_distances = []
        
        for i in range(batch_size):
            embedding = embeddings_np[i]
            
            ai_dist = self.ai_skeleton.distance_to_embedding(
                embedding, self.skeleton_distance_method
            )
            real_dist = self.real_skeleton.distance_to_embedding(
                embedding, self.skeleton_distance_method
            )
            
            ai_distances.append(ai_dist)
            real_distances.append(real_dist)
        
        ai_distances = torch.tensor(ai_distances, device=embeddings.device)
        real_distances = torch.tensor(real_distances, device=embeddings.device)
        
        # Convert distances to probabilities (closer = higher probability)
        # Use softmax on negative distances
        distance_logits = torch.stack([-real_distances, -ai_distances], dim=1)
        distance_probs = F.softmax(distance_logits, dim=1)
        
        # AI probability is the second column
        ai_probs = distance_probs[:, 1]
        
        return {
            'skeleton_probs': ai_probs,
            'ai_distances': ai_distances,
            'real_distances': real_distances,
            'distance_logits': distance_logits
        }
    
    def fused_predict(self, x: torch.Tensor, fusion_weight: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Predict using fusion of base classifier and skeleton matching.
        
        Args:
            x: Input frames
            fusion_weight: Weight for skeleton prediction (0=base only, 1=skeleton only)
            
        Returns:
            Fused prediction results
        """
        # Get base predictions
        results = self.forward(x)
        base_logits = results['logits']
        base_probs = torch.sigmoid(base_logits)
        
        # Get skeleton predictions
        skeleton_results = self.skeleton_predict(results['embeddings'])
        skeleton_probs = skeleton_results['skeleton_probs']
        
        # Fuse predictions
        fused_probs = (1 - fusion_weight) * base_probs + fusion_weight * skeleton_probs
        fused_logits = torch.log(fused_probs / (1 - fused_probs + 1e-8))
        
        return {
            'fused_logits': fused_logits,
            'fused_probs': fused_probs,
            'base_probs': base_probs,
            'skeleton_probs': skeleton_probs,
            'embeddings': results['embeddings'],
            **skeleton_results
        }
    
    def save_skeletons(self, save_dir: str):
        """Save computed skeletons to files."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        if self.ai_skeleton:
            self.ai_skeleton.save(str(save_path / "ai_skeleton.pkl"))
        
        if self.real_skeleton:
            self.real_skeleton.save(str(save_path / "real_skeleton.pkl"))
        
        # Save skeleton metadata
        metadata = {
            'feature_dim': self.feature_dim,
            'distance_method': self.skeleton_distance_method,
            'ai_samples': self.ai_skeleton.n_samples if self.ai_skeleton else 0,
            'real_samples': self.real_skeleton.n_samples if self.real_skeleton else 0
        }
        
        with open(save_path / "skeleton_metadata.json", 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved skeletons to {save_dir}")
    
    def load_skeletons(self, save_dir: str):
        """Load computed skeletons from files."""
        save_path = Path(save_dir)
        
        ai_skeleton_path = save_path / "ai_skeleton.pkl"
        real_skeleton_path = save_path / "real_skeleton.pkl"
        
        if ai_skeleton_path.exists():
            self.ai_skeleton = SkeletonEmbedding.load(str(ai_skeleton_path))
        
        if real_skeleton_path.exists():
            self.real_skeleton = SkeletonEmbedding.load(str(real_skeleton_path))
        
        logger.info(f"Loaded skeletons from {save_dir}")

class SkeletonLoss(nn.Module):
    """
    Combined loss function for skeleton-based training.
    """
    
    def __init__(
        self, 
        classification_weight: float = 1.0,
        prompt_weight: float = 0.1,
        skeleton_weight: float = 0.1
    ):
        super(SkeletonLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.prompt_weight = prompt_weight
        self.skeleton_weight = skeleton_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(
        self, 
        results: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            results: Model output dictionary
            labels: Ground truth labels
            prompt_embeddings: Optional prompt embeddings for multi-task loss
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Classification loss
        classification_loss = self.bce_loss(results['logits'], labels.float())
        losses['classification'] = classification_loss
        
        # Prompt prediction loss (multi-task)
        if 'predicted_prompts' in results and prompt_embeddings is not None:
            prompt_loss = self.mse_loss(results['predicted_prompts'], prompt_embeddings)
            losses['prompt'] = prompt_loss
        else:
            losses['prompt'] = torch.tensor(0.0, device=labels.device)
        
        # Skeleton consistency loss (encourage similar embeddings for same class)
        if 'embeddings' in results:
            skeleton_loss = self._compute_skeleton_loss(results['embeddings'], labels)
            losses['skeleton'] = skeleton_loss
        else:
            losses['skeleton'] = torch.tensor(0.0, device=labels.device)
        
        # Total loss
        total_loss = (
            self.classification_weight * losses['classification'] +
            self.prompt_weight * losses['prompt'] +
            self.skeleton_weight * losses['skeleton']
        )
        losses['total'] = total_loss
        
        return losses
    
    def _compute_skeleton_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute skeleton consistency loss."""
        # Separate AI and Real embeddings
        ai_mask = labels == 1
        real_mask = labels == 0
        
        if ai_mask.sum() == 0 or real_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        ai_embeddings = embeddings[ai_mask]
        real_embeddings = embeddings[real_mask]
        
        # Compute within-class similarity (should be high)
        ai_center = torch.mean(ai_embeddings, dim=0, keepdim=True)
        real_center = torch.mean(real_embeddings, dim=0, keepdim=True)
        
        ai_similarity = F.cosine_similarity(ai_embeddings, ai_center, dim=1)
        real_similarity = F.cosine_similarity(real_embeddings, real_center, dim=1)
        
        within_class_loss = -torch.mean(ai_similarity) - torch.mean(real_similarity)
        
        # Compute between-class dissimilarity (should be low)
        between_class_similarity = F.cosine_similarity(ai_center, real_center, dim=1)
        between_class_loss = torch.mean(between_class_similarity)
        
        return within_class_loss + between_class_loss

def create_skeleton_model(
    backbone: str = 'efficientnet_b3',
    num_frames: int = 5,
    freeze_backbone: bool = False,
    enable_multitask: bool = False
) -> SkeletonBasedDetector:
    """
    Factory function to create skeleton-based detector.
    
    Args:
        backbone: CNN backbone architecture
        num_frames: Number of frames per video
        freeze_backbone: Whether to freeze backbone weights
        enable_multitask: Enable multi-task prompt prediction
        
    Returns:
        Initialized skeleton-based detector
    """
    model = SkeletonBasedDetector(
        backbone=backbone,
        num_frames=num_frames,
        freeze_backbone=freeze_backbone,
        enable_multitask=enable_multitask
    )
    
    logger.info(f"Created skeleton-based detector:")
    logger.info(f"  Backbone: {backbone}")
    logger.info(f"  Frames: {num_frames}")
    logger.info(f"  Multi-task: {enable_multitask}")
    
    return model

if __name__ == "__main__":
    # Test skeleton model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_skeleton_model(
        backbone='efficientnet_b3',
        num_frames=5,
        enable_multitask=True
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 5, 3, 224, 224).to(device)
    dummy_prompts = torch.randint(0, 1000, (batch_size, 20)).to(device)
    
    with torch.no_grad():
        results = model(dummy_input, dummy_prompts)
    
    print("Model test results:")
    for key, value in results.items():
        print(f"  {key}: {value.shape}")
    
    # Test skeleton computation
    dummy_ai_embeddings = np.random.randn(100, model.feature_dim)
    dummy_real_embeddings = np.random.randn(100, model.feature_dim)
    
    model.compute_skeletons(dummy_ai_embeddings, dummy_real_embeddings)
    
    # Test skeleton prediction
    test_embeddings = torch.randn(batch_size, model.feature_dim).to(device)
    skeleton_results = model.skeleton_predict(test_embeddings)
    
    print("Skeleton prediction test:")
    for key, value in skeleton_results.items():
        print(f"  {key}: {value.shape}")