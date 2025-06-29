"""
Skeleton‑Based AI Detector (updated to use new AdvancedAIDetector interface).
"""
import logging, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from typing import Dict, Optional
from ai_detector.models.advanced_model import AdvancedAIDetector

logger = logging.getLogger(__name__)


class SkeletonEmbedding:
    def __init__(self, embeddings: np.ndarray, label: str):
        self.label = label
        self.n_samples = len(embeddings)
        self.feature_dim = embeddings.shape[1]
        self.mean = np.mean(embeddings, axis=0)
        self.std = np.std(embeddings, axis=0)
        self.cov = np.cov(embeddings.T)
        self.median = np.median(embeddings, axis=0)
        self.mad = np.median(np.abs(embeddings - self.median), axis=0)
        self.embeddings = embeddings.copy()


class PromptEncoder(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.projection(final_hidden)


class SkeletonBasedDetector(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", num_frames: int = 5,
                 freeze_backbone: bool = False, dropout_rate: float = 0.3,
                 enable_multitask: bool = False, prompt_embed_dim: int = 256):
        super().__init__()
        self.base_detector = AdvancedAIDetector(backbone, num_frames, freeze_backbone, dropout_rate)
        self.feature_dim = self.base_detector.feature_dim
        self.enable_multitask = enable_multitask
        if enable_multitask:
            self.prompt_encoder = PromptEncoder(embed_dim=prompt_embed_dim)
            self.prompt_predictor = nn.Sequential(
                nn.Linear(self.feature_dim, prompt_embed_dim * 2),
                nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(prompt_embed_dim * 2, prompt_embed_dim)
            )
        
        # Skeleton-based detection components
        self.ai_skeleton = None
        self.real_skeleton = None
        self.skeleton_initialized = False

    def forward(self, x: torch.Tensor, prompts: Optional[torch.Tensor] = None):
        logits = self.base_detector(x)
        B, T = x.shape[:2]
        x_flat = x.reshape(B * T, *x.shape[2:])
        frame_feat = self.base_detector.extract_frame_features(x_flat)
        frame_feat = frame_feat.view(B, T, self.feature_dim)
        video_emb = frame_feat.mean(dim=1)
        out = {"logits": logits, "embeddings": video_emb}
        if self.enable_multitask and prompts is not None:
            out["prompt_embeddings"] = self.prompt_encoder(prompts)
            out["predicted_prompts"] = self.prompt_predictor(video_emb)
        return out

    def compute_skeletons(self, ai_embeddings: np.ndarray, real_embeddings: np.ndarray):
        """Compute skeleton representations for AI and real embeddings."""
        self.ai_skeleton = SkeletonEmbedding(ai_embeddings, "ai")
        self.real_skeleton = SkeletonEmbedding(real_embeddings, "real")
        self.skeleton_initialized = True

    def skeleton_predict(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict using skeleton-based distance metrics."""
        if not self.skeleton_initialized:
            raise ValueError("Skeletons must be computed before skeleton prediction")
        
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Calculate distances to AI and real skeletons
        ai_distances = self._calculate_skeleton_distance(embeddings_np, self.ai_skeleton)
        real_distances = self._calculate_skeleton_distance(embeddings_np, self.real_skeleton)
        
        # Convert distances to probabilities (closer to AI skeleton = higher AI probability)
        total_distance = ai_distances + real_distances
        skeleton_probs = ai_distances / (total_distance + 1e-8)  # Avoid division by zero
        
        return {
            'skeleton_probs': torch.tensor(skeleton_probs, device=embeddings.device),
            'ai_distances': torch.tensor(ai_distances, device=embeddings.device),
            'real_distances': torch.tensor(real_distances, device=embeddings.device)
        }

    def fused_predict(self, frames: torch.Tensor, fusion_weight: float = 0.5) -> Dict[str, torch.Tensor]:
        """Combine base detector and skeleton predictions."""
        # Get base detector predictions
        base_outputs = self.forward(frames)
        base_probs = torch.sigmoid(base_outputs['logits'])
        
        # Get skeleton predictions
        embeddings = base_outputs['embeddings']
        skeleton_results = self.skeleton_predict(embeddings)
        skeleton_probs = skeleton_results['skeleton_probs']
        
        # Fuse predictions
        fused_probs = fusion_weight * skeleton_probs + (1 - fusion_weight) * base_probs
        
        return {
            'fused_probs': fused_probs,
            'base_probs': base_probs,
            'skeleton_probs': skeleton_probs
        }

    def _calculate_skeleton_distance(self, embeddings: np.ndarray, skeleton: SkeletonEmbedding) -> np.ndarray:
        """Calculate Mahalanobis distance to skeleton."""
        diff = embeddings - skeleton.mean
        try:
            # Use pseudo-inverse for numerical stability
            inv_cov = np.linalg.pinv(skeleton.cov)
            mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        except:
            # Fallback to Euclidean distance if covariance is singular
            mahal_dist = np.linalg.norm(diff, axis=1)
        
        return mahal_dist


class SkeletonLoss(nn.Module):
    def __init__(self, classification_weight: float = 1.0, prompt_weight: float = 0.1, skeleton_weight: float = 0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.prompt_weight = prompt_weight
        self.skeleton_weight = skeleton_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, results: Dict[str, torch.Tensor], labels: torch.Tensor,
                prompt_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        losses = {}
        losses['classification'] = self.bce_loss(results['logits'], labels.float())
        losses['prompt'] = (
            self.mse_loss(results['predicted_prompts'], prompt_embeddings)
            if 'predicted_prompts' in results and prompt_embeddings is not None
            else torch.tensor(0.0, device=labels.device)
        )
        losses['skeleton'] = torch.tensor(0.0, device=labels.device)
        losses['total'] = (
            self.classification_weight * losses['classification'] +
            self.prompt_weight * losses['prompt'] +
            self.skeleton_weight * losses['skeleton']
        )
        return losses


def create_skeleton_model(backbone: str = 'convnext_tiny',
                          num_frames: int = 5,
                          freeze_backbone: bool = False,
                          enable_multitask: bool = False) -> SkeletonBasedDetector:
    return SkeletonBasedDetector(
        backbone=backbone,
        num_frames=num_frames,
        freeze_backbone=freeze_backbone,
        enable_multitask=enable_multitask
    )