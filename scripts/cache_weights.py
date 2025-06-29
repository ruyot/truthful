"""
Skeleton model weights caching script.

This script loads the skeleton model weights into memory to ensure they're cached
in the Docker image layer, improving startup performance.

Usage:
    python scripts/cache_weights.py
"""

import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get model path from environment variable
    model_path = os.getenv("SKELETON_MODEL_PATH")
    
    if not model_path:
        logger.error("SKELETON_MODEL_PATH environment variable not set")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        sys.exit(1)
    
    logger.info(f"Loading model from {model_path} to cache weights...")
    
    try:
        # Load model weights into memory
        checkpoint = torch.load(model_path, map_location="cpu")
        logger.info(f"Successfully loaded model with {len(checkpoint['model_state_dict'])} layers")
        logger.info(f"Model config: {checkpoint.get('config', {}).get('model', {})}")
        logger.info("Model weights cached successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()