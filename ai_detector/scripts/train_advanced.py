#!/usr/bin/env python3
"""
Training script for advanced AI video classifier.

Usage examples:
    # Train with balanced configuration
    python train_advanced.py --model_config balanced --data_dir data --epochs 100

    # Train with custom backbone
    python train_advanced.py --backbone efficientnet_b3 --batch_size 8 --lr 5e-4

    # Train with focal loss and aggressive augmentation
    python train_advanced.py --loss focal --epochs 150 --max_videos 2000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_detector.training.advanced_trainer import main

if __name__ == '__main__':
    main()