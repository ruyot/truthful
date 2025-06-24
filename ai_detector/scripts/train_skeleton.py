#!/usr/bin/env python3
"""
Training script for skeleton-based AI video classifier.

Usage examples:
    # Basic training with VidProM + DFD
    python train_skeleton.py --dfd_dir data --vidprom_dir data --epochs 100

    # Training with multi-task learning
    python train_skeleton.py --enable_multitask --epochs 150 --batch_size 8

    # Custom backbone and settings
    python train_skeleton.py --backbone convnext_tiny --lr 5e-4 --max_videos 1000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_detector.training.skeleton_trainer import main

if __name__ == '__main__':
    main()