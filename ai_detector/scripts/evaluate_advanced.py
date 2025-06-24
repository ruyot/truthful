#!/usr/bin/env python3
"""
Evaluation script for advanced AI video classifier.

Usage examples:
    # Evaluate best model
    python evaluate_advanced.py --model_path weights/best_advanced_model.pt --data_dir data

    # Evaluate with custom output directory
    python evaluate_advanced.py --model_path weights/best_advanced_model.pt --output_dir my_results --save_plots
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_detector.evaluation.advanced_evaluator import main

if __name__ == '__main__':
    main()