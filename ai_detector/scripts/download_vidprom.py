#!/usr/bin/env python3
"""
Script to download and process VidProM dataset.

Usage examples:
    # Download and process VidProM dataset
    python download_vidprom.py --output_dir data --max_videos 1000

    # Process with more frames per video
    python download_vidprom.py --frames_per_video 7 --max_videos 500
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from ai_detector.datasets.vidprom_dataset import download_and_process_vidprom

def main():
    parser = argparse.ArgumentParser(description='Download and process VidProM dataset')
    
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for processed data')
    parser.add_argument('--max_videos', type=int, default=1000,
                       help='Maximum videos to process')
    parser.add_argument('--frames_per_video', type=int, default=5,
                       help='Frames to extract per video')
    
    args = parser.parse_args()
    
    print("Starting VidProM dataset download and processing...")
    
    summary = download_and_process_vidprom(
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        frames_per_video=args.frames_per_video
    )
    
    print("\nVidProM processing complete!")
    print(f"Summary: {summary}")

if __name__ == '__main__':
    main()