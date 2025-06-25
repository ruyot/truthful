#!/usr/bin/env python3
"""
Script to download 1000 real videos from UCF101 and extract frames.

Usage:
    python download_ucf101.py --output_dir data --max_videos 1000 --frames_per_video 5
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

VIDEO_EXTS = {'.avi', '.mp4', '.mov'}

def extract_frames(video_path, output_dir, frames_per_video=5):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < frames_per_video or total_frames == 0:
        cap.release()
        return 0

    step = total_frames // frames_per_video
    count = 0
    for i in range(frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = cap.read()
        if success:
            frame_name = f"{video_path.stem}_frame_{i+1:02d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            count += 1

    cap.release()
    os.remove(video_path)
    return count

def main():
    parser = argparse.ArgumentParser(description="Download UCF101 and extract frames")
    parser.add_argument("--output_dir", type=str, default="data", help="Where to store frames")
    parser.add_argument("--max_videos", type=int, default=1000, help="How many videos to use")
    parser.add_argument("--frames_per_video", type=int, default=5, help="Frames per video")
    args = parser.parse_args()

    real_dir = Path(args.output_dir) / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading UCF101 dataset...")
    dataset = load_dataset("ucf101", split="train[:{}]".format(args.max_videos))

    print(f"[✓] Loaded {len(dataset)} videos")

    extracted_total = 0
    for entry in tqdm(dataset, desc="Extracting frames", unit="video"):
        video = entry.get("video")
        if not video or not video.get("path"):
            continue
        video_path = Path(video["path"])
        if not video_path.exists():
            continue
        try:
            frames_extracted = extract_frames(video_path, real_dir, args.frames_per_video)
            extracted_total += frames_extracted
        except Exception as e:
            print(f"[!] Error with {video_path}: {e}")

    print(f"[✓] Done! Extracted {extracted_total} frames to: {real_dir}")

if __name__ == "__main__":
    main()