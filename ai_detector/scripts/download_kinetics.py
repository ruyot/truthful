#!/usr/bin/env python3
"""
Kinetics Dataset Frame Extraction Script

- Downloads Kinetics-400 dataset metadata from Hugging Face
- Extracts frames from 1000 random real videos
- Saves exactly `frames_per_video` frames per video
- Organizes output into `real/` directory for compatibility with training pipelines

Usage:
    python extract_kinetics_real.py --output_dir data --frames_per_video 5
"""

import os
import cv2
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}


def download_kinetics_videos(dataset_split, download_dir, max_videos):
    os.makedirs(download_dir, exist_ok=True)
    dataset = load_dataset("kinetics400", split=dataset_split)
    selected = dataset.shuffle(seed=42).select(range(max_videos))
    print(f"[✓] Selected {len(selected)} videos from Kinetics-400")

    video_paths = []
    for idx, sample in enumerate(tqdm(selected, desc="Downloading videos")):
        url = sample.get("url")
        if not url:
            continue

        video_path = Path(download_dir) / f"kinetics_{idx}.mp4"
        if video_path.exists():
            video_paths.append(video_path)
            continue

        try:
            os.system(f"wget -q '{url}' -O {video_path}")
            if video_path.exists():
                video_paths.append(video_path)
        except Exception as e:
            print(f"[!] Failed to download {url}: {e}")

    return video_paths


def extract_frames(video_path, output_dir, frames_per_video):
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_per_video:
            return 0

        step = max(total_frames // frames_per_video, 1)
        extracted = 0

        for i in range(frames_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            success, frame = cap.read()
            if not success:
                continue

            frame_path = output_dir / f"{video_path.stem}_frame_{i+1:02d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted += 1

        cap.release()
        os.remove(video_path)  # Remove video to save space
        return extracted
    except Exception as e:
        print(f"[!] Error extracting from {video_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Extract real frames from Kinetics dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--frames_per_video", type=int, default=5, help="Frames per video")
    parser.add_argument("--max_videos", type=int, default=1000, help="Number of real videos")
    args = parser.parse_args()

    real_dir = Path(args.output_dir) / "real"
    tmp_videos = Path(args.output_dir) / "tmp_videos"
    real_dir.mkdir(parents=True, exist_ok=True)
    tmp_videos.mkdir(parents=True, exist_ok=True)

    video_paths = download_kinetics_videos("train", tmp_videos, args.max_videos)

    print("[*] Extracting frames from real videos...")
    total_frames = 0
    for video_path in tqdm(video_paths, desc="Extracting", unit="video"):
        extracted = extract_frames(video_path, real_dir, args.frames_per_video)
        total_frames += extracted

    print(f"[✓] Finished extracting {total_frames} frames to {real_dir}")


if __name__ == "__main__":
    main()
