#!/usr/bin/env python3
"""
Comprehensive VidProM Dataset Download & Frame Extraction Script

- Downloads metadata and example videos from Hugging Face
- Extracts representative frames from each video
- Stores only extracted frames to save space
- Compatible with training pipelines (AI vs Real classification)

Usage:
    python download_vidprom_full_pipeline.py --output_dir data --max_videos 1000 --frames_per_video 5
"""

import os
import sys
import cv2
import json
import tarfile
import argparse
import urllib.request
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

EXAMPLE_TARS = [
    "https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/example/pika_videos_example.tar",
    "https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/example/vc2_videos_example.tar",
    "https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/example/opensora_videos_example.tar"
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

def download_tar(url, output_dir):
    filename = os.path.join(output_dir, os.path.basename(url))
    if os.path.exists(filename):
        print(f"[✓] Already downloaded: {filename}")
        return filename
    print(f"[↓] Downloading {url}")
    urllib.request.urlretrieve(url, filename)
    print(f"[✓] Saved to {filename}")
    return filename

def extract_tar(filepath, output_dir):
    print(f"[×] Extracting {filepath}")
    with tarfile.open(filepath, "r") as tar:
        tar.extractall(path=output_dir)
    print(f"[✓] Extracted to {output_dir}")

def extract_frames_from_video(video_path, output_dir, frames_per_video=5):
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_per_video:
            return 0  # Skip short videos

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
        os.remove(video_path)  # Clean up video after extracting
        return extracted
    except Exception as e:
        print(f"[!] Error extracting from {video_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Download and process VidProM dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max_videos", type=int, default=1000, help="Maximum metadata entries")
    parser.add_argument("--frames_per_video", type=int, default=5, help="Frames per video")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ai_dir = Path(args.output_dir) / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading metadata from Hugging Face...")
    dataset = load_dataset("WenhaoWang/VidProM", split="VidProM_unique")
    subset = dataset.select(range(min(args.max_videos, len(dataset))))
    subset.to_csv(os.path.join(args.output_dir, "VidProM_subset.csv"))
    print(f"[✓] Saved metadata CSV with {len(subset)} entries")

    for tar_url in EXAMPLE_TARS:
        tar_path = download_tar(tar_url, args.output_dir)
        extract_tar(tar_path, args.output_dir)

    print("[*] Extracting frames from example videos...")
    video_paths = []
    for root, _, files in os.walk(args.output_dir):
        for file in files:
            if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                video_paths.append(Path(root) / file)

    total_frames = 0
    for video_path in tqdm(video_paths, desc="Extracting frames", unit="video"):
        extracted = extract_frames_from_video(video_path, ai_dir, args.frames_per_video)
        total_frames += extracted

    print(f"[✓] Finished extracting frames: {total_frames} total")
    print(f"[✓] All frames saved in: {ai_dir}")

if __name__ == "__main__":
    main()
