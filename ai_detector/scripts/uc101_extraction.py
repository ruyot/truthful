"""
UC101 Dataset Frame Extraction Script

Usage:
    python uc101_extraction.py --output_dir data --frames_per_video 20
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, frames_per_video=20):
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
            frame_name = f"{video_path.stem}_frame_{i+1:03d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            extracted += 1

        cap.release()
        return extracted
    except Exception as e:
        print(f"[!] Error with {video_path.name}: {e}")
        return 0

def process_ucf101(input_dir, output_dir, frames_per_video=20):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = list(input_dir.rglob("*.avi"))
    print(f"Found {len(video_paths)} videos.")

    total_extracted = 0
    for video_path in tqdm(video_paths, desc="Extracting frames"):
        total_extracted += extract_frames_from_video(video_path, output_dir, frames_per_video)

    print(f"Total frames extracted: {total_extracted}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to UCF-101 extracted video folder")
    parser.add_argument("--output_dir", type=str, default="data/real", help="Where to save extracted frames")
    parser.add_argument("--frames_per_video", type=int, default=20, help="Number of frames per video")
    args = parser.parse_args()

    process_ucf101(args.input_dir, args.output_dir, args.frames_per_video)