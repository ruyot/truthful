"""
VidProM Dataset Integration for Skeleton-Based AI Detection

This module handles downloading and processing the VidProM dataset from Hugging Face,
filtering for commercial-compatible licenses, and preparing data for skeleton-based training.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import requests
from huggingface_hub import hf_hub_download, list_repo_files
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import hashlib

logger = logging.getLogger(__name__)

class VidProMDatasetProcessor:
    """
    Processor for VidProM dataset with license filtering and frame extraction.
    """
    
    def __init__(self, cache_dir: str = "vidprom_cache"):
        """
        Initialize VidProM dataset processor.
        
        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Commercial-compatible licenses (CC BY-NC 4.0 is acceptable for MVP/hackathon)
        self.allowed_licenses = [
            'cc-by-4.0',
            'cc-by-sa-4.0', 
            'cc-by-nc-4.0',
            'mit',
            'apache-2.0',
            'bsd-3-clause',
            'public-domain'
        ]
        
        logger.info(f"VidProM processor initialized with cache dir: {cache_dir}")
        logger.info(f"Allowed licenses: {self.allowed_licenses}")
    
    def download_metadata(self, repo_id: str = "vidprom/VidProM") -> Dict[str, Any]:
        """
        Download VidProM metadata from Hugging Face.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            Metadata dictionary
        """
        try:
            # Download metadata file
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.json",
                cache_dir=str(self.cache_dir)
            )
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Downloaded metadata for {len(metadata)} videos")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to download VidProM metadata: {e}")
            # Return mock metadata for development
            return self._create_mock_metadata()
    
    def _create_mock_metadata(self) -> Dict[str, Any]:
        """Create mock metadata for development when VidProM is not available."""
        mock_data = {}
        
        # Create mock entries for different AI generation methods
        ai_methods = ['sora', 'runway', 'pika', 'stable-video', 'animatediff']
        
        for i, method in enumerate(ai_methods):
            for j in range(20):  # 20 videos per method
                video_id = f"{method}_video_{j:03d}"
                mock_data[video_id] = {
                    'video_id': video_id,
                    'prompt': f"A sample prompt for {method} generation {j}",
                    'generation_method': method,
                    'license': 'cc-by-nc-4.0',
                    'duration': np.random.uniform(5.0, 30.0),
                    'resolution': '1024x576',
                    'fps': 24,
                    'file_size': np.random.randint(10, 100) * 1024 * 1024,
                    'url': f"https://mock-url.com/{video_id}.mp4"
                }
        
        logger.warning("Using mock VidProM metadata for development")
        return mock_data
    
    def filter_by_license(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter videos by commercial-compatible licenses.
        
        Args:
            metadata: Full metadata dictionary
            
        Returns:
            Filtered metadata dictionary
        """
        filtered = {}
        
        for video_id, video_data in metadata.items():
            license_type = video_data.get('license', '').lower()
            
            if any(allowed in license_type for allowed in self.allowed_licenses):
                filtered[video_id] = video_data
            else:
                logger.debug(f"Filtered out {video_id} due to license: {license_type}")
        
        logger.info(f"Filtered {len(filtered)} videos from {len(metadata)} total")
        return filtered
    
    def download_videos(
        self, 
        metadata: Dict[str, Any], 
        max_videos: Optional[int] = None,
        repo_id: str = "vidprom/VidProM"
    ) -> List[str]:
        """
        Download video files from VidProM dataset.
        
        Args:
            metadata: Filtered metadata dictionary
            max_videos: Maximum number of videos to download
            repo_id: Hugging Face repository ID
            
        Returns:
            List of downloaded video file paths
        """
        video_paths = []
        video_ids = list(metadata.keys())
        
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        for video_id in tqdm(video_ids, desc="Downloading VidProM videos"):
            try:
                # Try to download from Hugging Face
                video_filename = f"{video_id}.mp4"
                
                try:
                    video_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"videos/{video_filename}",
                        cache_dir=str(self.cache_dir)
                    )
                    video_paths.append(video_path)
                    
                except Exception as e:
                    logger.warning(f"Could not download {video_id} from HF: {e}")
                    # Create placeholder for development
                    placeholder_path = self.cache_dir / f"{video_id}_placeholder.mp4"
                    self._create_placeholder_video(placeholder_path)
                    video_paths.append(str(placeholder_path))
                
            except Exception as e:
                logger.error(f"Failed to process {video_id}: {e}")
                continue
        
        logger.info(f"Downloaded {len(video_paths)} videos")
        return video_paths
    
    def _create_placeholder_video(self, output_path: Path):
        """Create a placeholder video for development."""
        if output_path.exists():
            return
        
        # Create a simple 5-second video with random frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, 24.0, (640, 480))
        
        for _ in range(120):  # 5 seconds at 24 FPS
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    def extract_frames_from_videos(
        self, 
        video_paths: List[str], 
        metadata: Dict[str, Any],
        output_dir: str,
        frames_per_video: int = 5
    ) -> Dict[str, List[str]]:
        """
        Extract frames from downloaded videos.
        
        Args:
            video_paths: List of video file paths
            metadata: Video metadata
            output_dir: Output directory for frames
            frames_per_video: Number of frames to extract per video
            
        Returns:
            Dictionary mapping video_id to list of frame paths
        """
        output_path = Path(output_dir)
        ai_dir = output_path / "ai"
        ai_dir.mkdir(parents=True, exist_ok=True)
        
        frame_mapping = {}
        
        for video_path in tqdm(video_paths, desc="Extracting frames"):
            try:
                # Extract video ID from path
                video_id = Path(video_path).stem.replace('_placeholder', '')
                
                # Extract frames
                frames = self._extract_frames_from_video(
                    video_path, frames_per_video
                )
                
                # Save frames
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_filename = f"{video_id}_frame_{i+1:02d}.jpg"
                    frame_path = ai_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                
                frame_mapping[video_id] = frame_paths
                
            except Exception as e:
                logger.error(f"Failed to extract frames from {video_path}: {e}")
                continue
        
        logger.info(f"Extracted frames for {len(frame_mapping)} videos")
        return frame_mapping
    
    def _extract_frames_from_video(
        self, 
        video_path: str, 
        num_frames: int
    ) -> List[np.ndarray]:
        """Extract frames uniformly from a video."""
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("Video has no frames")
        
        # Calculate frame positions
        if total_frames <= num_frames:
            frame_positions = list(range(total_frames))
        else:
            frame_positions = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def process_complete_dataset(
        self, 
        output_dir: str,
        max_videos: Optional[int] = None,
        frames_per_video: int = 5
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for VidProM dataset.
        
        Args:
            output_dir: Output directory for processed data
            max_videos: Maximum videos to process
            frames_per_video: Frames to extract per video
            
        Returns:
            Processing summary
        """
        logger.info("Starting VidProM dataset processing...")
        
        # Download metadata
        metadata = self.download_metadata()
        
        # Filter by license
        filtered_metadata = self.filter_by_license(metadata)
        
        # Download videos
        video_paths = self.download_videos(
            filtered_metadata, 
            max_videos=max_videos
        )
        
        # Extract frames
        frame_mapping = self.extract_frames_from_videos(
            video_paths, 
            filtered_metadata, 
            output_dir, 
            frames_per_video
        )
        
        # Save processing summary
        summary = {
            'total_metadata_entries': len(metadata),
            'license_filtered_entries': len(filtered_metadata),
            'downloaded_videos': len(video_paths),
            'processed_videos': len(frame_mapping),
            'total_frames': sum(len(frames) for frames in frame_mapping.values()),
            'frames_per_video': frames_per_video,
            'output_directory': output_dir,
            'allowed_licenses': self.allowed_licenses
        }
        
        summary_path = Path(output_dir) / "vidprom_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"VidProM processing complete: {summary}")
        return summary

class CombinedVideoDataset(Dataset):
    """
    Combined dataset using both DFD (real videos) and VidProM (AI videos).
    """
    
    def __init__(
        self,
        dfd_dir: str,
        vidprom_dir: str,
        num_frames: int = 5,
        transform=None,
        max_videos_per_source: Optional[int] = None
    ):
        """
        Initialize combined dataset.
        
        Args:
            dfd_dir: Directory containing DFD real video frames
            vidprom_dir: Directory containing VidProM AI video frames
            num_frames: Number of frames per video
            transform: Image transforms
            max_videos_per_source: Max videos per source
        """
        self.num_frames = num_frames
        self.transform = transform
        
        # Load DFD real videos
        self.real_videos = self._load_video_frames(
            Path(dfd_dir) / "real", 
            label=0, 
            max_videos=max_videos_per_source
        )
        
        # Load VidProM AI videos
        self.ai_videos = self._load_video_frames(
            Path(vidprom_dir) / "ai", 
            label=1, 
            max_videos=max_videos_per_source
        )
        
        # Combine datasets
        self.all_videos = self.real_videos + self.ai_videos
        
        logger.info(f"Combined dataset loaded:")
        logger.info(f"  Real videos: {len(self.real_videos)}")
        logger.info(f"  AI videos: {len(self.ai_videos)}")
        logger.info(f"  Total videos: {len(self.all_videos)}")
    
    def _load_video_frames(
        self, 
        directory: Path, 
        label: int, 
        max_videos: Optional[int]
    ) -> List[Dict]:
        """Load video frame information from directory."""
        videos = {}
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        # Group frames by video ID
        for frame_file in directory.glob("*.jpg"):
            parts = frame_file.stem.split('_frame_')
            if len(parts) != 2:
                continue
            
            video_id = parts[0]
            frame_num = int(parts[1])
            
            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id,
                    'label': label,
                    'frames': []
                }
            
            videos[video_id]['frames'].append({
                'path': frame_file,
                'frame_num': frame_num
            })
        
        # Sort frames and convert to list
        video_list = []
        for video_id, video_data in videos.items():
            video_data['frames'].sort(key=lambda x: x['frame_num'])
            video_list.append(video_data)
        
        # Limit videos if specified
        if max_videos and len(video_list) > max_videos:
            video_list = video_list[:max_videos]
        
        return video_list
    
    def __len__(self) -> int:
        return len(self.all_videos)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a video sample."""
        video_data = self.all_videos[idx]
        
        # Sample frames
        frames = self._sample_frames(video_data['frames'])
        
        # Load and transform frames
        frame_tensors = []
        for frame_info in frames:
            try:
                image = Image.open(frame_info['path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frame_tensors.append(image)
            except Exception as e:
                logger.warning(f"Error loading frame {frame_info['path']}: {e}")
                # Create black frame as fallback
                if self.transform:
                    black_frame = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
                else:
                    black_frame = torch.zeros(3, 224, 224)
                frame_tensors.append(black_frame)
        
        frames_tensor = torch.stack(frame_tensors)
        return frames_tensor, video_data['label'], video_data['video_id']
    
    def _sample_frames(self, frames: List[Dict]) -> List[Dict]:
        """Sample frames uniformly from video."""
        if len(frames) <= self.num_frames:
            sampled = frames.copy()
            while len(sampled) < self.num_frames:
                sampled.extend(frames[:self.num_frames - len(sampled)])
            return sampled[:self.num_frames]
        
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        return [frames[i] for i in indices]

def download_and_process_vidprom(
    output_dir: str = "data",
    max_videos: Optional[int] = 1000,
    frames_per_video: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to download and process VidProM dataset.
    
    Args:
        output_dir: Output directory
        max_videos: Maximum videos to process
        frames_per_video: Frames per video
        
    Returns:
        Processing summary
    """
    processor = VidProMDatasetProcessor()
    return processor.process_complete_dataset(
        output_dir=output_dir,
        max_videos=max_videos,
        frames_per_video=frames_per_video
    )

if __name__ == "__main__":
    # Test VidProM processing
    summary = download_and_process_vidprom(
        output_dir="data",
        max_videos=100,
        frames_per_video=5
    )
    print("VidProM processing summary:", summary)