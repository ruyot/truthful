"""
DeepFakeDetection (DFD) Dataset Frame Extraction Script

This script processes the DFD dataset downloaded via kagglehub and extracts
exactly 3 frames per video (at 1/3, 1/2, and 2/3 timestamps) for training
a PyTorch-based AI vs. Real video classification model.

Usage:
    python extract_dfd_frames.py --dataset_path /path/to/dfd --output_dir data --max_videos 1000

Features:
- Extracts 3 representative frames per video
- Organizes frames into /data/ai/ and /data/real/ directories
- Progress tracking with tqdm
- Graceful handling of corrupted videos
- Batch processing for efficiency
- Resume capability for interrupted runs
- Detailed logging and statistics

Author: AI Detector Team
Date: 2024
"""

import os
import cv2
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from tqdm import tqdm
import hashlib
from datetime import datetime
import shutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dfd_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DFDFrameExtractor:
    """
    DeepFakeDetection dataset frame extractor.
    
    Processes DFD videos and extracts frames for training AI detection models.
    """
    
    def __init__(self, dataset_path: str, output_dir: str, max_videos: Optional[int] = None):
        """
        Initialize the frame extractor.
        
        Args:
            dataset_path: Path to DFD dataset root directory
            output_dir: Output directory for extracted frames
            max_videos: Maximum videos to process per category (None for all)
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.max_videos = max_videos
        
        # Dataset structure
        self.manipulated_dir = self.dataset_path / "DFD_manipulated_sequences"
        self.original_dir = self.dataset_path / "DFD_original_sequences"
        
        # Output structure
        self.ai_output_dir = self.output_dir / "ai"
        self.real_output_dir = self.output_dir / "real"
        
        # Statistics
        self.stats = {
            'total_videos_found': 0,
            'total_videos_processed': 0,
            'total_frames_extracted': 0,
            'ai_videos_processed': 0,
            'real_videos_processed': 0,
            'ai_frames_extracted': 0,
            'real_frames_extracted': 0,
            'corrupted_videos': 0,
            'processing_errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Progress tracking
        self.progress_file = self.output_dir / "extraction_progress.json"
        self.processed_videos = set()
        
        self._setup_directories()
        self._load_progress()
    
    def _setup_directories(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ai_output_dir.mkdir(parents=True, exist_ok=True)
        self.real_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created:")
        logger.info(f"  AI frames: {self.ai_output_dir}")
        logger.info(f"  Real frames: {self.real_output_dir}")
    
    def _load_progress(self):
        """Load progress from previous runs."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.processed_videos = set(progress_data.get('processed_videos', []))
                    logger.info(f"Loaded progress: {len(self.processed_videos)} videos already processed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
    
    def _save_progress(self):
        """Save current progress."""
        try:
            progress_data = {
                'processed_videos': list(self.processed_videos),
                'stats': self.stats,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def _get_video_files(self, directory: Path) -> List[Path]:
        """
        Get all video files from a directory.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of video file paths
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        video_files = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return video_files
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def _extract_frames_from_video(self, video_path: Path, output_dir: Path, 
                                 video_id: str, category: str) -> Tuple[bool, int, str]:
        """
        Extract 3 frames from a single video.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            video_id: Unique identifier for the video
            category: 'ai' or 'real'
            
        Returns:
            Tuple of (success, frames_extracted, error_message)
        """
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False, 0, f"Could not open video: {video_path}"
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames < 3:
                cap.release()
                return False, 0, f"Video too short: {total_frames} frames"
            
            # Calculate frame positions (1/3, 1/2, 2/3)
            frame_positions = [
                int(total_frames * 1/3),
                int(total_frames * 1/2),
                int(total_frames * 2/3)
            ]
            
            frames_extracted = 0
            
            for i, frame_pos in enumerate(frame_positions):
                # Seek to frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame {frame_pos} from {video_path}")
                    continue
                
                # Generate frame filename
                frame_filename = f"{video_id}_frame_{i+1:02d}.jpg"
                frame_path = output_dir / frame_filename
                
                # Save frame
                success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    frames_extracted += 1
                else:
                    logger.warning(f"Could not save frame to {frame_path}")
            
            cap.release()
            
            if frames_extracted == 0:
                return False, 0, "No frames could be extracted"
            
            return True, frames_extracted, ""
            
        except Exception as e:
            return False, 0, f"Error processing video: {str(e)}"
    
    def _generate_video_id(self, video_path: Path, category: str) -> str:
        """
        Generate a unique ID for a video.
        
        Args:
            video_path: Path to video file
            category: 'ai' or 'real'
            
        Returns:
            Unique video identifier
        """
        # Use filename and category to create unique ID
        filename_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        return f"{category}_{video_path.stem}_{filename_hash}"
    
    def _process_video_batch(self, video_batch: List[Tuple[Path, str]]) -> Dict:
        """
        Process a batch of videos.
        
        Args:
            video_batch: List of (video_path, category) tuples
            
        Returns:
            Batch processing statistics
        """
        batch_stats = {
            'processed': 0,
            'frames_extracted': 0,
            'errors': 0,
            'corrupted': 0
        }
        
        for video_path, category in video_batch:
            try:
                # Skip if already processed
                video_key = f"{category}_{video_path.name}"
                if video_key in self.processed_videos:
                    continue
                
                # Determine output directory
                output_dir = self.ai_output_dir if category == 'ai' else self.real_output_dir
                
                # Generate video ID
                video_id = self._generate_video_id(video_path, category)
                
                # Extract frames
                success, frames_extracted, error_msg = self._extract_frames_from_video(
                    video_path, output_dir, video_id, category
                )
                
                if success:
                    batch_stats['processed'] += 1
                    batch_stats['frames_extracted'] += frames_extracted
                    self.processed_videos.add(video_key)
                    
                    # Update global stats
                    if category == 'ai':
                        self.stats['ai_videos_processed'] += 1
                        self.stats['ai_frames_extracted'] += frames_extracted
                    else:
                        self.stats['real_videos_processed'] += 1
                        self.stats['real_frames_extracted'] += frames_extracted
                else:
                    batch_stats['errors'] += 1
                    if "could not open" in error_msg.lower() or "too short" in error_msg.lower():
                        batch_stats['corrupted'] += 1
                    
                    logger.warning(f"Failed to process {video_path}: {error_msg}")
                
            except Exception as e:
                batch_stats['errors'] += 1
                logger.error(f"Unexpected error processing {video_path}: {e}")
        
        return batch_stats
    
    def extract_frames(self, num_workers: int = 4, batch_size: int = 10) -> Dict:
        """
        Extract frames from all videos in the dataset.
        
        Args:
            num_workers: Number of worker threads
            batch_size: Number of videos to process per batch
            
        Returns:
            Extraction statistics
        """
        self.stats['start_time'] = datetime.now().isoformat()
        logger.info("Starting DFD frame extraction...")
        
        # Get video files
        logger.info("Scanning for video files...")
        manipulated_videos = self._get_video_files(self.manipulated_dir)
        original_videos = self._get_video_files(self.original_dir)
        
        logger.info(f"Found {len(manipulated_videos)} manipulated videos")
        logger.info(f"Found {len(original_videos)} original videos")
        
        # Apply max_videos limit
        if self.max_videos:
            manipulated_videos = manipulated_videos[:self.max_videos]
            original_videos = original_videos[:self.max_videos]
            logger.info(f"Limited to {self.max_videos} videos per category")
        
        # Create video processing list
        video_list = []
        video_list.extend([(path, 'ai') for path in manipulated_videos])
        video_list.extend([(path, 'real') for path in original_videos])
        
        self.stats['total_videos_found'] = len(video_list)
        
        # Filter out already processed videos
        unprocessed_videos = []
        for video_path, category in video_list:
            video_key = f"{category}_{video_path.name}"
            if video_key not in self.processed_videos:
                unprocessed_videos.append((video_path, category))
        
        logger.info(f"Total videos to process: {len(unprocessed_videos)}")
        
        if not unprocessed_videos:
            logger.info("All videos already processed!")
            return self.stats
        
        # Create batches
        batches = []
        for i in range(0, len(unprocessed_videos), batch_size):
            batch = unprocessed_videos[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Processing {len(batches)} batches with {num_workers} workers")
        
        # Process batches with progress bar
        with tqdm(total=len(unprocessed_videos), desc="Extracting frames", unit="video") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_video_batch, batch): batch 
                    for batch in batches
                }
                
                # Process completed batches
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_stats = future.result()
                        
                        # Update progress
                        pbar.update(batch_stats['processed'])
                        
                        # Update global stats
                        self.stats['total_videos_processed'] += batch_stats['processed']
                        self.stats['total_frames_extracted'] += batch_stats['frames_extracted']
                        self.stats['corrupted_videos'] += batch_stats['corrupted']
                        self.stats['processing_errors'] += batch_stats['errors']
                        
                        # Save progress periodically
                        self._save_progress()
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        self.stats['processing_errors'] += len(batch)
        
        self.stats['end_time'] = datetime.now().isoformat()
        self._save_progress()
        
        # Final statistics
        logger.info("Frame extraction completed!")
        self._print_final_stats()
        
        return self.stats
    
    def _print_final_stats(self):
        """Print final extraction statistics."""
        logger.info("="*60)
        logger.info("EXTRACTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total videos found: {self.stats['total_videos_found']}")
        logger.info(f"Total videos processed: {self.stats['total_videos_processed']}")
        logger.info(f"Total frames extracted: {self.stats['total_frames_extracted']}")
        logger.info("")
        logger.info(f"AI videos processed: {self.stats['ai_videos_processed']}")
        logger.info(f"AI frames extracted: {self.stats['ai_frames_extracted']}")
        logger.info(f"Real videos processed: {self.stats['real_videos_processed']}")
        logger.info(f"Real frames extracted: {self.stats['real_frames_extracted']}")
        logger.info("")
        logger.info(f"Corrupted videos: {self.stats['corrupted_videos']}")
        logger.info(f"Processing errors: {self.stats['processing_errors']}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            start_time = datetime.fromisoformat(self.stats['start_time'])
            end_time = datetime.fromisoformat(self.stats['end_time'])
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Processing time: {duration:.2f} seconds")
            
            if self.stats['total_videos_processed'] > 0:
                avg_time = duration / self.stats['total_videos_processed']
                logger.info(f"Average time per video: {avg_time:.2f} seconds")
        
        logger.info("="*60)
    
    def verify_extraction(self) -> Dict:
        """
        Verify the extracted frames.
        
        Returns:
            Verification statistics
        """
        logger.info("Verifying extracted frames...")
        
        verification_stats = {
            'ai_frames_found': 0,
            'real_frames_found': 0,
            'corrupted_frames': 0,
            'total_size_mb': 0.0
        }
        
        # Check AI frames
        if self.ai_output_dir.exists():
            ai_frames = list(self.ai_output_dir.glob("*.jpg"))
            verification_stats['ai_frames_found'] = len(ai_frames)
            
            for frame_path in ai_frames:
                try:
                    # Check if frame can be loaded
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        verification_stats['corrupted_frames'] += 1
                    else:
                        verification_stats['total_size_mb'] += frame_path.stat().st_size / (1024 * 1024)
                except Exception:
                    verification_stats['corrupted_frames'] += 1
        
        # Check Real frames
        if self.real_output_dir.exists():
            real_frames = list(self.real_output_dir.glob("*.jpg"))
            verification_stats['real_frames_found'] = len(real_frames)
            
            for frame_path in real_frames:
                try:
                    # Check if frame can be loaded
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        verification_stats['corrupted_frames'] += 1
                    else:
                        verification_stats['total_size_mb'] += frame_path.stat().st_size / (1024 * 1024)
                except Exception:
                    verification_stats['corrupted_frames'] += 1
        
        logger.info("Verification Results:")
        logger.info(f"  AI frames: {verification_stats['ai_frames_found']}")
        logger.info(f"  Real frames: {verification_stats['real_frames_found']}")
        logger.info(f"  Corrupted frames: {verification_stats['corrupted_frames']}")
        logger.info(f"  Total size: {verification_stats['total_size_mb']:.2f} MB")
        
        return verification_stats

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Extract frames from DFD dataset")
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to DFD dataset root directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for extracted frames (default: data)'
    )
    
    parser.add_argument(
        '--max_videos',
        type=int,
        default=None,
        help='Maximum videos to process per category (default: all)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker threads (default: 4)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )
    
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing extracted frames'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous extraction (default behavior)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = DFDFrameExtractor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_videos=args.max_videos
    )
    
    if args.verify_only:
        # Only run verification
        verification_stats = extractor.verify_extraction()
        return
    
    # Run extraction
    try:
        stats = extractor.extract_frames(
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )
        
        # Run verification
        verification_stats = extractor.verify_extraction()
        
        # Save final report
        final_report = {
            'extraction_stats': stats,
            'verification_stats': verification_stats,
            'extraction_completed': True,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = Path(args.output_dir) / "extraction_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"Final report saved to: {report_path}")
        
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        extractor._save_progress()
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        extractor._save_progress()
        raise

if __name__ == "__main__":
    main()