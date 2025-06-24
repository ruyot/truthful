"""
Kaggle DFD Dataset Downloader and Processor

This script downloads the DeepFakeDetection dataset from Kaggle using kagglehub
and then processes it to extract frames for training.

Usage:
    python kaggle_dfd_downloader.py --output_dir data --max_videos 1000

Requirements:
    - kagglehub library
    - Kaggle API credentials configured
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional
import kagglehub
from extract_dfd_frames import DFDFrameExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDFDProcessor:
    """
    Downloads and processes the DFD dataset from Kaggle.
    """
    
    def __init__(self, output_dir: str = "data", max_videos: Optional[int] = None):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory for extracted frames
            max_videos: Maximum videos to process per category
        """
        self.output_dir = Path(output_dir)
        self.max_videos = max_videos
        self.dataset_path = None
    
    def download_dataset(self) -> str:
        """
        Download the DFD dataset from Kaggle.
        
        Returns:
            Path to downloaded dataset
        """
        logger.info("Downloading DFD dataset from Kaggle...")
        logger.info("This may take a while (~24GB download)...")
        
        try:
            # Download latest version of the dataset
            path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
            logger.info(f"Dataset downloaded to: {path}")
            self.dataset_path = path
            return path
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Kaggle API credentials configured (~/.kaggle/kaggle.json)")
            logger.error("2. Accepted the dataset terms on Kaggle website")
            logger.error("3. Internet connection for large download")
            raise
    
    def process_dataset(self, num_workers: int = 4) -> dict:
        """
        Process the downloaded dataset to extract frames.
        
        Args:
            num_workers: Number of worker threads
            
        Returns:
            Processing statistics
        """
        if not self.dataset_path:
            raise ValueError("Dataset not downloaded yet. Call download_dataset() first.")
        
        logger.info("Processing dataset to extract frames...")
        
        # Initialize frame extractor
        extractor = DFDFrameExtractor(
            dataset_path=self.dataset_path,
            output_dir=str(self.output_dir),
            max_videos=self.max_videos
        )
        
        # Extract frames
        stats = extractor.extract_frames(num_workers=num_workers)
        
        # Verify extraction
        verification_stats = extractor.verify_extraction()
        
        return {
            'extraction_stats': stats,
            'verification_stats': verification_stats
        }
    
    def run_complete_pipeline(self, num_workers: int = 4) -> dict:
        """
        Run the complete download and processing pipeline.
        
        Args:
            num_workers: Number of worker threads
            
        Returns:
            Complete processing statistics
        """
        # Step 1: Download dataset
        dataset_path = self.download_dataset()
        
        # Step 2: Process dataset
        results = self.process_dataset(num_workers=num_workers)
        
        logger.info("Complete pipeline finished!")
        logger.info(f"Dataset location: {dataset_path}")
        logger.info(f"Frames extracted to: {self.output_dir}")
        
        return results

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Download and process DFD dataset from Kaggle")
    
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
        '--download_only',
        action='store_true',
        help='Only download the dataset, do not process'
    )
    
    parser.add_argument(
        '--process_only',
        action='store_true',
        help='Only process existing dataset (provide --dataset_path)'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to existing dataset (for --process_only)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.process_only:
            if not args.dataset_path:
                logger.error("--dataset_path required when using --process_only")
                return
            
            # Process existing dataset
            extractor = DFDFrameExtractor(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                max_videos=args.max_videos
            )
            
            stats = extractor.extract_frames(num_workers=args.num_workers)
            verification_stats = extractor.verify_extraction()
            
        else:
            # Initialize processor
            processor = KaggleDFDProcessor(
                output_dir=args.output_dir,
                max_videos=args.max_videos
            )
            
            if args.download_only:
                # Only download
                dataset_path = processor.download_dataset()
                logger.info(f"Download complete. Dataset at: {dataset_path}")
            else:
                # Run complete pipeline
                results = processor.run_complete_pipeline(num_workers=args.num_workers)
                
                # Print summary
                extraction_stats = results['extraction_stats']
                verification_stats = results['verification_stats']
                
                logger.info("\n" + "="*50)
                logger.info("FINAL SUMMARY")
                logger.info("="*50)
                logger.info(f"Videos processed: {extraction_stats['total_videos_processed']}")
                logger.info(f"Frames extracted: {extraction_stats['total_frames_extracted']}")
                logger.info(f"AI frames: {verification_stats['ai_frames_found']}")
                logger.info(f"Real frames: {verification_stats['real_frames_found']}")
                logger.info(f"Total size: {verification_stats['total_size_mb']:.2f} MB")
                logger.info("="*50)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()