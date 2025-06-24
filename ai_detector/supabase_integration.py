"""
Supabase integration for AI detector model.
Handles image tagging, labeling, and model improvement workflows.
"""

import os
import asyncio
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json
import numpy as np
from PIL import Image
import io
import base64
import cv2

# Supabase imports (install with: pip install supabase)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase not available. Install with: pip install supabase")

from inference import AIDetectorInference

logger = logging.getLogger(__name__)

class SupabaseAIDetectorIntegration:
    """Integration class for AI detector with Supabase."""
    
    def __init__(self, supabase_url: str, supabase_key: str, model_path: str):
        """
        Initialize Supabase integration.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            model_path: Path to trained AI detector model
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase not available. Install with: pip install supabase")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.detector = AIDetectorInference(model_path)
        
        # Initialize database tables if needed
        self._setup_tables()
        
        logger.info("Supabase AI detector integration initialized")
    
    def _setup_tables(self):
        """Setup required database tables."""
        # This would typically be done via migrations, but here's the schema:
        
        # Table: ai_detection_results
        # - id (uuid, primary key)
        # - user_id (uuid, foreign key)
        # - image_url (text, optional)
        # - image_data (text, base64 encoded image, optional)
        # - prediction (integer, 0=real, 1=ai)
        # - probability (float, 0-1)
        # - confidence (float, 0-1)
        # - model_version (text)
        # - created_at (timestamp)
        # - verified_label (integer, optional, human-verified label)
        # - feedback_score (integer, optional, user feedback)
        
        # Table: training_data_queue
        # - id (uuid, primary key)
        # - image_data (text, base64)
        # - predicted_label (integer)
        # - verified_label (integer, optional)
        # - confidence (float)
        # - needs_review (boolean)
        # - created_at (timestamp)
        
        logger.info("Database tables should be set up via migrations")
    
    async def analyze_and_store_image(
        self, 
        image_data: bytes, 
        user_id: str,
        image_url: Optional[str] = None
    ) -> Dict:
        """
        Analyze image and store results in Supabase.
        
        Args:
            image_data: Raw image bytes
            user_id: User ID
            image_url: Optional image URL
            
        Returns:
            Analysis results with database ID
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Run AI detection
            result = self.detector.predict_single(image)
            
            # Encode image as base64 for storage
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Store in database
            db_record = {
                'user_id': user_id,
                'image_url': image_url,
                'image_data': image_b64,
                'prediction': result['prediction'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'model_version': 'v1.0',
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = self.supabase.table('ai_detection_results').insert(db_record).execute()
            
            # Add database ID to result
            result['database_id'] = response.data[0]['id']
            result['stored_at'] = db_record['created_at']
            
            logger.info(f"Stored analysis result with ID: {result['database_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing and storing image: {e}")
            raise
    
    async def submit_user_feedback(
        self, 
        result_id: str, 
        user_feedback: int,
        verified_label: Optional[int] = None
    ) -> bool:
        """
        Submit user feedback on a prediction.
        
        Args:
            result_id: Database ID of the prediction
            user_feedback: Feedback score (1-5, where 5 is "very accurate")
            verified_label: Optional verified label (0=real, 1=ai)
            
        Returns:
            Success status
        """
        try:
            update_data = {'feedback_score': user_feedback}
            
            if verified_label is not None:
                update_data['verified_label'] = verified_label
            
            self.supabase.table('ai_detection_results').update(update_data).eq('id', result_id).execute()
            
            # If this is a correction, add to training queue
            if verified_label is not None:
                await self._add_to_training_queue(result_id, verified_label)
            
            logger.info(f"Updated feedback for result {result_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False
    
    async def _add_to_training_queue(self, result_id: str, verified_label: int):
        """Add corrected prediction to training queue."""
        try:
            # Get original prediction
            result = self.supabase.table('ai_detection_results').select('*').eq('id', result_id).execute()
            
            if result.data:
                record = result.data[0]
                
                # Add to training queue
                queue_record = {
                    'image_data': record['image_data'],
                    'predicted_label': record['prediction'],
                    'verified_label': verified_label,
                    'confidence': record['confidence'],
                    'needs_review': record['prediction'] != verified_label,  # Flag misclassifications
                    'created_at': datetime.utcnow().isoformat()
                }
                
                self.supabase.table('training_data_queue').insert(queue_record).execute()
                logger.info(f"Added correction to training queue")
                
        except Exception as e:
            logger.error(f"Error adding to training queue: {e}")
    
    async def get_training_data(self, limit: int = 1000) -> List[Dict]:
        """
        Get training data from the queue for model retraining.
        
        Args:
            limit: Maximum number of samples to retrieve
            
        Returns:
            List of training samples
        """
        try:
            response = self.supabase.table('training_data_queue').select('*').limit(limit).execute()
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return []
    
    async def get_user_analysis_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get user's analysis history.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            
        Returns:
            List of analysis results
        """
        try:
            response = self.supabase.table('ai_detection_results').select(
                'id, prediction, probability, confidence, created_at, image_url, feedback_score'
            ).eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
            
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []
    
    async def get_model_performance_stats(self) -> Dict:
        """
        Get model performance statistics from user feedback.
        
        Returns:
            Performance statistics
        """
        try:
            # Get all results with feedback
            response = self.supabase.table('ai_detection_results').select(
                'prediction, verified_label, feedback_score, confidence'
            ).not_.is_('feedback_score', 'null').execute()
            
            data = response.data
            
            if not data:
                return {'total_feedback': 0}
            
            # Calculate statistics
            total_feedback = len(data)
            avg_feedback_score = np.mean([r['feedback_score'] for r in data])
            
            # Accuracy where verified labels exist
            verified_data = [r for r in data if r['verified_label'] is not None]
            accuracy = 0.0
            if verified_data:
                correct = sum(1 for r in verified_data if r['prediction'] == r['verified_label'])
                accuracy = correct / len(verified_data)
            
            # Confidence statistics
            confidences = [r['confidence'] for r in data]
            avg_confidence = np.mean(confidences)
            
            return {
                'total_feedback': total_feedback,
                'average_feedback_score': float(avg_feedback_score),
                'verified_samples': len(verified_data),
                'accuracy_on_verified': float(accuracy),
                'average_confidence': float(avg_confidence),
                'needs_retraining': accuracy < 0.8 and len(verified_data) > 100
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {'error': str(e)}
    
    async def export_training_data(self, output_dir: str = 'exported_training_data'):
        """
        Export training data for model retraining.
        
        Args:
            output_dir: Directory to save exported data
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/ai", exist_ok=True)
            os.makedirs(f"{output_dir}/real", exist_ok=True)
            
            # Get training data
            training_data = await self.get_training_data()
            
            exported_count = {'ai': 0, 'real': 0}
            
            for i, sample in enumerate(training_data):
                if sample['verified_label'] is not None:
                    # Decode image
                    image_data = base64.b64decode(sample['image_data'])
                    
                    # Determine folder
                    folder = 'ai' if sample['verified_label'] == 1 else 'real'
                    
                    # Save image
                    filename = f"{folder}_{i:06d}.jpg"
                    filepath = os.path.join(output_dir, folder, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    
                    exported_count[folder] += 1
            
            # Save metadata
            metadata = {
                'export_date': datetime.utcnow().isoformat(),
                'total_samples': len(training_data),
                'ai_samples': exported_count['ai'],
                'real_samples': exported_count['real'],
                'export_directory': output_dir
            }
            
            with open(f"{output_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Exported {sum(exported_count.values())} training samples to {output_dir}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            raise

# Example usage and integration functions
async def analyze_video_frame_with_supabase(
    frame: np.ndarray,
    user_id: str,
    supabase_integration: SupabaseAIDetectorIntegration
) -> Dict:
    """
    Analyze video frame and optionally store in Supabase.
    
    Args:
        frame: Video frame
        user_id: User ID
        supabase_integration: Supabase integration instance
        
    Returns:
        Analysis results
    """
    try:
        # Convert frame to bytes
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Analyze and store
        result = await supabase_integration.analyze_and_store_image(img_bytes, user_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in video frame analysis: {e}")
        # Fallback to local inference
        detector = AIDetectorInference('weights/ai_detector.pt')
        return detector.predict_single(frame)

# Migration SQL for Supabase
SUPABASE_MIGRATION_SQL = """
-- AI Detection Results Table
CREATE TABLE IF NOT EXISTS ai_detection_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    image_url text,
    image_data text, -- base64 encoded image
    prediction integer NOT NULL CHECK (prediction IN (0, 1)),
    probability double precision NOT NULL CHECK (probability >= 0 AND probability <= 1),
    confidence double precision NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version text NOT NULL DEFAULT 'v1.0',
    verified_label integer CHECK (verified_label IN (0, 1)),
    feedback_score integer CHECK (feedback_score >= 1 AND feedback_score <= 5),
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Training Data Queue Table
CREATE TABLE IF NOT EXISTS training_data_queue (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    image_data text NOT NULL, -- base64 encoded image
    predicted_label integer NOT NULL CHECK (predicted_label IN (0, 1)),
    verified_label integer CHECK (verified_label IN (0, 1)),
    confidence double precision NOT NULL,
    needs_review boolean DEFAULT false,
    processed boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE ai_detection_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_data_queue ENABLE ROW LEVEL SECURITY;

-- RLS Policies for ai_detection_results
CREATE POLICY "Users can read own detection results"
    ON ai_detection_results FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own detection results"
    ON ai_detection_results FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own detection results"
    ON ai_detection_results FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- RLS Policies for training_data_queue (admin only)
CREATE POLICY "Admins can manage training queue"
    ON training_data_queue FOR ALL
    TO authenticated
    USING (auth.jwt() ->> 'role' = 'admin');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ai_detection_results_user_id ON ai_detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_detection_results_created_at ON ai_detection_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_data_queue_needs_review ON training_data_queue(needs_review) WHERE needs_review = true;

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_ai_detection_results_updated_at
    BEFORE UPDATE ON ai_detection_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""

if __name__ == "__main__":
    # Example usage
    print("Supabase integration module loaded")
    print("Migration SQL:")
    print(SUPABASE_MIGRATION_SQL)