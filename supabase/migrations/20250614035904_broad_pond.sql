/*
  # Create video analyses table

  1. New Tables
    - `video_analyses`
      - `id` (uuid, primary key)
      - `user_id` (uuid, foreign key to auth.users)
      - `video_url` (text, optional - for URL-based videos)
      - `video_filename` (text, optional - for uploaded videos)
      - `overall_likelihood` (float, AI detection percentage 0-100)
      - `analysis_results` (jsonb, detailed frame-by-frame results)
      - `created_at` (timestamp with timezone)
      - `updated_at` (timestamp with timezone)

  2. Security
    - Enable RLS on `video_analyses` table
    - Add policy for authenticated users to read/write their own analyses
    - Add policy for users to delete their own analyses

  3. Indexes
    - Index on user_id for fast queries
    - Index on created_at for chronological sorting
*/

CREATE TABLE IF NOT EXISTS video_analyses (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    video_url text,
    video_filename text,
    overall_likelihood float NOT NULL CHECK (overall_likelihood >= 0 AND overall_likelihood <= 100),
    analysis_results jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Add constraint to ensure either video_url or video_filename is provided
ALTER TABLE video_analyses ADD CONSTRAINT video_source_check 
CHECK (video_url IS NOT NULL OR video_filename IS NOT NULL);

-- Enable Row Level Security
ALTER TABLE video_analyses ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can read own video analyses"
    ON video_analyses
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own video analyses"
    ON video_analyses
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own video analyses"
    ON video_analyses
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own video analyses"
    ON video_analyses
    FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_video_analyses_user_id ON video_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_video_analyses_created_at ON video_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_analyses_likelihood ON video_analyses(overall_likelihood);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_video_analyses_updated_at
    BEFORE UPDATE ON video_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();