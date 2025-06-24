/*
  # Create video analyses table

  1. New Tables
    - `video_analyses`
      - `id` (uuid, primary key)
      - `user_id` (uuid, foreign key to auth.users)
      - `video_url` (text, optional - for URL-based videos)
      - `video_filename` (text, optional - for uploaded videos)
      - `overall_likelihood` (double precision, AI detection percentage 0-100)
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
    - Index on overall_likelihood for filtering

  4. Constraints
    - Check constraint to ensure either video_url or video_filename is provided
    - Check constraint to ensure overall_likelihood is between 0 and 100

  5. Triggers
    - Auto-update updated_at timestamp on row updates
*/

-- Create the video_analyses table if it doesn't exist
CREATE TABLE IF NOT EXISTS video_analyses (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    video_url text,
    video_filename text,
    overall_likelihood double precision NOT NULL,
    analysis_results jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Add foreign key constraint if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'video_analyses_user_id_fkey' 
        AND table_name = 'video_analyses'
    ) THEN
        ALTER TABLE video_analyses 
        ADD CONSTRAINT video_analyses_user_id_fkey 
        FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Add check constraint for overall_likelihood if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'video_analyses_overall_likelihood_check' 
        AND table_name = 'video_analyses'
    ) THEN
        ALTER TABLE video_analyses 
        ADD CONSTRAINT video_analyses_overall_likelihood_check 
        CHECK (overall_likelihood >= 0 AND overall_likelihood <= 100);
    END IF;
END $$;

-- Add constraint to ensure either video_url or video_filename is provided
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'video_source_check' 
        AND table_name = 'video_analyses'
    ) THEN
        ALTER TABLE video_analyses 
        ADD CONSTRAINT video_source_check 
        CHECK (video_url IS NOT NULL OR video_filename IS NOT NULL);
    END IF;
END $$;

-- Enable Row Level Security
ALTER TABLE video_analyses ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist and recreate them
DROP POLICY IF EXISTS "Users can read own video analyses" ON video_analyses;
DROP POLICY IF EXISTS "Users can insert own video analyses" ON video_analyses;
DROP POLICY IF EXISTS "Users can update own video analyses" ON video_analyses;
DROP POLICY IF EXISTS "Users can delete own video analyses" ON video_analyses;

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

-- Create indexes for performance (IF NOT EXISTS)
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

-- Drop existing trigger if it exists and recreate it
DROP TRIGGER IF EXISTS update_video_analyses_updated_at ON video_analyses;

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_video_analyses_updated_at
    BEFORE UPDATE ON video_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();