import { createClient, User } from '@supabase/supabase-js'

// Use placeholder values that won't cause errors if env vars are missing
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://placeholder.supabase.co'
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'placeholder-key'

// Only create client if we have real values
const hasValidConfig = supabaseUrl !== 'https://placeholder.supabase.co' && supabaseKey !== 'placeholder-key'

export const supabase = hasValidConfig 
  ? createClient(supabaseUrl, supabaseKey)
  : null

export type Database = {
  public: {
    Tables: {
      video_analyses: {
        Row: {
          id: string
          user_id: string
          video_url?: string
          video_filename?: string
          overall_likelihood: number
          analysis_results: Record<string, unknown>
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          video_url?: string
          video_filename?: string
          overall_likelihood: number
          analysis_results: Record<string, unknown>
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          video_url?: string
          video_filename?: string
          overall_likelihood?: number
          analysis_results?: Record<string, unknown>
          created_at?: string
          updated_at?: string
        }
      }
    }
  }
}

export type { User }