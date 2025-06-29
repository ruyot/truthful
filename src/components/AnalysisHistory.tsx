import React, { useState, useEffect } from 'react'
import { Clock, Video, ExternalLink, BarChart3, Trash2 } from 'lucide-react'
import { supabase } from '../lib/supabase'
import MultiColorText from './animations/MultiColorText'

export interface HistoryItem {
  id: string
  video_url?: string
  video_filename?: string
  overall_likelihood: number
  created_at: string
  analysis_results: Record<string, unknown>
}

interface AnalysisHistoryProps {
  userId: string
  onSelectAnalysis: (analysis: HistoryItem) => void
}

export const AnalysisHistory: React.FC<AnalysisHistoryProps> = ({ userId, onSelectAnalysis }) => {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (supabase && userId) {
      fetchHistory()
    } else {
      setLoading(false)
    }
  }, [userId])

  const fetchHistory = async () => {
    if (!supabase) return

    try {
      const { data, error } = await supabase
        .from('video_analyses')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(20)

      if (error) throw error
      setHistory(data || [])
    } catch (error) {
      console.error('Error fetching history:', error)
    } finally {
      setLoading(false)
    }
  }

  const deleteAnalysis = async (id: string) => {
    if (!supabase) return

    try {
      const { error } = await supabase
        .from('video_analyses')
        .delete()
        .eq('id', id)

      if (error) throw error
      setHistory(history.filter(item => item.id !== id))
    } catch (error) {
      console.error('Error deleting analysis:', error)
    }
  }

  const getResultBadge = (likelihood: number) => {
    if (likelihood < 30) return 'bg-green-100 text-green-800'
    if (likelihood < 70) return 'bg-yellow-100 text-yellow-800'
    return 'bg-red-100 text-red-800'
  }

  if (!supabase) {
    return (
      <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100 text-center">
        <Video className="mx-auto mb-4 text-gray-400" size={48} />
        <h2 className="text-xl font-bold text-gray-900 mb-2">History Unavailable</h2>
        <p className="text-gray-600">Configure Supabase to enable analysis history</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
        <h2 className="text-xl font-bold text-gray-900 mb-6 text-center">
          <MultiColorText
            text="Analysis History"
            colors={['#8B5CF6', '#EC4899', '#3B82F6']}
          />
        </h2>
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="bg-gray-100 rounded-lg p-4">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100 text-center">
        <Video className="mx-auto mb-4 text-gray-400" size={48} />
        <h2 className="text-xl font-bold text-gray-900 mb-2">No Analysis History</h2>
        <p className="text-gray-600">Your analyzed videos will appear here</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
      <h2 className="text-xl font-bold text-gray-900 mb-6 text-center">
        <MultiColorText
          text="Analysis History"
          colors={['#8B5CF6', '#EC4899', '#3B82F6']}
        />
      </h2>
      <div className="space-y-4">
        {history.map((item) => (
          <div key={item.id} className="border border-gray-200 bg-white rounded-lg p-4 hover:bg-gray-50 transition-colors transform hover:scale-105 transition-transform duration-300">
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <Video size={16} className="text-gray-600 flex-shrink-0" />
                  <h3 className="font-medium text-gray-900 truncate">
                    {item.video_filename || 'Video URL'}
                  </h3>
                  {item.video_url && (
                    <a 
                      href={item.video_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-purple-600 hover:text-purple-700 flex-shrink-0"
                    >
                      <ExternalLink size={14} />
                    </a>
                  )}
                </div>
                
                <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
                  <div className="flex items-center gap-1">
                    <Clock size={14} />
                    {new Date(item.created_at).toLocaleDateString()}
                  </div>
                  <div className="flex items-center gap-1">
                    <BarChart3 size={14} />
                    {(item.analysis_results?.total_frames as number) || 0} frames
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getResultBadge(item.overall_likelihood)}`}>
                    {item.overall_likelihood}% AI Generated
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => onSelectAnalysis(item)}
                      className="text-purple-600 hover:text-purple-700 text-sm font-medium"
                    >
                      View Details
                    </button>
                    <button
                      onClick={() => deleteAnalysis(item.id)}
                      className="text-red-600 hover:text-red-700 p-1"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}