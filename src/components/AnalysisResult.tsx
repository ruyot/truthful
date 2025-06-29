import React from 'react'
import { CheckCircle, AlertTriangle, XCircle, Clock, Eye, BarChart3, ExternalLink, Zap, Brain, Search, Shield, Target } from 'lucide-react'

interface AnalysisResult {
  overall_likelihood: number
  analysis_results: {
    method: string
    preprocessing_details?: {
      metadata_flag: boolean
      ocr_flag: boolean
      logo_flag: boolean
      final_decision: string
      confidence_score: number
      processing_time: number
      frames_analyzed: number
    }
    ml_analysis?: {
      timestamps: Array<{
        time: number
        likelihood: number
        confidence: number
        details?: {
          structural_similarity_to_ai?: number
          structural_match_score?: number
          distance_method?: string
        }
      }>
      total_frames: number
      overall_likelihood: number
      video_duration: number
      analysis_fps: number
    }
    early_detection: boolean
    total_frames: number
    processing_time: number
    video_duration?: number
    detection_source?: string
    skeleton_available?: boolean
    skeleton_enabled?: boolean
  }
  video_url?: string
  video_filename?: string
  created_at: string
}

interface AnalysisResultProps {
  result: AnalysisResult
}

export const AnalysisResult: React.FC<AnalysisResultProps> = ({ result }) => {
  const { overall_likelihood, analysis_results } = result
  
  const getResultColor = (likelihood: number) => {
    if (likelihood < 30) return 'text-green-600'
    if (likelihood < 70) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getResultBg = (likelihood: number) => {
    if (likelihood < 30) return 'bg-green-50 border-green-200'
    if (likelihood < 70) return 'bg-yellow-50 border-yellow-200'
    return 'bg-red-50 border-red-200'
  }

  const getResultIcon = (likelihood: number) => {
    if (likelihood < 30) return <CheckCircle size={24} className="text-green-600" />
    if (likelihood < 70) return <AlertTriangle size={24} className="text-yellow-600" />
    return <XCircle size={24} className="text-red-600" />
  }

  const getResultText = (likelihood: number) => {
    if (likelihood < 30) return 'Likely Authentic'
    if (likelihood < 70) return 'Uncertain'
    return 'Likely AI Generated'
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Get detection method info
  const getDetectionMethodInfo = () => {
    const method = analysis_results.method
    const earlyDetection = analysis_results.early_detection
    const preprocessing = analysis_results.preprocessing_details
    const skeletonEnabled = analysis_results.skeleton_enabled

    if (earlyDetection && preprocessing) {
      return {
        icon: <Zap className="text-purple-600" size={20} />,
        title: 'Fast Detection',
        description: `Detected via ${preprocessing.final_decision.replace('AI-generated (', '').replace(')', '')}`,
        color: 'text-purple-600',
        bgColor: 'bg-purple-50 border-purple-200'
      }
    } else if (skeletonEnabled) {
      return {
        icon: <Target className="text-blue-600" size={20} />,
        title: 'Skeleton-Based Detection',
        description: 'Structural matching with distance analysis',
        color: 'text-blue-600',
        bgColor: 'bg-blue-50 border-blue-200'
      }
    } else if (method.includes('combined')) {
      return {
        icon: <Brain className="text-blue-600" size={20} />,
        title: 'ML Classification',
        description: 'Deep learning analysis with preprocessing',
        color: 'text-blue-600',
        bgColor: 'bg-blue-50 border-blue-200'
      }
    } else {
      return {
        icon: <Search className="text-gray-600" size={20} />,
        title: 'Standard Analysis',
        description: 'Machine learning classification',
        color: 'text-gray-600',
        bgColor: 'bg-gray-50 border-gray-200'
      }
    }
  }

  const detectionMethod = getDetectionMethodInfo()

  // Get high confidence timestamps from ML analysis
  const highConfidenceTimestamps = analysis_results.ml_analysis?.timestamps?.filter(t => t.likelihood > 70) || []

  // Get structural similarity info
  const getStructuralSimilarityInfo = () => {
    const timestamps = analysis_results.ml_analysis?.timestamps || []
    const structuralScores = timestamps
      .map(t => t.details?.structural_match_score)
      .filter(score => score !== undefined) as number[]
    
    if (structuralScores.length > 0) {
      const avgScore = structuralScores.reduce((a, b) => a + b, 0) / structuralScores.length
      return {
        available: true,
        averageScore: avgScore,
        maxScore: Math.max(...structuralScores),
        minScore: Math.min(...structuralScores)
      }
    }
    
    return { available: false }
  }

  const structuralInfo = getStructuralSimilarityInfo()

  // Get video source display
  const getVideoSource = () => {
    if (result.video_url) {
      if (result.video_url.includes('youtube.com') || result.video_url.includes('youtu.be')) {
        return {
          type: 'YouTube Video',
          url: result.video_url,
          displayUrl: result.video_url.length > 50 ? result.video_url.substring(0, 50) + '...' : result.video_url
        }
      }
      return {
        type: 'Video URL',
        url: result.video_url,
        displayUrl: result.video_url.length > 50 ? result.video_url.substring(0, 50) + '...' : result.video_url
      }
    }
    return {
      type: 'Uploaded File',
      displayUrl: result.video_filename || 'Unknown file'
    }
  }

  const videoSource = getVideoSource()

  return (
    <div className="bg-white rounded-2xl p-8 shadow-lg">
      <div className="text-center mb-8">
        <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 ${getResultBg(overall_likelihood)}`}>
          {getResultIcon(overall_likelihood)}
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Complete</h2>
        <p className={`text-lg font-semibold ${getResultColor(overall_likelihood)}`}>
          {getResultText(overall_likelihood)}
        </p>
        
        {/* Detection Method Badge */}
        <div className={`inline-flex items-center gap-2 mt-3 px-3 py-1 rounded-full border ${detectionMethod.bgColor}`}>
          {detectionMethod.icon}
          <span className={`text-sm font-medium ${detectionMethod.color}`}>
            {detectionMethod.title}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="text-center p-4 bg-gray-50 rounded-xl">
          <BarChart3 className="mx-auto mb-2 text-purple-600" size={24} />
          <div className="text-2xl font-bold text-gray-900">{overall_likelihood}%</div>
          <div className="text-sm text-gray-600">AI Likelihood</div>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-xl">
          <Eye className="mx-auto mb-2 text-blue-600" size={24} />
          <div className="text-2xl font-bold text-gray-900">{analysis_results.total_frames}</div>
          <div className="text-sm text-gray-600">Frames Analyzed</div>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-xl">
          <Clock className="mx-auto mb-2 text-green-600" size={24} />
          <div className="text-2xl font-bold text-gray-900">{analysis_results.processing_time}s</div>
          <div className="text-sm text-gray-600">Processing Time</div>
        </div>
      </div>

      <div className={`border rounded-xl p-6 ${getResultBg(overall_likelihood)}`}>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Summary</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">Overall Confidence:</span>
            <span className={`font-semibold ${getResultColor(overall_likelihood)}`}>
              {overall_likelihood}% AI Generated
            </span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-gray-700">Detection Method:</span>
            <span className={`font-medium ${detectionMethod.color}`}>
              {detectionMethod.description}
            </span>
          </div>
          
          {/* Structural Similarity Score */}
          {structuralInfo.available && (
            <div className="flex justify-between items-center">
              <span className="text-gray-700">Structural Match Score:</span>
              <span className={`font-medium ${getResultColor(structuralInfo.averageScore ?? 0)}`}>
                {structuralInfo.averageScore?.toFixed(1) ?? '0.0'}% AI Profile
              </span>
            </div>
          )}
          
          <div className="flex justify-between items-start">
            <span className="text-gray-700">Video Source:</span>
            <div className="text-right">
              <div className="text-gray-900 font-medium">{videoSource.type}</div>
              {videoSource.url ? (
                <div className="flex items-center gap-1 mt-1">
                  <span className="text-xs text-gray-600">{videoSource.displayUrl}</span>
                  <a 
                    href={videoSource.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-purple-600 hover:text-purple-700"
                  >
                    <ExternalLink size={12} />
                  </a>
                </div>
              ) : (
                <div className="text-xs text-gray-600 mt-1">{videoSource.displayUrl}</div>
              )}
            </div>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-gray-700">Analysis Date:</span>
            <span className="text-gray-900">
              {new Date(result.created_at).toLocaleDateString()}
            </span>
          </div>
          
          {analysis_results.video_duration && (
            <div className="flex justify-between items-center">
              <span className="text-gray-700">Video Duration:</span>
              <span className="text-gray-900">
                {formatTime(analysis_results.video_duration)}
              </span>
            </div>
          )}
          
          {analysis_results.ml_analysis?.analysis_fps && (
            <div className="flex justify-between items-center">
              <span className="text-gray-700">Analysis FPS:</span>
              <span className="text-gray-900">
                {analysis_results.ml_analysis.analysis_fps} FPS
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Skeleton Detection Features */}
      {analysis_results.skeleton_available && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Target className="text-blue-600" size={20} />
            Skeleton-Based Detection
          </h3>
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-sm font-medium text-blue-800">
                  Structural Matching
                </div>
                <div className="text-xs text-blue-600 mt-1">
                  {analysis_results.skeleton_enabled ? 'Enabled' : 'Available but disabled'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm font-medium text-blue-800">
                  Distance Analysis
                </div>
                <div className="text-xs text-blue-600 mt-1">
                  {structuralInfo.available ? 'Active' : 'Not computed'}
                </div>
              </div>
            </div>
            {structuralInfo.available && (
              <div className="mt-3 text-center">
                <span className="text-xs text-blue-500">
                  Avg structural similarity: {structuralInfo.averageScore?.toFixed(1) ?? '0.0'}% 
                  (Range: {structuralInfo.minScore?.toFixed(1) ?? '0.0'}% - {structuralInfo.maxScore?.toFixed(1) ?? '0.0'}%)
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Preprocessing Details */}
      {analysis_results.preprocessing_details && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Shield className="text-purple-600" size={20} />
            Preprocessing Results
          </h3>
          <div className="bg-purple-50 border border-purple-200 rounded-xl p-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="text-center">
                <div className={`text-sm font-medium ${analysis_results.preprocessing_details.metadata_flag ? 'text-red-600' : 'text-green-600'}`}>
                  Metadata Analysis
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis_results.preprocessing_details.metadata_flag ? 'AI indicators found' : 'No indicators'}
                </div>
              </div>
              <div className="text-center">
                <div className={`text-sm font-medium ${analysis_results.preprocessing_details.ocr_flag ? 'text-red-600' : 'text-green-600'}`}>
                  OCR Detection
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis_results.preprocessing_details.ocr_flag ? 'Watermarks detected' : 'No watermarks'}
                </div>
              </div>
              <div className="text-center">
                <div className={`text-sm font-medium ${analysis_results.preprocessing_details.logo_flag ? 'text-red-600' : 'text-green-600'}`}>
                  Logo Detection
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis_results.preprocessing_details.logo_flag ? 'AI tool logos found' : 'No logos detected'}
                </div>
              </div>
            </div>
            <div className="mt-3 text-center">
              <span className="text-xs text-gray-500">
                Preprocessing time: {analysis_results.preprocessing_details.processing_time.toFixed(2)}s
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ML Analysis Timestamps */}
      {highConfidenceTimestamps.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Suspicious Timestamps ({highConfidenceTimestamps.length} found)
          </h3>
          <div className="bg-gray-50 rounded-xl p-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {highConfidenceTimestamps.slice(0, 8).map((timestamp, index) => (
                <div key={index} className="flex justify-between items-center bg-white rounded-lg p-3">
                  <span className="font-mono text-sm text-gray-700">
                    {formatTime(timestamp.time)}
                  </span>
                  <div className="text-right">
                    <span className={`text-sm font-semibold ${getResultColor(timestamp.likelihood)}`}>
                      {timestamp.likelihood}%
                    </span>
                    <div className="text-xs text-gray-500">
                      {timestamp.confidence}% confidence
                    </div>
                    {timestamp.details?.structural_match_score && (
                      <div className="text-xs text-blue-500">
                        {timestamp.details.structural_match_score.toFixed(1)}% structural
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {highConfidenceTimestamps.length > 8 && (
              <p className="text-center text-gray-500 text-sm mt-3">
                +{highConfidenceTimestamps.length - 8} more suspicious timestamps
              </p>
            )}
          </div>
        </div>
      )}

      {/* Clean Analysis Message */}
      {highConfidenceTimestamps.length === 0 && overall_likelihood < 30 && !analysis_results.early_detection && (
        <div className="mt-6 bg-green-50 border border-green-200 rounded-xl p-4">
          <h3 className="text-lg font-semibold text-green-800 mb-2 flex items-center gap-2">
            <CheckCircle size={20} />
            Clean Analysis
          </h3>
          <p className="text-green-700 text-sm">
            No suspicious timestamps detected. The video appears to be authentic with natural patterns throughout.
            {analysis_results.skeleton_enabled 
              ? ' Both structural matching and ML analysis found no significant AI indicators.'
              : ' Both preprocessing and ML analysis found no significant AI indicators.'
            }
          </p>
        </div>
      )}

      {/* Early Detection Success */}
      {analysis_results.early_detection && (
        <div className="mt-6 bg-purple-50 border border-purple-200 rounded-xl p-4">
          <h3 className="text-lg font-semibold text-purple-800 mb-2 flex items-center gap-2">
            <Zap size={20} />
            Fast Detection Success
          </h3>
          <p className="text-purple-700 text-sm">
            AI generation detected quickly through preprocessing pipeline. This saved significant processing time
            by identifying clear indicators before requiring deep ML analysis.
          </p>
        </div>
      )}

      {/*Skeleton Detection Info */}
      {analysis_results.skeleton_enabled && !analysis_results.early_detection && (
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
          <h3 className="text-lg font-semibold text-blue-800 mb-2 flex items-center gap-2">
            <Target size={20} />
            Skeleton-Based Analysis
          </h3>
          <p className="text-blue-700 text-sm">
            Advanced structural matching was used to compare this video against learned patterns from AI and real content.
            This approach provides improved generalization to novel AI generation methods through distance-based classification.
          </p>
        </div>
      )}
    </div>
  )
}