import React, { useState } from 'react'
import { CheckCircle, AlertTriangle, XCircle, Clock, Eye, BarChart3, ExternalLink, Zap, Brain, Search, Shield, Target } from 'lucide-react'
import MultiColorText from '../components/animations/MultiColorText'
import { NeuralNetworkDiagram } from '../neural/NeuralNetworkDiagram'
import AnimatedGradientText from '../components/animations/AnimatedGradient'

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

interface EnhancedAnalysisResultProps {
  result: AnalysisResult
}

export const EnhancedAnalysisResult: React.FC<EnhancedAnalysisResultProps> = ({ result }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'neural' | 'details'>('overview')
  const { overall_likelihood, analysis_results } = result
  
  const getResultColor = (likelihood: number) => {
    if (likelihood < 30) return 'text-green-600'
    if (likelihood < 70) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getResultBg = (likelihood: number) => {
    if (likelihood < 30) return 'bg-green-50'
    if (likelihood < 70) return 'bg-yellow-50'
    return 'bg-red-50'
  }

  const getResultIcon = (likelihood: number) => {
    if (likelihood < 30) return <CheckCircle size={32} className="text-green-600" />
    if (likelihood < 70) return <AlertTriangle size={32} className="text-yellow-600" />
    return <XCircle size={32} className="text-red-600" />
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

  const getDetectionMethodInfo = () => {
    const method = analysis_results.method
    const earlyDetection = analysis_results.early_detection
    const preprocessing = analysis_results.preprocessing_details
    const skeletonEnabled = analysis_results.skeleton_enabled

    if (earlyDetection && preprocessing) {
      return {
        icon: <Zap className="text-purple-600" size={24} />,
        title: 'Fast Detection',
        description: `Detected via ${preprocessing.final_decision.replace('AI-generated (', '').replace(')', '')}`,
        color: 'text-purple-600',
        gradient: 'from-purple-600 via-pink-600 to-blue-600'
      }
    } else if (skeletonEnabled) {
      return {
        icon: <Target className="text-blue-600" size={24} />,
        title: 'Skeleton-Based Detection',
        description: 'Structural matching with distance analysis',
        color: 'text-blue-600',
        gradient: 'from-blue-600 via-purple-600 to-indigo-600'
      }
    } else {
      return {
        icon: <Brain className="text-blue-600" size={24} />,
        title: 'ML Classification',
        description: 'Deep learning analysis with preprocessing',
        color: 'text-blue-600',
        gradient: 'from-blue-600 via-purple-600 to-indigo-600'
      }
    }
  }

  const detectionMethod = getDetectionMethodInfo()
  const highConfidenceTimestamps = analysis_results.ml_analysis?.timestamps?.filter(t => t.likelihood > 70) || []

  const tabContent = {
    overview: (
      <div className="space-y-6">
        {/* Main Result Card */}
        <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100 transform hover:scale-105 transition-transform duration-300">
          <div className="text-center mb-8">
            <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-6 ${getResultBg(overall_likelihood)} border border-${overall_likelihood < 30 ? 'green' : overall_likelihood < 70 ? 'yellow' : 'red'}-200`}>
              {getResultIcon(overall_likelihood)}
            </div>
            
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Analysis Complete
            </h2>
            
            <p className={`text-2xl font-semibold mb-4 ${getResultColor(overall_likelihood)}`}>
              <AnimatedGradientText gradient={overall_likelihood < 30 ? 'from-green-500 to-emerald-500' : overall_likelihood < 70 ? 'from-yellow-500 to-orange-500' : 'from-red-500 to-pink-500'}>
                {getResultText(overall_likelihood)}
              </AnimatedGradientText>
            </p>
            
            <div className="flex items-center justify-center gap-3 mb-6">
              {detectionMethod.icon}
              <span className={`text-lg font-medium ${detectionMethod.color}`}>
                {detectionMethod.title}
              </span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-purple-50 rounded-xl p-6 text-center border border-purple-100 transform hover:scale-105 transition-transform duration-300">
              <BarChart3 className="mx-auto mb-3 text-purple-600" size={32} />
              <div className="text-3xl font-bold text-purple-600">
                {overall_likelihood}%
              </div>
              <div className="text-sm text-gray-600 mt-1">AI Likelihood</div>
            </div>

            <div className="bg-blue-50 rounded-xl p-6 text-center border border-blue-100 transform hover:scale-105 transition-transform duration-300">
              <Eye className="mx-auto mb-3 text-blue-600" size={32} />
              <div className="text-3xl font-bold text-blue-600">
                {analysis_results.total_frames}
              </div>
              <div className="text-sm text-gray-600 mt-1">Frames Analyzed</div>
            </div>

            <div className="bg-green-50 rounded-xl p-6 text-center border border-green-100 transform hover:scale-105 transition-transform duration-300">
              <Clock className="mx-auto mb-3 text-green-600" size={32} />
              <div className="text-3xl font-bold text-green-600">
                {analysis_results.processing_time}s
              </div>
              <div className="text-sm text-gray-600 mt-1">Processing Time</div>
            </div>
          </div>
        </div>

        {/* Detection Summary */}
        <div className="bg-white rounded-2xl p-6 shadow-md border border-purple-100 transform hover:scale-105 transition-transform duration-300">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 text-center">
            <MultiColorText
              text="Detection Summary"
              colors={['#8B5CF6', '#EC4899', '#3B82F6']}
            />
          </h3>
          <div className="space-y-4">
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
            
            <div className="flex justify-between items-center">
              <span className="text-gray-700">Analysis Date:</span>
              <span className="text-gray-900">
                {new Date(result.created_at).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    ),
    neural: (
      <div className="space-y-6">
        <div className="bg-white rounded-2xl p-6 shadow-md border border-purple-100 transform hover:scale-105 transition-transform duration-300">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 text-center">
            <MultiColorText
              text="Neural Network Analysis"
              colors={['#8B5CF6', '#EC4899', '#3B82F6']}
            />
          </h3>
          <NeuralNetworkDiagram />
        </div>
      </div>
    ),
    details: (
      <div className="space-y-6">
        {/* Preprocessing Details */}
        {analysis_results.preprocessing_details && (
          <div className="bg-white rounded-2xl p-6 shadow-md border border-purple-100 transform hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2 justify-center">
              <Shield className="text-purple-600" size={24} />
              <MultiColorText
                text="Preprocessing Results"
                colors={['#8B5CF6', '#EC4899', '#3B82F6']}
              />
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-purple-50 rounded-xl p-4 text-center border border-purple-100">
                <div className={`text-sm font-medium ${analysis_results.preprocessing_details.metadata_flag ? 'text-red-600' : 'text-green-600'}`}>
                  Metadata Analysis
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis_results.preprocessing_details.metadata_flag ? 'AI indicators found' : 'No indicators'}
                </div>
              </div>
              
              <div className="bg-purple-50 rounded-xl p-4 text-center border border-purple-100">
                <div className={`text-sm font-medium ${analysis_results.preprocessing_details.ocr_flag ? 'text-red-600' : 'text-green-600'}`}>
                  OCR Detection
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis_results.preprocessing_details.ocr_flag ? 'Watermarks detected' : 'No watermarks'}
                </div>
              </div>
              
              <div className="bg-purple-50 rounded-xl p-4 text-center border border-purple-100">
                <div className={`text-sm font-medium ${analysis_results.preprocessing_details.logo_flag ? 'text-red-600' : 'text-green-600'}`}>
                  Logo Detection
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis_results.preprocessing_details.logo_flag ? 'AI tool logos found' : 'No logos detected'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Suspicious Timestamps */}
        {highConfidenceTimestamps.length > 0 && (
          <div className="bg-white rounded-2xl p-6 shadow-md border border-purple-100 transform hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-semibold text-gray-900 mb-4 text-center">
              <MultiColorText
                text={`Suspicious Timestamps (${highConfidenceTimestamps.length} found)`}
                colors={['#EF4444', '#F59E0B', '#F97316']}
              />
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {highConfidenceTimestamps.slice(0, 6).map((timestamp, index) => (
                <div key={index} className="bg-purple-50 rounded-xl p-4 border border-purple-100 transform hover:scale-105 transition-transform duration-300">
                  <div className="flex justify-between items-center">
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
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-white rounded-xl p-2 shadow-md border border-purple-100">
        <div className="flex rounded-lg">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'neural', label: 'Neural Network', icon: Brain },
            { id: 'details', label: 'Details', icon: Search }
          ].map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`
                  flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium
                  transition-all duration-300
                  ${activeTab === tab.id 
                    ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg' 
                    : 'text-gray-600 hover:text-purple-700 hover:bg-purple-50'
                  }
                `}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* Tab Content */}
      {tabContent[activeTab]}
    </div>
  )
}