import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Link, Video, AlertCircle, Loader2, Zap, Brain, Search, Eye } from 'lucide-react'
import MultiColorText from '../animations/MultiColorText'
import AnimatedGradientText from '../animations/AnimatedGradient'

interface EnhancedVideoUploadProps {
  onVideoSelect: (file: File | null, url: string) => void
  loading?: boolean
  progress?: {
    status: string
    progress: number
    message: string
  }
}

export const EnhancedVideoUpload: React.FC<EnhancedVideoUploadProps> = ({ 
  onVideoSelect, 
  loading = false, 
  progress 
}) => {
  const [urlInput, setUrlInput] = useState('')
  const [uploadMethod, setUploadMethod] = useState<'file' | 'url'>('file')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      onVideoSelect(file, '')
    }
  }, [onVideoSelect])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024,
    disabled: loading
  })

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (urlInput.trim()) {
      onVideoSelect(null, urlInput.trim())
    }
  }

  const getProgressColor = (status: string) => {
    switch (status) {
      case 'downloading': return 'from-blue-500 to-cyan-500'
      case 'extracting': return 'from-yellow-500 to-orange-500'
      case 'analyzing': return 'from-purple-500 to-pink-500'
      case 'completing': return 'from-green-500 to-emerald-500'
      default: return 'from-gray-500 to-gray-600'
    }
  }

  return (
    <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100 transform hover:scale-105 transition-transform duration-300">
      <div className="text-center mb-8">
        <div className="relative inline-block mb-6">
          <Eye className="mx-auto text-purple-600" size={64} />
        </div>
        
        <h2 className="text-3xl font-bold mb-4 text-center">
          <AnimatedGradientText
            gradient="from-purple-600 via-pink-600 to-blue-600"
          >
            Analyze Video
          </AnimatedGradientText>
        </h2>
        
        <p className="text-gray-600 mb-4 text-center">
          Upload a video file or paste a URL to detect AI generation with our advanced neural network
        </p>
        
        <div className="flex items-center justify-center gap-4 text-sm text-gray-500">
          <span className="flex items-center gap-1">
            <Zap size={16} className="text-purple-600" />
            AI Preprocessing
          </span>
          <span className="flex items-center gap-1">
            <Brain size={16} className="text-blue-600" />
            Neural Analysis
          </span>
          <span className="flex items-center gap-1">
            <Search size={16} className="text-green-600" />
            Face Detection
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      {loading && progress && (
        <div className="mb-8">
          <div className="bg-purple-50 rounded-xl p-6 border border-purple-100">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <Brain className="animate-spin text-purple-600" size={24} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">
                    {progress.message}
                  </h4>
                  <p className="text-sm text-gray-600">Neural network processing...</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-purple-600">
                  {Math.round(progress.progress)}%
                </div>
                <div className="text-xs text-gray-500">Complete</div>
              </div>
            </div>
            
            <div className="relative w-full h-4 bg-purple-100 rounded-full overflow-hidden">
              <div 
                className={`h-full bg-gradient-to-r ${getProgressColor(progress.status)} transition-all duration-500 relative`}
                style={{ width: `${progress.progress}%` }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
              </div>
            </div>
            
            <div className="mt-3 flex justify-between text-xs text-gray-500">
              <span>Status: {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}</span>
              <span>Enhanced AI Detection Active</span>
            </div>
          </div>
        </div>
      )}

      {/* Method Selection */}
      <div className="flex justify-center mb-8">
        <div className="bg-purple-50 rounded-lg p-1">
          <div className="flex rounded-lg">
            <button
              onClick={() => setUploadMethod('file')}
              disabled={loading}
              className={`
                px-6 py-3 rounded-lg text-sm font-medium transition-all duration-300
                ${uploadMethod === 'file' 
                  ? 'bg-white text-purple-700 shadow-md' 
                  : 'text-gray-600 hover:text-purple-700 hover:bg-white/50'
                }
                ${loading ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              <Upload className="inline mr-2" size={16} />
              Upload File
            </button>
            <button
              onClick={() => setUploadMethod('url')}
              disabled={loading}
              className={`
                px-6 py-3 rounded-lg text-sm font-medium transition-all duration-300
                ${uploadMethod === 'url' 
                  ? 'bg-white text-purple-700 shadow-md' 
                  : 'text-gray-600 hover:text-purple-700 hover:bg-white/50'
                }
                ${loading ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              <Link className="inline mr-2" size={16} />
              Video URL
            </button>
          </div>
        </div>
      </div>

      {uploadMethod === 'file' ? (
        <div>
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
              transition-all duration-300 relative overflow-hidden
              ${isDragActive 
                ? 'border-purple-400 bg-purple-50' 
                : loading 
                ? 'border-gray-200 bg-gray-50 cursor-not-allowed' 
                : 'border-purple-300 hover:border-purple-400 hover:bg-purple-50'
              }
            `}
          >
            <input {...getInputProps()} />
            
            <div className="relative z-10">
              {loading ? (
                <div className="space-y-4">
                  <Loader2 className="mx-auto text-purple-600 animate-spin" size={64} />
                  <div>
                    <p className="text-purple-600 font-medium text-lg">
                      Processing video...
                    </p>
                    <p className="text-gray-500 text-sm mt-2">Neural network analysis in progress</p>
                  </div>
                </div>
              ) : isDragActive ? (
                <div className="space-y-4">
                  <Upload className="mx-auto text-purple-600 animate-bounce" size={64} />
                  <p className="text-purple-600 font-medium text-lg">
                    Drop your video here
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="mx-auto text-gray-400 hover:text-purple-600 transition-colors" size={64} />
                  <div>
                    <p className="text-gray-900 font-medium text-lg mb-2">
                      Drop your video here, or click to browse
                    </p>
                    <p className="text-gray-500 text-sm mb-4">
                      Supports MP4, AVI, MOV, MKV, WebM (max 100MB)
                    </p>
                    <div className="flex items-center justify-center gap-6 text-xs text-gray-400">
                      <span className="flex items-center gap-1">
                        <Zap size={12} className="text-purple-600" />
                        Fast preprocessing
                      </span>
                      <span className="flex items-center gap-1">
                        <Brain size={12} className="text-blue-600" />
                        ML classification
                      </span>
                      <span className="flex items-center gap-1">
                        <Search size={12} className="text-green-600" />
                        Face analysis
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {fileRejections.length > 0 && (
            <div className="mt-4">
              <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                <div className="flex items-center text-red-800 mb-2">
                  <AlertCircle size={16} className="mr-2" />
                  <span className="text-sm font-medium">Upload Error</span>
                </div>
                {fileRejections.map(({ file, errors }, index) => (
                  <div key={index} className="text-sm text-red-700">
                    <p className="font-medium">{file.name}</p>
                    {errors.map((error, errorIndex) => (
                      <p key={errorIndex} className="text-xs">• {error.message}</p>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <form onSubmit={handleUrlSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3 text-center">
              <MultiColorText
                text="Video URL"
                colors={['#8B5CF6', '#EC4899', '#3B82F6']}
              />
            </label>
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              className="w-full px-6 py-4 border border-purple-200 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all bg-white shadow-sm"
              placeholder="https://youtube.com/watch?v=..."
              disabled={loading}
            />
            <div className="flex items-center justify-between mt-3">
              <p className="text-xs text-gray-500">
                Supports YouTube, Vimeo, Dailymotion, and direct video links
              </p>
              <div className="flex items-center gap-2 text-xs">
                <Eye size={12} className="text-purple-600" />
                <span className="text-gray-500">Enhanced detection</span>
              </div>
            </div>
          </div>
          
          <button
            type="submit"
            disabled={loading || !urlInput.trim()}
            className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 text-white py-4 rounded-xl font-medium transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-lg hover:shadow-2xl hover:scale-105 relative overflow-hidden"
          >
            <div className="relative z-10 flex items-center">
              {loading ? (
                <>
                  <Loader2 className="animate-spin mr-3" size={20} />
                  Analyzing with AI Pipeline...
                </>
              ) : (
                <>
                  <Brain className="mr-3" size={20} />
                  Analyze Video
                </>
              )}
            </div>
          </button>
        </form>
      )}

      {/* Enhanced Features Info */}
      <div className="mt-8">
        <div className="bg-purple-50 rounded-xl p-6 border border-purple-100">
          <h4 className="text-lg font-semibold mb-4 text-center text-gray-900">
            <MultiColorText
              text="🚀 Enhanced Detection v4.0"
              colors={['#8B5CF6', '#EC4899', '#3B82F6']}
            />
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Zap size={16} className="text-purple-600" />
              <span className="text-gray-700">Skeleton-based detection</span>
            </div>
            <div className="flex items-center gap-2">
              <Search size={16} className="text-blue-600" />
              <span className="text-gray-700">Structural matching</span>
            </div>
            <div className="flex items-center gap-2">
              <Brain size={16} className="text-green-600" />
              <span className="text-gray-700">VidProM integration</span>
            </div>
            <div className="flex items-center gap-2">
              <Video size={16} className="text-orange-600" />
              <span className="text-gray-700">Multi-frame analysis</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}