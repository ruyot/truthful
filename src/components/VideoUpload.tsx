import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Link, Video, AlertCircle, Loader2, Zap, Brain, Search } from 'lucide-react'

interface VideoUploadProps {
  onVideoSelect: (file: File | null, url: string) => void
  loading?: boolean
  progress?: {
    status: string
    progress: number
    message: string
  }
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ 
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
    maxSize: 100 * 1024 * 1024, // 100MB
    disabled: loading
  })

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (urlInput.trim()) {
      onVideoSelect(null, urlInput.trim())
    }
  }

  const isValidVideoUrl = (url: string) => {
    const videoUrlPattern = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be|vimeo\.com|dailymotion\.com)/i
    return videoUrlPattern.test(url) || url.startsWith('http://') || url.startsWith('https://')
  }

  const getProgressColor = (status: string) => {
    switch (status) {
      case 'downloading': return 'bg-blue-500'
      case 'extracting': return 'bg-yellow-500'
      case 'analyzing': return 'bg-purple-500'
      case 'completing': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  const getProgressIcon = (status: string) => {
    switch (status) {
      case 'downloading': return <Upload className="animate-bounce" size={16} />
      case 'extracting': return <Search className="animate-pulse" size={16} />
      case 'analyzing': return <Brain className="animate-spin" size={16} />
      case 'completing': return <Zap className="animate-pulse" size={16} />
      default: return <Loader2 className="animate-spin" size={16} />
    }
  }

  return (
    <div className="bg-white rounded-2xl p-8 shadow-lg">
      <div className="text-center mb-6">
        <Video className="mx-auto mb-4 text-purple-600" size={48} />
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Analyze Video</h2>
        <p className="text-gray-600">Upload a video file or paste a video URL to detect AI generation</p>
        <div className="mt-3 text-sm text-gray-500">
          <span className="inline-flex items-center gap-1">
            <Zap size={14} className="text-purple-500" />
            Enhanced with AI preprocessing pipeline
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      {loading && progress && (
        <div className="mb-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl p-6 border border-purple-100">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              {getProgressIcon(progress.status)}
              <span className="font-medium text-gray-900">{progress.message}</span>
            </div>
            <span className="text-sm text-gray-600 font-mono">{Math.round(progress.progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div 
              className={`h-3 rounded-full transition-all duration-500 ${getProgressColor(progress.status)}`}
              style={{ width: `${progress.progress}%` }}
            >
              <div className="h-full bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
            </div>
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
            <span>Status: {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}</span>
            <span>Processing with enhanced AI detection</span>
          </div>
        </div>
      )}

      <div className="flex justify-center mb-6">
        <div className="bg-gray-100 rounded-lg p-1 flex">
          <button
            onClick={() => setUploadMethod('file')}
            disabled={loading}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              uploadMethod === 'file' 
                ? 'bg-white text-purple-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <Upload className="inline mr-2" size={16} />
            Upload File
          </button>
          <button
            onClick={() => setUploadMethod('url')}
            disabled={loading}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              uploadMethod === 'url' 
                ? 'bg-white text-purple-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <Link className="inline mr-2" size={16} />
            Video URL
          </button>
        </div>
      </div>

      {uploadMethod === 'file' ? (
        <div>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
              isDragActive 
                ? 'border-purple-400 bg-purple-50' 
                : loading 
                ? 'border-gray-200 bg-gray-50 cursor-not-allowed' 
                : 'border-gray-300 hover:border-purple-400 hover:bg-purple-50'
            }`}
          >
            <input {...getInputProps()} />
            {loading ? (
              <Loader2 className="mx-auto mb-4 text-gray-400 animate-spin" size={48} />
            ) : (
              <Upload className={`mx-auto mb-4 ${isDragActive ? 'text-purple-600' : 'text-gray-400'}`} size={48} />
            )}
            {loading ? (
              <div>
                <p className="text-gray-500 font-medium">Processing video...</p>
                <p className="text-gray-400 text-sm mt-1">Using enhanced AI detection pipeline</p>
              </div>
            ) : isDragActive ? (
              <p className="text-purple-600 font-medium">Drop your video here</p>
            ) : (
              <div>
                <p className="text-gray-900 font-medium mb-1">Drop your video here, or click to browse</p>
                <p className="text-gray-500 text-sm mb-2">Supports MP4, AVI, MOV, MKV, WebM (max 100MB)</p>
                <div className="flex items-center justify-center gap-4 text-xs text-gray-400 mt-3">
                  <span className="flex items-center gap-1">
                    <Zap size={12} />
                    Fast preprocessing
                  </span>
                  <span className="flex items-center gap-1">
                    <Brain size={12} />
                    ML classification
                  </span>
                  <span className="flex items-center gap-1">
                    <Search size={12} />
                    Face analysis
                  </span>
                </div>
              </div>
            )}
          </div>

          {fileRejections.length > 0 && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3">
              <div className="flex items-center text-red-800">
                <AlertCircle size={16} className="mr-2" />
                <span className="text-sm font-medium">Upload Error</span>
              </div>
              {fileRejections.map(({ file, errors }, index) => (
                <div key={index} className="mt-2 text-sm text-red-700">
                  <p className="font-medium">{file.name}</p>
                  {errors.map((error, errorIndex) => (
                    <p key={errorIndex} className="text-xs">• {error.message}</p>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <form onSubmit={handleUrlSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Video URL</label>
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              placeholder="https://youtube.com/watch?v=..."
              disabled={loading}
            />
            <div className="flex items-center justify-between mt-2">
              <p className="text-xs text-gray-500">
                Supports YouTube, Vimeo, Dailymotion, and direct video links
              </p>
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <Zap size={10} />
                <span>Enhanced detection</span>
              </div>
            </div>
          </div>
          
          {urlInput && !isValidVideoUrl(urlInput) && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <div className="flex items-center text-yellow-800">
                <AlertCircle size={16} className="mr-2" />
                <span className="text-sm">Please enter a valid video URL</span>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !urlInput.trim() || !isValidVideoUrl(urlInput)}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-lg hover:shadow-xl"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin mr-2" size={20} />
                Analyzing with AI Pipeline...
              </>
            ) : (
              <>
                <Brain className="mr-2" size={20} />
                Analyze Video
              </>
            )}
          </button>
        </form>
      )}

      {/* Enhanced Features Info */}
      <div className="mt-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4 border border-purple-100">
        <h4 className="text-sm font-semibold text-gray-900 mb-2">🚀 Enhanced Detection v3.0</h4>
        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
          <div className="flex items-center gap-1">
            <Zap size={10} className="text-purple-500" />
            <span>Fast preprocessing</span>
          </div>
          <div className="flex items-center gap-1">
            <Search size={10} className="text-blue-500" />
            <span>Metadata analysis</span>
          </div>
          <div className="flex items-center gap-1">
            <Brain size={10} className="text-green-500" />
            <span>ML classification</span>
          </div>
          <div className="flex items-center gap-1">
            <Video size={10} className="text-orange-500" />
            <span>Face detection</span>
          </div>
        </div>
      </div>
    </div>
  )
}