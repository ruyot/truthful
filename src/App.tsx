import React, { useState, useEffect } from 'react'
import { Shield, LogOut, User, Menu, X } from 'lucide-react'
import { supabase, type Database, type User as SupabaseUser } from './lib/supabase'
import { AuthModal } from './components/AuthModal'
import { VideoUpload } from './components/VideoUpload'
import { AnalysisResult } from './components/AnalysisResult'
import { AnalysisHistory, type HistoryItem } from './components/AnalysisHistory'

type VideoAnalysis = Database['public']['Tables']['video_analyses']['Row']

interface AnalysisProgress {
  status: string
  progress: number
  message: string
}

function App() {
  const [user, setUser] = useState<SupabaseUser | null>(null)
  const [authModal, setAuthModal] = useState({ isOpen: false, mode: 'signin' as 'signin' | 'signup' })
  const [currentAnalysis, setCurrentAnalysis] = useState<VideoAnalysis | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress | null>(null)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [currentView, setCurrentView] = useState<'upload' | 'history'>('upload')
  const [supabaseConfigured, setSupabaseConfigured] = useState(false)

  useEffect(() => {
    // Check if Supabase is configured
    const hasSupabase = supabase !== null
    setSupabaseConfigured(hasSupabase)

    if (hasSupabase && supabase) {
      // Check initial auth state
      supabase.auth.getUser().then(({ data: { user } }) => {
        setUser(user)
      }).catch(console.error)

      // Listen for auth state changes
      const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
        setUser(session?.user ?? null)
      })

      return () => subscription.unsubscribe()
    }
  }, [])

  const handleSignOut = async () => {
    if (supabase) {
      await supabase.auth.signOut()
    }
    setUser(null)
    setCurrentAnalysis(null)
    setCurrentView('upload')
  }

  const simulateProgress = (duration: number) => {
    const steps = [
      { status: 'downloading', message: 'Downloading video...', progress: 0 },
      { status: 'extracting', message: 'Extracting frames...', progress: 25 },
      { status: 'analyzing', message: 'Analyzing with AI model...', progress: 50 },
      { status: 'analyzing', message: 'Processing face regions...', progress: 75 },
      { status: 'completing', message: 'Generating results...', progress: 90 },
    ]

    let currentStep = 0
    const stepDuration = duration / steps.length

    const updateProgress = () => {
      if (currentStep < steps.length) {
        setAnalysisProgress(steps[currentStep])
        currentStep++
        setTimeout(updateProgress, stepDuration)
      } else {
        setAnalysisProgress({ status: 'completing', message: 'Finalizing analysis...', progress: 100 })
      }
    }

    updateProgress()
  }

  const handleVideoAnalysis = async (file: File | null, url: string) => {
    if (!supabaseConfigured) {
      // Demo mode - call backend API directly
      setAnalyzing(true)
      setCurrentAnalysis(null)
      
      // Start progress simulation
      simulateProgress(8000) // 8 seconds

      try {
        const formData = new FormData()
        if (file) {
          formData.append('video', file)
        }
        if (url) {
          formData.append('video_url', url)
        }
        formData.append('user_id', 'demo-user')

        // Get backend URL from environment or use default
        const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
        const apiUrl = `${backendUrl}/analyze-video`

        console.log('Calling API:', apiUrl)
        console.log('Form data:', { 
          hasVideo: !!file, 
          videoUrl: url, 
          userId: 'demo-user' 
        })

        const response = await fetch(apiUrl, {
          method: 'POST',
          body: formData,
        })

        console.log('Response status:', response.status)

        if (!response.ok) {
          const errorText = await response.text()
          console.error('API Error Response:', errorText)
          
          let errorMessage = 'Analysis failed'
          try {
            const errorData = JSON.parse(errorText)
            errorMessage = errorData.detail || errorMessage
          } catch {
            errorMessage = errorText || errorMessage
          }
          
          throw new Error(errorMessage)
        }

        const result = await response.json()
        console.log('Analysis result:', result)
        
        // Format the result for display
        const mockResult: VideoAnalysis = {
          id: Date.now().toString(),
          user_id: 'demo-user',
          video_url: url || undefined,
          video_filename: file?.name || undefined,
          overall_likelihood: result.overall_likelihood,
          analysis_results: result.analysis_results,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }

        setCurrentAnalysis(mockResult)
      } catch (error: unknown) {
        console.error('Analysis failed:', error)
        const errorMessage = error instanceof Error ? error.message : String(error)
        alert(`Analysis failed: ${errorMessage}`)
      } finally {
        setAnalyzing(false)
        setAnalysisProgress(null)
      }
      return
    }

    if (!user) {
      setAuthModal({ isOpen: true, mode: 'signin' })
      return
    }

    setAnalyzing(true)
    setCurrentAnalysis(null)
    
    // Start progress simulation
    simulateProgress(10000) // 10 seconds for authenticated users

    try {
      const formData = new FormData()
      if (file) {
        formData.append('video', file)
      }
      if (url) {
        formData.append('video_url', url)
      }
      formData.append('user_id', user.id)

      // Get backend URL from environment or use default
      const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const apiUrl = `${backendUrl}/analyze-video`

      console.log('Calling API:', apiUrl)

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API Error Response:', errorText)
        
        let errorMessage = 'Analysis failed'
        try {
          const errorData = JSON.parse(errorText)
          errorMessage = errorData.detail || errorMessage
        } catch {
          errorMessage = errorText || errorMessage
        }
        
        throw new Error(errorMessage)
      }

      const result = await response.json()

      const analysisData = {
        user_id: user.id,
        video_url: url || undefined,
        video_filename: file?.name || undefined,
        overall_likelihood: result.overall_likelihood,
        analysis_results: result.analysis_results
      }

      // Save to Supabase if configured
      if (supabase) {
        const { data, error } = await supabase
          .from('video_analyses')
          .insert([analysisData])
          .select()

        if (error) throw error

        // Use the returned data with auto-generated ID
        setCurrentAnalysis(data[0])
      }
    } catch (error: unknown) {
      console.error('Analysis failed:', error)
      const errorMessage = error instanceof Error ? error.message : String(error)
      alert(`Analysis failed: ${errorMessage}`)
    } finally {
      setAnalyzing(false)
      setAnalysisProgress(null)
    }
  }

  const handleSelectAnalysis = (analysis: HistoryItem) => {
    // Convert HistoryItem to VideoAnalysis format
    const videoAnalysis: VideoAnalysis = {
      id: analysis.id,
      user_id: user?.id || 'unknown',
      video_url: analysis.video_url,
      video_filename: analysis.video_filename,
      overall_likelihood: analysis.overall_likelihood,
      analysis_results: analysis.analysis_results,
      created_at: analysis.created_at,
      updated_at: analysis.created_at // Use created_at as fallback
    }
    setCurrentAnalysis(videoAnalysis)
    setCurrentView('upload')
  }

  // Helper function to convert VideoAnalysis to AnalysisResult format
  const convertToAnalysisResult = (videoAnalysis: VideoAnalysis) => {
    const results = videoAnalysis.analysis_results as Record<string, unknown>
    
    // Type definitions for nested objects
    type PreprocessingDetails = {
      metadata_flag: boolean
      ocr_flag: boolean
      logo_flag: boolean
      final_decision: string
      confidence_score: number
      processing_time: number
      frames_analyzed: number
    } | undefined
    
    type MLAnalysis = {
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
    } | undefined
    
    return {
      overall_likelihood: videoAnalysis.overall_likelihood,
      analysis_results: {
        method: (results.method as string) || 'combined_analysis',
        preprocessing_details: results.preprocessing_details as PreprocessingDetails,
        ml_analysis: results.ml_analysis as MLAnalysis,
        early_detection: (results.early_detection as boolean) || false,
        total_frames: (results.total_frames as number) || 0,
        processing_time: (results.processing_time as number) || 0,
        video_duration: results.video_duration as number | undefined,
        detection_source: results.detection_source as string | undefined,
        skeleton_available: results.skeleton_available as boolean | undefined,
        skeleton_enabled: results.skeleton_enabled as boolean | undefined
      },
      video_url: videoAnalysis.video_url,
      video_filename: videoAnalysis.video_filename,
      created_at: videoAnalysis.created_at
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100">
      {/* Supabase Configuration Notice */}
      {!supabaseConfigured && (
        <div className="bg-yellow-50 border-b border-yellow-200 px-4 py-3">
          <div className="max-w-7xl mx-auto">
            <p className="text-yellow-800 text-sm">
              <strong>Demo Mode:</strong> Supabase not configured. Authentication and data persistence are disabled. 
              Set up your Supabase credentials in the .env file to enable full functionality.
            </p>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Shield className="text-purple-600 mr-3" size={32} />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Truthful
              </h1>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-6">
              {(user || !supabaseConfigured) && (
                <>
                  <button
                    onClick={() => setCurrentView('upload')}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      currentView === 'upload' 
                        ? 'bg-purple-100 text-purple-700' 
                        : 'text-gray-600 hover:text-purple-600'
                    }`}
                  >
                    Analyze
                  </button>
                  {supabaseConfigured && (
                    <button
                      onClick={() => setCurrentView('history')}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                        currentView === 'history' 
                          ? 'bg-purple-100 text-purple-700' 
                          : 'text-gray-600 hover:text-purple-600'
                      }`}
                    >
                      History
                    </button>
                  )}
                </>
              )}
              
              {supabaseConfigured && (
                user ? (
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2">
                      <User size={16} className="text-gray-600" />
                      <span className="text-gray-700 text-sm">{user.email}</span>
                    </div>
                    <button
                      onClick={handleSignOut}
                      className="flex items-center space-x-1 text-gray-600 hover:text-red-600 transition-colors"
                    >
                      <LogOut size={16} />
                      <span className="text-sm">Sign Out</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-x-3">
                    <button
                      onClick={() => setAuthModal({ isOpen: true, mode: 'signin' })}
                      className="text-gray-600 hover:text-purple-600 font-medium"
                    >
                      Sign In
                    </button>
                    <button
                      onClick={() => setAuthModal({ isOpen: true, mode: 'signup' })}
                      className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200"
                    >
                      Get Started
                    </button>
                  </div>
                )
              )}
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-gray-600 hover:text-gray-900"
              >
                {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-white border-t border-gray-200">
            <div className="px-4 py-3 space-y-3">
              {(user || !supabaseConfigured) && (
                <>
                  <button
                    onClick={() => {
                      setCurrentView('upload')
                      setMobileMenuOpen(false)
                    }}
                    className={`block w-full text-left px-3 py-2 rounded-lg font-medium ${
                      currentView === 'upload' 
                        ? 'bg-purple-100 text-purple-700' 
                        : 'text-gray-600'
                    }`}
                  >
                    Analyze
                  </button>
                  {supabaseConfigured && (
                    <button
                      onClick={() => {
                        setCurrentView('history')
                        setMobileMenuOpen(false)
                      }}
                      className={`block w-full text-left px-3 py-2 rounded-lg font-medium ${
                        currentView === 'history' 
                          ? 'bg-purple-100 text-purple-700' 
                          : 'text-gray-600'
                      }`}
                    >
                      History
                    </button>
                  )}
                  {supabaseConfigured && user && (
                    <div className="border-t pt-3">
                      <div className="flex items-center space-x-2 px-3 py-2">
                        <User size={16} className="text-gray-600" />
                        <span className="text-gray-700 text-sm">{user.email}</span>
                      </div>
                      <button
                        onClick={handleSignOut}
                        className="flex items-center space-x-2 px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg w-full"
                      >
                        <LogOut size={16} />
                        <span className="text-sm">Sign Out</span>
                      </button>
                    </div>
                  )}
                </>
              )}
              
              {supabaseConfigured && !user && (
                <div className="space-y-2">
                  <button
                    onClick={() => {
                      setAuthModal({ isOpen: true, mode: 'signin' })
                      setMobileMenuOpen(false)
                    }}
                    className="block w-full text-left px-3 py-2 text-gray-600 hover:text-purple-600 font-medium"
                  >
                    Sign In
                  </button>
                  <button
                    onClick={() => {
                      setAuthModal({ isOpen: true, mode: 'signup' })
                      setMobileMenuOpen(false)
                    }}
                    className="block w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white px-3 py-2 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200"
                  >
                    Get Started
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!supabaseConfigured || user ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-8">
              {currentView === 'upload' ? (
                <>
                  <VideoUpload 
                    onVideoSelect={handleVideoAnalysis} 
                    loading={analyzing}
                    progress={analysisProgress}
                  />
                  {currentAnalysis && <AnalysisResult result={convertToAnalysisResult(currentAnalysis)} />}
                </>
              ) : (
                supabaseConfigured && <AnalysisHistory userId={user?.id || ''} onSelectAnalysis={handleSelectAnalysis} />
              )}
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {currentView === 'upload' && supabaseConfigured && user && (
                <AnalysisHistory userId={user.id} onSelectAnalysis={handleSelectAnalysis} />
              )}
              
              {/* Info Panel */}
              <div className="bg-white rounded-2xl p-6 shadow-lg">
                <h3 className="font-semibold text-gray-900 mb-4">Enhanced Detection v2.0</h3>
                <div className="space-y-3 text-sm text-gray-600">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">1</div>
                    <p>Upload video or paste URL (YouTube, Vimeo, etc.)</p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">2</div>
                    <p>Enhanced AI analysis with face detection at 3 FPS</p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">3</div>
                    <p>Get detailed results with improved accuracy</p>
                  </div>
                </div>
              </div>

              {/* Enhanced Analysis Methods */}
              <div className="bg-white rounded-2xl p-6 shadow-lg">
                <h3 className="font-semibold text-gray-900 mb-4">Enhanced Detection Methods</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <div>• <strong>MediaPipe Face Analysis:</strong> Advanced face region detection</div>
                  <div>• <strong>Enhanced Texture Analysis:</strong> Multi-layer pattern detection</div>
                  <div>• <strong>Improved Edge Detection:</strong> Sophisticated boundary analysis</div>
                  <div>• <strong>Color Distribution:</strong> Advanced histogram analysis</div>
                  <div>• <strong>Frequency Analysis:</strong> Spectral signature detection</div>
                  <div>• <strong>Temporal Consistency:</strong> Frame-to-frame analysis</div>
                </div>
              </div>

              {/* Production Features */}
              <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-2xl p-6 border border-purple-100">
                <h3 className="font-semibold text-gray-900 mb-4">🚀 Production Ready</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <div>✅ Cloud deployment on Render & Netlify</div>
                  <div>✅ Enhanced AI detection algorithms</div>
                  <div>✅ Real-time progress tracking</div>
                  <div>✅ Face-focused analysis</div>
                  <div>✅ Increased sampling rate (3 FPS)</div>
                  <div>✅ Production monitoring</div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <Shield className="mx-auto mb-8 text-purple-600" size={64} />
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Detect AI-Generated Videos</h2>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Upload videos or paste URLs to analyze with our enhanced AI detection technology. 
              Get detailed insights and confidence scores with improved accuracy.
            </p>
            <button
              onClick={() => setAuthModal({ isOpen: true, mode: 'signup' })}
              className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              Start Detecting Now
            </button>
          </div>
        )}
      </main>

      {supabaseConfigured && (
        <AuthModal
          isOpen={authModal.isOpen}
          onClose={() => setAuthModal({ ...authModal, isOpen: false })}
          mode={authModal.mode}
          onToggleMode={() => setAuthModal({ ...authModal, mode: authModal.mode === 'signin' ? 'signup' : 'signin' })}
        />
      )}
    </div>
  )
}

export default App