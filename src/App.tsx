import React, { useState, useEffect } from 'react'
import { Eye, LogOut, User, Menu, X } from 'lucide-react'
import { supabase, type Database, type User as SupabaseUser } from './lib/supabase'
import { AuthModal } from './components/AuthModal'
import { EnhancedVideoUpload } from './enhanced/EnhancedVideoUpload'
import { EnhancedAnalysisResult } from './enhanced/EnhancedAnalysisResult'
import { AnalysisHistory, type HistoryItem } from './components/AnalysisHistory'
import Particles from './components/backgrounds/Particles'
import AnimatedEye from './components/animations/AnimatedEye'
import MultiColorText from './components/animations/MultiColorText'
import AnimatedGradientText from './components/animations/AnimatedGradient'
import AboutPage from './pages/AboutPage'

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
  const [currentView, setCurrentView] = useState<'upload' | 'history' | 'about'>('upload')
  const [supabaseConfigured, setSupabaseConfigured] = useState(false)

  useEffect(() => {
    // Check if Supabase is configured
    const hasSupabase = supabase !== null
    setSupabaseConfigured(hasSupabase)

    if (hasSupabase) {
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
      { status: 'analyzing', message: 'Analyzing with neural network...', progress: 50 },
      { status: 'analyzing', message: 'Processing structural patterns...', progress: 75 },
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
    const videoAnalysis: VideoAnalysis = {
      id: analysis.id,
      user_id: 'unknown',
      video_url: analysis.video_url,
      video_filename: analysis.video_filename,
      overall_likelihood: analysis.overall_likelihood,
      analysis_results: analysis.analysis_results,
      created_at: analysis.created_at,
      updated_at: analysis.created_at
    }
    setCurrentAnalysis(videoAnalysis)
    setCurrentView('upload')
  }

  // Show About page
  if (currentView === 'about') {
    return (
      <>
        <div className="fixed inset-0 -z-10 bg-white">
          <Particles
            particleColors={['#8B5CF6', '#A78BFA', '#C4B5FD', '#EC4899', '#3B82F6', '#10B981']}
            particleCount={100}
            particleSpread={10}
            speed={0.1}
            particleBaseSize={80}
            moveParticlesOnHover={true}
            alphaParticles={true}
          />
        </div>
        <AboutPage onBack={() => setCurrentView('upload')} />
      </>
    )
  }

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Particles Background */}
      <div className="fixed inset-0 -z-10 bg-white">
        <Particles
          particleColors={['#8B5CF6', '#A78BFA', '#C4B5FD', '#EC4899', '#3B82F6', '#10B981']}
          particleCount={100}
          particleSpread={10}
          speed={0.1}
          particleBaseSize={80}
          moveParticlesOnHover={true}
          alphaParticles={true}
        />
      </div>

      {/* Supabase Configuration Notice */}
      {!supabaseConfigured && (
        <div className="relative z-10 bg-yellow-50/90 backdrop-blur-sm border-b border-yellow-200 px-4 py-3">
          <div className="max-w-7xl mx-auto">
            <p className="text-yellow-800 text-sm">
              <strong>Demo Mode:</strong> Supabase not configured. Authentication and data persistence are disabled. 
              Set up your Supabase credentials in the .env file to enable full functionality.
            </p>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="relative z-20 sticky top-0">
        <div className="bg-white/80 backdrop-blur-md border-b border-purple-100 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <button
                  onClick={() => setCurrentView('about')}
                  className="flex items-center group"
                >
                  <div className="mr-3">
                    <AnimatedEye size={32} color="#8B5CF6" />
                  </div>
                  <h1 className="text-2xl font-bold">
                    <AnimatedGradientText gradient="from-purple-600 via-pink-600 to-blue-600">
                      Truthful
                    </AnimatedGradientText>
                  </h1>
                </button>
              </div>

              {/* Desktop Navigation */}
              <div className="hidden md:flex items-center space-x-6">
                {(user || !supabaseConfigured) && (
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={() => setCurrentView('upload')}
                      className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        currentView === 'upload' 
                          ? 'bg-purple-100 text-purple-700' 
                          : 'text-gray-600 hover:text-purple-700 hover:bg-purple-50'
                      }`}
                    >
                      Analyze
                    </button>
                    {supabaseConfigured && (
                      <button
                        onClick={() => setCurrentView('history')}
                        className={`px-4 py-2 rounded-lg font-medium transition-all ${
                          currentView === 'history' 
                            ? 'bg-purple-100 text-purple-700' 
                            : 'text-gray-600 hover:text-purple-700 hover:bg-purple-50'
                        }`}
                      >
                        History
                      </button>
                    )}
                  </div>
                )}
                
                {supabaseConfigured && (
                  user ? (
                    <div className="flex items-center space-x-3">
                      <div className="bg-purple-50 px-3 py-1 rounded-lg border border-purple-100">
                        <div className="flex items-center space-x-2">
                          <User size={16} className="text-purple-600" />
                          <span className="text-gray-700 text-sm">{user.email}</span>
                        </div>
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
                    <button
                      onClick={() => setAuthModal({ isOpen: true, mode: 'signin' })}
                      className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl"
                    >
                      Sign In
                    </button>
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
            <div className="md:hidden">
              <div className="bg-white/90 backdrop-blur-sm border-t border-purple-100 m-4 p-4 rounded-lg shadow-lg">
                <div className="space-y-3">
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
                      <button
                        onClick={() => {
                          setCurrentView('about')
                          setMobileMenuOpen(false)
                        }}
                        className={`block w-full text-left px-3 py-2 rounded-lg font-medium ${
                          currentView === 'about' as any
                            ? 'bg-purple-100 text-purple-700' 
                            : 'text-gray-600'
                        }`}
                      >
                        About
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!supabaseConfigured || user ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-8">
              {currentView === 'upload' ? (
                <>
                  <EnhancedVideoUpload 
                    onVideoSelect={handleVideoAnalysis} 
                    loading={analyzing}
                    progress={analysisProgress || undefined}
                  />
                  {currentAnalysis && <EnhancedAnalysisResult result={currentAnalysis as any} />}
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
              
              {/* Enhanced Info Panel */}
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 border border-purple-100 shadow-md transform hover:scale-105 transition-transform duration-300">
                <h3 className="font-semibold text-gray-900 mb-4 text-center">
                  <MultiColorText
                    text="Enhanced Detection v4.0"
                    colors={['#8B5CF6', '#EC4899', '#3B82F6']}
                  />
                </h3>
                <div className="space-y-3 text-sm text-gray-600">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">1</div>
                    <p>Upload video or paste URL (YouTube, Vimeo, etc.)</p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">2</div>
                    <p>Skeleton-based structural analysis with VidProM integration</p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">3</div>
                    <p>Get detailed results with distance-based matching</p>
                  </div>
                </div>
              </div>

              {/* Enhanced Detection Methods */}
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 border border-purple-100 shadow-md transform hover:scale-105 transition-transform duration-300">
                <h3 className="font-semibold text-gray-900 mb-4 text-center">
                  <MultiColorText
                    text="Advanced Detection Methods"
                    colors={['#3B82F6', '#8B5CF6', '#EC4899']}
                  />
                </h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <div>• <strong>Skeleton-Based Matching:</strong> Structural pattern analysis</div>
                  <div>• <strong>VidProM Integration:</strong> Novel AI content detection</div>
                  <div>• <strong>Distance Metrics:</strong> Mahalanobis, Euclidean, Cosine</div>
                  <div>• <strong>Multi-Frame Fusion:</strong> Temporal consistency analysis</div>
                  <div>• <strong>Neural Networks:</strong> Deep CNN with attention</div>
                  <div>• <strong>Fast Preprocessing:</strong> Metadata and watermark detection</div>
                </div>
              </div>

              {/* Production Features */}
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 border border-purple-100 shadow-md transform hover:scale-105 transition-transform duration-300">
                <h3 className="font-semibold text-gray-900 mb-4 text-center">
                  <MultiColorText
                    text="🚀 Production Ready"
                    colors={['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6']}
                  />
                </h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <div>✅ Skeleton-based structural matching</div>
                  <div>✅ VidProM dataset integration</div>
                  <div>✅ Enhanced generalization capability</div>
                  <div>✅ Real-time neural network visualization</div>
                  <div>✅ Interactive UI with fluid animations</div>
                  <div>✅ Production monitoring & deployment</div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-12 max-w-4xl mx-auto border border-purple-100 shadow-lg">
              <div className="mb-8">
                <AnimatedEye size={80} color="#8B5CF6" />
              </div>
              
              <MultiColorText
                text="Detect AI-Generated Videos"
                className="text-5xl font-bold mb-6"
                colors={['#8B5CF6', '#EC4899', '#3B82F6', '#EF4444', '#10B981']}
              />
              
              <p className="text-xl text-gray-600 mb-12">
                Your extra eyes in an ever-growing digital world
              </p>
              
              <button
                onClick={() => setAuthModal({ isOpen: true, mode: 'signup' })}
                className="bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 text-white px-12 py-6 rounded-2xl font-semibold text-xl hover:from-purple-700 hover:to-blue-700 transition-all duration-300 shadow-2xl hover:shadow-3xl transform hover:scale-105"
              >
                Start Detecting Now
              </button>
            </div>
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