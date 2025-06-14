# Truthful - AI Video Detection Tool v2.0

A production-ready AI video detection tool with enhanced analysis capabilities, featuring a React web app and Chrome extension that analyzes videos to detect if they're AI-generated.

## 🚀 New in v2.0

### Enhanced AI Detection
- **Increased Frame Sampling**: 3 FPS (up from 1 FPS) for better accuracy
- **MediaPipe Face Detection**: Advanced face region analysis
- **Enhanced Algorithms**: Multi-layer texture and pattern analysis
- **Improved Accuracy**: Better overall likelihood calculations

### Production Features
- **Cloud Deployment**: Ready for Render (backend) + Netlify (frontend)
- **Progress Tracking**: Real-time analysis progress with status updates
- **Production Monitoring**: Sentry integration for error tracking
- **Enhanced CORS**: Proper production domain support

### User Experience
- **Progress Bar**: Visual feedback during analysis
- **Status Messages**: "Extracting frames...", "Analyzing with AI model..."
- **Better Error Handling**: Comprehensive error messages and recovery
- **Mobile Responsive**: Optimized for all device sizes

## Features

### Web Application
- 🔍 **Enhanced Video Analysis**: Upload videos or paste URLs with 3x better sampling
- 📊 **Detailed Results**: Frame-by-frame analysis with confidence scores
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🔐 **User Authentication**: Email/password and Google OAuth via Supabase
- 📈 **Analysis History**: View and manage past video analyses
- 🎨 **Beautiful UI**: Modern gradient design with smooth animations
- ⚡ **Progress Tracking**: Real-time analysis progress with status updates

### Chrome Extension
- 🎬 **YouTube Integration**: Analyze videos directly on YouTube
- 📤 **File Upload**: Upload videos from popup
- 🔗 **URL Analysis**: Paste video URLs for analysis
- ⚡ **Quick Results**: Get AI detection results instantly

### Backend API
- 🤖 **Enhanced AI Detection**: Multi-method analysis with face detection
- 🎥 **Advanced Video Processing**: 3 FPS frame extraction using FFmpeg
- 🌐 **URL Support**: Download videos from YouTube and direct links
- 📊 **Detailed Analytics**: Timestamps, confidence scores, and statistics
- 🔍 **Face Analysis**: MediaPipe-powered face region detection
- 📈 **Production Ready**: Deployed on Render with monitoring

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Vite
- **Backend**: FastAPI (Python), OpenCV, MediaPipe, FFmpeg
- **Database**: Supabase (PostgreSQL) - Optional
- **Authentication**: Supabase Auth - Optional
- **Chrome Extension**: Manifest V3
- **Video Processing**: FFmpeg, yt-dlp, MediaPipe
- **Deployment**: Netlify (frontend), Render (backend)
- **Monitoring**: Sentry (optional)

## Quick Start

### Option 1: Use Deployed Version (Recommended)

1. **Frontend**: Visit the deployed Netlify URL
2. **Backend**: Automatically connects to deployed Render backend
3. **Start Analyzing**: Upload videos or paste YouTube URLs immediately

### Option 2: Local Development

#### Prerequisites
- Node.js 18+
- Python 3.9+
- FFmpeg
- yt-dlp

#### 1. Frontend Setup
```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Add your backend URL to .env
echo "VITE_BACKEND_URL=http://localhost:8000" >> .env

# Start development server
npm run dev
```

#### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python main.py
```

The backend runs on `http://localhost:8000`
The frontend runs on `http://localhost:5173`

## Production Deployment

### Backend (Render)

1. **Create Render Account**: Sign up at [render.com](https://render.com)
2. **Create Web Service**: Connect your GitHub repository
3. **Configure Settings**:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port 10000`
   - Python Version: 3.9.18

4. **Environment Variables**:
   ```
   PORT=10000
   PYTHON_VERSION=3.9.18
   SENTRY_DSN=your-sentry-dsn (optional)
   ```

### Frontend (Netlify)

1. **Create Netlify Account**: Sign up at [netlify.com](https://netlify.com)
2. **Deploy from Git**: Connect your GitHub repository
3. **Build Settings**:
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Node version: 18

4. **Environment Variables**:
   ```
   VITE_BACKEND_URL=https://your-render-app.onrender.com
   VITE_SUPABASE_URL=https://your-project-id.supabase.co (optional)
   VITE_SUPABASE_ANON_KEY=your-anon-key (optional)
   ```

## API Documentation

### Enhanced Analyze Video Endpoint

**POST** `/analyze-video`

Enhanced video analysis with face detection and improved algorithms.

**Parameters:**
- `video` (file, optional): Video file upload (max 100MB)
- `video_url` (string, optional): Video URL (YouTube, Vimeo, etc.)
- `user_id` (string, required): User identifier

**Response:**
```json
{
  "overall_likelihood": 75.5,
  "analysis_results": {
    "timestamps": [
      {
        "time": 0.0,
        "likelihood": 80.2,
        "confidence": 85.7,
        "details": {
          "texture_score": 45.2,
          "edge_density": 0.034,
          "face_regions_found": 1,
          "color_uniformity": 1250.5
        }
      }
    ],
    "total_frames": 180,
    "processing_time": 32.1,
    "video_duration": 60.0,
    "analysis_fps": 3.0
  },
  "processing_time": 32.1,
  "total_frames": 180
}
```

## Enhanced Detection Methods

### 1. MediaPipe Face Analysis
- Detects and crops face regions
- Analyzes facial texture patterns
- Identifies unnatural smoothness
- Checks edge consistency in faces

### 2. Multi-Layer Texture Analysis
- Laplacian variance for blur detection
- Sobel edge magnitude analysis
- Local Binary Pattern-like analysis
- Enhanced texture scoring

### 3. Advanced Color Analysis
- Multi-channel histogram analysis
- Color distribution uniformity
- Unnatural color pattern detection

### 4. Frequency Domain Analysis
- FFT-based spectral analysis
- High-frequency component analysis
- Compression artifact detection

### 5. Temporal Consistency
- Frame-to-frame analysis
- Motion pattern detection
- Temporal artifact identification

## Chrome Extension Usage

### Installation
1. Open Chrome → Extensions → Developer mode
2. Load unpacked → Select `extension` folder
3. Extension appears in toolbar

### YouTube Integration
1. Navigate to any YouTube video
2. Click "Truthful" button next to like/dislike
3. View results directly on the page

### Popup Interface
1. Click extension icon
2. Upload files or paste URLs
3. View detailed analysis results

## Database Schema (Optional)

### video_analyses table
- `id`: UUID primary key
- `user_id`: UUID reference to auth.users
- `video_url`: Optional video URL
- `video_filename`: Optional uploaded filename
- `overall_likelihood`: Float (0-100) AI detection percentage
- `analysis_results`: JSONB detailed frame analysis
- `created_at`: Timestamp
- `updated_at`: Timestamp

## Monitoring & Analytics

### Sentry Integration
- Real-time error tracking
- Performance monitoring
- User session replay
- Custom alerts

### Render Monitoring
- Application logs
- Performance metrics
- Uptime monitoring
- Resource usage

## Performance Optimizations

- **Frame Sampling**: 3 FPS for optimal speed/accuracy balance
- **Face Detection**: Targeted analysis of face regions
- **Memory Management**: Automatic cleanup of temporary files
- **Caching**: Optimized for repeated analyses
- **Compression**: Efficient video processing pipeline

## Security Features

- **Input Validation**: Comprehensive file and URL validation
- **CORS Protection**: Properly configured for production domains
- **Rate Limiting**: Built-in protection against abuse
- **Error Handling**: Secure error messages without data leakage
- **File Cleanup**: Automatic removal of temporary files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the deployment guide in `backend/deployment-guide.md`
- Review the API documentation at `/docs` (FastAPI auto-generated)

## Changelog

### v2.0.0
- Enhanced AI detection with MediaPipe face analysis
- Increased frame sampling to 3 FPS
- Added real-time progress tracking
- Production deployment on Render + Netlify
- Sentry monitoring integration
- Improved accuracy algorithms
- Better error handling and user experience

### v1.0.0
- Initial release with basic AI detection
- Supabase integration
- Chrome extension
- Basic video analysis capabilities