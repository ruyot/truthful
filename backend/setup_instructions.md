# Backend Setup Instructions

## Prerequisites
Make sure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package manager)
- FFmpeg
- yt-dlp

## Installation Steps

### 1. Install System Dependencies

**On macOS:**
```bash
# Install FFmpeg
brew install ffmpeg

# Install yt-dlp
brew install yt-dlp
```

**On Ubuntu/Debian:**
```bash
# Install FFmpeg
sudo apt update
sudo apt install ffmpeg

# Install yt-dlp
sudo apt install yt-dlp
```

**On Windows:**
```bash
# Install FFmpeg (using chocolatey)
choco install ffmpeg

# Install yt-dlp
pip install yt-dlp
```

### 2. Setup Python Environment

Navigate to the backend directory and create a virtual environment:

```bash
cd backend
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Backend Server

```bash
python main.py
```

The backend will start on `http://localhost:8000`

## Testing the Backend

You can test if the backend is working by visiting:
- `http://localhost:8000` - API information
- `http://localhost:8000/health` - Health check
- `http://localhost:8000/docs` - Interactive API documentation

## Troubleshooting

### Common Issues:

1. **FFmpeg not found**: Make sure FFmpeg is installed and in your PATH
2. **yt-dlp not found**: Install yt-dlp using pip or your system package manager
3. **OpenCV issues**: Try reinstalling with `pip install opencv-python --upgrade`
4. **Permission errors**: Make sure you have write permissions in the temp directory

### Testing Video Analysis:

You can test the analysis endpoint using curl:

```bash
curl -X POST "http://localhost:8000/analyze-video" \
  -H "Content-Type: multipart/form-data" \
  -F "video_url=https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  -F "user_id=test-user"
```