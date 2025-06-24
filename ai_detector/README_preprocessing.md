# AI Video Preprocessing Pipeline

A comprehensive Python pipeline for detecting AI-generated videos through metadata analysis, OCR watermark detection, logo recognition, and invisible watermark detection.

## **Overview**

This preprocessing pipeline serves as a **fast pre-screening step** before expensive ML classification. It can quickly identify AI-generated videos through various indicators:

1. **Metadata Analysis** - C2PA manifests, creation tools, encoding tags
2. **OCR Watermark Detection** - Text like "AI Generated", "© OpenAI", etc.
3. **Logo Detection** - Known AI tool logos and overlays
4. **Invisible Watermark Detection** - SynthID and similar technologies (placeholder)

## **Quick Start**

### **Installation**
```bash
cd ai_detector
pip install -r preprocessing_requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr ffmpeg

# macOS:
brew install tesseract ffmpeg

# Windows:
# Download and install Tesseract and FFmpeg manually
```

### **Basic Usage**
```python
from ai_detector.video_preprocessing import AIVideoPreprocessor

# Initialize preprocessor
preprocessor = AIVideoPreprocessor()

# Analyze video
result = preprocessor.analyze_video("path/to/video.mp4")

# Check result
if result["final_decision"] != "Unknown":
    print(f"AI detected: {result['final_decision']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
else:
    print("No clear AI indicators found, proceed to ML analysis")
```

### **Integration with Existing Pipeline**
```python
from ai_detector.video_preprocessing import preprocess_video_for_ai_detection

# In your existing video analysis function
def analyze_video_enhanced(video_path: str, user_id: str):
    # Stage 1: Fast preprocessing
    preprocessing_result = preprocess_video_for_ai_detection(video_path)
    
    if preprocessing_result['final_decision'] != 'Unknown':
        # Early detection - skip expensive ML analysis
        return {
            'method': 'preprocessing',
            'overall_likelihood': preprocessing_result['confidence_score'] * 100,
            'processing_time': preprocessing_result['processing_time'],
            'early_detection': True
        }
    
    # Stage 2: Continue with ML analysis
    return run_ml_analysis(video_path, user_id)
```

## **Detection Methods**

### **1. Metadata Analysis**
```python
# Detects AI indicators in:
metadata_indicators = {
    'creation_tools': [
        'openai', 'sora', 'dall-e', 'midjourney', 'stable diffusion',
        'adobe firefly', 'runway', 'pika labs', 'invideo', 'synthesia'
    ],
    'software_tags': [
        'ai generated', 'artificial intelligence', 'synthetic media'
    ],
    'encoding_hints': [
        'ai_generated', 'synthetic', 'deepfake', 'ml_enhanced'
    ]
}
```

**Sources Checked:**
- FFprobe metadata extraction
- EXIF data (when available)
- C2PA manifests (placeholder for future integration)

### **2. OCR Watermark Detection**
```python
# Detects text patterns like:
watermark_patterns = [
    'ai generated', 'ai-generated', '© openai', 'sora',
    'midjourney', 'stable diffusion', 'adobe firefly',
    'created with ai', 'synthetic media', 'not real'
]
```

**OCR Engines:**
- **EasyOCR** (primary) - Better accuracy, GPU support
- **Tesseract** (fallback) - Widely available, CPU-based

### **3. Logo Detection**
```python
# Template matching for known AI tool logos
# Uses OpenCV template matching and image hashing
# Configurable confidence thresholds
```

**Detection Methods:**
- OpenCV template matching
- Perceptual image hashing
- Multi-scale and rotation testing

### **4. Invisible Watermark Detection**
```python
# Placeholder for SynthID and similar technologies
# Frequency domain analysis (basic implementation)
# API integration hooks for specialized services
```

## **Configuration**

### **Custom Configuration**
```python
config = {
    'ocr_confidence_threshold': 0.5,
    'logo_confidence_threshold': 0.7,
    'watermark_api_endpoint': 'https://api.synthid.com/detect',
    'frame_extraction_count': 3,
    'enable_c2pa': True
}

preprocessor = AIVideoPreprocessor(config)
```

### **Logo Template Setup**
```bash
# Add your logo templates to:
ai_detector/logo_templates/
├── openai_logo.png
├── midjourney_logo.png
├── runway_logo.png
└── adobe_firefly_logo.png
```
## 🔗 **Integration Examples**

### **FastAPI Backend Integration**
```python
# Add to your backend/main.py
from ai_detector.video_preprocessing import preprocess_video_for_ai_detection

@app.post("/analyze-video")
async def analyze_video(video: UploadFile, user_id: str):
    # Save uploaded file
    video_path = save_uploaded_file(video)
    
    # Stage 1: Preprocessing
    preprocessing_result = preprocess_video_for_ai_detection(video_path)
    
    if preprocessing_result['final_decision'] != 'Unknown':
        # Early detection - return immediately
        return {
            'overall_likelihood': preprocessing_result['confidence_score'] * 100,
            'method': 'preprocessing',
            'processing_time': preprocessing_result['processing_time'],
            'early_detection': True,
            'details': preprocessing_result
        }
    
    # Stage 2: Continue with ML analysis
    return await run_ml_analysis(video_path, user_id)
```

### **Supabase Integration**
```python
# Store preprocessing results
async def store_preprocessing_result(result: Dict, user_id: str):
    supabase_data = {
        'user_id': user_id,
        'detection_method': 'preprocessing',
        'overall_likelihood': result['confidence_score'] * 100,
        'analysis_results': result,
        'processing_time': result['processing_time']
    }
    
    await supabase.table('video_analyses').insert(supabase_data)
```

## 🛠️ **Advanced Features**

### **Custom Watermark Patterns**
```python
# Add your own patterns
custom_patterns = [
    'your custom watermark',
    'proprietary ai tool',
    'internal ai system'
]

preprocessor.watermark_patterns.extend(custom_patterns)
```

### **API Integration for Invisible Watermarks**
```python
# Integrate with SynthID or similar services
async def detect_synthid_watermark(frame: np.ndarray) -> float:
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode()
    
    # Call SynthID API
    response = await httpx.post(
        'https://api.synthid.com/detect',
        json={'image': image_b64},
        headers={'Authorization': f'Bearer {api_key}'}
    )
    
    return response.json().get('confidence', 0.0)
```

### **Batch Processing**
```python
# Process multiple videos
def batch_preprocess_videos(video_paths: List[str]) -> List[Dict]:
    preprocessor = AIVideoPreprocessor()
    results = []
    
    for video_path in video_paths:
        result = preprocessor.analyze_video(video_path)
        results.append(result)
    
    return results
```

##  **Output Format**

### **Complete Result Structure**
```json
{
  "metadata_flag": true,
  "metadata_details": {
    "detected_indicator": "openai",
    "category": "creation_tools",
    "ffprobe": {"format": {"tags": {"creation_tool": "OpenAI Sora"}}}
  },
  "ocr_flag": false,
  "ocr_detections": [],
  "logo_flag": false,
  "logo_detections": [],
  "invisible_watermark_conf": 0.0,
  "invisible_watermark_details": {},
  "final_decision": "AI-generated (metadata)",
  "confidence_score": 0.95,
  "processing_time": 1.23,
  "frames_analyzed": 3,
  "timestamp": "2024-01-15T10:30:00"
}
```

### **Decision Logic**
1. **"AI-generated (metadata)"** - Confidence: 0.95
2. **"AI-generated (watermark)"** - Confidence: 0.90  
3. **"AI-generated (logo)"** - Confidence: 0.85
4. **"AI-generated (invisible watermark)"** - Confidence: Variable
5. **"Unknown"** - Proceed to ML analysis

## **Error Handling**

### **Common Issues**
```python
# Handle missing dependencies
if not EASYOCR_AVAILABLE:
    logger.warning("EasyOCR not available, using Tesseract only")

# Handle corrupted videos
try:
    frames = extract_key_frames(video_path)
except Exception as e:
    return create_error_result(f"Frame extraction failed: {e}")

# Handle API timeouts
try:
    result = await watermark_api_call(frame)
except asyncio.TimeoutError:
    logger.warning("Watermark API timeout, using fallback")
```

## **Future Enhancements**

### **Planned Features**
- [ ] **C2PA Integration** - Real C2PA manifest parsing
- [ ] **SynthID Integration** - Google's invisible watermark detection
- [ ] **Advanced Logo Detection** - Deep learning-based logo recognition
- [ ] **Temporal Analysis** - Cross-frame consistency checking
- [ ] **Audio Analysis** - Voice synthesis detection
- [ ] **Blockchain Verification** - Provenance chain validation

### **Performance Optimizations**
- [ ] **GPU Acceleration** - CUDA-based OCR and image processing
- [ ] **Parallel Processing** - Multi-threaded frame analysis
- [ ] **Caching** - Template and model caching
- [ ] **Streaming Analysis** - Real-time video stream processing

## **Support**

For issues and questions:
- Check the troubleshooting section in the main README
- Review the integration examples
- Test with known AI-generated videos
- Monitor processing times and accuracy

This preprocessing pipeline provides a significant performance boost by catching obvious AI-generated videos early, allowing your expensive ML models to focus on the challenging edge cases.