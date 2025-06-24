# Skeleton-Based AI Video Detection with VidProM Dataset

A state-of-the-art AI video detection system using skeleton-based structural matching with the VidProM dataset for improved generalization to novel AI content.

## **Key Features**

### **Skeleton-Based Architecture**
- **Distance-Based Matching**: Structural representations of AI vs Real video characteristics
- **Class-Level Embeddings**: Mean, variance, and covariance statistics for robust matching
- **Multiple Distance Metrics**: Euclidean, Cosine, Mahalanobis, and k-NN distance calculations
- **Fusion Capabilities**: Combines base CNN predictions with skeleton matching

### **VidProM Dataset Integration**
- **Hugging Face Integration**: Automated download from VidProM repository
- **License Filtering**: Commercial-compatible licenses (CC BY-NC 4.0 acceptable)
- **Combined Training**: DFD (real videos) + VidProM (AI videos) for comprehensive coverage
- **Multi-Source AI Content**: Sora, Runway, Pika Labs, Stable Video, and more

### **Advanced Training Pipeline**
- **Multi-Task Learning**: Optional prompt embedding prediction for semantic understanding
- **Skeleton Loss**: Encourages within-class similarity and between-class separation
- **Video-Level Splits**: Prevents data leakage during training and evaluation
- **Production Ready**: Optimized for deployment with configurable fusion weights

## **Project Structure**

```
ai_detector/
├── datasets/
│   └── vidprom_dataset.py          # VidProM dataset processing
├── models/
│   └── skeleton_model.py           # Skeleton-based detector
├── training/
│   └── skeleton_trainer.py         # Training pipeline
├── inference/
│   └── skeleton_inference.py       # Production inference
├── scripts/
│   ├── train_skeleton.py           # Training script
│   └── download_vidprom.py         # Dataset download script
├── requirements_skeleton.txt       # Dependencies
└── README_skeleton.md             # This file
```

## **Installation**

```bash
# Install skeleton detector dependencies
pip install -r ai_detector/requirements_skeleton.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## **Dataset Preparation**

### **1. Download VidProM Dataset**

```bash
# Download and process VidProM dataset
python ai_detector/scripts/download_vidprom.py \
    --output_dir data \
    --max_videos 1000 \
    --frames_per_video 5
```

### **2. Prepare DFD Dataset**

```bash
# Download and process DFD dataset (for real videos)
python ai_detector/kaggle_dfd_downloader.py \
    --output_dir data \
    --max_videos 1000
```

### **Expected Structure**

```
data/
├── ai/                    # VidProM AI-generated frames
│   ├── sora_video1_frame_01.jpg
│   ├── runway_video2_frame_01.jpg
│   └── ...
└── real/                  # DFD real video frames
    ├── original_video1_frame_01.jpg
    ├── camera_video2_frame_01.jpg
    └── ...
```

## **Quick Start**

### **1. Train Skeleton Model**

```bash
# Basic training with VidProM + DFD
python ai_detector/scripts/train_skeleton.py \
    --dfd_dir data \
    --vidprom_dir data \
    --epochs 100 \
    --batch_size 16

# Training with multi-task learning
python ai_detector/scripts/train_skeleton.py \
    --enable_multitask \
    --epochs 150 \
    --batch_size 8 \
    --lr 5e-4

# Custom backbone and settings
python ai_detector/scripts/train_skeleton.py \
    --backbone convnext_tiny \
    --max_videos 1000 \
    --epochs 200
```

### **2. Enable Skeleton Detection in Backend**

```bash
# Set environment variable to enable skeleton model
export USE_SKELETON_MODEL=true

# Start backend with skeleton detection
python backend/main.py
```

### **3. Run Inference**

```python
from ai_detector.inference.skeleton_inference import SkeletonBasedDetectorInference

# Initialize detector
detector = SkeletonBasedDetectorInference('weights/best_skeleton_model.pt')

# Analyze video with structural matching
result = detector.predict_video('path/to/video.mp4')
print(f"Prediction: {result['label']} ({result['probability']:.3f})")
print(f"Structural match score: {result['structural_match_score']:.1f}%")

# Detailed structural analysis
structural_analysis = detector.analyze_structural_similarity(frames)
print(f"Distance methods: {list(structural_analysis['distance_analysis'].keys())}")
```

## **Skeleton Detection Process**

### **1. Training Phase**

1. **Base Training**: Train CNN backbone on combined DFD + VidProM data
2. **Embedding Extraction**: Extract video-level embeddings for all training samples
3. **Skeleton Computation**: Calculate class-level statistics (mean, variance, covariance)
4. **Multi-Task Learning**: Optional prompt embedding prediction for semantic understanding

### **2. Inference Phase**

1. **Feature Extraction**: Extract video embedding using trained backbone
2. **Distance Calculation**: Compute distances to AI and Real skeletons
3. **Fusion**: Combine base CNN prediction with skeleton distances
4. **Structural Score**: Generate interpretable structural similarity score

### **3. Distance Methods**

```python
# Available distance metrics
distance_methods = [
    'euclidean',     # L2 distance to skeleton mean
    'cosine',        # Cosine similarity to skeleton mean
    'mahalanobis',   # Mahalanobis distance using covariance
    'knn'           # Distance to k nearest neighbors
]
```


## **Configuration Options**

### **Fusion Weight Tuning**

```python
# Configure fusion between base CNN and skeleton matching
detector = SkeletonBasedDetectorInference(
    model_path='weights/best_skeleton_model.pt',
    fusion_weight=0.5,  # 0=base only, 1=skeleton only
    distance_method='mahalanobis'
)
```

### **Distance Method Selection**

```python
# Test different distance methods
methods = ['euclidean', 'cosine', 'mahalanobis', 'knn']
for method in methods:
    detector.model.skeleton_distance_method = method
    result = detector.predict(frames)
    print(f"{method}: {result['probability']:.3f}")
```

### **Multi-Task Learning**

```bash
# Enable prompt embedding prediction during training
python ai_detector/scripts/train_skeleton.py \
    --enable_multitask \
    --epochs 150
```

## **Backend Integration**

### **Environment Configuration**

```bash
# Enable skeleton-based detection
export USE_SKELETON_MODEL=true

# Configure fusion weight (optional)
export SKELETON_FUSION_WEIGHT=0.6

# Set distance method (optional)
export SKELETON_DISTANCE_METHOD=mahalanobis
```

### **API Response Enhancement**

```json
{
  "overall_likelihood": 85.3,
  "analysis_results": {
    "method": "skeleton_enhanced",
    "structural_similarity": {
      "ai_distance": 2.34,
      "real_distance": 5.67,
      "skeleton_probability": 0.87,
      "base_probability": 0.82,
      "distance_method": "mahalanobis"
    },
    "structural_match_score": 86.2,
    "fusion_weight": 0.5,
    "skeleton_enabled": true
  }
}
```

### **Frontend Display**

The frontend automatically detects and displays skeleton-based analysis:

- **Structural Match Score**: "Structural similarity to AI profile: 86%"
- **Detection Method Badge**: "Skeleton-Based Detection"
- **Enhanced Details**: Distance analysis and fusion information

## **VidProM Dataset Details**

### **Supported AI Generation Methods**

- **Sora (OpenAI)**: High-quality video generation
- **Runway Gen-2**: Creative video synthesis
- **Pika Labs**: AI video creation platform
- **Stable Video Diffusion**: Open-source video generation
- **AnimateDiff**: Animation-focused generation

### **License Compatibility**

```python
# Commercial-compatible licenses
allowed_licenses = [
    'cc-by-4.0',      # Creative Commons Attribution
    'cc-by-sa-4.0',   # Creative Commons ShareAlike
    'cc-by-nc-4.0',   # Creative Commons Non-Commercial (MVP acceptable)
    'mit',            # MIT License
    'apache-2.0',     # Apache License
    'bsd-3-clause',   # BSD License
    'public-domain'   # Public Domain
]
```

## 🔧 **Advanced Usage**

### **Custom Skeleton Computation**

```python
# Compute skeletons from custom embeddings
ai_embeddings = extract_embeddings_from_videos(ai_video_paths)
real_embeddings = extract_embeddings_from_videos(real_video_paths)

model.compute_skeletons(ai_embeddings, real_embeddings)
model.save_skeletons('custom_skeletons')
```

### **Batch Video Analysis**

```python
# Analyze multiple videos efficiently
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
results = detector.predict_batch_videos(video_paths)

for result in results:
    print(f"{result['video_path']}: {result['structural_match_score']:.1f}%")
```

### **Structural Analysis Deep Dive**

```python
# Detailed structural similarity analysis
analysis = detector.analyze_structural_similarity(frames)

print("Distance Analysis:")
for method, distances in analysis['distance_analysis'].items():
    print(f"  {method}: AI={distances['ai_distance']:.2f}, Real={distances['real_distance']:.2f}")

print(f"Skeleton Statistics:")
print(f"  AI samples: {analysis['skeleton_stats']['ai_skeleton']['n_samples']}")
print(f"  Real samples: {analysis['skeleton_stats']['real_skeleton']['n_samples']}")
```

## **Troubleshooting**

### **Common Issues**

1. **VidProM Download Fails**
   ```bash
   # Check Hugging Face credentials
   huggingface-cli login
   
   # Use mock data for development
   python ai_detector/scripts/download_vidprom.py --max_videos 10
   ```

2. **Skeleton Not Available**
   ```bash
   # Check skeleton files exist
   ls weights/skeletons/
   
   # Retrain if missing
   python ai_detector/scripts/train_skeleton.py --epochs 50
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   --batch_size 8
   
   # Limit videos per source
   --max_videos 500
   ```

### **Performance Optimization**

1. **Distance Method Selection**: Mahalanobis generally performs best
2. **Fusion Weight Tuning**: 0.5-0.7 typically optimal
3. **Skeleton Size**: More training samples = better skeletons
4. **Multi-Task Learning**: Improves semantic understanding

## **Citation**

```bibtex
@software{skeleton_ai_video_detector,
  title={Skeleton-Based AI Video Detection with VidProM Dataset Integration},
  author={Truthful AI Detection Team},
  year={2024},
  url={https://github.com/your-repo/truthful-ai-detector}
}
```

## **Contributing**

1. **Dataset Expansion**: Add new AI generation methods to VidProM processing
2. **Distance Metrics**: Implement new distance calculation methods
3. **Fusion Strategies**: Develop advanced fusion techniques
4. **Evaluation**: Test on additional datasets and generation methods

## **Support**

For skeleton-based detection issues:
- Check skeleton files in `weights/skeletons/`
- Verify VidProM dataset processing logs
- Test with different fusion weights and distance methods
- Monitor structural match scores for consistency

This skeleton-based approach provides significant improvements in detecting novel AI-generated content through learned structural representations and distance-based matching.