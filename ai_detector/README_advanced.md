# Advanced AI Video Classifier

A AI vs Real video classifier using deep CNN backbones, multi-frame aggregation, and robust training strategies.

## **Key Features**

### **Advanced Architecture**
- **Deep CNN Backbones**: ResNet-50, EfficientNet-B3, ConvNeXt-Tiny
- **Multi-Frame Aggregation**: Attention-based temporal fusion
- **GPU Optimized**: Efficient for RTX 3060 and similar consumer GPUs
- **Flexible Configuration**: Fast, balanced, and accurate presets

### **Robust Training Pipeline**
- **Focal Loss**: Handles class imbalance and hard examples
- **Advanced Augmentation**: Realistic transformations for better generalization
- **Video-Level Splits**: Prevents data leakage during cross-validation
- **Balanced Sampling**: Ensures equal representation during training

### **Comprehensive Evaluation**
- **Cross-Validation**: Video-level splits for robust evaluation
- **Detailed Metrics**: Accuracy, F1, AUC, precision, recall, specificity
- **Source Analysis**: Performance breakdown by video generation method
- **Confidence Analysis**: Prediction reliability assessment

## **Project Structure**

```
ai_detector/
├── models/
│   └── advanced_model.py          # Advanced model architectures
├── datasets/
│   └── video_dataset.py           # Multi-frame video dataset
├── training/
│   └── advanced_trainer.py        # Robust training pipeline
├── evaluation/
│   └── advanced_evaluator.py      # Comprehensive evaluation
├── inference/
│   └── advanced_inference.py      # Production inference
├── scripts/
│   ├── train_advanced.py          # Training script
│   └── evaluate_advanced.py       # Evaluation script
├── requirements_advanced.txt       # Dependencies
└── README_advanced.md             # This file
```

## **Installation**

```bash
# Install advanced dependencies
pip install -r ai_detector/requirements_advanced.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## **Dataset Preparation**

Ensure your data follows this structure:

```
data/
├── ai/
│   ├── sora_video1_frame_01.jpg
│   ├── sora_video1_frame_02.jpg
│   ├── sora_video1_frame_03.jpg
│   ├── runway_video2_frame_01.jpg
│   └── ...
└── real/
    ├── camera_video1_frame_01.jpg
    ├── camera_video1_frame_02.jpg
    ├── camera_video1_frame_03.jpg
    └── ...
```

**Frame Naming Convention**: `{source}_{video_id}_frame_{frame_num:02d}.jpg`

## **Quick Start**

### **1. Train a Model**

```bash
# Fast training (ResNet-50, 3 frames, frozen backbone)
python ai_detector/scripts/train_advanced.py \
    --model_config fast \
    --data_dir data \
    --epochs 50 \
    --batch_size 32

# Balanced training (EfficientNet-B3, 5 frames, fine-tuning)
python ai_detector/scripts/train_advanced.py \
    --model_config balanced \
    --data_dir data \
    --epochs 100 \
    --batch_size 16

# Accurate training (ConvNeXt-Tiny, 7 frames, full training)
python ai_detector/scripts/train_advanced.py \
    --model_config accurate \
    --data_dir data \
    --epochs 150 \
    --batch_size 8 \
    --lr 5e-4
```

### **2. Evaluate Performance**

```bash
# Comprehensive evaluation with plots
python ai_detector/scripts/evaluate_advanced.py \
    --model_path weights/best_advanced_model.pt \
    --data_dir data \
    --save_plots \
    --output_dir evaluation_results
```

### **3. Run Inference**

```python
from ai_detector.inference.advanced_inference import AdvancedAIDetectorInference

# Initialize detector
detector = AdvancedAIDetectorInference('weights/best_advanced_model.pt')

# Analyze video
result = detector.predict_video('path/to/video.mp4')
print(f"Prediction: {result['label']} ({result['probability']:.3f})")

# Temporal analysis for detailed insights
temporal_result = detector.analyze_video_temporal('path/to/video.mp4')
print(f"Consistency: {temporal_result['temporal_analysis']['consistency']:.3f}")
```

## ⚙️ **Model Configurations**

### **Fast Configuration**
- **Backbone**: ResNet-50
- **Frames**: 3 per video
- **Training**: Frozen backbone
- **Use Case**: Quick prototyping, limited compute
- **Training Time**: ~2 hours (RTX 3060)

### **Balanced Configuration** (Recommended)
- **Backbone**: EfficientNet-B3
- **Frames**: 5 per video
- **Training**: Full fine-tuning
- **Use Case**: Production deployment
- **Training Time**: ~4 hours (RTX 3060)

### **Accurate Configuration**
- **Backbone**: ConvNeXt-Tiny
- **Frames**: 7 per video
- **Training**: Full training with aggressive augmentation
- **Use Case**: Maximum accuracy, research
- **Training Time**: ~6 hours (RTX 3060)

## 📈 **Training Features**

### **Focal Loss**
```python
# Handles class imbalance and focuses on hard examples
loss = FocalLoss(alpha=1.0, gamma=2.0)
```

### **Advanced Augmentation**
- **RandomResizedCrop**: Scale (0.7-1.0), Ratio (0.8-1.2)
- **ColorJitter**: Brightness, contrast, saturation, hue variation
- **RandomGrayscale**: 10% probability
- **GaussianBlur**: 20% probability
- **Noise Injection**: Subtle Gaussian noise

### **Multi-Frame Aggregation**
```python
# Attention-based temporal fusion
aggregator = MultiFrameAggregator(feature_dim=2048, num_frames=5)
```

### **Balanced Sampling**
- **WeightedRandomSampler**: Ensures equal class representation
- **Video-Level Splits**: Prevents data leakage
- **Cross-Validation**: Robust performance estimation

## **Advanced Usage**

### **Custom Training Configuration**

```python
from ai_detector.training.advanced_trainer import create_training_config, AdvancedTrainer

# Create custom config
config = create_training_config(
    model_config='balanced',
    data_dir='data',
    epochs=100,
    batch_size=16,
    learning_rate=1e-3
)

# Customize loss function
config['training']['loss'] = {
    'type': 'focal',
    'alpha': 0.75,  # Adjust for class imbalance
    'gamma': 2.5    # Increase focus on hard examples
}

# Train with custom config
trainer = AdvancedTrainer(config)
trainer.train()
```

### **Integration with Existing Pipeline**

```python
# Replace existing detection function
from ai_detector.inference.advanced_inference import integrate_with_existing_pipeline

def detect_frame_ai_likelihood_enhanced(frame, frame_time=0.0):
    return integrate_with_existing_pipeline(frame, frame_time)
```

### **Batch Video Processing**

```python
# Process multiple videos efficiently
detector = AdvancedAIDetectorInference('weights/best_advanced_model.pt')

video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
results = detector.predict_batch_videos(video_paths)

for result in results:
    print(f"{result['video_path']}: {result['label']} ({result['probability']:.3f})")
```

## **Monitoring and Optimization**

### **TensorBoard Monitoring**
```bash
# View training progress
tensorboard --logdir logs/
```

### **Memory Optimization**
```python
# Reduce batch size for limited GPU memory
config['training']['batch_size'] = 8

# Use gradient accumulation
config['training']['grad_accumulation_steps'] = 2
```

### **Performance Profiling**
```python
# Profile model inference
import torch.profiler

with torch.profiler.profile() as prof:
    result = detector.predict_video('test_video.mp4')

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## **Troubleshooting**

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 8
   
   # Use gradient checkpointing
   --grad_checkpoint
   ```

2. **Low Accuracy**
   ```bash
   # Increase training epochs
   --epochs 150
   
   # Use balanced configuration
   --model_config balanced
   
   # Check data quality and balance
   ```

3. **Slow Training**
   ```bash
   # Use fast configuration
   --model_config fast
   
   # Reduce number of workers
   --num_workers 2
   ```

### **Performance Tips**

1. **Data Loading**: Use SSD storage for faster I/O
2. **Batch Size**: Maximize GPU utilization without OOM
3. **Mixed Precision**: Enable for faster training (experimental)
4. **Model Pruning**: Remove unnecessary parameters post-training

## **Citation**

If you use this advanced AI video classifier in your research, please cite:

```bibtex
@software{advanced_ai_video_classifier,
  title={Advanced AI Video Classifier with Deep CNN and Multi-Frame Aggregation},
  author={Truthful AI Detection Team},
  year={2024},
  url={https://github.com/your-repo/truthful-ai-detector}
}
```

## **Contributing**

1. **Model Improvements**: New architectures, loss functions, aggregation methods
2. **Dataset Expansion**: Additional video sources and generation methods
3. **Optimization**: Speed and memory improvements
4. **Evaluation**: New metrics and analysis methods

## **Support**

For issues and questions:
- Check the troubleshooting section
- Review training logs in `training_advanced.log`
- Examine evaluation results in `evaluation_results/`
- Monitor training with TensorBoard

This advanced classifier provides state-of-the-art performance for AI video detection with robust training and comprehensive evaluation capabilities.