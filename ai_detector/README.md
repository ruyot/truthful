# AI Detector Model - CLIP-Inspired Binary Classifier

A lightweight, production-ready AI detector model for identifying AI-generated images and video frames. Built with a CLIP-inspired architecture using a pretrained ResNet18 backbone.

## 🚀 Features

- **CLIP-Inspired Architecture**: Uses pretrained ResNet18 with custom classification head
- **Efficient Training**: Can be trained on small datasets (1000+ images per class)
- **Production Ready**: Optimized for real-time inference
- **Supabase Integration**: Built-in support for data collection and model improvement
- **Comprehensive Evaluation**: Detailed metrics and visualization tools
- **Easy Integration**: Drop-in replacement for existing detection methods

## 📁 Project Structure

```
ai_detector/
├── model.py              # Model architecture and utilities
├── dataset.py            # Dataset loading and preprocessing
├── train.py              # Training script
├── evaluate.py           # Evaluation and metrics
├── inference.py          # Inference utilities
├── supabase_integration.py # Supabase integration
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your training data in the following structure:

```
data/
├── ai/
│   ├── ai_image_001.jpg
│   ├── ai_image_002.png
│   └── ...
└── real/
    ├── real_image_001.jpg
    ├── real_image_002.png
    └── ...
```

**Data Collection Tips:**
- **AI Images**: Use images from Midjourney, DALL-E, Stable Diffusion, etc.
- **Real Images**: Use authentic photos from cameras, phones, stock photos
- **Minimum**: 500+ images per class (1000+ recommended)
- **Balance**: Keep roughly equal numbers of AI and real images
- **Quality**: Use high-resolution images (224x224 minimum)

### 3. Train the Model

```bash
# Basic training (frozen backbone)
python train.py --data_dir data --epochs 50 --batch_size 32 --freeze_backbone

# Fine-tuning (unfreeze backbone for better accuracy)
python train.py --data_dir data --epochs 30 --batch_size 16 --learning_rate 1e-4

# Advanced training with custom parameters
python train.py \
    --data_dir data \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --dropout_rate 0.3 \
    --hidden_dim 256 \
    --max_samples_per_class 2000
```

**Training Parameters:**
- `--freeze_backbone`: Freeze ResNet18 weights (faster, good for small datasets)
- `--dropout_rate`: Regularization (0.1-0.5, default: 0.3)
- `--hidden_dim`: Classification head size (128-512, default: 256)
- `--max_samples_per_class`: Limit training data size

### 4. Evaluate the Model

```bash
python evaluate.py --model_path weights/ai_detector.pt --data_dir data --save_plots
```

This generates:
- Confusion matrix
- ROC curve
- Calibration curve
- Detailed metrics report

## 🔧 Integration with Your Application

### Replace Existing Detection Function

In your `backend/main.py`, replace the `detect_frame_ai_likelihood_enhanced` function:

```python
from ai_detector.inference import integrate_with_existing_pipeline

# Replace the existing function with:
def detect_frame_ai_likelihood_enhanced(frame: np.ndarray, frame_time: float = 0.0) -> Dict[str, float]:
    return integrate_with_existing_pipeline(frame, frame_time)
```

### Direct Inference

```python
from ai_detector.inference import AIDetectorInference

# Initialize detector
detector = AIDetectorInference('weights/ai_detector.pt')

# Analyze single image
result = detector.predict_single(image)
print(f"Prediction: {result['label']} ({result['probability']:.3f})")

# Analyze video frames
frames = [frame1, frame2, frame3]  # List of numpy arrays
video_result = detector.analyze_video_frames(frames)
print(f"Overall likelihood: {video_result['overall_likelihood']:.1f}%")
```

## 📊 Data Augmentation Strategies

The model includes built-in augmentation:

```python
# Training augmentations (automatically applied)
- Random crop (256→224)
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization (ImageNet stats)
```

**Additional Augmentation Tips:**
- **Noise Addition**: Add subtle Gaussian noise to real images
- **Compression**: Apply JPEG compression at different quality levels
- **Resize Artifacts**: Resize images to different resolutions
- **Blur**: Apply slight blur to simulate camera effects

## 🎯 Evaluation Metrics

The model provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **Calibration**: How well probabilities match actual accuracy

**Target Performance:**
- Accuracy: >85% on balanced test set
- AUC: >0.90
- Precision/Recall: >80% for both classes

## 🗄️ Supabase Integration

### 1. Setup Database Tables

Run the migration SQL from `supabase_integration.py`:

```sql
-- Copy the SUPABASE_MIGRATION_SQL and run in Supabase SQL editor
```

### 2. Initialize Integration

```python
from ai_detector.supabase_integration import SupabaseAIDetectorIntegration

# Initialize
integration = SupabaseAIDetectorIntegration(
    supabase_url="your-project-url",
    supabase_key="your-anon-key",
    model_path="weights/ai_detector.pt"
)

# Analyze and store image
result = await integration.analyze_and_store_image(image_bytes, user_id)

# Submit user feedback
await integration.submit_user_feedback(result_id, feedback_score=4, verified_label=1)

# Export training data for retraining
metadata = await integration.export_training_data()
```

### 3. Model Improvement Workflow

1. **Collect Feedback**: Users rate prediction accuracy (1-5 stars)
2. **Gather Corrections**: Users can correct wrong predictions
3. **Export Data**: Periodically export corrected data
4. **Retrain Model**: Use new data to improve the model
5. **Deploy Updates**: Replace model weights with improved version

## 🚀 Production Deployment

### 1. Model Optimization

```python
# Convert to TorchScript for faster inference
model = load_model('weights/ai_detector.pt')
scripted_model = torch.jit.script(model)
scripted_model.save('weights/ai_detector_scripted.pt')
```

### 2. Batch Processing

```python
# Process multiple frames efficiently
detector = AIDetectorInference('weights/ai_detector.pt')
results = detector.predict_batch(frames)  # List of frames
```

### 3. GPU Acceleration

```python
# Use GPU if available
detector = AIDetectorInference('weights/ai_detector.pt', device='cuda')
```

## 📈 Performance Tips

### Training Tips:
- Start with frozen backbone for quick prototyping
- Unfreeze backbone for final training with lower learning rate
- Use class weights if data is imbalanced
- Monitor validation loss for early stopping

### Inference Tips:
- Batch process multiple frames for efficiency
- Use GPU for real-time video analysis
- Cache model in memory for repeated use
- Consider model quantization for mobile deployment

### Data Tips:
- Collect diverse AI-generated content (different models, styles)
- Include edge cases (low quality, compressed images)
- Regular data collection from user feedback
- Periodic model retraining (monthly/quarterly)

## 🔍 Troubleshooting

### Common Issues:

1. **Low Accuracy**: 
   - Check data quality and balance
   - Increase training epochs
   - Try unfreezing backbone

2. **Overfitting**:
   - Increase dropout rate
   - Add more data augmentation
   - Reduce model complexity

3. **Slow Training**:
   - Reduce batch size
   - Use frozen backbone
   - Enable mixed precision training

4. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Process images at lower resolution

### Performance Benchmarks:

- **Training Time**: ~30 minutes for 50 epochs (1000 images/class, GPU)
- **Inference Speed**: ~50ms per image (CPU), ~5ms per image (GPU)
- **Model Size**: ~45MB (ResNet18 backbone)
- **Memory Usage**: ~2GB GPU memory during training

## 📝 License

This project is part of the Truthful AI Video Detection Tool. See the main project license for details.

## 🤝 Contributing

1. Collect and label more training data
2. Experiment with different architectures
3. Improve data augmentation strategies
4. Add new evaluation metrics
5. Optimize inference performance

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review training logs in `training.log`
- Examine evaluation plots in `evaluation_plots/`
- Monitor training progress with TensorBoard: `tensorboard --logdir logs/`