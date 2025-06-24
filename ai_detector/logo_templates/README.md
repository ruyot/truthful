# Logo Templates Directory

This directory contains template images for detecting known AI tool logos in video frames.

## Setup Instructions

1. **Collect Logo Images**: Gather clear, high-quality images of logos from:
   - OpenAI (Sora, DALL-E)
   - Midjourney
   - Runway ML
   - Adobe Firefly
   - Stable Diffusion
   - Pika Labs
   - InVideo
   - Synthesia
   - Other AI video generation tools

2. **Image Requirements**:
   - Format: PNG or JPG
   - Size: 50x50 to 200x200 pixels
   - Clear, unobstructed logo
   - Transparent background preferred (PNG)

3. **Naming Convention**:
   ```
   openai_logo.png
   midjourney_logo.png
   runway_logo.png
   adobe_firefly_logo.png
   stable_diffusion_logo.png
   pika_labs_logo.png
   invideo_logo.png
   synthesia_logo.png
   ```

4. **Template Matching**:
   - The system uses OpenCV template matching
   - Multiple scales and rotations are tested
   - Confidence threshold: 0.7 (adjustable)

## Legal Considerations

- Only use logos for detection purposes
- Respect trademark and copyright laws
- Consider fair use guidelines
- Do not redistribute logo images

## Performance Tips

- Keep templates small (under 200x200px) for speed
- Use grayscale templates when possible
- Test with various video qualities
- Adjust confidence thresholds based on accuracy needs

## Adding New Templates

1. Save logo image in this directory
2. Update the `logo_templates` list in `video_preprocessing.py`
3. Test detection accuracy with sample videos
4. Adjust confidence thresholds if needed

## Current Status

- ✅ Directory structure created
- ✅ Placeholder templates generated
- ⚠️ Real logo templates needed
- ⚠️ Testing with actual AI-generated videos required