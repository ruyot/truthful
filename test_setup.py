#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
and the project can be imported without errors.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    # Test core ML libraries
    try:
        import torch
        import torchvision
        print("✓ PyTorch and TorchVision imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe
        print("✓ MediaPipe imported successfully")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    # Test web framework
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI and Uvicorn imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    # Test data processing
    try:
        import pandas
        import numpy
        print("✓ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Data processing import failed: {e}")
        return False
    
    return True

def test_project_modules():
    """Test project-specific modules."""
    print("\nTesting project modules...")
    
    try:
        from backend.main import app
        print("✓ Backend app imported successfully")
    except ImportError as e:
        print(f"✗ Backend import failed: {e}")
        return False
    
    try:
        from ai_detector.model import AIDetectorModel
        print("✓ AI detector model imported successfully")
    except ImportError as e:
        print(f"✗ AI detector import failed: {e}")
        return False
    
    try:
        from ai_detector.video_preprocessing import AIVideoPreprocessor
        print("✓ Video preprocessor imported successfully")
    except ImportError as e:
        print(f"✗ Video preprocessor import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation and basic functionality."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from ai_detector.model import create_model
        
        # Test model creation
        model = create_model()
        print("✓ AI detector model created successfully")
        
        # Test basic forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print("✓ Model forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TRUTHFUL AI DETECTOR - SETUP VERIFICATION")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test project modules
    if not test_project_modules():
        success = False
    
    # Test model creation
    if not test_model_creation():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Setup is complete and working.")
        print("You can now run the application.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 