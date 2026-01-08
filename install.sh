#!/bin/bash

# ========================================
# CIVITAS UB DETECTION - AUTO INSTALLER
# ========================================

echo "🚀 Installing Civitas UB Detection System..."
echo "=============================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $python_version == *"Python 3"* ]]; then
    echo "✅ Python 3 found: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.7+"
    exit 1
fi

# Install required packages
echo ""
echo "📦 Installing required packages..."
pip3 install opencv-python numpy

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Packages installed successfully"
else
    echo "❌ Package installation failed"
    exit 1
fi

# Create required directories
echo ""
echo "📁 Creating required directories..."
mkdir -p haarcascades
mkdir -p templates

# Download Haar Cascade if not exists
if [ ! -f "haarcascades/haarcascade_frontalface_default.xml" ]; then
    echo "📥 Downloading Haar Cascade file..."
    curl -o haarcascades/haarcascade_frontalface_default.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
    
    if [ $? -eq 0 ]; then
        echo "✅ Haar Cascade downloaded"
    else
        echo "⚠️  Haar Cascade download failed, will use system default"
    fi
else
    echo "✅ Haar Cascade already exists"
fi

# Check if main file exists
if [ -f "main-jetson-civitas.py" ]; then
    echo "✅ Main program file found"
else
    echo "❌ main-jetson-civitas.py not found!"
    echo "Please make sure the main program file is in this directory"
    exit 1
fi

# Final check
echo ""
echo "🔍 Final system check..."
python3 -c "import cv2, numpy; print('✅ OpenCV and NumPy working')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ OpenCV or NumPy not working properly"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo "=============================================="
echo ""
echo "📝 Next steps:"
echo "1. Make sure you have UB logo templates in templates/ folder"
echo "2. Connect your camera/webcam"
echo "3. Run: python3 main-jetson-civitas.py"
echo ""
echo "📖 For detailed instructions, see USER_GUIDE_CIVITAS.md"
echo "⚡ For quick start, see QUICK_START.md"