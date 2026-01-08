#!/bin/bash

# ========================================
# CIVITAS UB DETECTION - ONE-CLICK SETUP
# ========================================

echo "🚀 Setting up Civitas UB Detection System..."
echo "=============================================="

# Install requirements
echo "📦 Installing dependencies from requirements..."
pip install -r requirements-civitas.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create directories if needed
mkdir -p haarcascades templates

echo "✅ Setup completed!"
echo ""
echo "🎬 Starting Civitas UB Detection..."
echo "Press 'q' to quit the program"
echo ""

# Run the main program
python main-jetson-civitas.py