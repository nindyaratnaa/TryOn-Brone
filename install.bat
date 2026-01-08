@echo off
REM ========================================
REM CIVITAS UB DETECTION - AUTO INSTALLER
REM ========================================

echo 🚀 Installing Civitas UB Detection System...
echo ==============================================

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

python --version
echo ✅ Python found

REM Install required packages
echo.
echo 📦 Installing required packages...
pip install opencv-python numpy

if %errorlevel% neq 0 (
    echo ❌ Package installation failed
    pause
    exit /b 1
)

echo ✅ Packages installed successfully

REM Create required directories
echo.
echo 📁 Creating required directories...
if not exist "haarcascades" mkdir haarcascades
if not exist "templates" mkdir templates

REM Check if main file exists
if exist "main-jetson-civitas.py" (
    echo ✅ Main program file found
) else (
    echo ❌ main-jetson-civitas.py not found!
    echo Please make sure the main program file is in this directory
    pause
    exit /b 1
)

REM Final check
echo.
echo 🔍 Final system check...
python -c "import cv2, numpy; print('✅ OpenCV and NumPy working')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ OpenCV or NumPy not working properly
    pause
    exit /b 1
)

echo.
echo 🎉 Installation completed successfully!
echo ==============================================
echo.
echo 📝 Next steps:
echo 1. Make sure you have UB logo templates in templates\ folder
echo 2. Connect your camera/webcam
echo 3. Run: python main-jetson-civitas.py
echo.
echo 📖 For detailed instructions, see USER_GUIDE_CIVITAS.md
echo ⚡ For quick start, see QUICK_START.md
echo.
pause