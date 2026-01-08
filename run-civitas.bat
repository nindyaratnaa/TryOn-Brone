@echo off
REM ========================================
REM CIVITAS UB DETECTION - ONE-CLICK SETUP
REM ========================================

echo Setting up Civitas UB Detection System...
echo ============================================

REM Install requirements
echo Installing dependencies from requirements...
pip install -r requirements-civitas.txt

if %errorlevel% neq 0 (
    echo X Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed successfully

REM Create directories if needed
if not exist "haarcascades" mkdir haarcascades
if not exist "templates" mkdir templates

echo ✓ Setup completed!
echo.
echo Starting Civitas UB Detection...
echo Press 'q' to quit the program
echo.

REM Run the main program
python main-jetson-civitas.py

pause