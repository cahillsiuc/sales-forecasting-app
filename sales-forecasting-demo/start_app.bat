@echo off
echo ========================================
echo   Sales Forecasting App Startup Script
echo ========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
echo Script location: %SCRIPT_DIR%

REM Navigate to the script directory (ensures we're in the right place)
cd /d "%SCRIPT_DIR%"
echo Current directory: %CD%
echo.

REM Check if we're in the correct directory
if not exist "app.py" (
    echo ERROR: app.py not found!
    echo Please make sure this script is in the sales-forecasting-demo folder
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\activate
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Verify streamlit is available
echo Checking if Streamlit is installed...
python -c "import streamlit; print('Streamlit is available')" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not found in virtual environment!
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Please create .env file with your OpenAI API key
    echo.
)

REM Start the Streamlit app
echo ========================================
echo   Starting Sales Forecasting App...
echo ========================================
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Login credentials:
echo Customer: customer / Customer123!
echo Admin: admin / Admin123!
echo.
echo Press Ctrl+C to stop the app
echo.

streamlit run app.py

echo.
echo App stopped. Press any key to exit...
pause >nul
