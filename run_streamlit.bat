@echo off
REM Quick Start Script for Trading Strategy Streamlit App

echo ========================================
echo Trading Strategy Analysis - Streamlit UI
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.py first.
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
echo Checking Streamlit installation...
.\.venv\Scripts\python.exe -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing...
    .\.venv\Scripts\python.exe -m pip install streamlit
    if errorlevel 1 (
        echo ERROR: Failed to install Streamlit
        pause
        exit /b 1
    )
    echo Streamlit installed successfully!
    echo.
)

echo Starting Streamlit app...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Run the Streamlit app
.\.venv\Scripts\python.exe -m streamlit run app.py

pause
