# Quick Start Script for Trading Strategy Streamlit App
# PowerShell version

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Trading Strategy Analysis - Streamlit UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.py first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if streamlit is installed
Write-Host "Checking Streamlit installation..." -ForegroundColor Yellow
$streamlitCheck = & .\.venv\Scripts\python.exe -m pip show streamlit 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit not found. Installing..." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe -m pip install streamlit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install Streamlit" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Streamlit installed successfully!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host ""
Write-Host "The app will open in your browser at: " -NoNewline
Write-Host "http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run the Streamlit app
& .\.venv\Scripts\python.exe -m streamlit run app.py

Write-Host ""
Read-Host "Press Enter to exit"
