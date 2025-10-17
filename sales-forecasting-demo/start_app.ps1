# Enhanced PowerShell script to start the Sales Forecasting App
param(
    [switch]$SkipChecks
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Sales Forecasting App Startup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Script location: $ScriptDir" -ForegroundColor Yellow

# Navigate to the script directory
Set-Location $ScriptDir
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: app.py not found!" -ForegroundColor Red
    Write-Host "Please make sure this script is in the sales-forecasting-demo folder" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    Write-Host "Then run: venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "Then run: pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated successfully" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify streamlit is available
Write-Host "Checking if Streamlit is installed..." -ForegroundColor Yellow
try {
    python -c "import streamlit; print('Streamlit is available')"
    Write-Host "Streamlit verification successful" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Streamlit not found in virtual environment!" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "Please create .env file with your OpenAI API key" -ForegroundColor Yellow
    Write-Host ""
}

# Start the Streamlit app
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Sales Forecasting App..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The app will open in your browser at: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Login credentials:" -ForegroundColor White
Write-Host "Customer: customer / Customer123!" -ForegroundColor White
Write-Host "Admin: admin / Admin123!" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the app" -ForegroundColor Yellow
Write-Host ""

streamlit run app.py
