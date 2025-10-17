@echo off
echo ========================================
echo   Sales Forecasting App - Quick Start
echo ========================================
echo.

REM Find the sales-forecasting-demo directory
set "PROJECT_DIR="
for /d %%i in ("%~dp0*") do (
    if exist "%%i\app.py" (
        set "PROJECT_DIR=%%i"
        goto :found
    )
)

REM If not found in current directory, try parent directory
for /d %%i in ("%~dp0..\*") do (
    if exist "%%i\app.py" (
        set "PROJECT_DIR=%%i"
        goto :found
    )
)

echo ERROR: Could not find sales-forecasting-demo directory!
echo Please make sure you're running this from the correct location.
pause
exit /b 1

:found
echo Found project directory: %PROJECT_DIR%
echo.

REM Navigate to project directory and run the main startup script
cd /d "%PROJECT_DIR%"
call start_app.bat
