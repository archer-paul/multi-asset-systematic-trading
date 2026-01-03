@echo off
REM Start the new Trading Bot frontend (frontend2)

echo ========================================
echo Trading Bot - Starting New Frontend
echo ========================================

cd /d "%~dp0frontend2"

echo Current directory: %cd%
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js version:
node --version
echo.

REM Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm is not available
    pause
    exit /b 1
)

echo npm version:
npm --version
echo.

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
)

REM Start the development server
echo Starting development server...
echo Frontend will be available at: http://localhost:5173
echo Press Ctrl+C to stop the server
echo.

call npm run dev

if errorlevel 1 (
    echo ERROR: Failed to start development server
    pause
    exit /b 1
)

pause