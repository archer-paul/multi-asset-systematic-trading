@echo off
echo ========================================
echo Trading Bot Frontend - Starting...
echo ========================================

REM Check for npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found!
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)

REM Start dev server
echo Starting development server...
echo Open: http://localhost:5173
npm run dev