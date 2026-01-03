#!/bin/bash

# Start the new Trading Bot frontend (frontend2)

echo "========================================"
echo "Trading Bot - Starting New Frontend"
echo "========================================"

# Change to frontend2 directory
cd "$(dirname "$0")/frontend2" || exit 1

echo "Current directory: $(pwd)"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "Node.js version:"
node --version
echo ""

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not available"
    exit 1
fi

echo "npm version:"
npm --version
echo ""

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo ""
fi

# Start the development server
echo "Starting development server..."
echo "Frontend will be available at: http://localhost:5173"
echo "Press Ctrl+C to stop the server"
echo ""

npm run dev