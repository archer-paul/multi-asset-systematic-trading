#!/usr/bin/env python3
"""
Startup script for the new Trading Bot frontend.
This script builds the frontend and starts a simple HTTP server.
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path
import http.server
import socketserver

# Configuration
FRONTEND_DIR = Path(__file__).parent
BUILD_DIR = FRONTEND_DIR / "dist"
PORT = 3000

def build_frontend():
    """Build the React frontend"""
    print("Building React frontend...")

    # Check if node_modules exists
    if not (FRONTEND_DIR / "node_modules").exists():
        print("Installing dependencies...")
        result = subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing dependencies: {result.stderr}")
            return False

    # Build the frontend
    result = subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building frontend: {result.stderr}")
        return False

    print("Frontend built successfully!")
    return True

def start_dev_server():
    """Start the development server"""
    print(f"Starting development server on port {PORT}...")

    try:
        result = subprocess.run(["npm", "run", "dev", "--", "--port", str(PORT)], cwd=FRONTEND_DIR)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nDevelopment server stopped by user.")
        return True
    except Exception as e:
        print(f"Error starting development server: {e}")
        return False

def start_production_server():
    """Start production server serving built files"""
    if not BUILD_DIR.exists():
        print("Build directory not found. Building first...")
        if not build_frontend():
            return False

    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(BUILD_DIR), **kwargs)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()

        def do_GET(self):
            # Handle SPA routing - serve index.html for non-existent files
            requested_path = BUILD_DIR / self.path.lstrip('/')
            if not requested_path.exists() and not self.path.startswith('/api'):
                self.path = '/index.html'

            return super().do_GET()

    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"Serving production build at http://localhost:{PORT}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nProduction server stopped by user.")
        return True
    except Exception as e:
        print(f"Error starting production server: {e}")
        return False

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Start the Trading Bot frontend")
    parser.add_argument("--mode", choices=["dev", "prod", "build"], default="dev",
                       help="Mode to run: dev (development server), prod (production server), build (build only)")
    parser.add_argument("--port", type=int, default=PORT,
                       help=f"Port to run the server on (default: {PORT})")

    args = parser.parse_args()

    global PORT
    PORT = args.port

    print("=" * 60)
    print("Trading Bot Frontend Server")
    print("=" * 60)
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Mode: {args.mode}")
    print(f"Port: {PORT}")
    print("=" * 60)

    if args.mode == "build":
        success = build_frontend()
        sys.exit(0 if success else 1)

    elif args.mode == "dev":
        success = start_dev_server()
        sys.exit(0 if success else 1)

    elif args.mode == "prod":
        success = start_production_server()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()