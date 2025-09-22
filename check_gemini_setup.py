#!/usr/bin/env python3
"""
Quick diagnostic to check Gemini AI setup
Run this to verify your GEMINI_API_KEY is working
"""

import os
import sys
from dotenv import load_dotenv

def check_gemini_setup():
    """Quick check for Gemini AI setup"""

    # Load .env file first
    load_dotenv()

    print("Gemini AI Setup Diagnostic")
    print("=" * 40)
    print("Loading .env file...")

    # Check 1: Environment variable
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"[OK] GEMINI_API_KEY found: {gemini_key[:8]}****")
    else:
        print("[ERROR] GEMINI_API_KEY not found in environment")
        print("  Set it with: $env:GEMINI_API_KEY='your-api-key-here'")
        return False

    # Check 2: API key format
    if gemini_key.startswith('AI') and len(gemini_key) > 35:
        print("[OK] API key format looks correct")
    else:
        print("[ERROR] API key format may be incorrect")
        print("  Should start with 'AI' and be ~39 characters")
        return False

    # Check 3: Try importing google.generativeai
    try:
        import google.generativeai as genai
        print("[OK] google-generativeai package available")
    except ImportError:
        print("[ERROR] google-generativeai package not installed")
        print("  Install with: pip install google-generativeai")
        return False

    # Check 4: Test basic connection (optional)
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("[OK] Gemini AI client can be initialized")

        # Optional: Test a simple generation
        print("\nTesting API call...")
        response = model.generate_content("Say 'API working' if you can read this.")
        if response and response.text:
            print(f"[OK] API test successful: {response.text.strip()}")
        else:
            print("[WARNING] API responded but no text returned")

    except Exception as e:
        print(f"[ERROR] Gemini AI test failed: {e}")
        print("  Check your API key and network connection")
        return False

    print("\n" + "=" * 40)
    print("SUCCESS: Gemini AI setup looks good!")
    print("Your trading bot should now use AI sentiment analysis.")
    return True

if __name__ == "__main__":
    success = check_gemini_setup()
    sys.exit(0 if success else 1)