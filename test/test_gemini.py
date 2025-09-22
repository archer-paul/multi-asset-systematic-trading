#!/usr/bin/env python3
"""
Test script to diagnose Gemini AI integration
Run this to check if Gemini is working properly
"""

import os
import asyncio
import logging
from core.config import Config
from analysis.sentiment_analyzer import SentimentAnalyzer

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gemini():
    """Test Gemini AI sentiment analysis"""

    print("=" * 60)
    print("GEMINI AI DIAGNOSTIC TEST")
    print("=" * 60)

    # 1. Check environment variables
    print("\n1. Environment Variables Check:")
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"   GEMINI_API_KEY: {gemini_key[:8]}**** (found)")
    else:
        print("   GEMINI_API_KEY: NOT SET")

    # 2. Check config
    print("\n2. Configuration Check:")
    config = Config()
    if hasattr(config, 'gemini_api_key') and config.gemini_api_key:
        print(f"   Config gemini_api_key: {config.gemini_api_key[:8]}**** (found)")
    else:
        print("   Config gemini_api_key: NOT FOUND")

    # 3. Initialize sentiment analyzer
    print("\n3. Sentiment Analyzer Initialization:")
    analyzer = SentimentAnalyzer(config)

    if analyzer.gemini_client:
        print("   Gemini client: INITIALIZED")
    else:
        print("   Gemini client: NOT INITIALIZED")

    # 4. Test sentiment analysis
    print("\n4. Sentiment Analysis Test:")
    test_text = "Apple reported strong quarterly earnings with revenue growth of 15% and increased iPhone sales."

    try:
        result = await analyzer.analyze_financial_sentiment(
            text=test_text,
            company="Apple",
            symbols=["AAPL"]
        )

        print("   Test successful!")
        print(f"   Sentiment Score: {result['sentiment_score']:.2f}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning'][:100]}...")
        print(f"   Key Themes: {result['key_themes']}")

    except Exception as e:
        print(f"   Test failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed. Check logs above for detailed information.")

if __name__ == "__main__":
    asyncio.run(test_gemini())