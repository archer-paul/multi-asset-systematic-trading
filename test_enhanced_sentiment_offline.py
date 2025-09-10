#!/usr/bin/env python3.13
"""
Test offline pour l'analyse de sentiment améliorée
Tests les fonctionnalités sans dépendre de sources externes
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_enhanced_sentiment_offline():
    """Test enhanced sentiment analysis offline"""
    print("📰 Testing Enhanced Sentiment Analysis (Offline)...")
    
    try:
        from analysis.enhanced_sentiment import EnhancedSentimentAnalyzer
        from analysis.sentiment_analyzer import SentimentAnalyzer
        from core.config import Config
        
        config = Config()
        base_sentiment = SentimentAnalyzer(config)
        enhanced_sentiment = EnhancedSentimentAnalyzer(config, base_sentiment)
        
        print("✅ Enhanced sentiment analyzer created")
        
        # Test symbol extraction
        test_text = "Apple (AAPL) stock surged today while Microsoft (MSFT) declined. Tesla earnings beat expectations."
        symbols = enhanced_sentiment._extract_symbols_from_text(test_text)
        print(f"✅ Symbol extraction: {symbols}")
        
        # Test sentiment aggregation with mock data
        from analysis.enhanced_sentiment import SentimentData
        
        mock_sentiment_data = [
            SentimentData(
                source="test_source",
                content="Apple stock looks strong",
                title="AAPL Analysis",
                sentiment_score=0.7,
                confidence=0.8,
                market_impact=0.6,
                urgency=0.5,
                symbols_mentioned=["AAPL"],
                timestamp=datetime.now(),
                url="test.com"
            ),
            SentimentData(
                source="test_source_2", 
                content="Market volatility concerns",
                title="Market Update",
                sentiment_score=-0.3,
                confidence=0.6,
                market_impact=0.7,
                urgency=0.8,
                symbols_mentioned=["AAPL", "MSFT"],
                timestamp=datetime.now(),
                url="test2.com"
            )
        ]
        
        # Test aggregation
        aggregated = enhanced_sentiment.aggregate_sentiment_by_symbol(mock_sentiment_data)
        print(f"✅ Sentiment aggregation: {len(aggregated)} symbols processed")
        
        for symbol, data in aggregated.items():
            print(f"  {symbol}: {data['sentiment_trend']} (confidence: {data['avg_confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced sentiment offline test failed: {e}")
        return False

async def test_feedparser_working():
    """Test that feedparser is working correctly"""
    print("📡 Testing feedparser functionality...")
    
    try:
        import feedparser
        
        # Test with a simple RSS feed
        test_feed_content = '''<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
        <channel>
        <title>Test Feed</title>
        <item>
        <title>Test Article About AAPL</title>
        <description>Apple stock analysis shows positive trends</description>
        <link>https://example.com/article1</link>
        </item>
        <item>
        <title>Market Update MSFT</title>
        <description>Microsoft earnings exceed expectations</description>
        <link>https://example.com/article2</link>
        </item>
        </channel>
        </rss>'''
        
        # Parse the test feed
        feed = feedparser.parse(test_feed_content)
        
        print(f"✅ Feedparser working: {len(feed.entries)} entries parsed")
        
        for entry in feed.entries:
            print(f"  - {entry.title}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feedparser test failed: {e}")
        return False

async def main():
    """Run offline sentiment tests"""
    print("🧪 Enhanced Sentiment - Offline Tests")
    print("=" * 40)
    
    tests = [
        ("Feedparser Functionality", test_feedparser_working),
        ("Enhanced Sentiment Offline", test_enhanced_sentiment_offline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}\n")
        except Exception as e:
            results[test_name] = False
            print(f"❌ ERROR - {test_name}: {e}\n")
    
    # Summary
    print("=" * 40)
    print("🎯 OFFLINE TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nOffline Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 Enhanced sentiment core functionality works!")
        print("💡 Network connectivity issues with RSS feeds are normal")
        print("   The system will work when connected to different networks")
    else:
        print(f"\n⚠️ {total-passed} core functionality test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)