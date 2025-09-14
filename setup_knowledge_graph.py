#!/usr/bin/env python3
"""
Setup script for Knowledge Graph installation and initialization
Run this script to set up the knowledge graph system in your trading bot
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for the setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'networkx',
        'google-generativeai',
        'flask',
        'flask-socketio'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logging.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logging.error(f"✗ {package} is missing")

    if missing_packages:
        logging.error(f"Missing packages: {', '.join(missing_packages)}")
        logging.info("Please run: pip install -r requirements.txt")
        return False

    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'knowledge_graph',
        'exports',
        'logs'
    ]

    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        logging.info(f"✓ Directory created/verified: {dir_path}")

async def test_knowledge_graph():
    """Test the knowledge graph system"""
    try:
        from knowledge_graph import EconomicKnowledgeGraph
        from core.config import Config

        logging.info("Testing Knowledge Graph initialization...")

        # Create a test configuration
        config = Config()

        # Initialize the knowledge graph
        kg = EconomicKnowledgeGraph(config)

        # Test basic functionality
        stats = kg.get_graph_statistics()
        logging.info(f"✓ Knowledge Graph initialized with {stats['nodes']} nodes and {stats['edges']} edges")

        # Test cascade analysis with a simple event
        test_event = {
            'type': 'policy_change',
            'entity': 'FED',
            'magnitude': 0.3
        }

        cascade_result = await kg.analyze_cascading_effects(test_event)
        logging.info(f"✓ Cascade analysis completed with {cascade_result['total_effects']} effects detected")

        # Test export functionality
        export_data = kg.export_for_visualization()
        logging.info(f"✓ Export functionality working - {len(export_data['nodes'])} nodes ready for visualization")

        logging.info("✅ Knowledge Graph system test completed successfully!")
        return True

    except Exception as e:
        logging.error(f"❌ Knowledge Graph test failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def test_api_integration():
    """Test API integration"""
    try:
        from knowledge_graph.kg_api import kg_api
        from flask import Flask

        logging.info("Testing Knowledge Graph API integration...")

        # Create a test Flask app
        app = Flask(__name__)
        app.register_blueprint(kg_api)

        logging.info("✓ Knowledge Graph API blueprint registered successfully")

        # List registered routes
        kg_routes = []
        for rule in app.url_map.iter_rules():
            if rule.rule.startswith('/api/knowledge-graph'):
                kg_routes.append(rule.rule)

        logging.info(f"✓ Knowledge Graph API routes registered: {len(kg_routes)} endpoints")
        for route in kg_routes[:5]:  # Show first 5 routes
            logging.info(f"  - {route}")

        logging.info("✅ API integration test completed successfully!")
        return True

    except Exception as e:
        logging.error(f"❌ API integration test failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("""
🎯 Knowledge Graph Setup Complete!

📋 Next Steps:
1. Start your trading bot with: python enhanced_main.py
2. Access the Knowledge Graph at: http://localhost:5000/knowledge-graph
3. Or integrate with your existing frontend

🔧 API Endpoints Available:
- GET  /api/knowledge-graph/status
- GET  /api/knowledge-graph/entities
- GET  /api/knowledge-graph/visualization-data
- POST /api/knowledge-graph/analyze-cascade
- GET  /api/knowledge-graph/search

📖 Features:
✓ Interactive graph visualization with vis-network
✓ Real-time cascade effect analysis
✓ Economic entity relationship mapping
✓ WebSocket support for live updates
✓ Filtering and search capabilities
✓ Export functionality (PNG, JSON)

🔗 Frontend Integration:
The system is integrated with your Next.js frontend at /knowledge-graph

🎨 Visualization Features:
- Interactive node-link diagram
- Entity filtering by type and region
- Real-time cascade analysis
- Detailed entity information panels
- Path finding between entities
- Export capabilities

💡 Example Usage:
1. Open the knowledge graph interface
2. Apply filters to focus on specific entity types (companies, countries, etc.)
3. Click on entities to see detailed information
4. Use cascade analysis to simulate economic events
5. Export visualizations for presentations

🐛 Troubleshooting:
- Check logs in the logs/ directory
- Ensure all dependencies are installed: pip install -r requirements.txt
- Verify your Gemini API key is set in .env for AI features
""")

async def main():
    """Main setup function"""
    setup_logging()

    print("🚀 Setting up Knowledge Graph System for Trading Bot")
    print("=" * 60)

    # Step 1: Check dependencies
    logging.info("Step 1: Checking dependencies...")
    if not check_dependencies():
        logging.error("❌ Setup failed - missing dependencies")
        return 1

    # Step 2: Create directories
    logging.info("Step 2: Creating directories...")
    create_directories()

    # Step 3: Test knowledge graph
    logging.info("Step 3: Testing Knowledge Graph system...")
    if not await test_knowledge_graph():
        logging.error("❌ Setup failed - Knowledge Graph test failed")
        return 1

    # Step 4: Test API integration
    logging.info("Step 4: Testing API integration...")
    if not test_api_integration():
        logging.error("❌ Setup failed - API integration test failed")
        return 1

    # Step 5: Show usage instructions
    logging.info("Step 5: Setup completed!")
    show_usage_instructions()

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)