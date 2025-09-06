#!/usr/bin/env python3
"""
Setup script for Advanced Trading Bot
Handles installation, configuration, and environment setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any
import click

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        click.echo(click.style("Error: Python 3.8 or higher is required", fg='red'))
        sys.exit(1)
    
    click.echo(click.style(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected", fg='green'))

def install_requirements():
    """Install Python requirements"""
    click.echo("Installing Python dependencies...")
    
    try:
        # Install core requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        click.echo(click.style("✓ Core dependencies installed", fg='green'))
        
        # Install optional social media dependencies
        social_deps = ["tweepy>=4.14.0", "praw>=7.6.0"]
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + social_deps)
            click.echo(click.style("✓ Social media dependencies installed", fg='green'))
        except subprocess.CalledProcessError:
            click.echo(click.style("⚠ Some social media dependencies failed to install", fg='yellow'))
        
    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"Error installing dependencies: {e}", fg='red'))
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'data',
        'data/cache',
        'data/models',
        'data/reports',
        'data/backups',
        'core',
        'analysis',
        'ml',
        'trading',
        'analytics'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python packages
        if directory in ['core', 'analysis', 'ml', 'trading', 'analytics']:
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('"""Trading bot module"""')
    
    click.echo(click.style("✓ Created project directories", fg='green'))

def create_env_template():
    """Create .env template file"""
    env_template = """# Trading Bot Environment Variables
# Copy this to .env and fill in your API keys

# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# News APIs (at least one recommended)
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
FINNHUB_KEY=your_finnhub_key_here

# Social Media APIs (optional)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=TradingBot/1.0

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost/trading_bot
REDIS_URL=redis://localhost:6379

# Bot Configuration
LOG_LEVEL=INFO
DEBUG_MODE=False
MAX_CYCLES=0
BACKTEST_MODE=False

# Trading Parameters
INITIAL_CAPITAL=10000.0
ENABLE_SOCIAL_SENTIMENT=False
"""
    
    env_file = Path('.env.template')
    env_file.write_text(env_template)
    
    if not Path('.env').exists():
        click.echo(click.style("✓ Created .env.template", fg='green'))
        click.echo(click.style("Please copy .env.template to .env and configure your API keys", fg='yellow'))
    else:
        click.echo(click.style("✓ .env file already exists", fg='green'))

def setup_database():
    """Setup database (optional)"""
    click.echo("\nDatabase setup (optional):")
    
    use_db = click.confirm("Do you want to set up PostgreSQL database?", default=False)
    
    if use_db:
        # Check if PostgreSQL is available
        try:
            subprocess.check_output(['psql', '--version'])
            click.echo(click.style("✓ PostgreSQL found", fg='green'))
            
            # Create database
            db_name = click.prompt("Database name", default="trading_bot")
            db_user = click.prompt("Database user", default="trading_user")
            
            click.echo(f"Please create the database manually:")
            click.echo(f"  createdb {db_name}")
            click.echo(f"  createuser {db_user}")
            click.echo(f"Then update DATABASE_URL in .env file")
            
        except FileNotFoundError:
            click.echo(click.style("PostgreSQL not found. You can install it later.", fg='yellow'))
    
    # Redis setup
    use_redis = click.confirm("Do you want to set up Redis for caching?", default=False)
    
    if use_redis:
        try:
            subprocess.check_output(['redis-cli', 'ping'])
            click.echo(click.style("✓ Redis is running", fg='green'))
        except (FileNotFoundError, subprocess.CalledProcessError):
            click.echo(click.style("Redis not found or not running. You can install it later.", fg='yellow'))

def validate_api_keys():
    """Validate API keys"""
    click.echo("\nValidating API configuration...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = {
        'GEMINI_API_KEY': 'Required for sentiment analysis',
        'NEWS_API_KEY': 'Recommended for news data',
        'ALPHA_VANTAGE_KEY': 'Recommended for financial data',
    }
    
    for key, description in api_keys.items():
        value = os.getenv(key)
        if value and value != f'your_{key.lower()}_here':
            click.echo(click.style(f"✓ {key} is configured", fg='green'))
        else:
            if key == 'GEMINI_API_KEY':
                click.echo(click.style(f"✗ {key} is required - {description}", fg='red'))
            else:
                click.echo(click.style(f"⚠ {key} not configured - {description}", fg='yellow'))

def run_tests():
    """Run basic tests"""
    click.echo("\nRunning basic tests...")
    
    try:
        # Test imports
        from core.config import Config
        from core.utils import setup_logging, create_directories
        
        # Test configuration
        config = Config()
        click.echo(click.style("✓ Configuration loading works", fg='green'))
        
        # Test logging
        setup_logging()
        click.echo(click.style("✓ Logging setup works", fg='green'))
        
        # Test utilities
        create_directories()
        click.echo(click.style("✓ Directory creation works", fg='green'))
        
    except ImportError as e:
        click.echo(click.style(f"✗ Import error: {e}", fg='red'))
        return False
    except Exception as e:
        click.echo(click.style(f"✗ Test failed: {e}", fg='red'))
        return False
    
    return True

def create_systemd_service():
    """Create systemd service file (Linux only)"""
    if sys.platform != 'linux':
        return
    
    if not click.confirm("Create systemd service for auto-start? (Linux only)", default=False):
        return
    
    service_content = f"""[Unit]
Description=Advanced Trading Bot
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'trading')}
WorkingDirectory={project_root}
Environment=PATH={project_root}/venv/bin
ExecStart={sys.executable} main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path('/tmp/trading-bot.service')
    service_file.write_text(service_content)
    
    click.echo(f"Service file created at {service_file}")
    click.echo("To install:")
    click.echo(f"  sudo cp {service_file} /etc/systemd/system/")
    click.echo("  sudo systemctl daemon-reload")
    click.echo("  sudo systemctl enable trading-bot.service")

@click.group()
def cli():
    """Advanced Trading Bot Setup Tool"""
    pass

@cli.command()
def install():
    """Full installation process"""
    click.echo(click.style("🤖 Advanced Trading Bot Setup", fg='blue', bold=True))
    click.echo("=" * 50)
    
    # Check requirements
    check_python_version()
    
    # Install dependencies
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Create configuration template
    create_env_template()
    
    # Setup database (optional)
    setup_database()
    
    # Create systemd service (optional)
    create_systemd_service()
    
    click.echo("\n" + "=" * 50)
    click.echo(click.style("✓ Installation completed!", fg='green', bold=True))
    click.echo("\nNext steps:")
    click.echo("1. Configure API keys in .env file")
    click.echo("2. Run: python setup.py validate")
    click.echo("3. Run: python main.py")

@cli.command()
def validate():
    """Validate installation and configuration"""
    click.echo(click.style("🔍 Validating Trading Bot Setup", fg='blue', bold=True))
    click.echo("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Validate API keys
    validate_api_keys()
    
    # Run tests
    if run_tests():
        click.echo("\n" + click.style("✓ All validations passed!", fg='green', bold=True))
        click.echo("Your trading bot is ready to run!")
    else:
        click.echo("\n" + click.style("✗ Some validations failed", fg='red', bold=True))
        click.echo("Please fix the issues before running the bot")

@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to test')
@click.option('--cycles', '-c', default=1, help='Number of test cycles')
def test():
    """Run trading bot in test mode"""
    click.echo(click.style("🧪 Running Trading Bot Test", fg='blue', bold=True))
    
    # Set environment variables for testing
    os.environ['MAX_CYCLES'] = str(cycles)
    os.environ['DEBUG_MODE'] = 'True'
    
    try:
        from main import main
        import asyncio
        
        click.echo("Starting test run...")
        asyncio.run(main())
        
    except KeyboardInterrupt:
        click.echo("\nTest interrupted by user")
    except Exception as e:
        click.echo(click.style(f"Test failed: {e}", fg='red'))

@cli.command()
def clean():
    """Clean up generated files and caches"""
    click.echo("Cleaning up...")
    
    # Directories to clean
    cleanup_dirs = ['logs', 'data/cache', '__pycache__']
    
    for directory in cleanup_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            if dir_path.is_file():
                dir_path.unlink()
            else:
                shutil.rmtree(dir_path)
            click.echo(f"Cleaned {directory}")
    
    # Clean Python cache files
    for cache_dir in Path('.').rglob('__pycache__'):
        shutil.rmtree(cache_dir)
    
    for pyc_file in Path('.').rglob('*.pyc'):
        pyc_file.unlink()
    
    click.echo(click.style("✓ Cleanup completed", fg='green'))

@cli.command()
def update():
    """Update dependencies"""
    click.echo("Updating dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"
        ])
        click.echo(click.style("✓ Dependencies updated", fg='green'))
    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"Update failed: {e}", fg='red'))

@cli.command()
def status():
    """Check trading bot status"""
    click.echo(click.style("📊 Trading Bot Status", fg='blue', bold=True))
    click.echo("=" * 40)
    
    # Check if bot is running
    import psutil
    
    bot_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'main.py' in ' '.join(proc.info['cmdline'] or []):
                bot_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if bot_processes:
        click.echo(click.style(f"✓ Bot is running ({len(bot_processes)} processes)", fg='green'))
        for proc in bot_processes:
            click.echo(f"  PID: {proc.info['pid']}")
    else:
        click.echo(click.style("○ Bot is not running", fg='yellow'))
    
    # Check log files
    log_file = Path('logs/trading_bot.log')
    if log_file.exists():
        size_mb = log_file.stat().st_size / (1024 * 1024)
        click.echo(f"Log file: {size_mb:.1f} MB")
    
    # Check data directories
    data_dirs = ['data', 'data/cache', 'data/models']
    for directory in data_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            file_count = len(list(dir_path.iterdir()))
            click.echo(f"{directory}: {file_count} files")

if __name__ == "__main__":
    cli()