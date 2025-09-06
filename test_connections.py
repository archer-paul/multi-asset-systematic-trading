#!/usr/bin/env python3.13
"""
Test script to verify PostgreSQL and Redis connections
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

def test_postgresql():
    """Test PostgreSQL connection"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        database_url = os.getenv('DATABASE_URL')
        print(f"Testing PostgreSQL connection: {database_url}")
        
        if not database_url:
            print("[ERROR] No DATABASE_URL configured")
            return False
            
        # Parse URL
        url = database_url.replace('postgresql://', '')
        auth, location = url.split('@')
        user, password = auth.split(':')
        host_port, database = location.split('/')
        host, port = host_port.split(':') if ':' in host_port else (host_port, '5432')
        
        db_config = {
            'user': user,
            'password': password,
            'host': host,
            'port': int(port),
            'database': database,
            'client_encoding': 'utf8'
        }
        
        conn = psycopg2.connect(**db_config, cursor_factory=RealDictCursor)
        conn.autocommit = True
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print("[SUCCESS] PostgreSQL connection successful")
        print(f"   Version: {version}")
        return True
        
    except Exception as e:
        print(f"[ERROR] PostgreSQL connection failed: {e}")
        return False

def test_redis():
    """Test Redis connection"""
    try:
        import redis
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        print(f"Testing Redis connection: {redis_url}")
        
        # Parse URL
        url = redis_url.replace('redis://', '')
        host, port = url.split(':') if ':' in url else (url, '6379')
        
        redis_config = {
            'host': host,
            'port': int(port),
            'decode_responses': True
        }
        
        client = redis.Redis(**redis_config)
        response = client.ping()
        
        # Test set/get
        client.set('test_key', 'test_value')
        value = client.get('test_key')
        client.delete('test_key')
        
        info = client.info('server')
        redis_version = info.get('redis_version', 'Unknown')
        
        client.close()
        
        print("[SUCCESS] Redis connection successful")
        print(f"   Version: {redis_version}")
        print(f"   Test set/get: {value}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Redis connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Database Connection Tests")
    print("=" * 50)
    
    postgres_ok = test_postgresql()
    print()
    redis_ok = test_redis()
    
    print()
    print("=" * 50)
    print("Summary:")
    print(f"PostgreSQL: {'[OK]' if postgres_ok else '[FAILED]'}")
    print(f"Redis:      {'[OK]' if redis_ok else '[FAILED]'}")
    
    if postgres_ok and redis_ok:
        print("\n[SUCCESS] All database connections are working!")
        return 0
    else:
        print("\n[WARNING] Some database connections failed. Check configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())