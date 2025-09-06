#!/usr/bin/env python3.13
"""
Simple PostgreSQL test with better error handling
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

def test_postgresql_simple():
    """Test PostgreSQL connection with simpler approach"""
    try:
        import psycopg2
        
        # Try direct connection parameters
        print("Testing PostgreSQL with individual parameters...")
        
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading_bot',
            user='trading_user',
            password='0012'
        )
        
        print("[SUCCESS] PostgreSQL connection successful!")
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test;")
        result = cursor.fetchone()
        print(f"Test query result: {result[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL error: {e}")
        print(f"Error code: {e.pgcode}")
        return False
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return False

if __name__ == "__main__":
    test_postgresql_simple()