#!/usr/bin/env python3.13
"""
Test PostgreSQL with encoding fixes
"""

import os
import sys
from pathlib import Path

def test_postgresql_fixed():
    """Test PostgreSQL connection with encoding fixes"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # Set environment variables for proper encoding
        os.environ['PGCLIENTENCODING'] = 'utf8'
        
        print("Testing PostgreSQL with encoding fixes...")
        
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading_bot',
            user='trading_user',
            password='0012',
            client_encoding='utf8'
        )
        
        # Set connection encoding explicitly
        conn.set_client_encoding('UTF8')
        conn.autocommit = True
        
        print("[SUCCESS] PostgreSQL connection successful!")
        
        # Test a simple query
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT version() as version_info;")
        result = cursor.fetchone()
        print(f"PostgreSQL version: {result['version_info'][:50]}...")
        
        # Test table creation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100)
            );
        """)
        print("Test table created successfully")
        
        # Clean up test table
        cursor.execute("DROP TABLE IF EXISTS test_table;")
        print("Test table cleaned up")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL error: {e}")
        if hasattr(e, 'pgcode'):
            print(f"Error code: {e.pgcode}")
        return False
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return False

if __name__ == "__main__":
    success = test_postgresql_fixed()
    if success:
        print("\n[SUCCESS] PostgreSQL is working correctly!")
    else:
        print("\n[FAILED] PostgreSQL connection failed")
    sys.exit(0 if success else 1)