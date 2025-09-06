-- Setup PostgreSQL for Trading Bot
-- Run this script as postgres user: psql -U postgres -f setup_postgres.sql

-- Grant privileges to trading_user
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
ALTER USER trading_user CREATEDB;
ALTER USER trading_user WITH PASSWORD '0012';

-- Connect to trading_bot database
\c trading_bot;

-- Grant schema privileges
GRANT ALL PRIVILEGES ON SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;

-- Show database encoding info
SELECT datname, encoding, datcollate, datctype FROM pg_database WHERE datname='trading_bot';