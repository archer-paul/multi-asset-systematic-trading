@echo off
echo ============================================================
echo CORRECTION DE LA CONFIGURATION POSTGRESQL
echo ============================================================

REM Définir les variables d'environnement
set PGCLIENTENCODING=UTF8
set LC_ALL=C
set LANG=C

echo.
echo 1. Configuration des privileges utilisateur...
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;"
psql -U postgres -c "ALTER USER trading_user CREATEDB;"
psql -U postgres -c "ALTER USER trading_user WITH PASSWORD '0012';"

echo.
echo 2. Configuration des privileges sur le schema...
psql -U postgres -d trading_bot -c "GRANT ALL PRIVILEGES ON SCHEMA public TO trading_user;"
psql -U postgres -d trading_bot -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;"
psql -U postgres -d trading_bot -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;"
psql -U postgres -d trading_bot -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;"

echo.
echo 3. Test de connexion avec trading_user...
psql -U trading_user -d trading_bot -c "SELECT 'Connection OK' as status;"

echo.
echo Configuration PostgreSQL terminee.
echo Vous pouvez maintenant tester le bot.
pause