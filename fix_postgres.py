#!/usr/bin/env python3.13
"""
Script pour corriger la configuration PostgreSQL
"""

import os
import subprocess
import sys
from pathlib import Path

def run_psql_command(command, as_user="postgres"):
    """Exécuter une commande psql"""
    try:
        # Définir les variables d'environnement pour éviter les problèmes d'encodage
        env = os.environ.copy()
        env['PGCLIENTENCODING'] = 'UTF8'
        env['LC_ALL'] = 'C'
        env['LANG'] = 'C'
        
        cmd = ['psql', '-U', as_user, '-c', command]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"[SUCCESS] {command}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"[ERROR] {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}")
        return False

def fix_postgres():
    """Corriger la configuration PostgreSQL"""
    print("=" * 60)
    print("CORRECTION DE LA CONFIGURATION POSTGRESQL")
    print("=" * 60)
    
    # 1. Vérifier l'encodage de la base de données
    print("\n1. Vérification de l'encodage de la base trading_bot...")
    run_psql_command("SELECT datname, encoding, datcollate, datctype FROM pg_database WHERE datname='trading_bot';")
    
    # 2. Donner tous les privilèges à l'utilisateur
    print("\n2. Configuration des privilèges utilisateur...")
    commands = [
        "GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;",
        "ALTER USER trading_user CREATEDB;",
        "ALTER USER trading_user WITH PASSWORD '0012';"
    ]
    
    for cmd in commands:
        run_psql_command(cmd)
    
    # 3. Se connecter à la base trading_bot et configurer les privilèges sur le schéma
    print("\n3. Configuration des privilèges sur le schéma public...")
    db_commands = [
        "GRANT ALL PRIVILEGES ON SCHEMA public TO trading_user;",
        "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;",
        "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;"
    ]
    
    for cmd in db_commands:
        run_psql_command(f"\\c trading_bot && {cmd}")
    
    # 4. Test de connexion avec l'utilisateur trading_user
    print("\n4. Test de connexion avec trading_user...")
    test_success = run_psql_command("SELECT version();", as_user="trading_user")
    
    if test_success:
        print("\n[SUCCESS] PostgreSQL configuré avec succès !")
    else:
        print("\n[ERROR] Des problèmes persistent avec PostgreSQL")
        print("\nSolutions alternatives :")
        print("1. Essayez de recréer la base avec l'encodage UTF8 :")
        print("   createdb -E UTF8 -l C --template=template0 trading_bot_new")
        print("2. Ou définissez les variables d'environnement :")
        print("   set PGCLIENTENCODING=UTF8")
        print("   set LC_ALL=C")
    
    return test_success

def create_postgres_config_file():
    """Créer un fichier de configuration PostgreSQL pour le bot"""
    config_content = """
# Configuration PostgreSQL pour le Trading Bot
# Variables d'environnement recommandées

set PGCLIENTENCODING=UTF8
set LC_ALL=C
set LANG=C

# Si les problèmes persistent, utilisez ces commandes :
# dropdb trading_bot
# createdb -E UTF8 -l C --template=template0 trading_bot
# psql -U postgres -d trading_bot -c "GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;"
"""
    
    config_file = Path(__file__).parent / "postgres_config.txt"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Configuration sauvegardée dans : {config_file}")

if __name__ == "__main__":
    try:
        success = fix_postgres()
        create_postgres_config_file()
        
        if success:
            print("\n[SUCCESS] Configuration PostgreSQL terminée avec succès !")
            print("Vous pouvez maintenant tester le bot.")
        else:
            print("\n[WARNING] Configuration PostgreSQL incomplète.")
            print("Le bot peut fonctionner sans PostgreSQL, mais certaines fonctionnalités seront limitées.")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Configuration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n[ERROR] Erreur lors de la configuration : {e}")
        
    input("\nAppuyez sur Entrée pour continuer...")