"""
Configuration élargie de l'univers de trading
Inclut les principaux indices mondiaux et actions émergentes
"""

# S&P 500 - Top 100 actions les plus importantes par capitalisation
SP500_TOP100 = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
    'CRM', 'ACN', 'ADBE', 'NOW', 'TXN', 'QCOM', 'INTU', 'CSCO', 'AMD', 'IBM',
    
    # Finance
    'BRK.B', 'JPM', 'V', 'MA', 'UNH', 'PG', 'JNJ', 'HD', 'CVX', 'LLY',
    'ABBV', 'AVGO', 'WMT', 'XOM', 'PFE', 'KO', 'COST', 'DIS', 'ABT', 'MRK',
    
    # Consumer & Retail
    'NFLX', 'VZ', 'ADBE', 'PYPL', 'INTC', 'CMCSA', 'PEP', 'T', 'TMUS', 'LOW',
    'MDT', 'UPS', 'QCOM', 'SBUX', 'GILD', 'CVS', 'MMM', 'CAT', 'GS', 'MS',
    
    # Industrial & Energy
    'BA', 'HON', 'UNP', 'GE', 'LMT', 'RTX', 'DE', 'AXP', 'BLK', 'SPGI',
    'BKNG', 'ISRG', 'NOW', 'ZTS', 'SYK', 'TJX', 'BSX', 'REGN', 'SCHW', 'MU',
    
    # Healthcare & Biotech
    'AMGN', 'TMO', 'DHR', 'VRTX', 'CI', 'BIIB', 'MDLZ', 'SO', 'DUK', 'EL',
    'CL', 'NSC', 'ITW', 'APD', 'EOG', 'SHW', 'CSX', 'WM', 'PSA', 'ICE'
]

# FTSE 100 - Actions britanniques principales
FTSE100_SYMBOLS = [
    # Format Yahoo Finance pour UK: [SYMBOL].L
    'SHEL.L', 'AZN.L', 'LSEG.L', 'UU.L', 'ULVR.L', 'RIO.L', 'BP.L', 'HSBA.L',
    'GLEN.L', 'DGE.L', 'NG.L', 'REL.L', 'VOD.L', 'BARC.L', 'BT-A.L', 'LLOY.L',
    'TSCO.L', 'GSK.L', 'BATS.L', 'PRU.L', 'ANTO.L', 'AAL.L', 'RKT.L', 'CRH.L',
    'NXT.L', 'EXPN.L', 'IAG.L', 'FLTR.L', 'JET.L', 'FRAS.L'
]

# CAC 40 - Actions françaises principales 
CAC40_SYMBOLS = [
    # Format Yahoo Finance pour France: [SYMBOL].PA
    'MC.PA', 'ASML.AS', 'SAP.DE', 'OR.PA', 'SAN.PA', 'AI.PA', 'SU.PA', 'BNP.PA',
    'EL.PA', 'RMS.PA', 'CAP.PA', 'ORA.PA', 'BN.PA', 'ML.PA', 'KER.PA', 'PUB.PA',
    'CS.PA', 'ACA.PA', 'DG.PA', 'DSY.PA', 'VK.PA', 'GLE.PA', 'STM.PA', 'RI.PA',
    'TEP.PA', 'WLN.PA', 'URW.PA', 'VIE.PA', 'MMT.PA', 'STLA.PA'
]

# DAX - Actions allemandes principales
DAX_SYMBOLS = [
    # Format Yahoo Finance pour Allemagne: [SYMBOL].DE
    'SAP.DE', 'ASME.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'MBG.DE', 'BMW.DE',
    'BAS.DE', 'VOW3.DE', 'ADS.DE', 'MUV2.DE', 'DB1.DE', 'LIN.DE', 'CON.DE',
    'HEN3.DE', 'IFX.DE', 'FME.DE', 'HEI.DE', 'MRK.DE', 'RWE.DE'
]

# Actions asiatiques majeures - Principaux marchés
ASIA_SYMBOLS = [
    # Japon (Nikkei 225 top)
    '7203.T', '6758.T', '9984.T', '6861.T', '8306.T', '4063.T',  # Toyota, Sony, SoftBank, Keyence, etc.
    
    # Chine / Hong Kong
    'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'TME',  # ADR américains
    '0700.HK', '0941.HK', '1299.HK', '2318.HK', '3690.HK',     # Actions HK directes
    
    # Corée du Sud
    '005930.KS', '000660.KS', '035420.KS',  # Samsung, SK Hynix, Naver
    
    # Taiwan
    'TSM', 'UMC', 'ASX',  # TSMC et autres via ADR
    
    # Inde
    'INFY', 'WIT', 'HDB', 'IBN', 'TCOM'  # Via ADR américains
]

# Actions émergentes à surveiller (petites caps prometteuses)
EMERGING_WATCHLIST = [
    # Biotech émergent
    'MRNA', 'BNTX', 'NVAX', 'SGEN', 'BMRN', 'ALNY', 'PTCT', 'RARE',
    
    # Fintech/Crypto
    'SQ', 'PYPL', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST',
    
    # Clean Energy/EV
    'ENPH', 'SEDG', 'PLUG', 'FCEL', 'BE', 'LCID', 'RIVN',
    
    # Cloud/AI émergent
    'SNOW', 'PLTR', 'AI', 'SOUN', 'BBAI', 'UPST', 'PATH',
    
    # Autres secteurs émergents
    'RBLX', 'UNITY', 'TWLO', 'OKTA', 'ZM', 'DOCN', 'NET'
]

# Configuration par région pour optimisation
REGIONS_CONFIG = {
    'US': {
        'symbols': SP500_TOP100 + EMERGING_WATCHLIST,
        'currency': 'USD',
        'timezone': 'America/New_York',
        'market_hours': (9, 16),  # 9:30 AM - 4:00 PM EST
        'priority': 1  # Priorité la plus haute
    },
    'UK': {
        'symbols': FTSE100_SYMBOLS,
        'currency': 'GBP', 
        'timezone': 'Europe/London',
        'market_hours': (8, 16),  # 8:00 AM - 4:30 PM GMT
        'priority': 2
    },
    'EU': {
        'symbols': CAC40_SYMBOLS + DAX_SYMBOLS,
        'currency': 'EUR',
        'timezone': 'Europe/Paris',
        'market_hours': (9, 17),  # 9:00 AM - 5:30 PM CET
        'priority': 2
    },
    'ASIA': {
        'symbols': ASIA_SYMBOLS,
        'currency': 'USD',  # Plupart via ADR
        'timezone': 'Asia/Tokyo',
        'market_hours': (9, 15),  # 9:00 AM - 3:00 PM JST
        'priority': 3
    }
}

def get_all_symbols():
    """Retourne tous les symboles de trading"""
    all_symbols = []
    for region_config in REGIONS_CONFIG.values():
        all_symbols.extend(region_config['symbols'])
    return list(set(all_symbols))  # Supprime les doublons

def get_symbols_by_region(region):
    """Retourne les symboles pour une région donnée"""
    return REGIONS_CONFIG.get(region, {}).get('symbols', [])

def get_high_priority_symbols():
    """Retourne les symboles prioritaires (US + émergents)"""
    return SP500_TOP100[:50] + EMERGING_WATCHLIST[:20]  # Top 70 pour commencer

def get_emerging_symbols():
    """Retourne uniquement les actions émergentes à surveiller"""
    return EMERGING_WATCHLIST

# Configuration du nombre de symboles à traiter par cycle selon les ressources
PROCESSING_LIMITS = {
    'fast_mode': 50,        # Mode rapide - symboles prioritaires
    'normal_mode': 150,     # Mode normal - toutes les régions importantes  
    'comprehensive_mode': 300  # Mode complet - tout l'univers
}