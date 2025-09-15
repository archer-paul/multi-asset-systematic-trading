"""
Congress Trading Data Analysis Module
Utilise Finnhub.io et Senate Trading APIs pour suivre les transactions des membres du Congrès américain
qui surperforment historiquement les fonds traditionnels
"""

import logging
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class CongressTradingAnalyzer:
    """Analyse des transactions des membres du Congrès américain via Finnhub.io et Senate Trading APIs"""
    
    def __init__(self, config):
        self.config = config
        
        # API disponible: Finnhub uniquement
        self.finnhub_key = getattr(config, 'finnhub_key', None) or getattr(config, 'FINNHUB_KEY', None)
        
        # URLs des APIs
        self.finnhub_base_url = "https://finnhub.io/api/v1"
        
        # Configuration
        self.lookback_days = getattr(config, 'congress_lookback_days', 90)
        self.min_transaction_value = getattr(config, 'min_congress_transaction', 15000)  # $15k minimum
        
        # Statut des APIs
        self.apis_available = {
            'finnhub': bool(self.finnhub_key)
        }
        
        if not self.finnhub_key:
            logger.warning("No Finnhub API key found. Congress analysis will be disabled.")
    
    async def get_recent_congress_trades(self, days: int = None) -> List[Dict[str, Any]]:
        """Récupère les transactions récentes du Congrès depuis l'API Finnhub"""
        
        if not self.finnhub_key:
            logger.error("No Finnhub API key configured - skipping Congress analysis")
            return []
        
        days = days or self.lookback_days
        all_trades = []
        
        # Récupérer depuis Finnhub Congress endpoint
        try:
            finnhub_trades = await self._get_finnhub_congress_trades(days)
            all_trades.extend(finnhub_trades)
            logger.info(f"Retrieved {len(finnhub_trades)} trades from Finnhub Congress API")
        except Exception as e:
            logger.error(f"Finnhub Congress API error: {e}")
        
        # Filtrer par valeur minimum (déjà fait dans le parsing)
        if all_trades:
            logger.info(f"Total Congress trades retrieved: {len(all_trades)}")
            return all_trades
        else:
            logger.warning("No Congress trading data retrieved from Finnhub")
            return []
    
    async def _get_finnhub_congress_trades(self, days: int) -> List[Dict[str, Any]]:
        """Récupère les transactions du Congrès depuis Finnhub.io"""
        
        if not self.finnhub_key:
            raise ValueError("Finnhub API key not configured")
        
        # Endpoint spécifique Congress Trading de Finnhub
        url = f"{self.finnhub_base_url}/stock/congress-trading"
        headers = {'X-Finnhub-Token': self.finnhub_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d')
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        trades = []
                        
                        for trade in data.get('data', []):
                            parsed_trade = self._parse_finnhub_congress_trade(trade)
                            if parsed_trade:
                                trades.append(parsed_trade)
                        
                        return trades
                        
                    elif response.status == 403:
                        logger.error("Finnhub Congress API: Access forbidden - may require premium subscription")
                        return []
                    elif response.status == 429:
                        logger.warning("Finnhub rate limit reached")
                        return []
                    else:
                        logger.warning(f"Finnhub Congress API returned {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Finnhub Congress data: {e}")
            return []
    
    
    def _is_recent_trade(self, trade: Dict, cutoff_date: datetime) -> bool:
        """Vérifie si une transaction est récente"""
        try:
            # Format de date possible: "2024-01-15" ou "2024-01-15T00:00:00"
            trade_date_str = trade.get('TransactionDate') or trade.get('Date')
            if not trade_date_str:
                return False
            
            # Parse la date
            if 'T' in trade_date_str:
                trade_date = datetime.fromisoformat(trade_date_str.split('T')[0])
            else:
                trade_date = datetime.fromisoformat(trade_date_str)
            
            return trade_date >= cutoff_date
        
        except Exception:
            return False
    
    def _parse_finnhub_congress_trade(self, trade: Dict) -> Optional[Dict[str, Any]]:
        """Parse une transaction du Congrès depuis Finnhub"""
        try:
            # Extraire les informations importantes depuis l'API Congress Trading de Finnhub
            symbol = trade.get('symbol', '').strip().upper()
            if not symbol:
                return None
            
            representative = trade.get('name', '').strip()
            transaction_type = trade.get('transactionType', '').strip().upper()  # BUY/SELL
            
            # Valeur de la transaction depuis Finnhub Congress API
            transaction_value = float(trade.get('amount', 0))
            
            if not transaction_value or transaction_value < self.min_transaction_value:
                return None
            
            return {
                'symbol': symbol,
                'representative': representative,
                'transaction_type': transaction_type,
                'transaction_value': transaction_value,
                'transaction_date': trade.get('transactionDate', ''),
                'party': trade.get('party', ''),
                'house': trade.get('house', ''),  # House or Senate
                'source': 'finnhub_congress'
            }
        
        except Exception as e:
            logger.debug(f"Error parsing Finnhub Congress trade: {e}")
            return None
    
    def _parse_transaction_amount(self, amount_str: str) -> Optional[float]:
        """Parse le montant d'une transaction (format: '$15,001 - $50,000')"""
        if not amount_str:
            return None
        
        try:
            # Nettoyer la chaîne
            amount_clean = amount_str.replace('$', '').replace(',', '')
            
            if ' - ' in amount_clean:
                # Range format: "15001 - 50000"
                parts = amount_clean.split(' - ')
                if len(parts) == 2:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    return (min_val + max_val) / 2  # Moyenne
            
            elif amount_clean.replace('.', '').isdigit():
                # Montant exact
                return float(amount_clean)
            
            return None
        
        except Exception:
            return None
    
    async def analyze_congress_signals(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyse les signaux basés sur les transactions du Congrès"""
        
        trades = await self.get_recent_congress_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'signals': {},
                'top_stocks': [],
                'summary': 'No Congress trading data available'
            }
        
        # Analyser les signaux
        signals = {}
        buy_activity = {}
        sell_activity = {}
        
        for trade in trades:
            symbol = trade['symbol']
            
            # Filtrer par symboles si spécifié
            if symbols and symbol not in symbols:
                continue
            
            if symbol not in signals:
                signals[symbol] = {
                    'symbol': symbol,
                    'buy_transactions': [],
                    'sell_transactions': [],
                    'total_buy_value': 0,
                    'total_sell_value': 0,
                    'net_activity': 0,
                    'representatives_count': set()
                }
            
            # Ajouter la transaction
            if trade['transaction_type'] in ['BUY', 'PURCHASE']:
                signals[symbol]['buy_transactions'].append(trade)
                signals[symbol]['total_buy_value'] += trade['transaction_value']
                buy_activity[symbol] = buy_activity.get(symbol, 0) + 1
            
            elif trade['transaction_type'] in ['SELL', 'SALE']:
                signals[symbol]['sell_transactions'].append(trade)
                signals[symbol]['total_sell_value'] += trade['transaction_value']
                sell_activity[symbol] = sell_activity.get(symbol, 0) + 1
            
            signals[symbol]['representatives_count'].add(trade['representative'])
        
        # Calculer les signaux nets
        for symbol, data in signals.items():
            data['net_activity'] = data['total_buy_value'] - data['total_sell_value']
            data['representatives_count'] = len(data['representatives_count'])
        
        # Identifier les top opportunités
        top_buys = sorted(
            [s for s in signals.values() if s['net_activity'] > 0],
            key=lambda x: x['net_activity'],
            reverse=True
        )[:10]
        
        return {
            'total_trades': len(trades),
            'signals': signals,
            'top_stocks': top_buys,
            'buy_activity_count': sum(buy_activity.values()),
            'sell_activity_count': sum(sell_activity.values()),
            'summary': f"Analyzed {len(trades)} Congress trades across {len(signals)} symbols"
        }
    
    async def get_congress_sentiment_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Obtient le sentiment du Congrès pour un symbole spécifique"""
        
        analysis = await self.analyze_congress_signals([symbol])
        
        if symbol not in analysis['signals']:
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'activity_score': 0,
                'details': 'No Congress trading activity found'
            }
        
        data = analysis['signals'][symbol]
        
        # Calculer le sentiment
        net_value = data['net_activity']
        total_value = data['total_buy_value'] + data['total_sell_value']
        
        if total_value == 0:
            sentiment = 'neutral'
            confidence = 0.0
        elif net_value > 0:
            sentiment = 'bullish'
            confidence = min(1.0, abs(net_value) / total_value)
        else:
            sentiment = 'bearish' 
            confidence = min(1.0, abs(net_value) / total_value)
        
        # Score d'activité basé sur le nombre de représentants et la valeur
        activity_score = min(100, data['representatives_count'] * 10 + (total_value / 50000))
        
        return {
            'symbol': symbol,
            'sentiment': sentiment,
            'confidence': confidence,
            'activity_score': activity_score,
            'net_value': net_value,
            'total_trades': len(data['buy_transactions']) + len(data['sell_transactions']),
            'representatives_count': data['representatives_count'],
            'details': f"${net_value:,.0f} net activity from {data['representatives_count']} representatives"
        }
    
    async def get_trending_congress_stocks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtient les actions tendance selon l'activité du Congrès"""
        
        analysis = await self.analyze_congress_signals()
        
        # Calculer un score de tendance
        trending_stocks = []
        
        for symbol, data in analysis['signals'].items():
            total_value = data['total_buy_value'] + data['total_sell_value']
            if total_value < self.min_transaction_value:
                continue
            
            # Score basé sur: valeur nette, nombre de représentants, récence
            trend_score = (
                data['net_activity'] / 1000 +  # Valeur nette en milliers
                data['representatives_count'] * 5 +  # Nombre de représentants
                (data['total_buy_value'] + data['total_sell_value']) / 10000  # Activité totale
            )
            
            trending_stocks.append({
                'symbol': symbol,
                'trend_score': trend_score,
                'net_activity': data['net_activity'],
                'representatives_count': data['representatives_count'],
                'total_value': total_value,
                'sentiment': 'bullish' if data['net_activity'] > 0 else 'bearish' if data['net_activity'] < 0 else 'neutral'
            })
        
        # Trier par score de tendance
        trending_stocks.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return trending_stocks[:limit]
    
    
    
    
    def _deduplicate_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Supprime les doublons entre les APIs"""
        seen = set()
        unique_trades = []
        
        for trade in trades:
            # Créer une clé unique basée sur les champs importants
            key = (
                trade.get('symbol', ''),
                trade.get('representative', ''),
                trade.get('transaction_date', ''),
                trade.get('transaction_value', 0)
            )
            
            if key not in seen:
                seen.add(key)
                unique_trades.append(trade)
        
        return unique_trades
    
    def get_api_status(self) -> Dict[str, Any]:
        """Vérifie le statut des APIs"""
        return {
            'apis_configured': self.apis_available,
            'finnhub_available': self.apis_available['finnhub'],
            'lookback_days': self.lookback_days,
            'min_transaction_value': self.min_transaction_value,
            'status': 'ready' if any(self.apis_available.values()) else 'no_apis_configured'
        }