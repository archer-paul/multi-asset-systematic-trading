"""
Commodities and Forex Analysis Module
Analyzes precious metals, commodities, and forex pairs using free APIs
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CommodityData:
    """Commodity price and analysis data"""
    symbol: str
    name: str
    price: float
    change_24h: float
    change_pct_24h: float
    volume_24h: Optional[float]
    market_cap: Optional[float]
    timestamp: datetime
    source: str

@dataclass
class ForexData:
    """Forex pair data"""
    pair: str
    rate: float
    change_24h: float
    change_pct_24h: float
    bid: Optional[float]
    ask: Optional[float]
    timestamp: datetime
    source: str

class CommoditiesForexAnalyzer:
    """Analyze commodities and forex using free APIs"""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        
        # Commodities to track (available on free APIs)
        self.commodities = {
            'gold': {
                'symbol': 'XAU',
                'name': 'Gold',
                'category': 'precious_metals',
                'keywords': ['gold', 'bullion', 'precious metals', 'safe haven']
            },
            'silver': {
                'symbol': 'XAG', 
                'name': 'Silver',
                'category': 'precious_metals',
                'keywords': ['silver', 'precious metals', 'industrial metals']
            },
            'oil_wti': {
                'symbol': 'CL',
                'name': 'WTI Crude Oil',
                'category': 'energy',
                'keywords': ['oil', 'crude', 'WTI', 'energy', 'petroleum']
            },
            'oil_brent': {
                'symbol': 'BZ',
                'name': 'Brent Crude Oil', 
                'category': 'energy',
                'keywords': ['oil', 'brent', 'crude', 'energy', 'petroleum']
            },
            'copper': {
                'symbol': 'HG',
                'name': 'Copper',
                'category': 'industrial_metals',
                'keywords': ['copper', 'industrial metals', 'construction']
            },
            'bitcoin': {
                'symbol': 'BTC',
                'name': 'Bitcoin',
                'category': 'cryptocurrency',
                'keywords': ['bitcoin', 'crypto', 'digital currency']
            }
        }
        
        # Major forex pairs to track
        self.forex_pairs = {
            'EUR/USD': {
                'base': 'EUR',
                'quote': 'USD', 
                'name': 'Euro/US Dollar',
                'keywords': ['euro', 'dollar', 'ECB', 'Federal Reserve', 'eurozone']
            },
            'GBP/USD': {
                'base': 'GBP',
                'quote': 'USD',
                'name': 'British Pound/US Dollar',
                'keywords': ['pound', 'sterling', 'UK', 'Bank of England', 'brexit']
            },
            'USD/JPY': {
                'base': 'USD', 
                'quote': 'JPY',
                'name': 'US Dollar/Japanese Yen',
                'keywords': ['yen', 'japan', 'Bank of Japan', 'BOJ']
            },
            'USD/CHF': {
                'base': 'USD',
                'quote': 'CHF', 
                'name': 'US Dollar/Swiss Franc',
                'keywords': ['franc', 'switzerland', 'swiss', 'safe haven']
            },
            'AUD/USD': {
                'base': 'AUD',
                'quote': 'USD',
                'name': 'Australian Dollar/US Dollar', 
                'keywords': ['aussie', 'australia', 'commodity currency', 'RBA']
            },
            'USD/CAD': {
                'base': 'USD',
                'quote': 'CAD',
                'name': 'US Dollar/Canadian Dollar',
                'keywords': ['loonie', 'canada', 'commodity currency', 'Bank of Canada']
            }
        }
        
        # Free API endpoints
        self.api_endpoints = {
            'exchange_rates_api': 'https://api.exchangerate-api.com/v4/latest/USD',
            'coinapi_free': 'https://rest.coinapi.io/v1/exchangerate', # Requires free key
            'fixer_free': 'http://data.fixer.io/api/latest', # Requires free key
            'currencylayer_free': 'http://apilayer.net/api/live', # Requires free key
            'alpha_vantage_fx': 'https://www.alphavantage.co/query',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart/',
        }
        
        # Rate limiting
        self.rate_limit_delay = 2.0  # seconds between requests
        
        logger.info(f"Commodities/Forex analyzer initialized")
        logger.info(f"Tracking {len(self.commodities)} commodities and {len(self.forex_pairs)} forex pairs")
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'TradingBot/1.0 (Market Analysis)'
            }
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _fetch_yahoo_commodity_data(self, yahoo_symbol: str) -> Optional[Dict]:
        """Fetch commodity data from Yahoo Finance"""
        try:
            url = f"{self.api_endpoints['yahoo_finance']}{yahoo_symbol}"
            params = {
                'interval': '1d',
                'range': '2d'  # Get last 2 days to calculate change
            }
            
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    chart_data = data['chart']['result'][0]
                    meta = chart_data['meta']
                    prices = chart_data['indicators']['quote'][0]
                    
                    # Get latest price
                    latest_close = prices['close'][-1] if prices['close'] else None
                    previous_close = prices['close'][-2] if len(prices['close']) > 1 else latest_close
                    
                    if latest_close:
                        change = latest_close - previous_close if previous_close else 0
                        change_pct = (change / previous_close * 100) if previous_close else 0
                        
                        return {
                            'price': latest_close,
                            'change_24h': change,
                            'change_pct_24h': change_pct,
                            'volume': prices.get('volume', [None])[-1],
                            'currency': meta.get('currency', 'USD'),
                            'timestamp': datetime.now()
                        }
                        
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {yahoo_symbol}: {e}")
        
        return None
    
    async def _fetch_exchange_rates_api_data(self) -> Dict[str, float]:
        """Fetch forex rates from exchangerate-api.com (free)"""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(self.api_endpoints['exchange_rates_api']) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('rates', {})
                    
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {e}")
        
        return {}
    
    async def _fetch_alpha_vantage_fx(self, from_currency: str, to_currency: str) -> Optional[Dict]:
        """Fetch forex data from Alpha Vantage (if API key available)"""
        try:
            if not hasattr(self.config, 'alpha_vantage_key') or not self.config.alpha_vantage_key:
                return None
            
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': self.config.alpha_vantage_key
            }
            
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(self.api_endpoints['alpha_vantage_fx'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    time_series = data.get('Time Series (Daily)', {})
                    if time_series:
                        # Get latest two days
                        dates = sorted(time_series.keys(), reverse=True)
                        latest_date = dates[0]
                        previous_date = dates[1] if len(dates) > 1 else latest_date
                        
                        latest_data = time_series[latest_date]
                        previous_data = time_series[previous_date]
                        
                        latest_close = float(latest_data['4. close'])
                        previous_close = float(previous_data['4. close'])
                        
                        change = latest_close - previous_close
                        change_pct = (change / previous_close * 100) if previous_close else 0
                        
                        return {
                            'rate': latest_close,
                            'change_24h': change,
                            'change_pct_24h': change_pct,
                            'high': float(latest_data['2. high']),
                            'low': float(latest_data['3. low']),
                            'timestamp': datetime.now()
                        }
                        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage FX data for {from_currency}/{to_currency}: {e}")
        
        return None
    
    async def collect_commodity_data(self) -> Dict[str, CommodityData]:
        """Collect current commodity prices"""
        
        if not self.session:
            await self.initialize()
        
        commodity_data = {}
        
        # Yahoo Finance symbols for commodities
        yahoo_symbols = {
            'gold': 'GC=F',  # Gold futures
            'silver': 'SI=F',  # Silver futures 
            'oil_wti': 'CL=F',  # WTI Crude Oil futures
            'oil_brent': 'BZ=F',  # Brent Crude Oil futures
            'copper': 'HG=F',  # Copper futures
        }
        
        logger.info("Collecting commodity price data...")
        
        # Fetch commodity prices from Yahoo Finance
        for commodity_id, commodity_info in self.commodities.items():
            if commodity_id == 'bitcoin':
                continue  # Handle crypto separately
            
            yahoo_symbol = yahoo_symbols.get(commodity_id)
            if not yahoo_symbol:
                continue
            
            try:
                price_data = await self._fetch_yahoo_commodity_data(yahoo_symbol)
                
                if price_data:
                    commodity_data[commodity_id] = CommodityData(
                        symbol=commodity_info['symbol'],
                        name=commodity_info['name'],
                        price=price_data['price'],
                        change_24h=price_data['change_24h'],
                        change_pct_24h=price_data['change_pct_24h'],
                        volume_24h=price_data.get('volume'),
                        market_cap=None,
                        timestamp=price_data['timestamp'],
                        source='yahoo_finance'
                    )
                    
                    logger.info(f"Collected {commodity_info['name']}: ${price_data['price']:.2f} "
                              f"({price_data['change_pct_24h']:+.2f}%)")
                
            except Exception as e:
                logger.error(f"Error collecting data for {commodity_info['name']}: {e}")
        
        # Add Bitcoin (using Yahoo Finance BTC-USD)
        try:
            btc_data = await self._fetch_yahoo_commodity_data('BTC-USD')
            if btc_data:
                commodity_data['bitcoin'] = CommodityData(
                    symbol='BTC',
                    name='Bitcoin',
                    price=btc_data['price'],
                    change_24h=btc_data['change_24h'], 
                    change_pct_24h=btc_data['change_pct_24h'],
                    volume_24h=btc_data.get('volume'),
                    market_cap=None,
                    timestamp=btc_data['timestamp'],
                    source='yahoo_finance'
                )
                
                logger.info(f"Collected Bitcoin: ${btc_data['price']:.2f} "
                          f"({btc_data['change_pct_24h']:+.2f}%)")
                
        except Exception as e:
            logger.error(f"Error collecting Bitcoin data: {e}")
        
        logger.info(f"Commodity data collection completed: {len(commodity_data)} commodities")
        return commodity_data
    
    async def collect_forex_data(self) -> Dict[str, ForexData]:
        """Collect current forex rates"""
        
        if not self.session:
            await self.initialize()
        
        forex_data = {}
        
        logger.info("Collecting forex rate data...")
        
        # First, try free exchange rates API
        exchange_rates = await self._fetch_exchange_rates_api_data()
        
        if exchange_rates:
            # Calculate rates for our pairs
            for pair_name, pair_info in self.forex_pairs.items():
                base = pair_info['base']
                quote = pair_info['quote']
                
                try:
                    if base == 'USD':
                        # USD is base, rate is direct from API
                        rate = exchange_rates.get(quote)
                    elif quote == 'USD': 
                        # USD is quote, rate is inverse
                        base_rate = exchange_rates.get(base)
                        rate = 1 / base_rate if base_rate else None
                    else:
                        # Cross rate calculation
                        base_rate = exchange_rates.get(base)
                        quote_rate = exchange_rates.get(quote)
                        rate = base_rate / quote_rate if (base_rate and quote_rate) else None
                    
                    if rate:
                        # Note: Free API doesn't provide 24h changes, set to 0
                        forex_data[pair_name] = ForexData(
                            pair=pair_name,
                            rate=rate,
                            change_24h=0.0,  # Not available in free API
                            change_pct_24h=0.0,  # Not available in free API
                            bid=None,
                            ask=None,
                            timestamp=datetime.now(),
                            source='exchangerate_api'
                        )
                        
                        logger.info(f"Collected {pair_name}: {rate:.4f}")
                
                except Exception as e:
                    logger.error(f"Error calculating rate for {pair_name}: {e}")
        
        # Try Alpha Vantage for more detailed data (if API key available)
        if hasattr(self.config, 'alpha_vantage_key') and self.config.alpha_vantage_key:
            logger.info("Fetching detailed forex data from Alpha Vantage...")
            
            for pair_name, pair_info in self.forex_pairs.items():
                try:
                    base = pair_info['base']
                    quote = pair_info['quote']
                    
                    av_data = await self._fetch_alpha_vantage_fx(base, quote)
                    
                    if av_data:
                        forex_data[pair_name] = ForexData(
                            pair=pair_name,
                            rate=av_data['rate'],
                            change_24h=av_data['change_24h'],
                            change_pct_24h=av_data['change_pct_24h'], 
                            bid=None,
                            ask=None,
                            timestamp=av_data['timestamp'],
                            source='alpha_vantage'
                        )
                        
                        logger.info(f"Updated {pair_name}: {av_data['rate']:.4f} "
                                  f"({av_data['change_pct_24h']:+.2f}%)")
                
                except Exception as e:
                    logger.error(f"Error fetching Alpha Vantage data for {pair_name}: {e}")
        
        logger.info(f"Forex data collection completed: {len(forex_data)} pairs")
        return forex_data
    
    def analyze_commodity_trends(self, commodity_data: Dict[str, CommodityData]) -> Dict[str, Any]:
        """Analyze commodity price trends and correlations"""
        
        analysis = {
            'precious_metals': {
                'trend': 'neutral',
                'avg_change': 0.0,
                'commodities': []
            },
            'energy': {
                'trend': 'neutral', 
                'avg_change': 0.0,
                'commodities': []
            },
            'industrial_metals': {
                'trend': 'neutral',
                'avg_change': 0.0,
                'commodities': []
            },
            'cryptocurrency': {
                'trend': 'neutral',
                'avg_change': 0.0, 
                'commodities': []
            },
            'market_signals': []
        }
        
        # Group by category and calculate trends
        for commodity_id, data in commodity_data.items():
            commodity_info = self.commodities[commodity_id]
            category = commodity_info['category']
            
            if category in analysis:
                analysis[category]['commodities'].append({
                    'name': data.name,
                    'symbol': data.symbol,
                    'change_pct': data.change_pct_24h,
                    'price': data.price
                })
        
        # Calculate category trends
        for category, info in analysis.items():
            if category == 'market_signals':
                continue
                
            if info['commodities']:
                changes = [c['change_pct'] for c in info['commodities']]
                avg_change = sum(changes) / len(changes)
                info['avg_change'] = avg_change
                
                if avg_change > 1.0:
                    info['trend'] = 'bullish'
                elif avg_change < -1.0:
                    info['trend'] = 'bearish'
                else:
                    info['trend'] = 'neutral'
        
        # Generate market signals
        signals = []
        
        # Gold vs Dollar correlation
        if 'gold' in commodity_data:
            gold_change = commodity_data['gold'].change_pct_24h
            if gold_change > 2.0:
                signals.append({
                    'type': 'safe_haven_demand',
                    'description': f'Gold up {gold_change:.1f}% - potential risk-off sentiment',
                    'impact': 'bearish_stocks'
                })
            elif gold_change < -2.0:
                signals.append({
                    'type': 'risk_on_sentiment', 
                    'description': f'Gold down {gold_change:.1f}% - potential risk-on sentiment',
                    'impact': 'bullish_stocks'
                })
        
        # Oil price impact
        oil_changes = []
        for commodity_id in ['oil_wti', 'oil_brent']:
            if commodity_id in commodity_data:
                oil_changes.append(commodity_data[commodity_id].change_pct_24h)
        
        if oil_changes:
            avg_oil_change = sum(oil_changes) / len(oil_changes)
            if avg_oil_change > 3.0:
                signals.append({
                    'type': 'energy_price_spike',
                    'description': f'Oil up {avg_oil_change:.1f}% - potential inflationary pressure', 
                    'impact': 'mixed_stocks'
                })
            elif avg_oil_change < -3.0:
                signals.append({
                    'type': 'energy_price_drop',
                    'description': f'Oil down {avg_oil_change:.1f}% - potential deflationary pressure',
                    'impact': 'mixed_stocks'
                })
        
        analysis['market_signals'] = signals
        analysis['analysis_timestamp'] = datetime.now()
        
        return analysis
    
    def analyze_forex_trends(self, forex_data: Dict[str, ForexData]) -> Dict[str, Any]:
        """Analyze forex trends and their market implications"""
        
        analysis = {
            'dollar_strength': {
                'trend': 'neutral',
                'score': 0.0,
                'pairs_analyzed': []
            },
            'major_moves': [],
            'market_implications': [],
            'analysis_timestamp': datetime.now()
        }
        
        # Calculate USD strength
        usd_changes = []
        for pair_name, data in forex_data.items():
            pair_info = self.forex_pairs[pair_name]
            
            # Determine USD direction
            if pair_info['base'] == 'USD':
                # USD is base - positive change means USD strength
                usd_impact = data.change_pct_24h
            else:
                # USD is quote - negative change means USD strength  
                usd_impact = -data.change_pct_24h
            
            usd_changes.append(usd_impact)
            analysis['dollar_strength']['pairs_analyzed'].append({
                'pair': pair_name,
                'change': data.change_pct_24h,
                'usd_impact': usd_impact
            })
        
        # Calculate overall USD strength
        if usd_changes:
            avg_usd_strength = sum(usd_changes) / len(usd_changes)
            analysis['dollar_strength']['score'] = avg_usd_strength
            
            if avg_usd_strength > 0.5:
                analysis['dollar_strength']['trend'] = 'strengthening'
            elif avg_usd_strength < -0.5:
                analysis['dollar_strength']['trend'] = 'weakening'
            else:
                analysis['dollar_strength']['trend'] = 'neutral'
        
        # Identify major moves
        for pair_name, data in forex_data.items():
            if abs(data.change_pct_24h) > 1.0:  # > 1% move is significant for forex
                analysis['major_moves'].append({
                    'pair': pair_name,
                    'change_pct': data.change_pct_24h,
                    'rate': data.rate,
                    'direction': 'up' if data.change_pct_24h > 0 else 'down'
                })
        
        # Market implications
        implications = []
        
        dollar_trend = analysis['dollar_strength']['trend']
        if dollar_trend == 'strengthening':
            implications.append({
                'type': 'dollar_strength',
                'description': 'USD strengthening - potential headwind for US exports and commodities',
                'impact_on_stocks': 'mixed',
                'impact_on_commodities': 'bearish'
            })
        elif dollar_trend == 'weakening':
            implications.append({
                'type': 'dollar_weakness',
                'description': 'USD weakening - potential tailwind for US exports and commodities',
                'impact_on_stocks': 'mixed',
                'impact_on_commodities': 'bullish'
            })
        
        # EUR/USD specific implications
        if 'EUR/USD' in forex_data:
            eur_change = forex_data['EUR/USD'].change_pct_24h
            if abs(eur_change) > 0.5:
                implications.append({
                    'type': 'eur_usd_move',
                    'description': f'EUR/USD moved {eur_change:+.2f}% - watch European vs US market performance',
                    'impact_on_stocks': 'regional'
                })
        
        analysis['market_implications'] = implications
        
        return analysis
    
    async def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive commodities and forex analysis"""
        
        logger.info("Starting comprehensive commodities and forex analysis...")
        
        try:
            # Collect data in parallel
            commodity_task = self.collect_commodity_data()
            forex_task = self.collect_forex_data()
            
            commodity_data, forex_data = await asyncio.gather(
                commodity_task, forex_task, return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(commodity_data, Exception):
                logger.error(f"Commodity data collection failed: {commodity_data}")
                commodity_data = {}
            
            if isinstance(forex_data, Exception):
                logger.error(f"Forex data collection failed: {forex_data}")
                forex_data = {}
            
            # Analyze trends
            commodity_analysis = self.analyze_commodity_trends(commodity_data)
            forex_analysis = self.analyze_forex_trends(forex_data)
            
            # Combine analysis
            comprehensive_analysis = {
                'commodities': {
                    'data': commodity_data,
                    'analysis': commodity_analysis
                },
                'forex': {
                    'data': forex_data,
                    'analysis': forex_analysis
                },
                'cross_asset_signals': self._generate_cross_asset_signals(
                    commodity_data, forex_data, commodity_analysis, forex_analysis
                ),
                'collection_timestamp': datetime.now(),
                'success': True
            }
            
            logger.info("Comprehensive analysis completed successfully")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'commodities': {'data': {}, 'analysis': {}},
                'forex': {'data': {}, 'analysis': {}},
                'cross_asset_signals': [],
                'error': str(e),
                'success': False
            }
    
    def _generate_cross_asset_signals(self, commodity_data, forex_data, 
                                    commodity_analysis, forex_analysis) -> List[Dict[str, Any]]:
        """Generate signals from cross-asset relationships"""
        
        signals = []
        
        try:
            # USD vs Gold relationship
            if ('gold' in commodity_data and 
                'EUR/USD' in forex_data and 
                forex_analysis['dollar_strength']['trend'] != 'neutral'):
                
                gold_change = commodity_data['gold'].change_pct_24h
                usd_trend = forex_analysis['dollar_strength']['trend']
                
                if usd_trend == 'strengthening' and gold_change < -1.0:
                    signals.append({
                        'type': 'usd_gold_correlation',
                        'description': 'USD strength + Gold weakness - confirms dollar trend',
                        'confidence': 0.7,
                        'market_impact': 'bearish_precious_metals'
                    })
                elif usd_trend == 'weakening' and gold_change > 1.0:
                    signals.append({
                        'type': 'usd_gold_correlation', 
                        'description': 'USD weakness + Gold strength - confirms dollar trend',
                        'confidence': 0.7,
                        'market_impact': 'bullish_precious_metals'
                    })
            
            # Oil vs USD/CAD relationship (Canada is oil exporter)
            if ('oil_wti' in commodity_data and 'USD/CAD' in forex_data):
                oil_change = commodity_data['oil_wti'].change_pct_24h
                usd_cad_change = forex_data['USD/CAD'].change_pct_24h
                
                # Oil up should strengthen CAD (USD/CAD down)
                if oil_change > 2.0 and usd_cad_change < -0.3:
                    signals.append({
                        'type': 'oil_cad_correlation',
                        'description': 'Oil strength + CAD strength - consistent with commodity currency relationship',
                        'confidence': 0.6,
                        'market_impact': 'positive_commodity_currencies'
                    })
            
            # Risk-on/Risk-off signals
            risk_indicators = []
            
            # Gold as safe haven
            if 'gold' in commodity_data:
                gold_change = commodity_data['gold'].change_pct_24h
                if gold_change > 1.5:
                    risk_indicators.append('risk_off')
                elif gold_change < -1.5:
                    risk_indicators.append('risk_on')
            
            # USD strength as potential risk-off
            usd_trend = forex_analysis.get('dollar_strength', {}).get('trend')
            if usd_trend == 'strengthening':
                risk_indicators.append('risk_off')
            elif usd_trend == 'weakening':
                risk_indicators.append('risk_on')
            
            # Generate risk sentiment signal
            if len(risk_indicators) >= 2:
                most_common = max(set(risk_indicators), key=risk_indicators.count)
                if most_common == 'risk_off':
                    signals.append({
                        'type': 'risk_sentiment',
                        'description': 'Multiple indicators suggest risk-off sentiment',
                        'confidence': 0.6,
                        'market_impact': 'bearish_risk_assets'
                    })
                elif most_common == 'risk_on':
                    signals.append({
                        'type': 'risk_sentiment',
                        'description': 'Multiple indicators suggest risk-on sentiment',
                        'confidence': 0.6,
                        'market_impact': 'bullish_risk_assets'
                    })
        
        except Exception as e:
            logger.error(f"Error generating cross-asset signals: {e}")
        
        return signals