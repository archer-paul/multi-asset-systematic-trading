"""
Economic Knowledge Graph - Système complet de graphe de connaissance économique
Analyse les relations complexes entre entités économiques et prédit les effets de cascade
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import requests
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

logger = logging.getLogger(__name__)

class RelationType(Enum):
    """Types de relations dans le knowledge graph"""
    TRADE_DEPENDENCY = "trade_dependency"
    SUPPLY_CHAIN = "supply_chain"
    POLITICAL_ALLIANCE = "political_alliance"
    ECONOMIC_PARTNERSHIP = "economic_partnership"
    CURRENCY_CORRELATION = "currency_correlation"
    COMMODITY_DEPENDENCY = "commodity_dependency"
    INSTITUTIONAL_CONTROL = "institutional_control"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    SECTOR_CORRELATION = "sector_correlation"
    LEADERSHIP_INFLUENCE = "leadership_influence"

class EntityType(Enum):
    """Types d'entités dans le knowledge graph"""
    COMPANY = "company"
    COUNTRY = "country"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    POLITICIAN = "politician"
    INSTITUTION = "institution"
    SECTOR = "sector"
    EVENT = "event"

@dataclass
class Entity:
    """Entité du knowledge graph"""
    id: str
    name: str
    type: EntityType
    region: str
    metadata: Dict[str, Any]
    importance_score: float = 0.0

class Relationship:
    """Relation entre deux entités"""
    def __init__(self, source_id: str, target_id: str, relation_type: RelationType,
                 strength: float, metadata: Dict[str, Any] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.strength = strength  # 0.0 to 1.0
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

@dataclass
class CascadeEffect:
    """Effet de cascade calculé"""
    origin_entity: str
    affected_entity: str
    impact_magnitude: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    propagation_path: List[str]
    time_horizon: str  # 'immediate', 'short_term', 'medium_term', 'long_term'
    explanation: str

class EconomicKnowledgeGraph:
    """Graphe de connaissance pour relations économiques complexes"""

    def __init__(self, config=None):
        self.config = config
        self.graph = nx.MultiDiGraph()

        # Entités organisées par type
        self.entities = {
            'companies': {},      # Entreprises et leurs secteurs
            'countries': {},      # Pays et leurs indicateurs
            'commodities': {},    # Matières premières
            'currencies': {},     # Devises
            'politicians': {},    # Dirigeants politiques
            'institutions': {},   # Institutions (FMI, BCE, etc.)
            'sectors': {},        # Secteurs économiques
            'events': {}          # Événements économiques/politiques
        }

        # Relations organisées par type
        self.relationships = {
            'trade_dependencies': {},   # Dépendances commerciales
            'supply_chains': {},       # Chaînes d'approvisionnement
            'political_alliances': {}, # Alliances politiques
            'economic_partnerships': {}, # Partenariats économiques
            'currency_correlations': {}, # Corrélations devises
            'commodity_dependencies': {}, # Dépendances matières premières
            'institutional_controls': {}, # Contrôles institutionnels
            'geographic_proximities': {}, # Proximités géographiques
            'sector_correlations': {},   # Corrélations sectorielles
            'leadership_influences': {}  # Influences de leadership
        }

        # Cache pour les effets de cascade
        self.cascade_cache = {}
        self.cache_timeout = timedelta(hours=1)

        # IA pour l'analyse contextuelle
        if config and hasattr(config, 'gemini_api_key') and config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.ai_model = genai.GenerativeModel('gemini-pro')
        else:
            self.ai_model = None
            logger.warning("Gemini API key not configured - AI analysis disabled")

        # Données externes
        self.external_apis = {
            'world_bank': 'https://api.worldbank.org/v2/',
            'trading_economics': 'https://api.tradingeconomics.com/',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'finnhub': 'https://finnhub.io/api/v1/',
            'news_api': 'https://newsapi.org/v2/'
        }

        # Initialiser le graphe avec des données de base
        self._initialize_base_knowledge()

    def _initialize_base_knowledge(self):
        """Initialise le graphe avec des connaissances de base"""

        # Entreprises majeures avec leurs secteurs et régions
        major_companies = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'country': 'US', 'market_cap': 3000},
            'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Technology', 'country': 'US', 'market_cap': 2800},
            'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'country': 'US', 'market_cap': 1800},
            'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Technology', 'country': 'US', 'market_cap': 1800},
            'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive', 'country': 'US', 'market_cap': 800},
            'ASML': {'name': 'ASML Holding', 'sector': 'Technology', 'country': 'NL', 'market_cap': 300},
            'SAP': {'name': 'SAP SE', 'sector': 'Technology', 'country': 'DE', 'market_cap': 150},
            'LVMH': {'name': 'LVMH', 'sector': 'Luxury', 'country': 'FR', 'market_cap': 400},
            'TSM': {'name': 'Taiwan Semi', 'sector': 'Technology', 'country': 'TW', 'market_cap': 500}
        }

        for symbol, data in major_companies.items():
            self.add_entity(
                id=symbol,
                name=data['name'],
                entity_type=EntityType.COMPANY,
                region=data['country'],
                metadata={
                    'sector': data['sector'],
                    'market_cap_billion': data['market_cap'],
                    'symbol': symbol
                }
            )

        # Pays majeurs avec leurs indicateurs
        major_countries = {
            'US': {'name': 'United States', 'gdp': 25000, 'population': 330, 'currency': 'USD'},
            'CN': {'name': 'China', 'gdp': 17000, 'population': 1400, 'currency': 'CNY'},
            'DE': {'name': 'Germany', 'gdp': 4200, 'population': 83, 'currency': 'EUR'},
            'JP': {'name': 'Japan', 'gdp': 4200, 'population': 125, 'currency': 'JPY'},
            'FR': {'name': 'France', 'gdp': 2900, 'population': 68, 'currency': 'EUR'},
            'NL': {'name': 'Netherlands', 'gdp': 1000, 'population': 17, 'currency': 'EUR'},
            'TW': {'name': 'Taiwan', 'gdp': 800, 'population': 23, 'currency': 'TWD'}
        }

        for code, data in major_countries.items():
            self.add_entity(
                id=code,
                name=data['name'],
                entity_type=EntityType.COUNTRY,
                region=code,
                metadata={
                    'gdp_billion': data['gdp'],
                    'population_million': data['population'],
                    'primary_currency': data['currency']
                }
            )

        # Devises principales
        major_currencies = {
            'USD': {'name': 'US Dollar', 'strength': 1.0, 'volatility': 0.1},
            'EUR': {'name': 'Euro', 'strength': 0.9, 'volatility': 0.15},
            'JPY': {'name': 'Japanese Yen', 'strength': 0.007, 'volatility': 0.2},
            'CNY': {'name': 'Chinese Yuan', 'strength': 0.14, 'volatility': 0.3},
            'GBP': {'name': 'British Pound', 'strength': 1.25, 'volatility': 0.25}
        }

        for code, data in major_currencies.items():
            self.add_entity(
                id=code,
                name=data['name'],
                entity_type=EntityType.CURRENCY,
                region='GLOBAL',
                metadata={
                    'relative_strength': data['strength'],
                    'volatility_index': data['volatility']
                }
            )

        # Matières premières
        major_commodities = {
            'CRUDE_OIL': {'name': 'Crude Oil', 'unit': 'barrel', 'volatility': 0.4},
            'GOLD': {'name': 'Gold', 'unit': 'ounce', 'volatility': 0.2},
            'WHEAT': {'name': 'Wheat', 'unit': 'bushel', 'volatility': 0.5},
            'NATURAL_GAS': {'name': 'Natural Gas', 'unit': 'mmbtu', 'volatility': 0.6},
            'COPPER': {'name': 'Copper', 'unit': 'pound', 'volatility': 0.3},
            'LITHIUM': {'name': 'Lithium', 'unit': 'ton', 'volatility': 0.7}
        }

        for code, data in major_commodities.items():
            self.add_entity(
                id=code,
                name=data['name'],
                entity_type=EntityType.COMMODITY,
                region='GLOBAL',
                metadata={
                    'unit': data['unit'],
                    'volatility_index': data['volatility']
                }
            )

        # Institutions internationales
        major_institutions = {
            'FED': {'name': 'Federal Reserve', 'type': 'central_bank', 'region': 'US'},
            'ECB': {'name': 'European Central Bank', 'type': 'central_bank', 'region': 'EU'},
            'BOJ': {'name': 'Bank of Japan', 'type': 'central_bank', 'region': 'JP'},
            'IMF': {'name': 'International Monetary Fund', 'type': 'international', 'region': 'GLOBAL'},
            'WTO': {'name': 'World Trade Organization', 'type': 'trade', 'region': 'GLOBAL'},
            'OPEC': {'name': 'Organization of Petroleum Exporting Countries', 'type': 'commodity', 'region': 'GLOBAL'}
        }

        for code, data in major_institutions.items():
            self.add_entity(
                id=code,
                name=data['name'],
                entity_type=EntityType.INSTITUTION,
                region=data['region'],
                metadata={'institution_type': data['type']}
            )

        # Établir les relations de base
        self._establish_base_relationships()

    def _establish_base_relationships(self):
        """Établit les relations de base entre les entités"""

        # Relations entreprise-pays
        company_country_relations = [
            ('AAPL', 'US', 0.9), ('MSFT', 'US', 0.9), ('GOOGL', 'US', 0.9), ('NVDA', 'US', 0.9),
            ('TSLA', 'US', 0.8), ('ASML', 'NL', 0.9), ('SAP', 'DE', 0.9), ('LVMH', 'FR', 0.9),
            ('TSM', 'TW', 0.9)
        ]

        for company, country, strength in company_country_relations:
            self.add_relationship(
                company, country, RelationType.ECONOMIC_PARTNERSHIP, strength,
                {'type': 'headquarters', 'tax_dependency': True}
            )

        # Relations secteur-technologie (chaînes d'approvisionnement)
        tech_supply_chain = [
            ('NVDA', 'TSM', 0.8),  # NVIDIA dépend de Taiwan Semi pour la fabrication
            ('AAPL', 'TSM', 0.7),  # Apple dépend de Taiwan Semi
            ('ASML', 'TSM', 0.9),  # ASML fournit les machines à TSM
            ('ASML', 'NVDA', 0.6)  # ASML impact indirect sur NVIDIA
        ]

        for source, target, strength in tech_supply_chain:
            self.add_relationship(
                source, target, RelationType.SUPPLY_CHAIN, strength,
                {'type': 'semiconductor_supply', 'critical': True}
            )

        # Dépendances aux matières premières
        commodity_dependencies = [
            ('TSLA', 'LITHIUM', 0.8),  # Tesla dépend du lithium
            ('NVDA', 'COPPER', 0.6),   # NVIDIA dépend du cuivre
            ('US', 'CRUDE_OIL', 0.7),  # US dépendance pétrole
            ('DE', 'NATURAL_GAS', 0.8) # Allemagne dépendance gaz
        ]

        for entity, commodity, strength in commodity_dependencies:
            self.add_relationship(
                entity, commodity, RelationType.COMMODITY_DEPENDENCY, strength,
                {'essential': True, 'strategic': True}
            )

        # Relations institutionnelles
        institutional_relations = [
            ('FED', 'USD', 0.9),  # Fed contrôle le dollar
            ('ECB', 'EUR', 0.9),  # BCE contrôle l'euro
            ('BOJ', 'JPY', 0.9),  # BOJ contrôle le yen
            ('OPEC', 'CRUDE_OIL', 0.8)  # OPEC influence le pétrole
        ]

        for institution, target, strength in institutional_relations:
            self.add_relationship(
                institution, target, RelationType.INSTITUTIONAL_CONTROL, strength,
                {'policy_influence': True, 'price_control': True}
            )

        # Corrélations de devises
        currency_correlations = [
            ('EUR', 'USD', -0.7),  # Corrélation négative EUR/USD
            ('JPY', 'USD', -0.5),  # Corrélation négative JPY/USD
            ('CNY', 'USD', -0.6),  # Corrélation négative CNY/USD
            ('GOLD', 'USD', -0.6)  # Or inversement corrélé au dollar
        ]

        for curr1, curr2, correlation in currency_correlations:
            strength = abs(correlation)
            self.add_relationship(
                curr1, curr2, RelationType.CURRENCY_CORRELATION, strength,
                {'correlation_coefficient': correlation, 'inverse': correlation < 0}
            )

    def add_entity(self, id: str, name: str, entity_type: EntityType,
                   region: str, metadata: Dict[str, Any], importance_score: float = 0.0):
        """Ajoute une entité au graphe de connaissance"""
        entity = Entity(id, name, entity_type, region, metadata, importance_score)

        # Stocker dans la catégorie appropriée
        type_key = entity_type.value + 's'
        if type_key in self.entities:
            self.entities[type_key][id] = entity

        # Ajouter au graphe NetworkX
        self.graph.add_node(id, **{
            'name': name,
            'type': entity_type.value,
            'region': region,
            'importance': importance_score,
            **metadata
        })

        logger.debug(f"Added entity: {name} ({id}) of type {entity_type.value}")

    def add_relationship(self, source_id: str, target_id: str,
                        relation_type: RelationType, strength: float,
                        metadata: Dict[str, Any] = None):
        """Ajoute une relation entre deux entités"""
        relationship = Relationship(source_id, target_id, relation_type, strength, metadata)

        # Stocker dans la catégorie appropriée
        type_key = relation_type.value + 's'
        if type_key in self.relationships:
            if source_id not in self.relationships[type_key]:
                self.relationships[type_key][source_id] = []
            self.relationships[type_key][source_id].append(relationship)

        # Ajouter au graphe NetworkX
        self.graph.add_edge(source_id, target_id, **{
            'type': relation_type.value,
            'strength': strength,
            'metadata': metadata or {},
            'created_at': relationship.created_at.isoformat()
        })

        logger.debug(f"Added relationship: {source_id} -> {target_id} ({relation_type.value}, strength: {strength})")

    async def enrich_with_external_data(self):
        """Enrichit le graphe avec des données externes"""
        logger.info("Enriching knowledge graph with external data...")

        tasks = [
            self._fetch_economic_indicators(),
            self._fetch_market_correlations(),
            self._fetch_supply_chain_data(),
            self._fetch_political_events(),
            self._fetch_commodity_dependencies()
        ]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Knowledge graph enrichment completed")
        except Exception as e:
            logger.error(f"Error enriching knowledge graph: {e}")

    async def _fetch_economic_indicators(self):
        """Récupère les indicateurs économiques"""
        try:
            # Utiliser les APIs disponibles dans le bot
            if self.config:
                # Intégration avec le data collector existant
                pass
        except Exception as e:
            logger.debug(f"Could not fetch economic indicators: {e}")

    async def _fetch_market_correlations(self):
        """Calcule les corrélations de marché en temps réel"""
        try:
            # Calculer les corrélations entre actions
            symbols = list(self.entities['companies'].keys())

            # Simuler des corrélations pour la démo
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    # Corrélation simulée basée sur les secteurs
                    sector1 = self.entities['companies'][symbol1].metadata.get('sector', 'Unknown')
                    sector2 = self.entities['companies'][symbol2].metadata.get('sector', 'Unknown')

                    if sector1 == sector2:
                        correlation = 0.6 + np.random.normal(0, 0.2)
                    else:
                        correlation = 0.1 + np.random.normal(0, 0.3)

                    correlation = np.clip(correlation, -1.0, 1.0)

                    if abs(correlation) > 0.3:  # Seulement les corrélations significatives
                        self.add_relationship(
                            symbol1, symbol2, RelationType.SECTOR_CORRELATION,
                            abs(correlation),
                            {'correlation_coefficient': correlation, 'period': '30d'}
                        )

        except Exception as e:
            logger.debug(f"Could not fetch market correlations: {e}")

    async def _fetch_supply_chain_data(self):
        """Récupère les données de chaîne d'approvisionnement"""
        # Ajouter des relations de chaîne d'approvisionnement connues
        known_supply_chains = [
            ('AAPL', 'CN', 0.8, {'type': 'manufacturing', 'percentage': 80}),
            ('TSLA', 'CN', 0.6, {'type': 'battery_supply', 'percentage': 60}),
            ('NVDA', 'TW', 0.9, {'type': 'chip_manufacturing', 'critical': True}),
            ('US', 'CRUDE_OIL', 0.4, {'type': 'energy_import', 'strategic': True}),
            ('DE', 'NATURAL_GAS', 0.6, {'type': 'energy_import', 'strategic': True})
        ]

        for source, target, strength, metadata in known_supply_chains:
            self.add_relationship(source, target, RelationType.SUPPLY_CHAIN, strength, metadata)

    async def _fetch_political_events(self):
        """Récupère les événements politiques récents"""
        # Ajouter des événements politiques significatifs
        political_events = [
            {
                'id': 'US_ELECTION_2024',
                'name': 'US Presidential Election 2024',
                'type': EntityType.EVENT,
                'region': 'US',
                'impact_level': 0.9,
                'affected_entities': ['USD', 'US', 'AAPL', 'MSFT', 'TSLA']
            },
            {
                'id': 'EU_POLICY_TECH',
                'name': 'EU Tech Regulation Package',
                'type': EntityType.EVENT,
                'region': 'EU',
                'impact_level': 0.7,
                'affected_entities': ['GOOGL', 'AAPL', 'MSFT', 'EUR']
            }
        ]

        for event in political_events:
            self.add_entity(
                event['id'], event['name'], event['type'],
                event['region'], {'impact_level': event['impact_level']}
            )

            # Connecter l'événement aux entités affectées
            for entity_id in event['affected_entities']:
                self.add_relationship(
                    event['id'], entity_id, RelationType.LEADERSHIP_INFLUENCE,
                    event['impact_level'], {'event_type': 'political'}
                )

    async def _fetch_commodity_dependencies(self):
        """Met à jour les dépendances aux matières premières"""
        # Ajouter des dépendances détaillées
        detailed_dependencies = [
            ('TSLA', 'LITHIUM', 0.9, {'usage': 'battery_production', 'critical': True}),
            ('NVDA', 'COPPER', 0.5, {'usage': 'chip_production', 'moderate': True}),
            ('AAPL', 'COPPER', 0.4, {'usage': 'device_production', 'moderate': True}),
            ('ASML', 'GOLD', 0.3, {'usage': 'precision_components', 'low': True})
        ]

        for company, commodity, strength, metadata in detailed_dependencies:
            self.add_relationship(company, commodity, RelationType.COMMODITY_DEPENDENCY, strength, metadata)

    async def analyze_cascading_effects(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse des effets de cascade d'un événement"""

        event_type = event.get('type', 'unknown')
        origin_entity = event.get('entity', 'unknown')
        magnitude = event.get('magnitude', 0.5)

        logger.info(f"Analyzing cascading effects for event: {event_type} on {origin_entity}")

        # Vérifier le cache
        cache_key = f"{event_type}_{origin_entity}_{magnitude}".replace(' ', '_')
        if cache_key in self.cascade_cache:
            cache_entry = self.cascade_cache[cache_key]
            if datetime.now() - cache_entry['timestamp'] < self.cache_timeout:
                logger.debug(f"Using cached cascade analysis for {cache_key}")
                return cache_entry['data']

        # Analyser les effets selon le type d'événement
        cascade_effects = []

        if event_type == 'war' or 'conflict' in event_type.lower():
            cascade_effects = await self._analyze_conflict_effects(origin_entity, magnitude)
        elif event_type == 'election' or 'political' in event_type.lower():
            cascade_effects = await self._analyze_political_effects(origin_entity, magnitude)
        elif event_type == 'banking_crisis' or 'financial' in event_type.lower():
            cascade_effects = await self._analyze_financial_crisis_effects(origin_entity, magnitude)
        elif event_type == 'supply_disruption':
            cascade_effects = await self._analyze_supply_disruption_effects(origin_entity, magnitude)
        elif event_type == 'policy_change':
            cascade_effects = await self._analyze_policy_change_effects(origin_entity, magnitude)
        else:
            cascade_effects = await self._analyze_generic_effects(origin_entity, magnitude)

        # Enrichir avec l'analyse IA si disponible
        if self.ai_model:
            cascade_effects = await self._enrich_with_ai_analysis(event, cascade_effects)

        # Organiser les résultats
        result = {
            'event': event,
            'total_effects': len(cascade_effects),
            'immediate_effects': [e for e in cascade_effects if e.time_horizon == 'immediate'],
            'short_term_effects': [e for e in cascade_effects if e.time_horizon == 'short_term'],
            'medium_term_effects': [e for e in cascade_effects if e.time_horizon == 'medium_term'],
            'long_term_effects': [e for e in cascade_effects if e.time_horizon == 'long_term'],
            'all_effects': cascade_effects,
            'analysis_timestamp': datetime.now().isoformat()
        }

        # Mettre en cache
        self.cascade_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now()
        }

        return result

    async def _analyze_conflict_effects(self, origin_entity: str, magnitude: float) -> List[CascadeEffect]:
        """Analyse les effets d'un conflit (ex: guerre en Ukraine)"""
        effects = []

        # Exemples d'effets typiques d'un conflit
        conflict_impacts = {
            'CRUDE_OIL': (0.3, 'Energy supply disruption from conflict zone'),
            'WHEAT': (0.4, 'Agricultural export disruption from conflict region'),
            'NATURAL_GAS': (0.5, 'Energy supply chain disruption'),
            'GOLD': (0.2, 'Safe haven demand increase during geopolitical tension'),
            'USD': (0.1, 'Flight to safety strengthens reserve currency'),
            'EUR': (-0.2, 'Regional currency weakness due to proximity to conflict'),
            'NVDA': (-0.1, 'Supply chain concerns affect tech manufacturing'),
            'TSLA': (-0.15, 'Commodity price increases affect production costs')
        }

        # Calculer les effets propagés
        for entity_id, (base_impact, reason) in conflict_impacts.items():
            if entity_id in [node for node in self.graph.nodes()]:
                adjusted_impact = base_impact * magnitude
                confidence = 0.7 + (magnitude * 0.2)

                # Trouver le chemin de propagation
                try:
                    path = nx.shortest_path(self.graph, origin_entity, entity_id)
                except nx.NetworkXNoPath:
                    path = [origin_entity, entity_id]  # Direct impact

                effect = CascadeEffect(
                    origin_entity=origin_entity,
                    affected_entity=entity_id,
                    impact_magnitude=adjusted_impact,
                    confidence=confidence,
                    propagation_path=path,
                    time_horizon='immediate' if abs(adjusted_impact) > 0.2 else 'short_term',
                    explanation=reason
                )
                effects.append(effect)

        return effects

    async def _analyze_political_effects(self, origin_entity: str, magnitude: float) -> List[CascadeEffect]:
        """Analyse les effets d'un événement politique"""
        effects = []

        if origin_entity == 'US':  # Élection américaine
            political_impacts = {
                'USD': (0.1, 'Policy uncertainty affects currency strength'),
                'AAPL': (0.05, 'Tech policy changes may affect major tech companies'),
                'TSLA': (0.1, 'Clean energy policies impact electric vehicle companies'),
                'CRUDE_OIL': (-0.1, 'Energy policy changes affect oil demand'),
                'NVDA': (0.05, 'Infrastructure spending may benefit chip companies'),
                'CNY': (-0.05, 'Trade policy uncertainty affects China relations')
            }
        elif origin_entity in ['DE', 'FR', 'EU']:  # Événement européen
            political_impacts = {
                'EUR': (0.1, 'European policy changes affect regional currency'),
                'ASML': (0.1, 'EU tech policies affect European tech leaders'),
                'SAP': (0.05, 'Regional policy changes impact local companies'),
                'NATURAL_GAS': (0.2, 'Energy policy affects European energy markets')
            }
        else:
            political_impacts = {}

        for entity_id, (base_impact, reason) in political_impacts.items():
            if entity_id in [node for node in self.graph.nodes()]:
                adjusted_impact = base_impact * magnitude
                confidence = 0.6 + (magnitude * 0.3)

                try:
                    path = nx.shortest_path(self.graph, origin_entity, entity_id)
                except nx.NetworkXNoPath:
                    path = [origin_entity, entity_id]

                effect = CascadeEffect(
                    origin_entity=origin_entity,
                    affected_entity=entity_id,
                    impact_magnitude=adjusted_impact,
                    confidence=confidence,
                    propagation_path=path,
                    time_horizon='medium_term',
                    explanation=reason
                )
                effects.append(effect)

        return effects

    async def _analyze_financial_crisis_effects(self, origin_entity: str, magnitude: float) -> List[CascadeEffect]:
        """Analyse les effets d'une crise financière"""
        effects = []

        # Contagion financière typique
        crisis_impacts = {
            'USD': (0.2, 'Flight to safety strengthens dollar during crisis'),
            'GOLD': (0.3, 'Safe haven demand increases during financial uncertainty'),
            'EUR': (-0.15, 'Regional banking crisis affects currency'),
            'AAPL': (-0.1, 'Credit tightening affects consumer discretionary spending'),
            'TSLA': (-0.2, 'High-growth companies sensitive to credit conditions'),
            'NVDA': (-0.15, 'Capital intensive sector affected by credit tightening'),
            'CRUDE_OIL': (-0.1, 'Economic slowdown reduces energy demand')
        }

        for entity_id, (base_impact, reason) in crisis_impacts.items():
            if entity_id in [node for node in self.graph.nodes()]:
                adjusted_impact = base_impact * magnitude
                confidence = 0.8  # High confidence in financial contagion patterns

                try:
                    path = nx.shortest_path(self.graph, origin_entity, entity_id)
                except nx.NetworkXNoPath:
                    path = [origin_entity, entity_id]

                effect = CascadeEffect(
                    origin_entity=origin_entity,
                    affected_entity=entity_id,
                    impact_magnitude=adjusted_impact,
                    confidence=confidence,
                    propagation_path=path,
                    time_horizon='immediate',
                    explanation=reason
                )
                effects.append(effect)

        return effects

    async def _analyze_supply_disruption_effects(self, origin_entity: str, magnitude: float) -> List[CascadeEffect]:
        """Analyse les effets d'une disruption de supply chain"""
        effects = []

        # Trouver les entités dépendantes
        dependent_entities = []

        for source in self.graph.nodes():
            for target in self.graph.successors(source):
                edge_data = self.graph[source][target]
                for edge_key in edge_data:
                    edge_info = edge_data[edge_key]
                    if (edge_info.get('type') == 'supply_chain' and
                        target == origin_entity and
                        edge_info.get('strength', 0) > 0.3):
                        dependent_entities.append((source, edge_info.get('strength', 0.5)))

        # Calculer l'impact sur les entités dépendantes
        for entity_id, dependency_strength in dependent_entities:
            impact = -magnitude * dependency_strength  # Impact négatif
            confidence = dependency_strength

            path = [origin_entity, entity_id]

            effect = CascadeEffect(
                origin_entity=origin_entity,
                affected_entity=entity_id,
                impact_magnitude=impact,
                confidence=confidence,
                propagation_path=path,
                time_horizon='short_term',
                explanation=f"Supply chain dependency disruption affects {entity_id}"
            )
            effects.append(effect)

        return effects

    async def _analyze_policy_change_effects(self, origin_entity: str, magnitude: float) -> List[CascadeEffect]:
        """Analyse les effets d'un changement de politique"""
        effects = []

        # Identifier les entités affectées par les politiques de cette région
        policy_impacts = {}

        if origin_entity in ['FED', 'ECB', 'BOJ']:  # Politique monétaire
            if origin_entity == 'FED':
                policy_impacts = {
                    'USD': (magnitude * 0.3, 'Monetary policy directly affects currency'),
                    'AAPL': (magnitude * 0.1, 'Interest rate changes affect growth stocks'),
                    'TSLA': (magnitude * 0.15, 'Rate sensitive growth company'),
                    'NVDA': (magnitude * 0.12, 'Growth stock sensitive to rates')
                }
            elif origin_entity == 'ECB':
                policy_impacts = {
                    'EUR': (magnitude * 0.3, 'ECB policy directly affects Euro'),
                    'ASML': (magnitude * 0.1, 'European company affected by regional policy'),
                    'SAP': (magnitude * 0.08, 'Regional monetary policy impact')
                }

        for entity_id, (impact, reason) in policy_impacts.items():
            if entity_id in [node for node in self.graph.nodes()]:
                confidence = 0.75

                path = [origin_entity, entity_id]

                effect = CascadeEffect(
                    origin_entity=origin_entity,
                    affected_entity=entity_id,
                    impact_magnitude=impact,
                    confidence=confidence,
                    propagation_path=path,
                    time_horizon='short_term',
                    explanation=reason
                )
                effects.append(effect)

        return effects

    async def _analyze_generic_effects(self, origin_entity: str, magnitude: float) -> List[CascadeEffect]:
        """Analyse générique basée sur les relations du graphe"""
        effects = []

        # Utiliser la propagation de graphe pour estimer les effets
        if origin_entity in self.graph.nodes():
            # Effet direct sur les voisins
            for neighbor in self.graph.successors(origin_entity):
                edge_data = self.graph[origin_entity][neighbor]
                for edge_key in edge_data:
                    edge_info = edge_data[edge_key]
                    strength = edge_info.get('strength', 0.5)

                    impact = magnitude * strength * 0.5  # Facteur d'atténuation
                    confidence = strength * 0.8

                    path = [origin_entity, neighbor]

                    effect = CascadeEffect(
                        origin_entity=origin_entity,
                        affected_entity=neighbor,
                        impact_magnitude=impact,
                        confidence=confidence,
                        propagation_path=path,
                        time_horizon='short_term',
                        explanation=f"Direct relationship impact via {edge_info.get('type', 'connection')}"
                    )
                    effects.append(effect)

        return effects

    async def _enrich_with_ai_analysis(self, event: Dict[str, Any],
                                     cascade_effects: List[CascadeEffect]) -> List[CascadeEffect]:
        """Enrichit l'analyse avec l'IA pour des insights plus nuancés"""
        if not self.ai_model:
            return cascade_effects

        try:
            # Construire le prompt pour l'IA
            event_description = f"Event: {event.get('type', 'unknown')} affecting {event.get('entity', 'unknown')} with magnitude {event.get('magnitude', 0.5)}"

            effects_summary = "\n".join([
                f"- {effect.affected_entity}: {effect.impact_magnitude:.3f} impact ({effect.explanation})"
                for effect in cascade_effects[:10]  # Limiter pour éviter la surcharge
            ])

            prompt = f"""
            Analyze this economic event and its cascading effects:

            {event_description}

            Current identified effects:
            {effects_summary}

            Based on your knowledge of economic relationships, are there any missing important effects or corrections to the analysis?
            Focus on:
            1. Indirect effects through complex relationships
            2. Time delays in propagation
            3. Confidence adjustments based on historical patterns
            4. Sectoral spill-over effects

            Respond in JSON format with suggestions for additional effects or modifications.
            """

            # Appel à l'IA (avec gestion d'erreur)
            response = await asyncio.to_thread(self.ai_model.generate_content, prompt)

            # Traiter la réponse (simplifié pour la démo)
            ai_insights = response.text
            logger.debug(f"AI analysis insights: {ai_insights}")

            # Pour le moment, retourner les effets originaux
            # Dans une implémentation complète, on parserait les suggestions de l'IA

        except Exception as e:
            logger.debug(f"AI enrichment failed: {e}")

        return cascade_effects

    def get_entity_importance(self, entity_id: str) -> float:
        """Calcule l'importance d'une entité dans le graphe"""
        if entity_id not in self.graph.nodes():
            return 0.0

        # Calculer l'importance basée sur plusieurs facteurs
        factors = {
            'degree_centrality': nx.degree_centrality(self.graph).get(entity_id, 0) * 0.3,
            'betweenness_centrality': nx.betweenness_centrality(self.graph).get(entity_id, 0) * 0.4,
            'pagerank': nx.pagerank(self.graph).get(entity_id, 0) * 0.3
        }

        # Facteur additionnel basé sur les métadonnées
        node_data = self.graph.nodes[entity_id]
        if node_data.get('type') == 'company':
            market_cap = node_data.get('market_cap_billion', 0)
            factors['market_cap'] = min(market_cap / 1000, 1.0) * 0.2
        elif node_data.get('type') == 'country':
            gdp = node_data.get('gdp_billion', 0)
            factors['gdp'] = min(gdp / 25000, 1.0) * 0.2

        importance = sum(factors.values())

        # Mettre à jour le score d'importance
        if entity_id in self.entities.get(node_data.get('type', 'unknown') + 's', {}):
            self.entities[node_data.get('type') + 's'][entity_id].importance_score = importance

        return importance

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le graphe de connaissance"""
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'entities_by_type': {}
        }

        # Compter les entités par type
        for entity_type, entities in self.entities.items():
            stats['entities_by_type'][entity_type] = len(entities)

        # Statistiques des relations
        relation_counts = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            relation_type = data.get('type', 'unknown')
            relation_counts[relation_type] += 1

        stats['relationships_by_type'] = dict(relation_counts)

        # Top entités par importance
        importances = {}
        for node in self.graph.nodes():
            importances[node] = self.get_entity_importance(node)

        stats['top_entities'] = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

        return stats

    def export_for_visualization(self) -> Dict[str, Any]:
        """Exporte le graphe dans un format adapté à la visualisation web"""
        nodes = []
        links = []

        # Préparer les noeuds
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            importance = self.get_entity_importance(node_id)

            node = {
                'id': node_id,
                'name': node_data.get('name', node_id),
                'type': node_data.get('type', 'unknown'),
                'region': node_data.get('region', 'unknown'),
                'importance': importance,
                'size': 10 + (importance * 30),  # Taille basée sur l'importance
                'color': self._get_node_color(node_data.get('type', 'unknown')),
                'metadata': {k: v for k, v in node_data.items()
                           if k not in ['name', 'type', 'region']}
            }
            nodes.append(node)

        # Préparer les liens
        for source, target in self.graph.edges():
            edge_data = self.graph[source][target]
            for edge_key in edge_data:
                edge_info = edge_data[edge_key]

                link = {
                    'source': source,
                    'target': target,
                    'type': edge_info.get('type', 'unknown'),
                    'strength': edge_info.get('strength', 0.5),
                    'width': edge_info.get('strength', 0.5) * 5,
                    'color': self._get_edge_color(edge_info.get('type', 'unknown')),
                    'metadata': edge_info.get('metadata', {})
                }
                links.append(link)

        return {
            'nodes': nodes,
            'links': links,
            'statistics': self.get_graph_statistics(),
            'layout': 'force-directed',  # Recommandation de layout
            'export_timestamp': datetime.now().isoformat()
        }

    def _get_node_color(self, node_type: str) -> str:
        """Retourne une couleur pour un type de noeud"""
        colors = {
            'company': '#3498db',      # Bleu
            'country': '#e74c3c',      # Rouge
            'currency': '#f1c40f',     # Jaune
            'commodity': '#8e44ad',    # Violet
            'institution': '#2ecc71',  # Vert
            'politician': '#ff7f50',   # Orange saumon
            'sector': '#1abc9c',       # Turquoise
            'event': '#e67e22',        # Orange
            'unknown': '#95a5a6'       # Gris
        }
        return colors.get(node_type, colors['unknown'])

    def _get_edge_color(self, edge_type: str) -> str:
        """Retourne une couleur pour un type de lien"""
        colors = {
            'supply_chain': '#e74c3c',           # Rouge - critique
            'trade_dependency': '#f39c12',       # Orange - important
            'economic_partnership': '#3498db',   # Bleu - coopération
            'currency_correlation': '#9b59b6',   # Violet - financier
            'commodity_dependency': '#27ae60',   # Vert - ressources
            'institutional_control': '#34495e',  # Gris foncé - autorité
            'political_alliance': '#e67e22',     # Orange - politique
            'sector_correlation': '#1abc9c',     # Turquoise - marché
            'geographic_proximity': '#95a5a6',   # Gris - géographie
            'leadership_influence': '#c0392b',   # Rouge foncé - influence
            'unknown': '#bdc3c7'                 # Gris clair
        }
        return colors.get(edge_type, colors['unknown'])