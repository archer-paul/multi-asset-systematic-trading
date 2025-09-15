'''
Knowledge Graph Module for Trading Bot

This module is responsible for building, maintaining, and querying a knowledge graph
of economic entities (companies, sectors, countries, events) and their relationships.
It supports cascade analysis to simulate impact propagation.
'''

import logging
from typing import Dict, List, Any, Optional
import networkx as nx
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Manages the economic knowledge graph and its analysis capabilities."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.graph = nx.DiGraph() # Directed graph for relationships
        self.is_initialized = False
        self.last_update = None

        # Placeholder for entities and relationships
        self._mock_entities = {
            "AAPL": {"id": "AAPL", "name": "Apple Inc.", "type": "Company", "region": "US", "importance": 0.9, "metadata": {"sector": "Technology"}},
            "MSFT": {"id": "MSFT", "name": "Microsoft Corp.", "type": "Company", "region": "US", "importance": 0.85, "metadata": {"sector": "Technology"}},
            "US": {"id": "US", "name": "United States", "type": "Country", "region": "North America", "importance": 0.95, "metadata": {}},
            "Technology": {"id": "Technology", "name": "Technology Sector", "type": "Sector", "region": "Global", "importance": 0.9, "metadata": {}},
            "TradeWar": {"id": "TradeWar", "name": "US-China Trade War", "type": "Event", "region": "Global", "importance": 0.7, "metadata": {"start_date": "2018-01-01"}}
        }
        self._mock_relationships = [
            {"source": "AAPL", "target": "Technology", "type": "BELONGS_TO", "strength": 1.0},
            {"source": "MSFT", "target": "Technology", "type": "BELONGS_TO", "strength": 1.0},
            {"source": "AAPL", "target": "US", "type": "OPERATES_IN", "strength": 0.9},
            {"source": "MSFT", "target": "US", "type": "OPERATES_IN", "strength": 0.9},
            {"source": "TradeWar", "target": "AAPL", "type": "IMPACTS", "strength": 0.7},
            {"source": "TradeWar", "target": "MSFT", "type": "IMPACTS", "strength": 0.6}
        ]

    async def initialize(self):
        """Initializes the knowledge graph, loading initial data."""
        logger.info("Initializing Knowledge Graph...")
        # In a real scenario, this would load from a database or external source
        for entity_id, data in self._mock_entities.items():
            self.graph.add_node(entity_id, **data)
        for rel in self._mock_relationships:
            self.graph.add_edge(rel['source'], rel['target'], type=rel['type'], strength=rel['strength'])
        
        self.is_initialized = True
        self.last_update = datetime.now()
        logger.info(f"Knowledge Graph initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    async def get_status(self) -> Dict[str, Any]:
        """Returns current status and statistics of the knowledge graph."""
        if not self.is_initialized:
            return {"status": "not_initialized", "message": "Knowledge Graph not yet initialized."}

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        density = nx.density(self.graph) if nodes > 1 else 0.0
        is_connected = nx.is_connected(self.graph.to_undirected()) if nodes > 0 else False

        entities_by_type = {node_type: sum(1 for n, data in self.graph.nodes(data=True) if data.get('type') == node_type) for node_type in set(nx.get_node_attributes(self.graph, 'type').values())}
        relationships_by_type = {edge_type: sum(1 for u, v, data in self.graph.edges(data=True) if data.get('type') == edge_type) for edge_type in set(nx.get_edge_attributes(self.graph, 'type').values())}

        # Simple top entities by importance
        top_entities = sorted([(data['name'], data['importance']) for n, data in self.graph.nodes(data=True)], key=lambda x: x[1], reverse=True)[:5]

        return {
            "status": "active",
            "last_update": self.last_update.isoformat(),
            "statistics": {
                "nodes": nodes,
                "edges": edges,
                "density": density,
                "is_connected": is_connected,
                "entities_by_type": entities_by_type,
                "relationships_by_type": relationships_by_type,
                "top_entities": top_entities
            }
        }

    async def get_entity_details(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed information about a specific entity and its direct relations."""
        if not self.is_initialized or entity_id not in self.graph:
            return None

        entity_data = self.graph.nodes[entity_id].copy()
        
        outgoing_relations = []
        for neighbor in self.graph.successors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            outgoing_relations.append({
                "target": neighbor,
                "target_name": self.graph.nodes[neighbor].get('name', neighbor),
                "type": edge_data.get('type', 'UNKNOWN'),
                "strength": edge_data.get('strength', 1.0)
            })

        incoming_relations = []
        for neighbor in self.graph.predecessors(entity_id):
            edge_data = self.graph.get_edge_data(neighbor, entity_id)
            incoming_relations.append({
                "source": neighbor,
                "source_name": self.graph.nodes[neighbor].get('name', neighbor),
                "type": edge_data.get('type', 'UNKNOWN'),
                "strength": edge_data.get('strength', 1.0)
            })

        total_connections = len(outgoing_relations) + len(incoming_relations)

        # Centrality metrics (can be computationally intensive for large graphs)
        degree_centrality = nx.degree_centrality(self.graph).get(entity_id, 0.0)
        betweenness_centrality = nx.betweenness_centrality(self.graph).get(entity_id, 0.0)

        entity_data['relations'] = {
            "outgoing": outgoing_relations,
            "incoming": incoming_relations,
            "total_connections": total_connections
        }
        entity_data['centrality_metrics'] = {
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality
        }

        return entity_data

    async def analyze_cascade(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates the propagation of an event through the knowledge graph."""
        if not self.is_initialized:
            return {"error": "Knowledge Graph not initialized."}

        event_type = event_data.get('type')
        entity_id = event_data.get('entity')
        magnitude = event_data.get('magnitude', 1.0)

        if entity_id not in self.graph:
            return {"error": f"Entity {entity_id} not found in graph."}

        logger.info(f"Starting cascade analysis for event {event_type} on {entity_id} with magnitude {magnitude}")

        # Simple BFS-like propagation for demonstration
        impacts = []
        visited = set()
        queue = [(entity_id, magnitude, 0, [entity_id])] # (node, current_magnitude, depth, path)

        while queue:
            current_node, current_mag, depth, path = queue.pop(0)
            if current_node in visited: continue
            visited.add(current_node)

            # Record impact
            if current_node != entity_id: # Don't record the source event as an impact
                impacts.append({
                    "affected_entity": self.graph.nodes[current_node].get('name', current_node),
                    "impact_magnitude": current_mag,
                    "confidence": max(0.1, 1.0 - (depth * 0.2)), # Confidence decreases with depth
                    "time_horizon": self._get_time_horizon(depth),
                    "explanation": f"Impact from {event_type} event on {entity_id} propagated to {current_node}.",
                    "propagation_path": path
                })

            # Propagate to neighbors
            for neighbor in self.graph.successors(current_node):
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                propagation_strength = edge_data.get('strength', 0.5) # How much impact propagates
                
                new_magnitude = current_mag * propagation_strength * (0.8 if edge_data.get('type') == 'BELONGS_TO' else 1.0)
                
                if new_magnitude > 0.05 and neighbor not in visited: # Only propagate significant impacts
                    queue.append((neighbor, new_magnitude, depth + 1, path + [neighbor]))
        
        total_effects = len(impacts)
        effects_by_horizon = {
            'immediate': sum(1 for i in impacts if i['time_horizon'] == 'immediate'),
            'short_term': sum(1 for i in impacts if i['time_horizon'] == 'short_term'),
            'medium_term': sum(1 for i in impacts if i['time_horizon'] == 'medium_term'),
            'long_term': sum(1 for i in impacts if i['time_horizon'] == 'long_term'),
        }

        logger.info(f"Cascade analysis completed. Total effects: {total_effects}")

        return {
            "event": event_data,
            "total_effects": total_effects,
            "effects_by_horizon": effects_by_horizon,
            "top_impacts": sorted(impacts, key=lambda x: abs(x['impact_magnitude']), reverse=True)[:5],
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _get_time_horizon(self, depth: int) -> str:
        """Determines time horizon based on propagation depth."""
        if depth == 0: return 'immediate'
        if depth == 1: return 'short_term'
        if depth == 2: return 'medium_term'
        return 'long_term'

    async def update_graph_from_analysis(self, macro_analysis: Dict[str, Any], geopolitical_risks: Dict[str, Any]):
        """Updates the knowledge graph with new information from analysis modules."""
        logger.info("Updating Knowledge Graph with latest analysis...")
        # Example: Add/update nodes for key themes or impacted sectors
        # This is a placeholder for more sophisticated graph updates
        if 'overall_sentiment' in macro_analysis and 'key_themes' in macro_analysis['overall_sentiment']:
            for theme, count in macro_analysis['overall_sentiment']['key_themes']:
                theme_id = theme.replace(' ', '_').upper()
                if theme_id not in self.graph:
                    self.graph.add_node(theme_id, id=theme_id, name=theme, type="Theme", importance=count/10.0, region="Global")
                    logger.debug(f"Added new theme node: {theme}")

        if 'risks' in geopolitical_risks:
            for risk in geopolitical_risks['risks']:
                risk_id = risk['risk_type'].upper() + "_" + risk['source'].upper()
                if risk_id not in self.graph:
                    self.graph.add_node(risk_id, id=risk_id, name=risk['title'], type="GeopoliticalRisk", importance=risk['risk_score'], region="Global")
                    logger.debug(f"Added new geopolitical risk node: {risk['title']}")
                
                # Example: Link risk to impacted sectors
                for sector in risk['impact_sectors']:
                    sector_id = sector.upper() + "_SECTOR"
                    if sector_id not in self.graph:
                        self.graph.add_node(sector_id, id=sector_id, name=sector, type="Sector", importance=0.5, region="Global")
                    if not self.graph.has_edge(risk_id, sector_id):
                        self.graph.add_edge(risk_id, sector_id, type="IMPACTS", strength=risk['risk_score'])
                        logger.debug(f"Added edge from {risk_id} to {sector_id}")

        self.last_update = datetime.now()
        logger.info("Knowledge Graph updated with latest analysis.")

