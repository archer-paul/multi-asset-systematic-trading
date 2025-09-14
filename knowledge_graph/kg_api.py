"""
Knowledge Graph API - Endpoints Flask pour le système de graphe de connaissance
Fournit des API REST et WebSocket pour interagir avec le knowledge graph
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Blueprint, jsonify, request, current_app
from flask_socketio import emit, join_room, leave_room
import numpy as np

from .economic_knowledge_graph import EconomicKnowledgeGraph, EntityType, RelationType

logger = logging.getLogger(__name__)

# Create Blueprint pour les API knowledge graph
kg_api = Blueprint('kg_api', __name__, url_prefix='/api/knowledge-graph')

# Instance globale du knowledge graph (sera initialisée au démarrage)
knowledge_graph: Optional[EconomicKnowledgeGraph] = None

def init_knowledge_graph(config):
    """Initialise le knowledge graph global"""
    global knowledge_graph
    try:
        knowledge_graph = EconomicKnowledgeGraph(config)

        # Enrichir avec des données externes en arrière-plan
        import threading
        def enrich_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(knowledge_graph.enrich_with_external_data())
            loop.close()

        thread = threading.Thread(target=enrich_async, daemon=True)
        thread.start()

        logger.info("Knowledge graph initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize knowledge graph: {e}")
        return False

@kg_api.route('/status', methods=['GET'])
def get_status():
    """Statut du knowledge graph"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Knowledge graph not initialized'
        }), 503

    try:
        stats = knowledge_graph.get_graph_statistics()
        return jsonify({
            'status': 'active',
            'initialized_at': datetime.now().isoformat(),
            'statistics': stats,
            'cache_size': len(knowledge_graph.cascade_cache),
            'last_enrichment': 'N/A'  # TODO: track enrichment timestamp
        })
    except Exception as e:
        logger.error(f"Error getting KG status: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/entities', methods=['GET'])
def get_entities():
    """Récupère la liste des entités avec filtrage optionnel"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        entity_type = request.args.get('type')  # 'company', 'country', etc.
        region = request.args.get('region')     # 'US', 'EU', etc.
        limit = request.args.get('limit', 100, type=int)

        entities = []

        # Parcourir toutes les entités
        for type_key, type_entities in knowledge_graph.entities.items():
            for entity_id, entity in type_entities.items():
                # Appliquer les filtres
                if entity_type and entity.type.value != entity_type:
                    continue
                if region and entity.region != region:
                    continue

                # Calculer l'importance
                importance = knowledge_graph.get_entity_importance(entity_id)

                entity_data = {
                    'id': entity.id,
                    'name': entity.name,
                    'type': entity.type.value,
                    'region': entity.region,
                    'importance': importance,
                    'metadata': entity.metadata
                }
                entities.append(entity_data)

        # Trier par importance et limiter
        entities = sorted(entities, key=lambda x: x['importance'], reverse=True)[:limit]

        return jsonify({
            'entities': entities,
            'total_count': len(entities),
            'filters_applied': {
                'type': entity_type,
                'region': region,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error getting entities: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/relationships', methods=['GET'])
def get_relationships():
    """Récupère les relations avec filtrage optionnel"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        relation_type = request.args.get('type')  # Type de relation
        entity_id = request.args.get('entity')    # Relations d'une entité spécifique
        min_strength = request.args.get('min_strength', 0.0, type=float)
        limit = request.args.get('limit', 200, type=int)

        relationships = []

        # Parcourir les relations dans le graphe NetworkX
        for source, target in knowledge_graph.graph.edges():
            edge_data = knowledge_graph.graph[source][target]

            for edge_key in edge_data:
                edge_info = edge_data[edge_key]

                # Appliquer les filtres
                if relation_type and edge_info.get('type') != relation_type:
                    continue
                if entity_id and source != entity_id and target != entity_id:
                    continue

                strength = edge_info.get('strength', 0.5)
                if strength < min_strength:
                    continue

                rel_data = {
                    'source': source,
                    'target': target,
                    'type': edge_info.get('type', 'unknown'),
                    'strength': strength,
                    'metadata': edge_info.get('metadata', {}),
                    'created_at': edge_info.get('created_at')
                }
                relationships.append(rel_data)

        # Limiter les résultats
        relationships = relationships[:limit]

        return jsonify({
            'relationships': relationships,
            'total_count': len(relationships),
            'filters_applied': {
                'type': relation_type,
                'entity': entity_id,
                'min_strength': min_strength,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/analyze-cascade', methods=['POST'])
def analyze_cascade():
    """Analyse les effets de cascade d'un événement"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Valider les données d'entrée
        required_fields = ['type', 'entity', 'magnitude']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Valider la magnitude
        magnitude = data['magnitude']
        if not isinstance(magnitude, (int, float)) or not -1.0 <= magnitude <= 1.0:
            return jsonify({'error': 'Magnitude must be a number between -1.0 and 1.0'}), 400

        # Valider que l'entité existe
        entity_id = data['entity']
        if entity_id not in knowledge_graph.graph.nodes():
            return jsonify({'error': f'Entity {entity_id} not found in knowledge graph'}), 404

        # Lancer l'analyse en arrière-plan
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                knowledge_graph.analyze_cascading_effects(data)
            )
        finally:
            loop.close()

        # Formater les résultats pour l'API
        formatted_result = {
            'event': result['event'],
            'analysis_timestamp': result['analysis_timestamp'],
            'total_effects': result['total_effects'],
            'effects_by_horizon': {
                'immediate': len(result['immediate_effects']),
                'short_term': len(result['short_term_effects']),
                'medium_term': len(result['medium_term_effects']),
                'long_term': len(result['long_term_effects'])
            },
            'top_impacts': []
        }

        # Sélectionner les impacts les plus significatifs
        all_effects = result['all_effects']
        top_impacts = sorted(all_effects, key=lambda e: abs(e.impact_magnitude), reverse=True)[:10]

        for effect in top_impacts:
            formatted_result['top_impacts'].append({
                'affected_entity': effect.affected_entity,
                'impact_magnitude': effect.impact_magnitude,
                'confidence': effect.confidence,
                'time_horizon': effect.time_horizon,
                'explanation': effect.explanation,
                'propagation_path': effect.propagation_path
            })

        return jsonify(formatted_result)

    except Exception as e:
        logger.error(f"Error analyzing cascade effects: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/visualization-data', methods=['GET'])
def get_visualization_data():
    """Récupère les données formatées pour la visualisation"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        # Paramètres de filtrage
        entity_types = request.args.getlist('entity_types')  # ['company', 'country']
        regions = request.args.getlist('regions')            # ['US', 'EU']
        min_importance = request.args.get('min_importance', 0.0, type=float)
        max_nodes = request.args.get('max_nodes', 100, type=int)
        include_labels = request.args.get('include_labels', 'true').lower() == 'true'

        # Obtenir les données de visualisation de base
        viz_data = knowledge_graph.export_for_visualization()

        # Appliquer les filtres
        filtered_nodes = []
        for node in viz_data['nodes']:
            # Filtre par type d'entité
            if entity_types and node['type'] not in entity_types:
                continue
            # Filtre par région
            if regions and node['region'] not in regions:
                continue
            # Filtre par importance
            if node['importance'] < min_importance:
                continue

            # Ajouter les labels si demandé
            if not include_labels:
                node.pop('name', None)

            filtered_nodes.append(node)

        # Limiter le nombre de noeuds
        filtered_nodes = sorted(filtered_nodes, key=lambda x: x['importance'], reverse=True)[:max_nodes]
        node_ids = {node['id'] for node in filtered_nodes}

        # Filtrer les liens pour ne garder que ceux entre les noeuds filtrés
        filtered_links = []
        for link in viz_data['links']:
            if link['source'] in node_ids and link['target'] in node_ids:
                filtered_links.append(link)

        # Calculer des métriques pour le frontend
        centrality_metrics = {}
        if knowledge_graph.graph.number_of_nodes() > 0:
            import networkx as nx
            try:
                degree_centrality = nx.degree_centrality(knowledge_graph.graph)
                betweenness_centrality = nx.betweenness_centrality(knowledge_graph.graph)

                for node in filtered_nodes:
                    node_id = node['id']
                    centrality_metrics[node_id] = {
                        'degree_centrality': degree_centrality.get(node_id, 0),
                        'betweenness_centrality': betweenness_centrality.get(node_id, 0)
                    }
            except Exception as e:
                logger.debug(f"Could not calculate centrality metrics: {e}")

        result = {
            'nodes': filtered_nodes,
            'links': filtered_links,
            'statistics': viz_data['statistics'],
            'centrality_metrics': centrality_metrics,
            'layout_recommendations': {
                'algorithm': 'force-directed',
                'node_size_metric': 'importance',
                'link_width_metric': 'strength',
                'color_scheme': 'by_type'
            },
            'filters_applied': {
                'entity_types': entity_types,
                'regions': regions,
                'min_importance': min_importance,
                'max_nodes': max_nodes,
                'total_nodes_before_filter': len(viz_data['nodes']),
                'total_nodes_after_filter': len(filtered_nodes)
            },
            'export_timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/entity/<entity_id>', methods=['GET'])
def get_entity_details(entity_id):
    """Récupère les détails d'une entité spécifique"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        if entity_id not in knowledge_graph.graph.nodes():
            return jsonify({'error': f'Entity {entity_id} not found'}), 404

        # Données de base de l'entité
        node_data = knowledge_graph.graph.nodes[entity_id]
        importance = knowledge_graph.get_entity_importance(entity_id)

        # Relations sortantes
        outgoing_relations = []
        for target in knowledge_graph.graph.successors(entity_id):
            edge_data = knowledge_graph.graph[entity_id][target]
            for edge_key in edge_data:
                edge_info = edge_data[edge_key]
                outgoing_relations.append({
                    'target': target,
                    'target_name': knowledge_graph.graph.nodes[target].get('name', target),
                    'type': edge_info.get('type', 'unknown'),
                    'strength': edge_info.get('strength', 0.5)
                })

        # Relations entrantes
        incoming_relations = []
        for source in knowledge_graph.graph.predecessors(entity_id):
            edge_data = knowledge_graph.graph[source][entity_id]
            for edge_key in edge_data:
                edge_info = edge_data[edge_key]
                incoming_relations.append({
                    'source': source,
                    'source_name': knowledge_graph.graph.nodes[source].get('name', source),
                    'type': edge_info.get('type', 'unknown'),
                    'strength': edge_info.get('strength', 0.5)
                })

        # Métriques de centralité
        import networkx as nx
        centrality_metrics = {}
        try:
            degree_centrality = nx.degree_centrality(knowledge_graph.graph)
            betweenness_centrality = nx.betweenness_centrality(knowledge_graph.graph)
            closeness_centrality = nx.closeness_centrality(knowledge_graph.graph)

            centrality_metrics = {
                'degree_centrality': degree_centrality.get(entity_id, 0),
                'betweenness_centrality': betweenness_centrality.get(entity_id, 0),
                'closeness_centrality': closeness_centrality.get(entity_id, 0)
            }
        except Exception as e:
            logger.debug(f"Could not calculate centrality for {entity_id}: {e}")

        result = {
            'id': entity_id,
            'name': node_data.get('name', entity_id),
            'type': node_data.get('type', 'unknown'),
            'region': node_data.get('region', 'unknown'),
            'importance': importance,
            'metadata': {k: v for k, v in node_data.items() if k not in ['name', 'type', 'region']},
            'relations': {
                'outgoing': outgoing_relations,
                'incoming': incoming_relations,
                'total_connections': len(outgoing_relations) + len(incoming_relations)
            },
            'centrality_metrics': centrality_metrics,
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting entity details for {entity_id}: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/search', methods=['GET'])
def search_entities():
    """Recherche d'entités par nom ou métadonnées"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        query = request.args.get('q', '').strip().lower()
        entity_type = request.args.get('type')
        limit = request.args.get('limit', 20, type=int)

        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400

        results = []

        # Rechercher dans toutes les entités
        for node_id in knowledge_graph.graph.nodes():
            node_data = knowledge_graph.graph.nodes[node_id]

            # Filtre par type si spécifié
            if entity_type and node_data.get('type') != entity_type:
                continue

            # Recherche dans le nom et l'ID
            name = node_data.get('name', node_id).lower()
            if query in name or query in node_id.lower():
                importance = knowledge_graph.get_entity_importance(node_id)

                results.append({
                    'id': node_id,
                    'name': node_data.get('name', node_id),
                    'type': node_data.get('type', 'unknown'),
                    'region': node_data.get('region', 'unknown'),
                    'importance': importance,
                    'match_score': 1.0 if query == name else 0.8 if name.startswith(query) else 0.5
                })

        # Trier par pertinence puis par importance
        results = sorted(results, key=lambda x: (x['match_score'], x['importance']), reverse=True)[:limit]

        return jsonify({
            'query': query,
            'results': results,
            'total_count': len(results),
            'filters_applied': {
                'type': entity_type,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return jsonify({'error': str(e)}), 500

@kg_api.route('/shortest-path/<source>/<target>', methods=['GET'])
def find_shortest_path(source, target):
    """Trouve le chemin le plus court entre deux entités"""
    global knowledge_graph

    if not knowledge_graph:
        return jsonify({'error': 'Knowledge graph not initialized'}), 503

    try:
        import networkx as nx

        if source not in knowledge_graph.graph.nodes():
            return jsonify({'error': f'Source entity {source} not found'}), 404
        if target not in knowledge_graph.graph.nodes():
            return jsonify({'error': f'Target entity {target} not found'}), 404

        try:
            path = nx.shortest_path(knowledge_graph.graph, source, target)
            path_length = len(path) - 1

            # Détails du chemin
            path_details = []
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]

                # Récupérer les détails de la relation
                edge_data = knowledge_graph.graph[current][next_node]
                edge_info = list(edge_data.values())[0]  # Prendre la première relation

                path_details.append({
                    'from': current,
                    'from_name': knowledge_graph.graph.nodes[current].get('name', current),
                    'to': next_node,
                    'to_name': knowledge_graph.graph.nodes[next_node].get('name', next_node),
                    'relation_type': edge_info.get('type', 'unknown'),
                    'strength': edge_info.get('strength', 0.5)
                })

            return jsonify({
                'source': source,
                'target': target,
                'path_exists': True,
                'path': path,
                'path_length': path_length,
                'path_details': path_details,
                'total_strength': sum(detail['strength'] for detail in path_details) / len(path_details) if path_details else 0
            })

        except nx.NetworkXNoPath:
            return jsonify({
                'source': source,
                'target': target,
                'path_exists': False,
                'message': 'No path found between the entities'
            })

    except Exception as e:
        logger.error(f"Error finding shortest path: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events pour les mises à jour en temps réel
def init_kg_websocket_events(socketio):
    """Initialise les événements WebSocket pour le knowledge graph"""

    @socketio.on('kg_subscribe')
    def handle_kg_subscribe():
        """S'abonner aux mises à jour du knowledge graph"""
        join_room('kg_updates')
        emit('kg_subscription', {'status': 'subscribed'})
        logger.debug("Client subscribed to knowledge graph updates")

    @socketio.on('kg_unsubscribe')
    def handle_kg_unsubscribe():
        """Se désabonner des mises à jour du knowledge graph"""
        leave_room('kg_updates')
        emit('kg_subscription', {'status': 'unsubscribed'})
        logger.debug("Client unsubscribed from knowledge graph updates")

    @socketio.on('kg_request_analysis')
    def handle_kg_analysis_request(data):
        """Demande d'analyse en temps réel"""
        try:
            global knowledge_graph
            if not knowledge_graph:
                emit('kg_analysis_error', {'error': 'Knowledge graph not initialized'})
                return

            # Lancer l'analyse en arrière-plan
            def run_analysis():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        knowledge_graph.analyze_cascading_effects(data)
                    )
                    # Émettre le résultat
                    socketio.emit('kg_analysis_result', result, room='kg_updates')
                except Exception as e:
                    socketio.emit('kg_analysis_error', {'error': str(e)}, room='kg_updates')
                finally:
                    loop.close()

            import threading
            thread = threading.Thread(target=run_analysis, daemon=True)
            thread.start()

            emit('kg_analysis_started', {'message': 'Analysis started'})

        except Exception as e:
            logger.error(f"Error handling KG analysis request: {e}")
            emit('kg_analysis_error', {'error': str(e)})

    # Fonction pour diffuser des mises à jour
    def broadcast_kg_update(update_data):
        """Diffuse une mise à jour du knowledge graph"""
        socketio.emit('kg_update', update_data, room='kg_updates')

    # Attacher la fonction de broadcast à l'objet socketio
    socketio.broadcast_kg_update = broadcast_kg_update

    logger.info("Knowledge Graph WebSocket events initialized")