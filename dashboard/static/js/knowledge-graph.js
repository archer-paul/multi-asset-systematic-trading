/**
 * Knowledge Graph Visualization
 * Interface JavaScript pour l'exploration interactive du graphe de connaissance économique
 */

class KnowledgeGraphVisualizer {
    constructor() {
        this.network = null;
        this.nodes = null;
        this.edges = null;
        this.socket = io();
        this.selectedNode = null;
        this.physicsEnabled = false;

        this.init();
    }

    init() {
        console.log('Initializing Knowledge Graph Visualizer...');

        // Configuration initiale
        this.setupSocketEvents();
        this.setupEventListeners();
        this.updateConnectionStatus('connecting');

        // Charger les données initiales
        this.loadGraphData();

        // Auto-refresh périodique
        setInterval(() => {
            if (this.isConnected()) {
                this.refreshStatistics();
            }
        }, 60000); // Toutes les minutes
    }

    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to Knowledge Graph server');
            this.updateConnectionStatus('connected');
            this.socket.emit('kg_subscribe');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from Knowledge Graph server');
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('kg_update', (data) => {
            console.log('Knowledge Graph update received:', data);
            this.handleGraphUpdate(data);
        });

        this.socket.on('kg_analysis_result', (data) => {
            console.log('Cascade analysis result:', data);
            this.displayAnalysisResults(data);
        });

        this.socket.on('kg_analysis_error', (error) => {
            console.error('Analysis error:', error);
            this.showError('Erreur d\'analyse: ' + error.error);
            this.hideAnalysisLoading();
        });
    }

    setupEventListeners() {
        // Contrôles de visualisation
        document.getElementById('zoom-in').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoom-out').addEventListener('click', () => this.zoomOut());
        document.getElementById('fit-network').addEventListener('click', () => this.fitNetwork());
        document.getElementById('physics-toggle').addEventListener('click', () => this.togglePhysics());
        document.getElementById('refresh-data').addEventListener('click', () => this.loadGraphData());
        document.getElementById('export-png').addEventListener('click', () => this.exportToPNG());

        // Filtres
        document.getElementById('apply-filters').addEventListener('click', () => this.applyFilters());

        // Sliders
        const importanceSlider = document.getElementById('importance-slider');
        importanceSlider.addEventListener('input', (e) => {
            document.getElementById('importance-value').textContent = parseFloat(e.target.value).toFixed(2);
        });

        const magnitudeSlider = document.getElementById('magnitude-slider');
        magnitudeSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('magnitude-value').textContent = value.toFixed(1);
            document.getElementById('magnitude-label').textContent = this.getMagnitudeLabel(value);
        });

        // Analyse de cascade
        document.getElementById('analyze-cascade').addEventListener('click', () => this.analyzeCascade());

        // Recherche d'entité
        const originEntityInput = document.getElementById('origin-entity');
        originEntityInput.addEventListener('input', (e) => this.updateEntitySuggestions(e.target.value));
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        const indicator = statusElement.querySelector('.status-indicator');

        indicator.className = 'status-indicator';

        switch (status) {
            case 'connected':
                indicator.classList.add('status-connected');
                statusElement.innerHTML = '<span class="status-indicator status-connected"></span>Connecté';
                statusElement.className = 'badge bg-success';
                break;
            case 'disconnected':
                indicator.classList.add('status-disconnected');
                statusElement.innerHTML = '<span class="status-indicator status-disconnected"></span>Déconnecté';
                statusElement.className = 'badge bg-danger';
                break;
            case 'connecting':
                indicator.classList.add('status-loading');
                statusElement.innerHTML = '<span class="status-indicator status-loading"></span>Connexion...';
                statusElement.className = 'badge bg-warning';
                break;
        }
    }

    async loadGraphData() {
        try {
            console.log('Loading graph data...');

            // Obtenir les données de visualisation
            const filters = this.getCurrentFilters();
            const queryParams = new URLSearchParams(filters).toString();

            const response = await fetch(`/api/knowledge-graph/visualization-data?${queryParams}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            console.log('Graph data loaded:', data);

            // Mettre à jour les statistiques
            this.updateStatistics(data.statistics);

            // Initialiser la visualisation
            this.initializeNetwork(data);

            // Mettre à jour les suggestions d'entités
            this.updateEntitySuggestions();

        } catch (error) {
            console.error('Error loading graph data:', error);
            this.showError('Erreur de chargement: ' + error.message);
        }
    }

    initializeNetwork(data) {
        const container = document.getElementById('network-container');

        // Préparer les données pour vis-network
        const nodes = new vis.DataSet(data.nodes.map(node => ({
            id: node.id,
            label: node.name,
            title: this.createNodeTooltip(node),
            shape: this.getNodeShape(node.type),
            color: {
                background: node.color,
                border: this.darkenColor(node.color, 0.3),
                highlight: {
                    background: this.lightenColor(node.color, 0.2),
                    border: node.color
                }
            },
            size: Math.max(15, Math.min(50, node.size)),
            font: {
                color: '#e6edf3',
                size: 12,
                face: 'Arial'
            },
            metadata: node.metadata
        })));

        const edges = new vis.DataSet(data.links.map((link, index) => ({
            id: index,
            from: link.source,
            to: link.target,
            title: this.createEdgeTooltip(link),
            color: {
                color: link.color,
                opacity: 0.7
            },
            width: Math.max(1, Math.min(8, link.width)),
            arrows: {
                to: { enabled: true, scaleFactor: 0.8 }
            },
            smooth: { type: 'continuous' },
            metadata: link.metadata
        })));

        this.nodes = nodes;
        this.edges = edges;

        // Configuration de la visualisation
        const options = {
            physics: {
                enabled: this.physicsEnabled,
                stabilization: { iterations: 100 },
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 95,
                    springConstant: 0.04,
                    damping: 0.09
                }
            },
            interaction: {
                hover: true,
                selectConnectedEdges: false,
                tooltipDelay: 200
            },
            nodes: {
                borderWidth: 2,
                shadow: true,
                chosen: {
                    node: (values, id, selected, hovering) => {
                        values.shadow = true;
                        values.shadowSize = 10;
                        values.shadowColor = '#58a6ff';
                    }
                }
            },
            edges: {
                shadow: true,
                selectionWidth: 3,
                hoverWidth: 2,
                chosen: {
                    edge: (values, id, selected, hovering) => {
                        values.shadow = true;
                        values.shadowColor = '#58a6ff';
                    }
                }
            },
            layout: {
                improvedLayout: false
            }
        };

        // Créer le réseau
        this.network = new vis.Network(container, { nodes, edges }, options);

        // Événements de sélection
        this.network.on('selectNode', (params) => {
            if (params.nodes.length > 0) {
                this.onNodeSelected(params.nodes[0]);
            }
        });

        this.network.on('deselectNode', () => {
            this.hideEntityDetails();
        });

        this.network.on('doubleClick', (params) => {
            if (params.nodes.length > 0) {
                this.showEntityAnalysis(params.nodes[0]);
            }
        });

        // Finaliser l'initialisation
        this.network.once('stabilized', () => {
            console.log('Network stabilized');
            this.fitNetwork();
        });

        console.log(`Network initialized with ${nodes.length} nodes and ${edges.length} edges`);
    }

    createNodeTooltip(node) {
        return `
            <div style="padding: 8px; background: #21262d; border-radius: 4px; color: #e6edf3; max-width: 300px;">
                <div style="font-weight: bold; margin-bottom: 4px;">${node.name}</div>
                <div style="color: #8b949e; font-size: 12px;">Type: ${node.type}</div>
                <div style="color: #8b949e; font-size: 12px;">Région: ${node.region}</div>
                <div style="color: #58a6ff; font-size: 12px;">Importance: ${node.importance.toFixed(3)}</div>
                ${node.metadata.sector ? `<div style="color: #8b949e; font-size: 12px;">Secteur: ${node.metadata.sector}</div>` : ''}
                ${node.metadata.market_cap_billion ? `<div style="color: #8b949e; font-size: 12px;">Capitalisation: $${node.metadata.market_cap_billion}B</div>` : ''}
            </div>
        `;
    }

    createEdgeTooltip(link) {
        return `
            <div style="padding: 8px; background: #21262d; border-radius: 4px; color: #e6edf3;">
                <div style="font-weight: bold; margin-bottom: 4px;">${link.type.replace('_', ' ')}</div>
                <div style="color: #58a6ff; font-size: 12px;">Force: ${link.strength.toFixed(2)}</div>
                ${link.metadata.critical ? '<div style="color: #f85149; font-size: 12px;">🔴 Critique</div>' : ''}
            </div>
        `;
    }

    getNodeShape(nodeType) {
        const shapes = {
            'company': 'dot',
            'country': 'square',
            'currency': 'diamond',
            'commodity': 'triangle',
            'institution': 'star',
            'politician': 'hexagon',
            'sector': 'ellipse',
            'event': 'database'
        };
        return shapes[nodeType] || 'dot';
    }

    darkenColor(hex, factor) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(255 * factor);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return '#' + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    lightenColor(hex, factor) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(255 * factor);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return '#' + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    async onNodeSelected(nodeId) {
        this.selectedNode = nodeId;
        console.log('Node selected:', nodeId);

        try {
            const response = await fetch(`/api/knowledge-graph/entity/${nodeId}`);
            const entityData = await response.json();

            if (entityData.error) {
                throw new Error(entityData.error);
            }

            this.showEntityDetails(entityData);
        } catch (error) {
            console.error('Error loading entity details:', error);
            this.showError('Erreur lors du chargement des détails: ' + error.message);
        }
    }

    showEntityDetails(entityData) {
        const detailsPanel = document.getElementById('entity-details');

        let html = `
            <div class="entity-info">
                <h5>${entityData.name} <span class="badge bg-primary">${entityData.type}</span></h5>
                <div class="row">
                    <div class="col-md-6">
                        <strong>Région:</strong> ${entityData.region}<br>
                        <strong>Importance:</strong> ${entityData.importance.toFixed(3)}<br>
                        <strong>Connexions totales:</strong> ${entityData.relations.total_connections}
                    </div>
                    <div class="col-md-6">
                        ${Object.entries(entityData.metadata).map(([key, value]) =>
                            `<strong>${key}:</strong> ${value}<br>`
                        ).join('')}
                    </div>
                </div>
            </div>

            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Relations sortantes (${entityData.relations.outgoing.length})</h6>
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${entityData.relations.outgoing.map(rel => `
                            <div class="relation-item">
                                → <strong>${rel.target_name}</strong>
                                <span class="badge bg-secondary">${rel.type}</span>
                                <small class="text-info">(${rel.strength.toFixed(2)})</small>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="col-md-6">
                    <h6>Relations entrantes (${entityData.relations.incoming.length})</h6>
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${entityData.relations.incoming.map(rel => `
                            <div class="relation-item">
                                ← <strong>${rel.source_name}</strong>
                                <span class="badge bg-secondary">${rel.type}</span>
                                <small class="text-info">(${rel.strength.toFixed(2)})</small>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>

            ${Object.keys(entityData.centrality_metrics).length > 0 ? `
                <div class="mt-3">
                    <h6>Métriques de centralité</h6>
                    <div class="row">
                        <div class="col-md-4">
                            <div class="text-center">
                                <div class="stat-value">${entityData.centrality_metrics.degree_centrality.toFixed(3)}</div>
                                <div class="stat-label">Degree Centrality</div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <div class="stat-value">${entityData.centrality_metrics.betweenness_centrality.toFixed(3)}</div>
                                <div class="stat-label">Betweenness</div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <div class="stat-value">${entityData.centrality_metrics.closeness_centrality?.toFixed(3) || 'N/A'}</div>
                                <div class="stat-label">Closeness</div>
                            </div>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;

        detailsPanel.innerHTML = html;
        detailsPanel.style.display = 'block';
    }

    hideEntityDetails() {
        const detailsPanel = document.getElementById('entity-details');
        detailsPanel.style.display = 'none';
        this.selectedNode = null;
    }

    updateStatistics(stats) {
        const statsContainer = document.getElementById('graph-stats');

        const html = `
            <div class="stat-card">
                <div class="stat-value">${stats.nodes}</div>
                <div class="stat-label">Entités</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.edges}</div>
                <div class="stat-label">Relations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(stats.density * 100).toFixed(1)}%</div>
                <div class="stat-label">Densité</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.is_connected ? 'Oui' : 'Non'}</div>
                <div class="stat-label">Connecté</div>
            </div>
        `;

        statsContainer.innerHTML = html;
    }

    getCurrentFilters() {
        const entityTypes = Array.from(document.getElementById('entity-type-filter').selectedOptions)
            .map(option => option.value);
        const regions = Array.from(document.getElementById('region-filter').selectedOptions)
            .map(option => option.value);
        const minImportance = parseFloat(document.getElementById('importance-slider').value);

        return {
            entity_types: entityTypes,
            regions: regions,
            min_importance: minImportance,
            max_nodes: 100,
            include_labels: 'true'
        };
    }

    applyFilters() {
        console.log('Applying filters...');
        this.loadGraphData();
    }

    async updateEntitySuggestions(query = '') {
        if (!query || query.length < 2) return;

        try {
            const response = await fetch(`/api/knowledge-graph/search?q=${encodeURIComponent(query)}&limit=10`);
            const data = await response.json();

            if (data.error) return;

            const datalist = document.getElementById('entity-suggestions');
            datalist.innerHTML = data.results.map(entity =>
                `<option value="${entity.id}">${entity.name} (${entity.type})</option>`
            ).join('');

        } catch (error) {
            console.debug('Error updating entity suggestions:', error);
        }
    }

    getMagnitudeLabel(value) {
        if (value > 0.7) return 'Très positif';
        if (value > 0.3) return 'Positif modéré';
        if (value > 0.1) return 'Légèrement positif';
        if (value > -0.1) return 'Neutre';
        if (value > -0.3) return 'Légèrement négatif';
        if (value > -0.7) return 'Négatif modéré';
        return 'Très négatif';
    }

    async analyzeCascade() {
        const eventType = document.getElementById('event-type').value;
        const originEntity = document.getElementById('origin-entity').value;
        const magnitude = parseFloat(document.getElementById('magnitude-slider').value);

        if (!originEntity) {
            this.showError('Veuillez spécifier une entité d\'origine');
            return;
        }

        this.showAnalysisLoading();

        const analysisData = {
            type: eventType,
            entity: originEntity,
            magnitude: magnitude,
            timestamp: new Date().toISOString()
        };

        try {
            // Utiliser l'API REST pour l'analyse
            const response = await fetch('/api/knowledge-graph/analyze-cascade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(analysisData)
            });

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            this.hideAnalysisLoading();
            this.displayAnalysisResults(result);

        } catch (error) {
            console.error('Error analyzing cascade:', error);
            this.showError('Erreur d\'analyse: ' + error.message);
            this.hideAnalysisLoading();
        }
    }

    showAnalysisLoading() {
        const spinner = document.querySelector('#analyze-cascade .loading-spinner');
        const button = document.getElementById('analyze-cascade');

        spinner.classList.add('show');
        button.disabled = true;
    }

    hideAnalysisLoading() {
        const spinner = document.querySelector('#analyze-cascade .loading-spinner');
        const button = document.getElementById('analyze-cascade');

        spinner.classList.remove('show');
        button.disabled = false;
    }

    displayAnalysisResults(data) {
        const resultsContainer = document.getElementById('analysis-results');

        let html = `
            <div class="mb-4">
                <h6>Événement analysé</h6>
                <div class="entity-info">
                    <strong>Type:</strong> ${data.event.type}<br>
                    <strong>Entité d'origine:</strong> ${data.event.entity}<br>
                    <strong>Magnitude:</strong> ${data.event.magnitude}<br>
                    <strong>Effets totaux détectés:</strong> ${data.total_effects}
                </div>
            </div>

            <div class="mb-4">
                <h6>Répartition par horizon temporel</h6>
                <div class="row">
                    <div class="col-3">
                        <div class="stat-card">
                            <div class="stat-value">${data.effects_by_horizon.immediate}</div>
                            <div class="stat-label">Immédiats</div>
                        </div>
                    </div>
                    <div class="col-3">
                        <div class="stat-card">
                            <div class="stat-value">${data.effects_by_horizon.short_term}</div>
                            <div class="stat-label">Court terme</div>
                        </div>
                    </div>
                    <div class="col-3">
                        <div class="stat-card">
                            <div class="stat-value">${data.effects_by_horizon.medium_term}</div>
                            <div class="stat-label">Moyen terme</div>
                        </div>
                    </div>
                    <div class="col-3">
                        <div class="stat-card">
                            <div class="stat-value">${data.effects_by_horizon.long_term}</div>
                            <div class="stat-label">Long terme</div>
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <h6>Impacts les plus significatifs</h6>
                <div class="analysis-results">
                    ${data.top_impacts.map(impact => `
                        <div class="cascade-effect ${impact.impact_magnitude > 0 ? 'impact-positive' : impact.impact_magnitude < 0 ? 'impact-negative' : 'impact-neutral'}">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div>
                                    <strong>${impact.affected_entity}</strong>
                                    <span class="badge bg-secondary ms-2">${impact.time_horizon}</span>
                                </div>
                                <div class="text-end">
                                    <div class="fw-bold ${impact.impact_magnitude > 0 ? 'text-success' : impact.impact_magnitude < 0 ? 'text-danger' : 'text-warning'}">
                                        ${impact.impact_magnitude > 0 ? '+' : ''}${impact.impact_magnitude.toFixed(3)}
                                    </div>
                                    <small class="text-muted">Confiance: ${(impact.confidence * 100).toFixed(0)}%</small>
                                </div>
                            </div>
                            <div class="text-small text-muted mb-2">${impact.explanation}</div>
                            <div class="text-small">
                                <strong>Chemin:</strong> ${impact.propagation_path.join(' → ')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        resultsContainer.innerHTML = html;

        // Afficher la modal
        const modal = new bootstrap.Modal(document.getElementById('analysisModal'));
        modal.show();

        // Optionnel: mettre en évidence les entités affectées dans le graphe
        this.highlightAffectedEntities(data.top_impacts);
    }

    highlightAffectedEntities(impacts) {
        if (!this.network || !this.nodes) return;

        // Réinitialiser les couleurs
        const updates = [];

        // Mettre en évidence les entités affectées
        impacts.forEach(impact => {
            const nodeId = impact.affected_entity;
            try {
                const node = this.nodes.get(nodeId);
                if (node) {
                    const highlightColor = impact.impact_magnitude > 0 ? '#3fb950' :
                                         impact.impact_magnitude < 0 ? '#f85149' : '#d29922';

                    updates.push({
                        id: nodeId,
                        color: {
                            background: highlightColor,
                            border: this.darkenColor(highlightColor, 0.3),
                            highlight: {
                                background: this.lightenColor(highlightColor, 0.2),
                                border: highlightColor
                            }
                        }
                    });
                }
            } catch (error) {
                console.debug('Could not highlight node:', nodeId, error);
            }
        });

        if (updates.length > 0) {
            this.nodes.update(updates);

            // Revenir aux couleurs normales après 10 secondes
            setTimeout(() => {
                this.loadGraphData();
            }, 10000);
        }
    }

    // Contrôles de visualisation
    zoomIn() {
        if (this.network) {
            const scale = this.network.getScale();
            this.network.moveTo({ scale: scale * 1.2 });
        }
    }

    zoomOut() {
        if (this.network) {
            const scale = this.network.getScale();
            this.network.moveTo({ scale: scale * 0.8 });
        }
    }

    fitNetwork() {
        if (this.network) {
            this.network.fit({ animation: true });
        }
    }

    togglePhysics() {
        this.physicsEnabled = !this.physicsEnabled;
        const icon = document.getElementById('physics-icon');

        if (this.network) {
            this.network.setOptions({ physics: { enabled: this.physicsEnabled } });
        }

        icon.className = this.physicsEnabled ? 'fas fa-pause' : 'fas fa-play';
    }

    async exportToPNG() {
        if (this.network) {
            try {
                const canvas = this.network.canvas.frame.canvas;
                const dataURL = canvas.toDataURL('image/png');

                const link = document.createElement('a');
                link.download = `knowledge-graph-${new Date().toISOString().slice(0, 10)}.png`;
                link.href = dataURL;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                console.log('Graph exported to PNG');
            } catch (error) {
                console.error('Error exporting to PNG:', error);
                this.showError('Erreur lors de l\'export: ' + error.message);
            }
        }
    }

    async refreshStatistics() {
        try {
            const response = await fetch('/api/knowledge-graph/status');
            const data = await response.json();

            if (data.status === 'active' && data.statistics) {
                this.updateStatistics(data.statistics);
            }
        } catch (error) {
            console.debug('Error refreshing statistics:', error);
        }
    }

    showError(message) {
        // Créer une alerte temporaire
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show position-fixed';
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alert);

        // Auto-suppression après 5 secondes
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 5000);
    }

    isConnected() {
        return this.socket && this.socket.connected;
    }

    handleGraphUpdate(updateData) {
        // Gérer les mises à jour en temps réel du graphe
        console.log('Handling graph update:', updateData);

        // Recharger les données si nécessaire
        if (updateData.reload_required) {
            this.loadGraphData();
        }
    }
}

// Initialiser le visualiseur une fois le DOM chargé
document.addEventListener('DOMContentLoaded', () => {
    window.kgVisualizer = new KnowledgeGraphVisualizer();
});