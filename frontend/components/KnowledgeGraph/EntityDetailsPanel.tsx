'use client'

import { motion } from 'framer-motion'
import { XMarkIcon, ArrowRightIcon, ArrowLeftIcon } from '@heroicons/react/24/outline'

interface EntityDetails {
  id: string
  name: string
  type: string
  region: string
  importance: number
  metadata: Record<string, any>
  relations: {
    outgoing: Array<{
      target: string
      target_name: string
      type: string
      strength: number
    }>
    incoming: Array<{
      source: string
      source_name: string
      type: string
      strength: number
    }>
    total_connections: number
  }
  centrality_metrics: {
    degree_centrality: number
    betweenness_centrality: number
    closeness_centrality?: number
  }
}

interface EntityDetailsPanelProps {
  entity: EntityDetails
  onClose: () => void
}

export default function EntityDetailsPanel({ entity, onClose }: EntityDetailsPanelProps) {
  const getRelationTypeColor = (relationType: string) => {
    const colors = {
      'supply_chain': 'bg-red-900 text-red-200',
      'trade_dependency': 'bg-orange-900 text-orange-200',
      'economic_partnership': 'bg-blue-900 text-blue-200',
      'currency_correlation': 'bg-purple-900 text-purple-200',
      'commodity_dependency': 'bg-green-900 text-green-200',
      'institutional_control': 'bg-gray-900 text-gray-200',
      'political_alliance': 'bg-pink-900 text-pink-200',
      'sector_correlation': 'bg-teal-900 text-teal-200',
      'geographic_proximity': 'bg-indigo-900 text-indigo-200',
      'leadership_influence': 'bg-yellow-900 text-yellow-200'
    }
    return colors[relationType as keyof typeof colors] || 'bg-gray-900 text-gray-200'
  }

  const formatRelationType = (type: string) => {
    return type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  const getEntityTypeIcon = (type: string) => {
    const icons = {
      'company': '🏢',
      'country': '🌍',
      'currency': '💱',
      'commodity': '📦',
      'institution': '🏛️',
      'politician': '👤',
      'sector': '🎯',
      'event': '📅'
    }
    return icons[type as keyof typeof icons] || '📍'
  }

  const renderMetadata = () => {
    const metadataEntries = Object.entries(entity.metadata).filter(([key]) =>
      !['name', 'type', 'region'].includes(key)
    )

    if (metadataEntries.length === 0) return null

    return (
      <div className="bg-dark-300 sharp-card p-4">
        <h4 className="font-semibold text-white mb-3">Informations détaillées</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {metadataEntries.map(([key, value]) => (
            <div key={key} className="flex justify-between text-sm">
              <span className="text-dark-400 capitalize">
                {key.replace('_', ' ')}:
              </span>
              <span className="text-white font-medium">
                {typeof value === 'number'
                  ? key.includes('billion')
                    ? `$${value}B`
                    : key.includes('million')
                    ? `${value}M`
                    : value.toFixed(key.includes('percentage') || key.includes('ratio') ? 1 : 0)
                  : String(value)
                }
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const renderCentralityMetrics = () => {
    if (!entity.centrality_metrics || Object.keys(entity.centrality_metrics).length === 0) {
      return null
    }

    return (
      <div className="bg-dark-300 sharp-card p-4">
        <h4 className="font-semibold text-white mb-3">Métriques de centralité</h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-accent-blue">
              {entity.centrality_metrics.degree_centrality.toFixed(3)}
            </div>
            <div className="text-xs text-dark-400 mt-1">Degree</div>
            <div className="text-xs text-dark-500">Centrality</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-accent-purple">
              {entity.centrality_metrics.betweenness_centrality.toFixed(3)}
            </div>
            <div className="text-xs text-dark-400 mt-1">Betweenness</div>
            <div className="text-xs text-dark-500">Centrality</div>
          </div>
          {entity.centrality_metrics.closeness_centrality !== undefined && (
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-green">
                {entity.centrality_metrics.closeness_centrality.toFixed(3)}
              </div>
              <div className="text-xs text-dark-400 mt-1">Closeness</div>
              <div className="text-xs text-dark-500">Centrality</div>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="text-2xl">{getEntityTypeIcon(entity.type)}</div>
          <div>
            <h3 className="text-xl font-bold text-white">{entity.name}</h3>
            <div className="flex items-center space-x-2 text-sm text-dark-400">
              <span className="capitalize">{entity.type}</span>
              <span>•</span>
              <span>{entity.region}</span>
              <span>•</span>
              <span className="text-accent-blue">
                Importance: {entity.importance.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 sharp-card hover:bg-dark-300 text-dark-400 hover:text-white transition-colors"
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
      </div>

      {/* Content */}
      <div className="space-y-6">
        {/* Basic Info */}
        <div className="bg-dark-300 sharp-card p-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-blue">
                {entity.relations.total_connections}
              </div>
              <div className="text-sm text-dark-400">Connexions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-green">
                {entity.relations.outgoing.length}
              </div>
              <div className="text-sm text-dark-400">Sortantes</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-purple">
                {entity.relations.incoming.length}
              </div>
              <div className="text-sm text-dark-400">Entrantes</div>
            </div>
          </div>
        </div>

        {/* Metadata */}
        {renderMetadata()}

        {/* Centrality Metrics */}
        {renderCentralityMetrics()}

        {/* Relations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Outgoing Relations */}
          <div>
            <h4 className="font-semibold text-white mb-3 flex items-center">
              <ArrowRightIcon className="w-4 h-4 mr-2 text-accent-green" />
              Relations sortantes ({entity.relations.outgoing.length})
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {entity.relations.outgoing.length === 0 ? (
                <p className="text-dark-400 text-sm italic">Aucune relation sortante</p>
              ) : (
                entity.relations.outgoing
                  .sort((a, b) => b.strength - a.strength)
                  .map((relation, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="bg-dark-200 sharp-card p-3 border-l-4 border-accent-green"
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="font-medium text-white text-sm">
                            {relation.target_name}
                          </div>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className={`px-2 py-1 text-xs sharp-button${getRelationTypeColor(relation.type)}`}>
                              {formatRelationType(relation.type)}
                            </span>
                            <span className="text-accent-blue text-xs font-medium">
                              {relation.strength.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))
              )}
            </div>
          </div>

          {/* Incoming Relations */}
          <div>
            <h4 className="font-semibold text-white mb-3 flex items-center">
              <ArrowLeftIcon className="w-4 h-4 mr-2 text-accent-purple" />
              Relations entrantes ({entity.relations.incoming.length})
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {entity.relations.incoming.length === 0 ? (
                <p className="text-dark-400 text-sm italic">Aucune relation entrante</p>
              ) : (
                entity.relations.incoming
                  .sort((a, b) => b.strength - a.strength)
                  .map((relation, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="bg-dark-200 sharp-card p-3 border-l-4 border-accent-purple"
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="font-medium text-white text-sm">
                            {relation.source_name}
                          </div>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className={`px-2 py-1 text-xs sharp-button${getRelationTypeColor(relation.type)}`}>
                              {formatRelationType(relation.type)}
                            </span>
                            <span className="text-accent-blue text-xs font-medium">
                              {relation.strength.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))
              )}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}