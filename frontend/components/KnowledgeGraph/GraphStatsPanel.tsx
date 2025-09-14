'use client'

import { motion } from 'framer-motion'
import {
  ChartBarIcon,
  CircleStackIcon,
  ArrowsRightLeftIcon,
  LinkIcon,
  TrophyIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline'

interface GraphStats {
  nodes: number
  edges: number
  density: number
  is_connected: boolean
  entities_by_type: Record<string, number>
  relationships_by_type: Record<string, number>
  top_entities: Array<[string, number]>
}

interface GraphStatsPanelProps {
  stats: GraphStats
}

const entityTypeIcons: Record<string, string> = {
  'companies': '🏢',
  'countries': '🌍',
  'currencies': '💱',
  'commodities': '📦',
  'institutions': '🏛️',
  'politicians': '👤',
  'sectors': '🎯',
  'events': '📅'
}

const relationshipTypeColors: Record<string, string> = {
  'supply_chain': 'text-red-400',
  'trade_dependency': 'text-orange-400',
  'economic_partnership': 'text-blue-400',
  'currency_correlation': 'text-purple-400',
  'commodity_dependency': 'text-green-400',
  'institutional_control': 'text-gray-400',
  'political_alliance': 'text-pink-400',
  'sector_correlation': 'text-teal-400',
  'geographic_proximity': 'text-indigo-400',
  'leadership_influence': 'text-yellow-400'
}

export default function GraphStatsPanel({ stats }: GraphStatsPanelProps) {
  const formatRelationType = (type: string) => {
    return type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  const getConnectivityStatus = (isConnected: boolean) => {
    return {
      status: isConnected ? 'Connecté' : 'Fragmenté',
      color: isConnected ? 'text-green-400' : 'text-yellow-400',
      icon: isConnected ? '🔗' : '🔸'
    }
  }

  const getDensityLevel = (density: number) => {
    if (density > 0.7) return { level: 'Très dense', color: 'text-red-400' }
    if (density > 0.4) return { level: 'Dense', color: 'text-orange-400' }
    if (density > 0.2) return { level: 'Modéré', color: 'text-yellow-400' }
    if (density > 0.1) return { level: 'Éparse', color: 'text-blue-400' }
    return { level: 'Très éparse', color: 'text-gray-400' }
  }

  const connectivity = getConnectivityStatus(stats.is_connected)
  const densityInfo = getDensityLevel(stats.density)

  return (
    <div className="bg-dark-300 rounded-lg border border-dark-400">
      <div className="p-4 border-b border-dark-400">
        <h3 className="text-lg font-semibold text-white flex items-center">
          <ChartBarIcon className="w-5 h-5 mr-2 text-accent-blue" />
          Statistiques
        </h3>
      </div>

      <div className="p-4 space-y-6">
        {/* Overview Stats */}
        <div className="grid grid-cols-2 gap-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-dark-200 rounded-lg p-4 text-center"
          >
            <CircleStackIcon className="w-6 h-6 mx-auto text-accent-blue mb-2" />
            <div className="text-2xl font-bold text-white">{stats.nodes}</div>
            <div className="text-sm text-dark-400">Entités</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-dark-200 rounded-lg p-4 text-center"
          >
            <ArrowsRightLeftIcon className="w-6 h-6 mx-auto text-accent-purple mb-2" />
            <div className="text-2xl font-bold text-white">{stats.edges}</div>
            <div className="text-sm text-dark-400">Relations</div>
          </motion.div>
        </div>

        {/* Network Properties */}
        <div className="space-y-3">
          <div className="bg-dark-200 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-dark-400 flex items-center">
                <LinkIcon className="w-4 h-4 mr-2" />
                Connectivité
              </span>
              <span className={`text-sm font-medium ${connectivity.color} flex items-center`}>
                <span className="mr-1">{connectivity.icon}</span>
                {connectivity.status}
              </span>
            </div>
          </div>

          <div className="bg-dark-200 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-dark-400 flex items-center">
                <GlobeAltIcon className="w-4 h-4 mr-2" />
                Densité
              </span>
              <div className="text-right">
                <div className={`text-sm font-medium ${densityInfo.color}`}>
                  {densityInfo.level}
                </div>
                <div className="text-xs text-dark-400">
                  {(stats.density * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Entity Types Distribution */}
        <div>
          <h4 className="text-sm font-medium text-white mb-3">Types d'entités</h4>
          <div className="space-y-2">
            {Object.entries(stats.entities_by_type)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 6)
              .map(([type, count], index) => (
                <motion.div
                  key={type}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between bg-dark-200 rounded-lg p-2"
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{entityTypeIcons[type] || '📍'}</span>
                    <span className="text-sm text-white capitalize">
                      {type.replace('ies', 'ies').replace('s', 's')}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="text-sm font-medium text-accent-blue">{count}</div>
                    <div className="w-16 bg-dark-400 rounded-full h-2">
                      <div
                        className="bg-accent-blue h-2 rounded-full"
                        style={{
                          width: `${Math.min(100, (count / Math.max(...Object.values(stats.entities_by_type))) * 100)}%`
                        }}
                      />
                    </div>
                  </div>
                </motion.div>
              ))}
          </div>
        </div>

        {/* Relationship Types Distribution */}
        <div>
          <h4 className="text-sm font-medium text-white mb-3">Types de relations</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {Object.entries(stats.relationships_by_type)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 8)
              .map(([type, count], index) => (
                <motion.div
                  key={type}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between bg-dark-200 rounded-lg p-2"
                >
                  <div className="flex-1">
                    <div className={`text-xs font-medium ${relationshipTypeColors[type] || 'text-white'}`}>
                      {formatRelationType(type)}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="text-sm font-medium text-white">{count}</div>
                    <div className="w-12 bg-dark-400 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${
                          relationshipTypeColors[type]?.replace('text-', 'bg-') || 'bg-gray-400'
                        }`}
                        style={{
                          width: `${Math.min(100, (count / Math.max(...Object.values(stats.relationships_by_type))) * 100)}%`
                        }}
                      />
                    </div>
                  </div>
                </motion.div>
              ))}
          </div>
        </div>

        {/* Top Entities */}
        <div>
          <h4 className="text-sm font-medium text-white mb-3 flex items-center">
            <TrophyIcon className="w-4 h-4 mr-2 text-yellow-400" />
            Entités les plus importantes
          </h4>
          <div className="space-y-2">
            {stats.top_entities.slice(0, 5).map(([entityId, importance], index) => (
              <motion.div
                key={entityId}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between bg-dark-200 rounded-lg p-3"
              >
                <div className="flex items-center space-x-2">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                    index === 0 ? 'bg-yellow-500 text-black' :
                    index === 1 ? 'bg-gray-400 text-black' :
                    index === 2 ? 'bg-orange-600 text-white' :
                    'bg-dark-400 text-white'
                  }`}>
                    {index + 1}
                  </div>
                  <span className="text-sm font-medium text-white">{entityId}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold text-accent-blue">
                    {importance.toFixed(3)}
                  </div>
                  <div className="w-16 bg-dark-400 rounded-full h-1.5 mt-1">
                    <div
                      className="bg-accent-blue h-1.5 rounded-full"
                      style={{
                        width: `${Math.min(100, (importance / Math.max(...stats.top_entities.map(([,imp]) => imp))) * 100)}%`
                      }}
                    />
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Network Health Indicator */}
        <div className="bg-gradient-to-r from-dark-300 to-dark-200 rounded-lg p-4 border border-dark-400">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-white mb-1">Santé du réseau</h4>
              <div className="text-xs text-dark-400">
                {stats.nodes > 0 && stats.edges > 0 ? (
                  stats.is_connected && stats.density > 0.1 ?
                    'Réseau bien connecté et analysable' :
                    'Réseau fragmenté ou très éparse'
                ) : 'Données insuffisantes'}
              </div>
            </div>
            <div className="text-2xl">
              {stats.nodes > 0 && stats.edges > 0 ? (
                stats.is_connected && stats.density > 0.1 ? '💚' : '⚠️'
              ) : '❌'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}