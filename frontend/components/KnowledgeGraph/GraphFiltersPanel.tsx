'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  FunnelIcon,
  ArrowPathIcon,
  EyeIcon,
  EyeSlashIcon
} from '@heroicons/react/24/outline'

interface GraphFilters {
  entity_types: string[]
  regions: string[]
  min_importance: number
  max_nodes: number
}

interface GraphFiltersPanelProps {
  onFiltersChange: (filters: GraphFilters) => void
}

const entityTypeOptions = [
  { value: 'company', label: 'Entreprises', icon: '🏢', color: 'bg-blue-500' },
  { value: 'country', label: 'Pays', icon: '🌍', color: 'bg-red-500' },
  { value: 'currency', label: 'Devises', icon: '💱', color: 'bg-yellow-500' },
  { value: 'commodity', label: 'Matières premières', icon: '📦', color: 'bg-purple-500' },
  { value: 'institution', label: 'Institutions', icon: '🏛️', color: 'bg-green-500' },
  { value: 'politician', label: 'Dirigeants', icon: '👤', color: 'bg-pink-500' },
  { value: 'sector', label: 'Secteurs', icon: '🎯', color: 'bg-indigo-500' },
  { value: 'event', label: 'Événements', icon: '📅', color: 'bg-orange-500' }
]

const regionOptions = [
  { value: 'US', label: 'États-Unis', icon: '🇺🇸' },
  { value: 'EU', label: 'Europe', icon: '🇪🇺' },
  { value: 'CN', label: 'Chine', icon: '🇨🇳' },
  { value: 'JP', label: 'Japon', icon: '🇯🇵' },
  { value: 'UK', label: 'Royaume-Uni', icon: '🇬🇧' },
  { value: 'DE', label: 'Allemagne', icon: '🇩🇪' },
  { value: 'FR', label: 'France', icon: '🇫🇷' },
  { value: 'GLOBAL', label: 'Global', icon: '🌐' }
]

export default function GraphFiltersPanel({ onFiltersChange }: GraphFiltersPanelProps) {
  const [filters, setFilters] = useState<GraphFilters>({
    entity_types: ['company', 'country', 'currency', 'commodity'],
    regions: ['US', 'EU', 'GLOBAL'],
    min_importance: 0,
    max_nodes: 100
  })

  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    onFiltersChange(filters)
  }, [filters, onFiltersChange])

  const handleEntityTypeToggle = (entityType: string) => {
    setFilters(prev => ({
      ...prev,
      entity_types: prev.entity_types.includes(entityType)
        ? prev.entity_types.filter(t => t !== entityType)
        : [...prev.entity_types, entityType]
    }))
  }

  const handleRegionToggle = (region: string) => {
    setFilters(prev => ({
      ...prev,
      regions: prev.regions.includes(region)
        ? prev.regions.filter(r => r !== region)
        : [...prev.regions, region]
    }))
  }

  const handleImportanceChange = (value: number) => {
    setFilters(prev => ({
      ...prev,
      min_importance: value
    }))
  }

  const handleMaxNodesChange = (value: number) => {
    setFilters(prev => ({
      ...prev,
      max_nodes: value
    }))
  }

  const resetFilters = () => {
    setFilters({
      entity_types: ['company', 'country', 'currency', 'commodity'],
      regions: ['US', 'EU', 'GLOBAL'],
      min_importance: 0,
      max_nodes: 100
    })
  }

  const selectAllEntityTypes = () => {
    setFilters(prev => ({
      ...prev,
      entity_types: entityTypeOptions.map(opt => opt.value)
    }))
  }

  const selectAllRegions = () => {
    setFilters(prev => ({
      ...prev,
      regions: regionOptions.map(opt => opt.value)
    }))
  }

  return (
    <div className="bg-dark-300 sharp-card border border-dark-400">
      <div className="p-4 border-b border-dark-400">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <FunnelIcon className="w-5 h-5 mr-2 text-accent-blue" />
            Filtres
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={resetFilters}
              className="p-1 sharp-buttonhover:bg-dark-200 text-dark-400 hover:text-white transition-colors"
              title="Réinitialiser"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="p-1 sharp-buttonhover:bg-dark-200 text-dark-400 hover:text-white transition-colors"
            >
              {collapsed ? <EyeIcon className="w-4 h-4" /> : <EyeSlashIcon className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      <motion.div
        initial={{ height: collapsed ? 0 : 'auto' }}
        animate={{ height: collapsed ? 0 : 'auto' }}
        transition={{ duration: 0.3 }}
        className="overflow-hidden"
      >
        <div className="p-4 space-y-6">
          {/* Entity Types */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-medium text-dark-300">
                Types d'entités
              </label>
              <button
                onClick={selectAllEntityTypes}
                className="text-xs text-accent-blue hover:text-accent-blue/80 transition-colors"
              >
                Tout sélectionner
              </button>
            </div>

            <div className="space-y-2">
              {entityTypeOptions.map((option) => {
                const isSelected = filters.entity_types.includes(option.value)
                return (
                  <motion.button
                    key={option.value}
                    onClick={() => handleEntityTypeToggle(option.value)}
                    className={`w-full p-2 sharp-card border-2 transition-all text-left ${
                      isSelected
                        ? 'border-accent-blue bg-accent-blue/10'
                        : 'border-dark-400 hover:border-dark-300 hover:bg-dark-200'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${option.color}`} />
                      <span className="text-lg">{option.icon}</span>
                      <span className="text-sm text-white flex-1">{option.label}</span>
                      {isSelected && (
                        <div className="w-2 h-2 bg-accent-blue rounded-full" />
                      )}
                    </div>
                  </motion.button>
                )
              })}
            </div>

            <div className="mt-2 text-xs text-dark-400">
              {filters.entity_types.length} type(s) sélectionné(s)
            </div>
          </div>

          {/* Regions */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-medium text-dark-300">
                Régions
              </label>
              <button
                onClick={selectAllRegions}
                className="text-xs text-accent-blue hover:text-accent-blue/80 transition-colors"
              >
                Tout sélectionner
              </button>
            </div>

            <div className="grid grid-cols-1 gap-2">
              {regionOptions.map((option) => {
                const isSelected = filters.regions.includes(option.value)
                return (
                  <motion.button
                    key={option.value}
                    onClick={() => handleRegionToggle(option.value)}
                    className={`p-2 sharp-card border transition-all text-left ${
                      isSelected
                        ? 'border-accent-blue bg-accent-blue/10'
                        : 'border-dark-400 hover:border-dark-300 hover:bg-dark-200'
                    }`}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{option.icon}</span>
                      <span className="text-sm text-white flex-1">{option.label}</span>
                      {isSelected && (
                        <div className="w-2 h-2 bg-accent-blue rounded-full" />
                      )}
                    </div>
                  </motion.button>
                )
              })}
            </div>

            <div className="mt-2 text-xs text-dark-400">
              {filters.regions.length} région(s) sélectionnée(s)
            </div>
          </div>

          {/* Importance Threshold */}
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-3">
              Importance minimale
            </label>

            <div className="space-y-3">
              <div className="relative">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={filters.min_importance}
                  onChange={(e) => handleImportanceChange(parseFloat(e.target.value))}
                  className="w-full h-2 bg-dark-400 sharp-card appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-dark-400 mt-1">
                  <span>0</span>
                  <span>0.5</span>
                  <span>1</span>
                </div>
              </div>

              <div className="bg-dark-200 sharp-card p-2 text-center">
                <div className="text-accent-blue font-bold">
                  {filters.min_importance.toFixed(2)}
                </div>
                <div className="text-xs text-dark-400">
                  {filters.min_importance === 0 ? 'Tous les nœuds' :
                   filters.min_importance < 0.3 ? 'Nœuds importants' :
                   filters.min_importance < 0.7 ? 'Nœuds très importants' : 'Nœuds critiques'}
                </div>
              </div>
            </div>
          </div>

          {/* Max Nodes */}
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-3">
              Nombre maximum de nœuds
            </label>

            <div className="space-y-3">
              <div className="relative">
                <input
                  type="range"
                  min="20"
                  max="200"
                  step="10"
                  value={filters.max_nodes}
                  onChange={(e) => handleMaxNodesChange(parseInt(e.target.value))}
                  className="w-full h-2 bg-dark-400 sharp-card appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-dark-400 mt-1">
                  <span>20</span>
                  <span>100</span>
                  <span>200</span>
                </div>
              </div>

              <div className="bg-dark-200 sharp-card p-2 text-center">
                <div className="text-accent-blue font-bold">
                  {filters.max_nodes}
                </div>
                <div className="text-xs text-dark-400">
                  nœuds maximum
                </div>
              </div>
            </div>
          </div>

          {/* Filter Summary */}
          <div className="bg-dark-200 sharp-card p-3 border border-dark-400">
            <h4 className="text-sm font-medium text-white mb-2">Résumé des filtres</h4>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-dark-400">Types d'entités:</span>
                <span className="text-accent-blue">{filters.entity_types.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-dark-400">Régions:</span>
                <span className="text-accent-blue">{filters.regions.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-dark-400">Importance min:</span>
                <span className="text-accent-blue">{filters.min_importance.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-dark-400">Nœuds max:</span>
                <span className="text-accent-blue">{filters.max_nodes}</span>
              </div>
            </div>
          </div>

          {/* Apply Filters Info */}
          <div className="text-xs text-dark-400 text-center italic">
            Les filtres sont appliqués automatiquement
          </div>
        </div>
      </motion.div>
    </div>
  )
}