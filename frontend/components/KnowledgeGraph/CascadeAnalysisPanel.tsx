'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  BoltIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'

interface CascadeAnalysisPanelProps {
  onAnalyze: (analysisData: {
    type: string
    entity: string
    magnitude: number
  }) => void
  isAnalyzing: boolean
}

const eventTypes = [
  { value: 'war', label: 'Conflit/Guerre', icon: '⚔️', description: 'Impact géopolitique majeur' },
  { value: 'election', label: 'Élection', icon: '🗳️', description: 'Changement politique' },
  { value: 'banking_crisis', label: 'Crise bancaire', icon: '🏦', description: 'Instabilité financière' },
  { value: 'supply_disruption', label: 'Rupture d\'approvisionnement', icon: '🚢', description: 'Perturbation supply chain' },
  { value: 'policy_change', label: 'Changement de politique', icon: '📋', description: 'Nouvelle réglementation' },
  { value: 'natural_disaster', label: 'Catastrophe naturelle', icon: '🌪️', description: 'Événement climatique' },
  { value: 'technology_disruption', label: 'Disruption technologique', icon: '🚀', description: 'Innovation majeure' },
  { value: 'pandemic', label: 'Pandémie', icon: '🦠', description: 'Crise sanitaire globale' }
]

const commonEntities = [
  { id: 'US', name: 'États-Unis', type: 'country', flag: '🇺🇸' },
  { id: 'CN', name: 'Chine', type: 'country', flag: '🇨🇳' },
  { id: 'DE', name: 'Allemagne', type: 'country', flag: '🇩🇪' },
  { id: 'JP', name: 'Japon', type: 'country', flag: '🇯🇵' },
  { id: 'AAPL', name: 'Apple Inc.', type: 'company', flag: '🍎' },
  { id: 'TSLA', name: 'Tesla', type: 'company', flag: '⚡' },
  { id: 'NVDA', name: 'NVIDIA', type: 'company', flag: '🎮' },
  { id: 'USD', name: 'Dollar américain', type: 'currency', flag: '💵' },
  { id: 'EUR', name: 'Euro', type: 'currency', flag: '💶' },
  { id: 'CRUDE_OIL', name: 'Pétrole brut', type: 'commodity', flag: '🛢️' },
  { id: 'GOLD', name: 'Or', type: 'commodity', flag: '🥇' },
  { id: 'FED', name: 'Réserve fédérale', type: 'institution', flag: '🏛️' }
]

export default function CascadeAnalysisPanel({ onAnalyze, isAnalyzing }: CascadeAnalysisPanelProps) {
  const [selectedEventType, setSelectedEventType] = useState('war')
  const [selectedEntity, setSelectedEntity] = useState('US')
  const [magnitude, setMagnitude] = useState(0.5)
  const [customEntity, setCustomEntity] = useState('')
  const [useCustomEntity, setUseCustomEntity] = useState(false)
  const [showHelp, setShowHelp] = useState(false)

  const getMagnitudeLabel = (value: number) => {
    if (value > 0.7) return { label: 'Très positif', color: 'text-green-400' }
    if (value > 0.3) return { label: 'Positif modéré', color: 'text-green-300' }
    if (value > 0.1) return { label: 'Légèrement positif', color: 'text-green-200' }
    if (value > -0.1) return { label: 'Neutre', color: 'text-yellow-400' }
    if (value > -0.3) return { label: 'Légèrement négatif', color: 'text-red-200' }
    if (value > -0.7) return { label: 'Négatif modéré', color: 'text-red-300' }
    return { label: 'Très négatif', color: 'text-red-400' }
  }

  const handleAnalyze = () => {
    const entityId = useCustomEntity ? customEntity.trim() : selectedEntity

    if (!entityId) {
      return
    }

    onAnalyze({
      type: selectedEventType,
      entity: entityId,
      magnitude: magnitude
    })
  }

  const selectedEventInfo = eventTypes.find(et => et.value === selectedEventType)
  const magnitudeInfo = getMagnitudeLabel(magnitude)

  return (
    <div className="bg-dark-300 sharp-card border border-dark-400">
      <div className="p-4 border-b border-dark-400">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <BoltIcon className="w-5 h-5 mr-2 text-yellow-400" />
            Analyse d'Impact
          </h3>
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="p-1 sharp-buttonhover:bg-dark-200 text-dark-400 hover:text-white transition-colors"
          >
            <InformationCircleIcon className="w-5 h-5" />
          </button>
        </div>

        {showHelp && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 p-3 bg-blue-900/20 sharp-card border border-blue-700/50"
          >
            <p className="text-sm text-blue-200">
              L'analyse d'impact prédit comment un événement économique ou géopolitique
              se propage à travers le réseau de relations. Plus la magnitude est élevée,
              plus l'impact est important.
            </p>
          </motion.div>
        )}
      </div>

      <div className="p-4 space-y-6">
        {/* Event Type Selection */}
        <div>
          <label className="block text-sm font-medium text-dark-300 mb-3">
            Type d'événement
          </label>
          <div className="grid grid-cols-1 gap-2">
            {eventTypes.map((eventType) => (
              <motion.button
                key={eventType.value}
                onClick={() => setSelectedEventType(eventType.value)}
                className={`p-3 sharp-card border-2 transition-all text-left ${
                  selectedEventType === eventType.value
                    ? 'border-accent-blue bg-accent-blue/10 shadow-md'
                    : 'border-dark-400 hover:border-dark-300 hover:bg-dark-200'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-lg">{eventType.icon}</span>
                  <div className="flex-1">
                    <div className="font-medium text-white text-sm">
                      {eventType.label}
                    </div>
                    <div className="text-xs text-dark-400">
                      {eventType.description}
                    </div>
                  </div>
                  {selectedEventType === eventType.value && (
                    <div className="w-2 h-2 bg-accent-blue rounded-full" />
                  )}
                </div>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Entity Selection */}
        <div>
          <label className="block text-sm font-medium text-dark-300 mb-3">
            Entité d'origine
          </label>

          <div className="space-y-3">
            {/* Toggle between predefined and custom */}
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setUseCustomEntity(false)}
                className={`px-3 py-1 sharp-buttontext-sm transition-colors ${
                  !useCustomEntity
                    ? 'bg-accent-blue text-white'
                    : 'bg-dark-400 text-dark-200 hover:text-white'
                }`}
              >
                Prédéfinies
              </button>
              <button
                onClick={() => setUseCustomEntity(true)}
                className={`px-3 py-1 sharp-buttontext-sm transition-colors ${
                  useCustomEntity
                    ? 'bg-accent-blue text-white'
                    : 'bg-dark-400 text-dark-200 hover:text-white'
                }`}
              >
                Personnalisé
              </button>
            </div>

            {/* Entity Selection */}
            {useCustomEntity ? (
              <input
                type="text"
                value={customEntity}
                onChange={(e) => setCustomEntity(e.target.value)}
                placeholder="ex: GOOGL, BTC, OIL..."
                className="w-full px-3 py-2 bg-dark-200 border border-dark-400 sharp-card text-white text-sm focus:border-accent-blue focus:ring-1 focus:ring-accent-blue outline-none"
              />
            ) : (
              <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto">
                {commonEntities.map((entity) => (
                  <motion.button
                    key={entity.id}
                    onClick={() => setSelectedEntity(entity.id)}
                    className={`p-2 sharp-card border transition-all text-left ${
                      selectedEntity === entity.id
                        ? 'border-accent-blue bg-accent-blue/10'
                        : 'border-dark-400 hover:border-dark-300 hover:bg-dark-200'
                    }`}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{entity.flag}</span>
                      <div className="flex-1">
                        <div className="text-sm font-medium text-white">
                          {entity.name}
                        </div>
                        <div className="text-xs text-dark-400 capitalize">
                          {entity.type} • {entity.id}
                        </div>
                      </div>
                      {selectedEntity === entity.id && (
                        <div className="w-2 h-2 bg-accent-blue rounded-full" />
                      )}
                    </div>
                  </motion.button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Magnitude Slider */}
        <div>
          <label className="block text-sm font-medium text-dark-300 mb-3">
            Magnitude d'impact
          </label>

          <div className="space-y-3">
            <div className="relative">
              <input
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={magnitude}
                onChange={(e) => setMagnitude(parseFloat(e.target.value))}
                className="w-full h-2 bg-dark-400 sharp-card appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right,
                    #ef4444 0%,
                    #f59e0b 25%,
                    #10b981 50%,
                    #3b82f6 75%,
                    #8b5cf6 100%
                  )`
                }}
              />
              <div className="flex justify-between text-xs text-dark-400 mt-1">
                <span>-1.0</span>
                <span>0</span>
                <span>+1.0</span>
              </div>
            </div>

            <div className="bg-dark-200 sharp-card p-3 text-center">
              <div className={`text-lg font-bold ${magnitudeInfo.color}`}>
                {magnitude > 0 ? '+' : ''}{magnitude.toFixed(1)}
              </div>
              <div className={`text-sm ${magnitudeInfo.color}`}>
                {magnitudeInfo.label}
              </div>
            </div>
          </div>
        </div>

        {/* Current Selection Summary */}
        {selectedEventInfo && (
          <div className="bg-dark-200 sharp-card p-3 border border-dark-400">
            <h4 className="text-sm font-medium text-white mb-2">Analyse prévue</h4>
            <div className="space-y-1 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-dark-400">Événement:</span>
                <span className="text-white flex items-center">
                  <span className="mr-1">{selectedEventInfo.icon}</span>
                  {selectedEventInfo.label}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-dark-400">Entité:</span>
                <span className="text-white">
                  {useCustomEntity ? customEntity || '(non spécifiée)' : selectedEntity}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-dark-400">Impact:</span>
                <span className={`font-medium ${magnitudeInfo.color}`}>
                  {magnitude > 0 ? '+' : ''}{magnitude.toFixed(1)} ({magnitudeInfo.label})
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Analyze Button */}
        <motion.button
          onClick={handleAnalyze}
          disabled={isAnalyzing || (!useCustomEntity && !selectedEntity) || (useCustomEntity && !customEntity.trim())}
          className={`w-full py-3 px-4 sharp-card font-medium transition-all flex items-center justify-center space-x-2 ${
            isAnalyzing
              ? 'bg-yellow-600 text-white cursor-not-allowed'
              : ((!useCustomEntity && !selectedEntity) || (useCustomEntity && !customEntity.trim()))
              ? 'bg-dark-400 text-dark-500 cursor-not-allowed'
              : 'bg-gradient-to-r from-yellow-600 to-red-600 text-white hover:from-yellow-500 hover:to-red-500 shadow-lg hover:shadow-xl'
          }`}
          whileHover={!isAnalyzing ? { scale: 1.02 } : {}}
          whileTap={!isAnalyzing ? { scale: 0.98 } : {}}
        >
          {isAnalyzing ? (
            <>
              <ArrowPathIcon className="w-5 h-5 animate-spin" />
              <span>Analyse en cours...</span>
            </>
          ) : (
            <>
              <BoltIcon className="w-5 h-5" />
              <span>Analyser les effets de cascade</span>
            </>
          )}
        </motion.button>

        {/* Warning for high magnitude */}
        {Math.abs(magnitude) > 0.7 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-start space-x-2 p-3 bg-yellow-900/20 sharp-card border border-yellow-700/50"
          >
            <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-yellow-200">
              <strong>Magnitude élevée:</strong> Cette analyse simule un événement majeur
              avec des impacts significatifs sur l'économie mondiale. Les résultats peuvent
              inclure des effets de cascade importants.
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

// CSS pour le slider personnalisé
const styles = `
.slider::-webkit-slider-thumb {
  appearance: none;
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: #58a6ff;
  cursor: pointer;
  box-shadow: 0 0 10px rgba(88, 166, 255, 0.5);
  border: 2px solid #0d1117;
}

.slider::-moz-range-thumb {
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: #58a6ff;
  cursor: pointer;
  box-shadow: 0 0 10px rgba(88, 166, 255, 0.5);
  border: 2px solid #0d1117;
}
`

// Inject styles
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement("style")
  styleSheet.innerText = styles
  document.head.appendChild(styleSheet)
}