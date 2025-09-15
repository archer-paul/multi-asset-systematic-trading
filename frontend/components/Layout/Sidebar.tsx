'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import {
  HomeIcon,
  CpuChipIcon,
  ShieldCheckIcon,
  BriefcaseIcon,
  ChevronLeftIcon,
  ChartBarIcon,
  ShareIcon,
  SparklesIcon,
  ChevronDownIcon,
  BuildingLibraryIcon,
  TrendingUpIcon,
  ClockIcon,
  PresentationChartLineIcon,
} from '@heroicons/react/24/outline'

const coreMenuItems = [
  {
    id: 'dashboard',
    name: 'Dashboard',
    href: '/',
    icon: HomeIcon,
    description: 'Overview & Performance',
  },
  {
    id: 'sentiment-analysis',
    name: 'Sentiment Analysis',
    href: '/sentiment-analysis',
    icon: SparklesIcon,
    description: 'Market & Geopolitical Mood',
  },
  {
    id: 'machine-learning',
    name: 'Machine Learning',
    href: '/machine-learning',
    icon: CpuChipIcon,
    description: 'Models & Analytics',
  },
  {
    id: 'risk-management',
    name: 'Risk Management',
    href: '/risk-management',
    icon: ShieldCheckIcon,
    description: 'Risk Analysis & Control',
  },
  {
    id: 'portfolio',
    name: 'Portfolio',
    href: '/portfolio',
    icon: BriefcaseIcon,
    description: 'Holdings & Allocation',
  },
  {
    id: 'knowledge-graph',
    name: 'Knowledge Graph',
    href: '/knowledge-graph',
    icon: ShareIcon,
    description: 'Economic Relations',
  },
]

const advancedMenuItems = [
  {
    id: 'congress-trading',
    name: 'Congress Trading',
    href: '/congress-trading',
    icon: BuildingLibraryIcon,
    description: 'Congressional Trading Analysis',
  },
  {
    id: 'emerging-stocks',
    name: 'Emerging Stocks',
    href: '/emerging-stocks',
    icon: TrendingUpIcon,
    description: 'AI Growth Stock Detection',
  },
  {
    id: 'long-term-analysis',
    name: 'Long-term Analysis',
    href: '/long-term-analysis',
    icon: ClockIcon,
    description: '3-5 Year Investment Analysis',
  },
  {
    id: 'technical-analysis',
    name: 'Technical Analysis',
    href: '/technical-analysis',
    icon: PresentationChartLineIcon,
    description: 'Multi-timeframe TA',
  },
]

interface SidebarProps {
  isCollapsed: boolean
  setIsCollapsed: (collapsed: boolean) => void
}

export default function Sidebar({ isCollapsed, setIsCollapsed }: SidebarProps) {
  const pathname = usePathname()
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)
  const [advancedSectionExpanded, setAdvancedSectionExpanded] = useState(false)

  return (
    <motion.aside
      initial={{ width: isCollapsed ? 80 : 280 }}
      animate={{ width: isCollapsed ? 80 : 280 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="bg-dark-200 border-r border-dark-300 flex flex-col relative z-50"
    >
      {/* Header */}
      <div className="p-6 border-b border-dark-300">
        <div className="flex items-center justify-between">
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.2 }}
                className="flex items-center space-x-3"
              >
                <div className="w-10 h-10 bg-gradient-to-r from-accent-blue to-accent-purple sharp-card flex items-center justify-center">
                  <ChartBarIcon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">QAE</h1>
                  <p className="text-xs text-dark-500">Alpha Engine</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-2 sharp-button hover:bg-dark-300 transition-colors duration-200"
          >
            <motion.div
              animate={{ rotate: isCollapsed ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChevronLeftIcon className="w-5 h-5 text-dark-500" />
            </motion.div>
          </button>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 py-6 overflow-y-auto">
        <div className="space-y-2 px-3">
          {/* Core Menu Items */}
          {coreMenuItems.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon

            return (
              <div key={item.id} className="relative">
                <Link href={item.href}>
                  <motion.div
                    className={`
                      group flex items-center px-3 py-3 sharp-button transition-all duration-200 cursor-pointer
                      ${isActive
                        ? 'bg-gradient-to-r from-accent-blue/20 to-accent-purple/20 border-l-4 border-accent-blue'
                        : 'hover:bg-dark-300/50'
                      }
                    `}
                    onHoverStart={() => setHoveredItem(item.id)}
                    onHoverEnd={() => setHoveredItem(null)}
                    whileHover={{ x: isCollapsed ? 0 : 4 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center min-w-0 flex-1">
                      <div className={`
                        flex-shrink-0 w-8 h-8 flex items-center justify-center sharp-card transition-all duration-200
                        ${isActive
                          ? 'bg-accent-blue text-white glow'
                          : 'text-dark-500 group-hover:text-accent-blue group-hover:bg-accent-blue/10'
                        }
                      `}>
                        <Icon className="w-5 h-5" />
                      </div>

                      <AnimatePresence mode="wait">
                        {!isCollapsed && (
                          <motion.div
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            transition={{ duration: 0.2 }}
                            className="ml-4 min-w-0 flex-1"
                          >
                            <p className={`
                              text-sm font-medium transition-colors duration-200
                              ${isActive ? 'text-white' : 'text-dark-500 group-hover:text-white'}
                            `}>
                              {item.name}
                            </p>
                            <p className={`
                              text-xs transition-colors duration-200
                              ${isActive ? 'text-accent-blue' : 'text-dark-500 group-hover:text-dark-400'}
                            `}>
                              {item.description}
                            </p>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {isActive && !isCollapsed && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="w-2 h-2 bg-accent-blue"
                      />
                    )}
                  </motion.div>
                </Link>

                {/* Tooltip for collapsed state */}
                <AnimatePresence>
                  {isCollapsed && hoveredItem === item.id && (
                    <motion.div
                      initial={{ opacity: 0, x: -10, scale: 0.9 }}
                      animate={{ opacity: 1, x: 0, scale: 1 }}
                      exit={{ opacity: 0, x: -10, scale: 0.9 }}
                      transition={{ duration: 0.2 }}
                      className="absolute left-16 top-0 bg-dark-300 text-white px-3 py-2 sharp-card shadow-lg border border-dark-400 z-50 whitespace-nowrap"
                    >
                      <div className="text-sm font-medium">{item.name}</div>
                      <div className="text-xs text-dark-500">{item.description}</div>
                      <div className="absolute left-0 top-1/2 transform -translate-x-1 -translate-y-1/2 w-2 h-2 bg-dark-300 border-l border-b border-dark-400"></div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )
          })}

          {/* Advanced Analytics Section */}
          {!isCollapsed && (
            <div className="pt-4">
              <motion.div
                className="px-3 py-2 cursor-pointer group"
                onClick={() => setAdvancedSectionExpanded(!advancedSectionExpanded)}
                whileHover={{ x: 2 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-6 h-6 flex items-center justify-center">
                      <motion.div
                        animate={{ rotate: advancedSectionExpanded ? 180 : 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <ChevronDownIcon className="w-4 h-4 text-dark-500 group-hover:text-accent-blue transition-colors" />
                      </motion.div>
                    </div>
                    <span className="text-xs font-semibold text-dark-500 group-hover:text-accent-blue transition-colors uppercase tracking-wider">
                      Advanced Analytics
                    </span>
                  </div>
                </div>
              </motion.div>

              <AnimatePresence>
                {advancedSectionExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: 'easeInOut' }}
                    className="overflow-hidden"
                  >
                    <div className="space-y-1 pt-2">
                      {advancedMenuItems.map((item) => {
                        const isActive = pathname === item.href
                        const Icon = item.icon

                        return (
                          <div key={item.id} className="relative">
                            <Link href={item.href}>
                              <motion.div
                                className={`
                                  group flex items-center px-3 py-2 ml-4 sharp-button transition-all duration-200 cursor-pointer
                                  ${isActive
                                    ? 'bg-gradient-to-r from-accent-purple/20 to-accent-blue/20 border-l-2 border-accent-purple'
                                    : 'hover:bg-dark-300/30'
                                  }
                                `}
                                onHoverStart={() => setHoveredItem(item.id)}
                                onHoverEnd={() => setHoveredItem(null)}
                                whileHover={{ x: 4 }}
                                whileTap={{ scale: 0.98 }}
                              >
                                <div className="flex items-center min-w-0 flex-1">
                                  <div className={`
                                    flex-shrink-0 w-6 h-6 flex items-center justify-center sharp-card transition-all duration-200
                                    ${isActive
                                      ? 'bg-accent-purple text-white glow'
                                      : 'text-dark-500 group-hover:text-accent-purple group-hover:bg-accent-purple/10'
                                    }
                                  `}>
                                    <Icon className="w-4 h-4" />
                                  </div>

                                  <div className="ml-3 min-w-0 flex-1">
                                    <p className={`
                                      text-xs font-medium transition-colors duration-200
                                      ${isActive ? 'text-white' : 'text-dark-500 group-hover:text-white'}
                                    `}>
                                      {item.name}
                                    </p>
                                    <p className={`
                                      text-xs transition-colors duration-200
                                      ${isActive ? 'text-accent-purple' : 'text-dark-600 group-hover:text-dark-400'}
                                    `}>
                                      {item.description}
                                    </p>
                                  </div>
                                </div>

                                {isActive && (
                                  <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className="w-1.5 h-1.5 bg-accent-purple rounded-full"
                                  />
                                )}
                              </motion.div>
                            </Link>
                          </div>
                        )
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Advanced items for collapsed sidebar */}
          {isCollapsed && advancedMenuItems.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon

            return (
              <div key={item.id} className="relative">
                <Link href={item.href}>
                  <motion.div
                    className={`
                      group flex items-center px-3 py-2 sharp-button transition-all duration-200 cursor-pointer
                      ${isActive
                        ? 'bg-gradient-to-r from-accent-purple/20 to-accent-blue/20 border-l-4 border-accent-purple'
                        : 'hover:bg-dark-300/50'
                      }
                    `}
                    onHoverStart={() => setHoveredItem(item.id)}
                    onHoverEnd={() => setHoveredItem(null)}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center min-w-0 flex-1 justify-center">
                      <div className={`
                        flex-shrink-0 w-8 h-8 flex items-center justify-center sharp-card transition-all duration-200
                        ${isActive
                          ? 'bg-accent-purple text-white glow'
                          : 'text-dark-500 group-hover:text-accent-purple group-hover:bg-accent-purple/10'
                        }
                      `}>
                        <Icon className="w-5 h-5" />
                      </div>
                    </div>
                  </motion.div>
                </Link>

                {/* Tooltip for collapsed state */}
                <AnimatePresence>
                  {isCollapsed && hoveredItem === item.id && (
                    <motion.div
                      initial={{ opacity: 0, x: -10, scale: 0.9 }}
                      animate={{ opacity: 1, x: 0, scale: 1 }}
                      exit={{ opacity: 0, x: -10, scale: 0.9 }}
                      transition={{ duration: 0.2 }}
                      className="absolute left-16 top-0 bg-dark-300 text-white px-3 py-2 sharp-card shadow-lg border border-dark-400 z-50 whitespace-nowrap"
                    >
                      <div className="text-sm font-medium">{item.name}</div>
                      <div className="text-xs text-dark-500">{item.description}</div>
                      <div className="absolute left-0 top-1/2 transform -translate-x-1 -translate-y-1/2 w-2 h-2 bg-dark-300 border-l border-b border-dark-400"></div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )
          })}
        </div>
      </nav>

      {/* Status Indicator */}
      <div className="p-4 border-t border-dark-300">
        <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'space-x-3'}`}>
          <div className="relative">
            <div className="w-3 h-3 bg-trading-profit status-online"></div>
            <div className="absolute inset-0 w-3 h-3 bg-trading-profit animate-ping opacity-75"></div>
          </div>

          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                <p className="text-xs text-dark-500">System Online</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.aside>
  )
}