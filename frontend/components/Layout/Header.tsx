'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  WifiIcon,
  ClockIcon,
  SignalIcon,
} from '@heroicons/react/24/outline'

interface HeaderProps {
  title: string
  subtitle?: string
  isConnected?: boolean
}

export default function Header({ title, subtitle, isConnected = true }: HeaderProps) {
  const [currentTime, setCurrentTime] = useState<string>('')
  const [lastUpdate, setLastUpdate] = useState<string>('')

  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      setCurrentTime(now.toLocaleTimeString())
    }

    const updateLastUpdate = () => {
      const now = new Date()
      setLastUpdate(now.toLocaleTimeString())
    }

    // Update time every second
    const timeInterval = setInterval(updateTime, 1000)

    // Update last update time every 30 seconds (simulating data refresh)
    const updateInterval = setInterval(updateLastUpdate, 30000)

    // Initialize
    updateTime()
    updateLastUpdate()

    return () => {
      clearInterval(timeInterval)
      clearInterval(updateInterval)
    }
  }, [])

  return (
    <header className="bg-dark-200 border-b border-dark-300 py-4 px-6">
      <div className="flex items-center justify-between">
        {/* Left side - Page Info */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-2xl font-bold text-white mb-1">{title}</h1>
          {subtitle && (
            <p className="text-sm text-dark-500">{subtitle}</p>
          )}
        </motion.div>

        {/* Right side - Status Info */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="flex items-center space-x-6"
        >
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <motion.div
              animate={{
                scale: isConnected ? [1, 1.1, 1] : 1,
                opacity: isConnected ? 1 : 0.5
              }}
              transition={{
                duration: 2,
                repeat: isConnected ? Infinity : 0
              }}
              className={`
                flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium
                ${isConnected
                  ? 'bg-trading-profit/20 text-trading-profit border border-trading-profit/30'
                  : 'bg-trading-loss/20 text-trading-loss border border-trading-loss/30'
                }
              `}
            >
              <div className={`
                w-2 h-2 rounded-full
                ${isConnected ? 'bg-trading-profit' : 'bg-trading-loss'}
              `} />
              <WifiIcon className="w-4 h-4" />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </motion.div>
          </div>

          {/* Last Update */}
          <div className="flex items-center space-x-2 text-dark-500">
            <ClockIcon className="w-4 h-4" />
            <span className="text-xs">
              Last Update: <span className="text-white font-mono">{lastUpdate || '--:--:--'}</span>
            </span>
          </div>

          {/* Current Time */}
          <div className="flex items-center space-x-2 text-dark-500">
            <SignalIcon className="w-4 h-4" />
            <span className="text-xs">
              <span className="text-white font-mono">{currentTime}</span> UTC
            </span>
          </div>

          {/* Live Indicator */}
          <motion.div
            className="flex items-center space-x-2"
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
            <span className="text-xs text-red-500 font-medium">LIVE</span>
          </motion.div>
        </motion.div>
      </div>
    </header>
  )
}