'use client'

import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUpIcon,
  TrendingDownIcon,
  MinusIcon,
} from '@heroicons/react/24/outline'

interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  changeType?: 'increase' | 'decrease' | 'neutral'
  icon?: ReactNode
  formatValue?: (value: string | number) => string
  loading?: boolean
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple'
  size?: 'sm' | 'md' | 'lg'
}

export default function MetricCard({
  title,
  value,
  change,
  changeType,
  icon,
  formatValue,
  loading = false,
  color = 'blue',
  size = 'md',
}: MetricCardProps) {

  const colorClasses = {
    blue: 'text-accent-blue border-accent-blue bg-accent-blue/10',
    green: 'text-trading-profit border-trading-profit bg-trading-profit/10',
    red: 'text-trading-loss border-trading-loss bg-trading-loss/10',
    yellow: 'text-accent-yellow border-accent-yellow bg-accent-yellow/10',
    purple: 'text-accent-purple border-accent-purple bg-accent-purple/10',
  }

  const sizeClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  }

  const titleSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  }

  const valueSizeClasses = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-3xl',
  }

  const formatDisplayValue = (val: string | number) => {
    if (formatValue) return formatValue(val)
    if (typeof val === 'number') {
      if (Math.abs(val) >= 1000000) {
        return `${(val / 1000000).toFixed(1)}M`
      }
      if (Math.abs(val) >= 1000) {
        return `${(val / 1000).toFixed(1)}K`
      }
      return val.toFixed(2)
    }
    return val.toString()
  }

  const getChangeIcon = () => {
    if (!change) return null

    switch (changeType) {
      case 'increase':
        return <TrendingUpIcon className="w-4 h-4 text-trading-profit" />
      case 'decrease':
        return <TrendingDownIcon className="w-4 h-4 text-trading-loss" />
      default:
        return <MinusIcon className="w-4 h-4 text-trading-neutral" />
    }
  }

  const getChangeColor = () => {
    switch (changeType) {
      case 'increase':
        return 'text-trading-profit'
      case 'decrease':
        return 'text-trading-loss'
      default:
        return 'text-trading-neutral'
    }
  }

  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`bg-dark-200 rounded-xl border border-dark-300 ${sizeClasses[size]}`}
      >
        <div className="animate-pulse">
          <div className="h-4 bg-dark-300 rounded mb-3"></div>
          <div className="h-8 bg-dark-300 rounded mb-2"></div>
          <div className="h-3 bg-dark-300 rounded w-1/2"></div>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
      className={`
        bg-dark-200 rounded-xl border border-dark-300 ${sizeClasses[size]}
        hover:border-dark-400 transition-all duration-200
        group relative overflow-hidden
      `}
    >
      {/* Background gradient effect */}
      <div className={`
        absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300
        ${colorClasses[color]}
      `} />

      {/* Content */}
      <div className="relative z-10">
        {/* Header with title and icon */}
        <div className="flex items-center justify-between mb-3">
          <h3 className={`
            font-medium text-dark-500 group-hover:text-white transition-colors duration-200
            ${titleSizeClasses[size]}
          `}>
            {title}
          </h3>

          {icon && (
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className={`text-dark-400 group-hover:${colorClasses[color].split(' ')[0]} transition-colors duration-200`}
            >
              {icon}
            </motion.div>
          )}
        </div>

        {/* Value */}
        <div className={`
          font-bold text-white mb-2 font-mono
          ${valueSizeClasses[size]}
        `}>
          {formatDisplayValue(value)}
        </div>

        {/* Change indicator */}
        {change !== undefined && (
          <div className="flex items-center space-x-2">
            {getChangeIcon()}
            <span className={`text-sm font-medium ${getChangeColor()}`}>
              {Math.abs(change).toFixed(2)}%
              {changeType === 'increase' && ' ↑'}
              {changeType === 'decrease' && ' ↓'}
            </span>
            <span className="text-xs text-dark-500">vs last period</span>
          </div>
        )}

        {/* Animated bottom border */}
        <motion.div
          className={`
            absolute bottom-0 left-0 h-1 bg-gradient-to-r
            ${color === 'blue' ? 'from-accent-blue to-accent-purple' : ''}
            ${color === 'green' ? 'from-trading-profit to-accent-blue' : ''}
            ${color === 'red' ? 'from-trading-loss to-accent-yellow' : ''}
            ${color === 'yellow' ? 'from-accent-yellow to-accent-purple' : ''}
            ${color === 'purple' ? 'from-accent-purple to-accent-blue' : ''}
          `}
          initial={{ width: 0 }}
          animate={{ width: '100%' }}
          transition={{ duration: 1, delay: 0.2 }}
        />
      </div>
    </motion.div>
  )
}