'use client'

import React, { Component, ReactNode, ErrorInfo } from 'react'
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error Boundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="min-h-screen flex items-center justify-center bg-dark-100">
          <div className="bg-dark-200 sharp-card p-8 border border-dark-300 max-w-md w-full mx-4">
            <div className="flex items-center space-x-3 mb-4">
              <ExclamationTriangleIcon className="w-8 h-8 text-yellow-400" />
              <h2 className="text-xl font-semibold text-white">Something went wrong</h2>
            </div>

            <p className="text-dark-500 mb-6">
              An error occurred while rendering this component. Please refresh the page to try again.
            </p>

            {this.state.error && (
              <details className="mb-6">
                <summary className="text-sm text-dark-400 cursor-pointer hover:text-white">
                  Error details
                </summary>
                <pre className="mt-2 text-xs text-red-400 bg-dark-300 p-3 rounded overflow-auto">
                  {this.state.error.message}
                </pre>
              </details>
            )}

            <button
              onClick={() => window.location.reload()}
              className="w-full bg-accent-blue hover:bg-accent-blue/80 text-white font-medium py-2 px-4 sharp-button transition-colors"
            >
              Refresh Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary