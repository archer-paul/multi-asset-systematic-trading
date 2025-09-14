'use client'

import { ReactNode, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster } from 'react-hot-toast'
import Sidebar from './Sidebar'
import Header from './Header'
import Footer from './Footer'

interface LayoutProps {
  children: ReactNode
  title: string
  subtitle?: string
}

export default function Layout({ children, title, subtitle }: LayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isConnected, setIsConnected] = useState(true)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // Simulate connection status checks
    const checkConnection = () => {
      // In real implementation, this would check actual connection to backend
      setIsConnected(Math.random() > 0.1) // 90% chance of being connected
    }

    const interval = setInterval(checkConnection, 10000) // Check every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const contentVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: 'easeOut'
      }
    }
  }

  return (
    <div className="flex h-screen bg-dark-100 overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        isCollapsed={sidebarCollapsed}
        setIsCollapsed={setSidebarCollapsed}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <Header
          title={title}
          subtitle={subtitle}
          isConnected={isConnected}
        />

        {/* Main Content */}
        <main className="flex-1 overflow-auto bg-dark-100">
          <AnimatePresence mode="wait">
            <motion.div
              key={title}
              variants={contentVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="p-6 h-full"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>

        {/* Footer */}
        <Footer />
      </div>

      {/* Loading Overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-dark-100/80 backdrop-blur-sm flex items-center justify-center z-50"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="bg-dark-200 rounded-xl p-8 shadow-2xl border border-dark-300"
            >
              <div className="loading-dots">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
              </div>
              <p className="text-white text-center mt-4 font-medium">Loading...</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#2d3748',
            color: '#fff',
            border: '1px solid #4a5568',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </div>
  )
}

// Higher-order component to wrap pages
export function withLayout<P extends object>(
  Component: React.ComponentType<P>,
  title: string,
  subtitle?: string
) {
  return function LayoutWrappedComponent(props: P) {
    return (
      <Layout title={title} subtitle={subtitle}>
        <Component {...props} />
      </Layout>
    )
  }
}