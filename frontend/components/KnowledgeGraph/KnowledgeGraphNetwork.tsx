'use client'

import { useEffect, useRef, useImperativeHandle, forwardRef, useState } from 'react'
import { Network } from 'vis-network'
import { DataSet } from 'vis-data'
import { knowledgeGraphAPI } from '../../lib/api'
import toast from 'react-hot-toast'

interface Node {
  id: string
  name: string
  type: string
  region: string
  importance: number
  size: number
  color: string
  metadata: Record<string, any>
}

interface Link {
  source: string
  target: string
  type: string
  strength: number
  width: number
  color: string
  metadata: Record<string, any>
}

interface GraphData {
  nodes: Node[]
  links: Link[]
}

interface GraphFilters {
  entity_types: string[]
  regions: string[]
  min_importance: number
  max_nodes: number
}

interface KnowledgeGraphNetworkProps {
  onEntitySelect: (entityId: string | null) => void
  physicsEnabled: boolean
  onLoading: (loading: boolean) => void
  onMockModeChanged?: (isUsingMock: boolean) => void
}

export interface KnowledgeGraphNetworkRef {
  loadData: (filters?: GraphFilters) => Promise<void>
  applyFilters: (filters: GraphFilters) => Promise<void>
  exportToPNG: () => void
  fitNetwork: () => void
  togglePhysics: (enabled: boolean) => void
}

const KnowledgeGraphNetwork = forwardRef<KnowledgeGraphNetworkRef, KnowledgeGraphNetworkProps>(
  ({ onEntitySelect, physicsEnabled, onLoading, onMockModeChanged }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null)
    const networkRef = useRef<Network | null>(null)
    const nodesRef = useRef<DataSet<any> | null>(null)
    const edgesRef = useRef<DataSet<any> | null>(null)
    const [currentFilters, setCurrentFilters] = useState<GraphFilters>({
      entity_types: ['company', 'country', 'currency', 'commodity', 'institution'],
      regions: ['US', 'EU', 'GLOBAL'],
      min_importance: 0,
      max_nodes: 100
    })
    const [isUsingMockData, setIsUsingMockData] = useState(false)

    const getNodeShape = (nodeType: string) => {
      const shapes = {
        'company': 'dot',
        'country': 'box',
        'currency': 'diamond',
        'commodity': 'triangle',
        'institution': 'star',
        'politician': 'hexagon',
        'sector': 'ellipse',
        'event': 'database'
      }
      return shapes[nodeType as keyof typeof shapes] || 'dot'
    }

    const createNodeTooltip = (node: Node) => {
      let tooltip = `<div style="padding: 12px; background: #161b22; border-radius: 8px; color: #e6edf3; max-width: 300px; border: 1px solid #30363d;">
        <div style="font-weight: bold; margin-bottom: 6px; color: #f0f6fc;">${node.name}</div>
        <div style="color: #8b949e; font-size: 12px; margin-bottom: 2px;">Type: <span style="color: #58a6ff;">${node.type}</span></div>
        <div style="color: #8b949e; font-size: 12px; margin-bottom: 2px;">Région: <span style="color: #58a6ff;">${node.region}</span></div>
        <div style="color: #8b949e; font-size: 12px; margin-bottom: 4px;">Importance: <span style="color: #f85149;">${node.importance.toFixed(3)}</span></div>`

      if (node.metadata.sector) {
        tooltip += `<div style="color: #8b949e; font-size: 12px;">Secteur: <span style="color: #58a6ff;">${node.metadata.sector}</span></div>`
      }
      if (node.metadata.market_cap_billion) {
        tooltip += `<div style="color: #8b949e; font-size: 12px;">Cap. boursière: <span style="color: #3fb950;">$${node.metadata.market_cap_billion}B</span></div>`
      }

      tooltip += '</div>'
      return tooltip
    }

    const createEdgeTooltip = (edge: any) => {
      return `<div style="padding: 8px; background: #161b22; border-radius: 6px; color: #e6edf3; border: 1px solid #30363d;">
        <div style="font-weight: bold; margin-bottom: 4px; color: #f0f6fc;">${edge.type.replace('_', ' ').toUpperCase()}</div>
        <div style="color: #58a6ff; font-size: 12px;">Force: ${edge.strength?.toFixed(2) || 'N/A'}</div>
        ${edge.metadata?.critical ? '<div style="color: #f85149; font-size: 12px;">🔴 Critique</div>' : ''}
      </div>`
    }

    const darkenColor = (hex: string, factor: number) => {
      const num = parseInt(hex.replace('#', ''), 16)
      const amt = Math.round(255 * factor)
      const R = Math.max(0, (num >> 16) - amt)
      const G = Math.max(0, (num >> 8 & 0x00FF) - amt)
      const B = Math.max(0, (num & 0x0000FF) - amt)
      return '#' + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1)
    }

    const lightenColor = (hex: string, factor: number) => {
      const num = parseInt(hex.replace('#', ''), 16)
      const amt = Math.round(255 * factor)
      const R = Math.min(255, (num >> 16) + amt)
      const G = Math.min(255, (num >> 8 & 0x00FF) + amt)
      const B = Math.min(255, (num & 0x0000FF) + amt)
      return '#' + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1)
    }

    const loadData = async (filters?: GraphFilters) => {
      const filtersToUse = filters || currentFilters
      setCurrentFilters(filtersToUse)
      onLoading(true)

      // If we're already using mock data, don't try API again
      if (isUsingMockData) {
        console.log('Already using mock data, skipping API call')
        onLoading(false)
        return
      }

      try {
        console.log('Loading graph data with filters:', filtersToUse)
        const response = await knowledgeGraphAPI.getVisualizationData(filtersToUse)
        const data: GraphData = response.data

        console.log(`Loaded ${data.nodes.length} nodes and ${data.links.length} links`)

        // Prepare nodes for vis-network
        const visNodes = data.nodes.map(node => ({
          id: node.id,
          label: node.name,
          title: createNodeTooltip(node),
          shape: getNodeShape(node.type),
          color: {
            background: node.color,
            border: darkenColor(node.color, 0.3),
            highlight: {
              background: lightenColor(node.color, 0.2),
              border: node.color
            },
            hover: {
              background: lightenColor(node.color, 0.1),
              border: node.color
            }
          },
          size: Math.max(15, Math.min(50, node.size)),
          font: {
            color: '#e6edf3',
            size: 12,
            face: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
          },
          borderWidth: 2,
          shadow: {
            enabled: true,
            color: 'rgba(0,0,0,0.3)',
            size: 8,
            x: 2,
            y: 2
          },
          metadata: node.metadata
        }))

        // Prepare edges for vis-network
        const visEdges = data.links.map((link, index) => ({
          id: index,
          from: link.source,
          to: link.target,
          title: createEdgeTooltip(link),
          color: {
            color: link.color,
            opacity: 0.7,
            highlight: lightenColor(link.color, 0.3),
            hover: lightenColor(link.color, 0.2)
          },
          width: Math.max(1, Math.min(6, link.width)),
          arrows: {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: 'arrow'
            }
          },
          smooth: {
            type: 'continuous',
            forceDirection: 'none',
            roundness: 0.3
          },
          shadow: true,
          metadata: link.metadata
        }))

        // Update datasets
        if (nodesRef.current) {
          nodesRef.current.clear()
          nodesRef.current.add(visNodes)
        } else {
          nodesRef.current = new DataSet(visNodes)
        }

        if (edgesRef.current) {
          edgesRef.current.clear()
          edgesRef.current.add(visEdges)
        } else {
          edgesRef.current = new DataSet(visEdges)
        }

        // Create or update network
        if (!networkRef.current && containerRef.current) {
          initializeNetwork()
        }

        setIsUsingMockData(false)
        onMockModeChanged?.(false)
        toast.success(`Graphe chargé: ${data.nodes.length} entités, ${data.links.length} relations`)

      } catch (error: any) {
        console.error('Error loading graph data:', error)

        // Fallback to mock data when API is not available
        console.log('API not available, using mock data for knowledge graph')
        const mockData: GraphData = {
          nodes: [
            {
              id: '1',
              name: 'Apple Inc.',
              type: 'company',
              region: 'US',
              importance: 0.95,
              size: 40,
              color: '#3498db',
              metadata: { sector: 'Technology', market_cap_billion: 3000 }
            },
            {
              id: '2',
              name: 'United States',
              type: 'country',
              region: 'US',
              importance: 0.98,
              size: 50,
              color: '#e74c3c',
              metadata: { gdp_trillion: 25.4 }
            },
            {
              id: '3',
              name: 'USD',
              type: 'currency',
              region: 'GLOBAL',
              importance: 0.99,
              size: 45,
              color: '#f1c40f',
              metadata: { symbol: '$' }
            },
            {
              id: '4',
              name: 'Gold',
              type: 'commodity',
              region: 'GLOBAL',
              importance: 0.85,
              size: 35,
              color: '#8e44ad',
              metadata: { price_per_oz: 2000 }
            },
            {
              id: '5',
              name: 'Federal Reserve',
              type: 'institution',
              region: 'US',
              importance: 0.92,
              size: 42,
              color: '#2ecc71',
              metadata: { type: 'central_bank' }
            },
            {
              id: '6',
              name: 'Microsoft Corp.',
              type: 'company',
              region: 'US',
              importance: 0.90,
              size: 38,
              color: '#3498db',
              metadata: { sector: 'Technology', market_cap_billion: 2800 }
            },
            {
              id: '7',
              name: 'European Union',
              type: 'country',
              region: 'EU',
              importance: 0.88,
              size: 45,
              color: '#e74c3c',
              metadata: { gdp_trillion: 17.1 }
            },
            {
              id: '8',
              name: 'EUR',
              type: 'currency',
              region: 'EU',
              importance: 0.82,
              size: 40,
              color: '#f1c40f',
              metadata: { symbol: '€' }
            }
          ],
          links: [
            {
              source: '1',
              target: '2',
              type: 'headquartered_in',
              strength: 0.9,
              width: 3,
              color: '#95a5a6',
              metadata: { critical: true }
            },
            {
              source: '1',
              target: '3',
              type: 'trades_in',
              strength: 0.85,
              width: 2,
              color: '#95a5a6',
              metadata: {}
            },
            {
              source: '2',
              target: '3',
              type: 'issues',
              strength: 0.95,
              width: 4,
              color: '#95a5a6',
              metadata: { critical: true }
            },
            {
              source: '5',
              target: '3',
              type: 'controls',
              strength: 0.98,
              width: 5,
              color: '#95a5a6',
              metadata: { critical: true }
            },
            {
              source: '6',
              target: '2',
              type: 'headquartered_in',
              strength: 0.9,
              width: 3,
              color: '#95a5a6',
              metadata: {}
            },
            {
              source: '1',
              target: '6',
              type: 'competes_with',
              strength: 0.75,
              width: 2,
              color: '#95a5a6',
              metadata: {}
            },
            {
              source: '7',
              target: '8',
              type: 'issues',
              strength: 0.92,
              width: 4,
              color: '#95a5a6',
              metadata: { critical: true }
            },
            {
              source: '3',
              target: '4',
              type: 'correlates_with',
              strength: 0.65,
              width: 2,
              color: '#95a5a6',
              metadata: {}
            }
          ]
        }

        // Use the same processing logic as for real data
        const visNodes = mockData.nodes.map(node => ({
          id: node.id,
          label: node.name,
          title: createNodeTooltip(node),
          shape: getNodeShape(node.type),
          color: {
            background: node.color,
            border: darkenColor(node.color, 0.3),
            highlight: {
              background: lightenColor(node.color, 0.2),
              border: node.color
            },
            hover: {
              background: lightenColor(node.color, 0.1),
              border: node.color
            }
          },
          size: Math.max(15, Math.min(50, node.size)),
          font: {
            color: '#e6edf3',
            size: 12,
            face: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
          },
          borderWidth: 2,
          shadow: {
            enabled: true,
            color: 'rgba(0,0,0,0.3)',
            size: 8,
            x: 2,
            y: 2
          },
          metadata: node.metadata
        }))

        const visEdges = mockData.links.map((link, index) => ({
          id: index,
          from: link.source,
          to: link.target,
          title: createEdgeTooltip(link),
          color: {
            color: link.color,
            opacity: 0.7,
            highlight: lightenColor(link.color, 0.3),
            hover: lightenColor(link.color, 0.2)
          },
          width: Math.max(1, Math.min(6, link.width)),
          arrows: {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: 'arrow'
            }
          },
          smooth: {
            type: 'continuous',
            forceDirection: 'none',
            roundness: 0.3
          },
          shadow: true,
          metadata: link.metadata
        }))

        // Update datasets with mock data
        if (nodesRef.current) {
          nodesRef.current.clear()
          nodesRef.current.add(visNodes)
        } else {
          nodesRef.current = new DataSet(visNodes)
        }

        if (edgesRef.current) {
          edgesRef.current.clear()
          edgesRef.current.add(visEdges)
        } else {
          edgesRef.current = new DataSet(visEdges)
        }

        // Create or update network
        if (!networkRef.current && containerRef.current) {
          initializeNetwork()
        }

        setIsUsingMockData(true)
        onMockModeChanged?.(true)
        toast.success(`Graphe de démonstration chargé: ${mockData.nodes.length} entités, ${mockData.links.length} relations (backend indisponible)`)
      } finally {
        onLoading(false)
      }
    }

    const initializeNetwork = () => {
      if (!containerRef.current || !nodesRef.current || !edgesRef.current) return

      const options = {
        physics: {
          enabled: physicsEnabled,
          solver: 'barnesHut',
          barnesHut: {
            gravitationalConstant: -2000,
            centralGravity: 0.3,
            springLength: 95,
            springConstant: 0.04,
            damping: 0.09,
            avoidOverlap: 0.5
          },
          stabilization: {
            enabled: true,
            iterations: 100,
            updateInterval: 25
          }
        },
        interaction: {
          hover: true,
          selectConnectedEdges: false,
          tooltipDelay: 200,
          zoomView: true,
          dragView: true,
          multiselect: false
        },
        nodes: {
          borderWidth: 2,
          shadow: true,
          chosen: true
        },
        edges: {
          shadow: true,
          selectionWidth: 3,
          hoverWidth: 2,
          chosen: true
        },
        layout: {
          improvedLayout: false,
          randomSeed: 42
        },
        groups: {
          company: { color: { background: '#3498db' } },
          country: { color: { background: '#e74c3c' } },
          currency: { color: { background: '#f1c40f' } },
          commodity: { color: { background: '#8e44ad' } },
          institution: { color: { background: '#2ecc71' } }
        }
      }

      networkRef.current = new Network(
        containerRef.current,
        { nodes: nodesRef.current, edges: edgesRef.current },
        options
      )

      // Event handlers
      networkRef.current.on('selectNode', (params) => {
        if (params.nodes.length > 0) {
          onEntitySelect(params.nodes[0])
        }
      })

      networkRef.current.on('deselectNode', () => {
        onEntitySelect(null)
      })

      networkRef.current.on('doubleClick', (params) => {
        if (params.nodes.length > 0) {
          // Double click to focus on node
          networkRef.current?.focus(params.nodes[0], {
            scale: 1.2,
            animation: { duration: 800, easingFunction: 'easeInOutQuad' }
          })
        }
      })

      networkRef.current.on('stabilized', () => {
        console.log('Network stabilized')
        setTimeout(() => fitNetwork(), 100)
      })

      networkRef.current.on('animationFinished', () => {
        console.log('Animation finished')
      })

      console.log('Network initialized')
    }

    const applyFilters = async (filters: GraphFilters) => {
      // If we're already using mock data, don't try to reload from API
      if (isUsingMockData) {
        console.log('Already using mock data, skipping filter application')
        return
      }

      try {
        await loadData(filters)
      } catch (error) {
        console.error('Error applying filters, skipping:', error)
        // Don't retry infinitely, just log and skip
      }
    }

    const exportToPNG = () => {
      if (networkRef.current) {
        try {
          const canvas = (networkRef.current as any).canvas.frame.canvas
          const dataURL = canvas.toDataURL('image/png')

          const link = document.createElement('a')
          link.download = `knowledge-graph-${new Date().toISOString().slice(0, 10)}.png`
          link.href = dataURL
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)

          toast.success('Graphe exporté en PNG')
        } catch (error) {
          console.error('Export error:', error)
          toast.error('Erreur lors de l\'export')
        }
      }
    }

    const fitNetwork = () => {
      if (networkRef.current) {
        networkRef.current.fit({
          animation: { duration: 800, easingFunction: 'easeInOutQuad' }
        })
      }
    }

    const togglePhysics = (enabled: boolean) => {
      if (networkRef.current) {
        networkRef.current.setOptions({ physics: { enabled } })
        if (enabled) {
          toast.success('Physique activée')
        } else {
          toast.success('Physique désactivée')
        }
      }
    }

    useImperativeHandle(ref, () => ({
      loadData,
      applyFilters,
      exportToPNG,
      fitNetwork,
      togglePhysics
    }))

    useEffect(() => {
      // Only load data once when component mounts
      if (!isUsingMockData) {
        loadData()
      }

      return () => {
        if (networkRef.current) {
          networkRef.current.destroy()
          networkRef.current = null
        }
      }
    }, []) // Empty dependency array to run only once

    useEffect(() => {
      if (networkRef.current) {
        networkRef.current.setOptions({ physics: { enabled: physicsEnabled } })
      }
    }, [physicsEnabled])

    return (
      <div
        ref={containerRef}
        className="w-full h-full bg-dark-100 relative overflow-hidden sharp-card border border-dark-300"
        style={{ minHeight: '400px' }}
      >
        {/* Network will be rendered here by vis-network */}
      </div>
    )
  }
)

KnowledgeGraphNetwork.displayName = 'KnowledgeGraphNetwork'

export default KnowledgeGraphNetwork