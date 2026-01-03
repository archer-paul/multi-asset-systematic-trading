import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Network } from "lucide-react";

interface Node {
  id: string;
  name: string;
  type: 'company' | 'country' | 'commodity' | 'person' | 'event';
  size: number;
  x: number;
  y: number;
  connections: string[];
}

interface Edge {
  from: string;
  to: string;
  strength: number;
  type: string;
}

const nodes: Node[] = [
  { id: 'nvidia', name: 'NVIDIA', type: 'company', size: 30, x: 200, y: 150, connections: ['tsmc', 'usa', 'ai-boom'] },
  { id: 'tsmc', name: 'TSMC', type: 'company', size: 25, x: 350, y: 100, connections: ['nvidia', 'taiwan', 'semiconductor'] },
  { id: 'usa', name: 'USA', type: 'country', size: 35, x: 100, y: 200, connections: ['nvidia', 'china', 'fed-policy'] },
  { id: 'china', name: 'China', type: 'country', size: 35, x: 400, y: 250, connections: ['usa', 'taiwan', 'trade-war'] },
  { id: 'taiwan', name: 'Taiwan', type: 'country', size: 20, x: 350, y: 200, connections: ['tsmc', 'china', 'semiconductor'] },
  { id: 'ai-boom', name: 'AI Boom', type: 'event', size: 15, x: 150, y: 80, connections: ['nvidia', 'fed-policy'] },
  { id: 'fed-policy', name: 'Fed Policy', type: 'event', size: 20, x: 50, y: 120, connections: ['usa', 'ai-boom'] },
  { id: 'trade-war', name: 'Trade War', type: 'event', size: 18, x: 250, y: 300, connections: ['usa', 'china'] },
  { id: 'semiconductor', name: 'Semiconductors', type: 'commodity', size: 22, x: 300, y: 150, connections: ['nvidia', 'tsmc', 'taiwan'] }
];

const edges: Edge[] = [
  { from: 'nvidia', to: 'tsmc', strength: 0.9, type: 'Supply Chain' },
  { from: 'nvidia', to: 'usa', strength: 0.7, type: 'Domicile' },
  { from: 'nvidia', to: 'ai-boom', strength: 0.95, type: 'Beneficiary' },
  { from: 'usa', to: 'china', strength: 0.8, type: 'Trade Dispute' },
  { from: 'china', to: 'taiwan', strength: 0.6, type: 'Geopolitical Tension' },
  { from: 'tsmc', to: 'taiwan', strength: 0.9, type: 'Domicile' },
  { from: 'usa', to: 'fed-policy', strength: 0.95, type: 'Policy Control' },
  { from: 'ai-boom', to: 'fed-policy', strength: 0.4, type: 'Economic Impact' }
];

const getNodeColor = (type: Node['type']) => {
  switch (type) {
    case 'company': return 'hsl(var(--chart-1))';
    case 'country': return 'hsl(var(--chart-2))';
    case 'commodity': return 'hsl(var(--chart-3))';
    case 'person': return 'hsl(var(--chart-4))';
    case 'event': return 'hsl(var(--chart-5))';
    default: return 'hsl(var(--muted-foreground))';
  }
};

export function InteractiveKnowledgeGraph() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredNode(nodeId);
  }, []);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(selectedNode === nodeId ? null : nodeId);
  }, [selectedNode]);

  const getHighlightedEdges = () => {
    const activeNode = hoveredNode || selectedNode;
    if (!activeNode) return [];
    
    const node = nodes.find(n => n.id === activeNode);
    return node ? node.connections : [];
  };

  const highlightedEdges = getHighlightedEdges();
  const activeNode = hoveredNode || selectedNode;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="w-5 h-5" />
          Interactive Economic Knowledge Graph
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <svg width="500" height="350" className="border border-border rounded-lg bg-muted/10">
            {/* Render edges */}
            {edges.map((edge, i) => {
              const fromNode = nodes.find(n => n.id === edge.from);
              const toNode = nodes.find(n => n.id === edge.to);
              if (!fromNode || !toNode) return null;
              
              const isHighlighted = highlightedEdges.includes(edge.to) || highlightedEdges.includes(edge.from);
              
              return (
                <g key={i}>
                  <line
                    x1={fromNode.x}
                    y1={fromNode.y}
                    x2={toNode.x}
                    y2={toNode.y}
                    stroke={isHighlighted ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'}
                    strokeWidth={isHighlighted ? 3 : 1}
                    strokeOpacity={isHighlighted ? 1 : 0.3}
                    className="transition-all duration-300"
                  />
                  {isHighlighted && (
                    <text
                      x={(fromNode.x + toNode.x) / 2}
                      y={(fromNode.y + toNode.y) / 2}
                      fill="hsl(var(--primary))"
                      fontSize="10"
                      textAnchor="middle"
                      className="font-medium"
                    >
                      {edge.type}
                    </text>
                  )}
                </g>
              );
            })}
            
            {/* Render nodes */}
            {nodes.map((node) => {
              const isActive = activeNode === node.id;
              const isConnected = highlightedEdges.includes(node.id);
              const opacity = !activeNode || isActive || isConnected ? 1 : 0.3;
              
              return (
                <g key={node.id}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={isActive ? node.size + 5 : node.size}
                    fill={getNodeColor(node.type)}
                    fillOpacity={opacity}
                    stroke={isActive ? 'hsl(var(--primary))' : 'hsl(var(--background))'}
                    strokeWidth={isActive ? 3 : 2}
                    className="cursor-pointer transition-all duration-300 hover:stroke-primary"
                    onMouseEnter={() => handleNodeHover(node.id)}
                    onMouseLeave={() => handleNodeHover(null)}
                    onClick={() => handleNodeClick(node.id)}
                  />
                  <text
                    x={node.x}
                    y={node.y + node.size + 15}
                    fill="hsl(var(--foreground))"
                    fontSize="11"
                    textAnchor="middle"
                    className="font-medium pointer-events-none"
                    fillOpacity={opacity}
                  >
                    {node.name}
                  </text>
                </g>
              );
            })}
          </svg>
          
          {/* Legend */}
          <div className="mt-4 flex flex-wrap gap-2">
            {['company', 'country', 'commodity', 'person', 'event'].map((type) => (
              <Badge key={type} variant="outline" className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: getNodeColor(type as Node['type']) }}
                />
                <span className="capitalize">{type}</span>
              </Badge>
            ))}
          </div>
          
          {/* Selected node info */}
          {selectedNode && (
            <div className="mt-4 p-3 bg-muted/20 rounded-lg border border-border">
              <div className="font-semibold">
                {nodes.find(n => n.id === selectedNode)?.name}
              </div>
              <div className="text-sm text-muted-foreground capitalize">
                {nodes.find(n => n.id === selectedNode)?.type}
              </div>
              <div className="text-sm mt-1">
                Connected to: {nodes.find(n => n.id === selectedNode)?.connections.length} entities
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}