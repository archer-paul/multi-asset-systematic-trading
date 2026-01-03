import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Brain, Zap, Target, Activity } from "lucide-react";

interface ModelWeight {
  name: string;
  weight: number;
  performance: number;
  confidence: number;
  type: 'transformer' | 'ensemble' | 'classical';
  status: 'active' | 'training' | 'optimizing';
}

const modelWeights: ModelWeight[] = [
  {
    name: "Transformer Alpha",
    weight: 34.2,
    performance: 94.7,
    confidence: 92.3,
    type: 'transformer',
    status: 'active'
  },
  {
    name: "Ensemble Beta",
    weight: 28.5,
    performance: 89.1,
    confidence: 87.4,
    type: 'ensemble',
    status: 'active'
  },
  {
    name: "Random Forest",
    weight: 15.3,
    performance: 82.6,
    confidence: 79.8,
    type: 'classical',
    status: 'active'
  },
  {
    name: "XGBoost",
    weight: 12.7,
    performance: 85.3,
    confidence: 81.2,
    type: 'classical',
    status: 'optimizing'
  },
  {
    name: "LSTM Network",
    weight: 9.3,
    performance: 78.9,
    confidence: 75.6,
    type: 'ensemble',
    status: 'training'
  }
];

const getTypeIcon = (type: ModelWeight['type']) => {
  switch (type) {
    case 'transformer':
      return <Brain className="w-4 h-4" />;
    case 'ensemble':
      return <Zap className="w-4 h-4" />;
    case 'classical':
      return <Target className="w-4 h-4" />;
  }
};

const getStatusColor = (status: ModelWeight['status']) => {
  switch (status) {
    case 'active':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'training':
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'optimizing':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
  }
};

export function ModelWeightVisualization() {
  const [hoveredModel, setHoveredModel] = useState<string | null>(null);
  const totalWeight = modelWeights.reduce((sum, model) => sum + model.weight, 0);

  const pieData = modelWeights.map((model, index) => ({
    name: model.name,
    value: model.weight,
    fill: model.type === 'transformer' ? 'hsl(var(--chart-1))' : 
          model.type === 'ensemble' ? 'hsl(var(--chart-2))' : 'hsl(var(--chart-3))',
    ...model
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Meta-Learner Model Weights
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Interactive Donut Chart */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-sm font-medium">Weight Distribution</span>
            <span className="text-sm text-muted-foreground">Total: {totalWeight.toFixed(1)}%</span>
          </div>
          
          <div className="grid grid-cols-2 gap-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    onMouseEnter={(data) => setHoveredModel(data.name)}
                    onMouseLeave={() => setHoveredModel(null)}
                  >
                    {pieData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.fill}
                        stroke={hoveredModel === entry.name ? 'hsl(var(--primary))' : 'transparent'}
                        strokeWidth={hoveredModel === entry.name ? 3 : 0}
                        className="transition-all duration-300 cursor-pointer"
                        fillOpacity={hoveredModel === null || hoveredModel === entry.name ? 1 : 0.6}
                      />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                    formatter={(value: number, name: string) => [
                      `${value.toFixed(1)}%`,
                      name
                    ]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium text-sm">Model Details</h4>
              {pieData.map((model, index) => (
                <div 
                  key={index} 
                  className={`flex items-center gap-3 p-2 rounded-lg transition-all cursor-pointer ${
                    hoveredModel === model.name ? 'bg-muted/50' : 'hover:bg-muted/30'
                  }`}
                  onMouseEnter={() => setHoveredModel(model.name)}
                  onMouseLeave={() => setHoveredModel(null)}
                >
                  <div 
                    className="w-4 h-4 rounded-full" 
                    style={{ backgroundColor: model.fill }}
                  />
                  <div className="flex-1">
                    <div className="font-medium text-sm">{model.name}</div>
                    <div className="text-xs text-muted-foreground capitalize">{model.type}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm font-bold">{model.weight}%</div>
                    <div className="text-xs text-muted-foreground">{model.performance}% acc</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Individual Model Details */}
        <div className="space-y-4">
          {modelWeights.map((model, index) => (
            <div key={index} className="p-4 bg-muted/20 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="text-muted-foreground">
                    {getTypeIcon(model.type)}
                  </div>
                  <div>
                    <div className="font-medium">{model.name}</div>
                    <div className="text-sm text-muted-foreground capitalize">{model.type}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(model.status)} variant="outline">
                    {model.status}
                  </Badge>
                  <div className="text-right">
                    <div className="text-lg font-mono font-bold">{model.weight}%</div>
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">Performance</span>
                    <span className="font-mono">{model.performance}%</span>
                  </div>
                  <Progress value={model.performance} className="h-1.5" />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">Confidence</span>
                    <span className="font-mono">{model.confidence}%</span>
                  </div>
                  <Progress value={model.confidence} className="h-1.5" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Summary Stats */}
        <div className="mt-6 p-4 bg-primary/5 border border-primary/20 rounded-lg">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-primary">
                {(modelWeights.reduce((sum, m) => sum + (m.weight * m.performance / 100), 0) / totalWeight * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Weighted Performance</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-primary">
                {modelWeights.filter(m => m.status === 'active').length}
              </div>
              <div className="text-xs text-muted-foreground">Active Models</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-primary">
                {(modelWeights.reduce((sum, m) => sum + (m.weight * m.confidence / 100), 0) / totalWeight * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Avg Confidence</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}