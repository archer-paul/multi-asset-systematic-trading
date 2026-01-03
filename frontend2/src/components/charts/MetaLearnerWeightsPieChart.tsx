import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Brain, TrendingUp, Zap } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient, type MLMetrics } from "@/lib/api";

// Mock data for meta-learner weights
const mockMetaLearnerWeights = [
  { name: 'Transformer', weight: 45, color: '#3B82F6', description: 'Attention-based deep learning' },
  { name: 'XGBoost', weight: 20, color: '#10B981', description: 'Gradient boosting' },
  { name: 'Random Forest', weight: 15, color: '#F59E0B', description: 'Tree ensemble' },
  { name: 'Neural Network', weight: 12, color: '#EF4444', description: 'Multi-layer perceptron' },
  { name: 'SVM', weight: 5, color: '#8B5CF6', description: 'Support vector machine' },
  { name: 'Ridge', weight: 3, color: '#6B7280', description: 'Linear regression' }
];

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280'];

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-card p-3 border border-border rounded-lg shadow-lg">
        <p className="font-semibold text-card-foreground">{data.name}</p>
        <p className="text-sm text-muted-foreground">
          Weight: {data.weight.toFixed(1)}%
        </p>
        <p className="text-xs text-muted-foreground">
          {data.description}
        </p>
      </div>
    );
  }
  return null;
};

const CustomLabelContent = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }: any) => {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  // Only show label if percentage is significant enough
  if (percent < 0.08) return null;

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      className="text-xs font-medium"
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export function MetaLearnerWeightsPieChart() {
  const { data: mlData, isLoading, error } = useQuery({
    queryKey: ['ml-metrics'],
    queryFn: async () => {
      const result = await apiClient.getMLMetrics();
      return result.data || {
        ensemble_weights: {
          'transformer': 0.45,
          'xgboost': 0.20,
          'random_forest': 0.15,
          'neural_network': 0.12,
          'svm': 0.05,
          'ridge': 0.03
        },
        ensemble_accuracy: 84.2,
        last_updated: new Date().toISOString()
      } as MLMetrics;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: false,
  });

  // Transform API data or use mock data
  const chartData = mlData?.ensemble_weights
    ? Object.entries(mlData.ensemble_weights).map(([name, weight], index) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        weight: weight * 100,
        color: COLORS[index % COLORS.length],
        description: getModelDescription(name)
      }))
    : mockMetaLearnerWeights;

  const totalModels = chartData.length;
  const bestModel = chartData.reduce((max, model) => model.weight > max.weight ? model : max, chartData[0]);
  const ensembleAccuracy = mlData?.ensemble_accuracy ?? 84.2;

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Meta-Learner Model Weights
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-64 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : error ? (
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <p>Error loading ML weights data</p>
          </div>
        ) : (
          <>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={CustomLabelContent}
                    outerRadius={80}
                    innerRadius={25} // Modern donut chart style
                    fill="#8884d8"
                    dataKey="weight"
                    stroke="none"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Summary Statistics */}
            <div className="mt-4 space-y-3">
              <div className="grid grid-cols-2 gap-4 p-3 bg-muted/20 rounded-lg">
                <div className="text-center">
                  <div className="text-lg font-bold text-primary">
                    {ensembleAccuracy.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Ensemble Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-chart-1">
                    {totalModels}
                  </div>
                  <div className="text-xs text-muted-foreground">Active Models</div>
                </div>
              </div>

              {/* Best performing model */}
              <div className="p-3 bg-gradient-to-r from-primary/10 to-chart-1/10 rounded-lg border border-primary/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-primary" />
                    <span className="font-medium text-sm">Best Model</span>
                  </div>
                  <TrendingUp className="w-4 h-4 text-green-500" />
                </div>
                <div className="mt-1">
                  <div className="font-semibold text-primary">{bestModel.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {bestModel.weight.toFixed(1)}% weight contribution
                  </div>
                </div>
              </div>

              {/* Model breakdown list */}
              <div className="space-y-2">
                {chartData.slice(0, 4).map((model, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: model.color }}
                      />
                      <span className="text-foreground">{model.name}</span>
                    </div>
                    <div className="font-medium text-primary">
                      {model.weight.toFixed(1)}%
                    </div>
                  </div>
                ))}
                {chartData.length > 4 && (
                  <div className="text-xs text-muted-foreground text-center pt-1">
                    +{chartData.length - 4} more models
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function getModelDescription(modelName: string): string {
  const descriptions: Record<string, string> = {
    'transformer': 'Attention-based deep learning',
    'xgboost': 'Gradient boosting framework',
    'random_forest': 'Tree ensemble method',
    'neural_network': 'Multi-layer perceptron',
    'svm': 'Support vector machine',
    'ridge': 'Linear regression with L2',
    'lasso': 'Linear regression with L1',
    'adaboost': 'Adaptive boosting',
    'gradient_boosting': 'Gradient boosting trees',
    'extra_trees': 'Extremely randomized trees'
  };

  return descriptions[modelName.toLowerCase()] || 'Machine learning model';
}