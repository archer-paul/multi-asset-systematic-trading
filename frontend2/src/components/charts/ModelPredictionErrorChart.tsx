import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingDown, Target, AlertTriangle } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";

// Mock data for prediction error over training epochs
const mockPredictionErrorData = [
  { epoch: 0, validation: 0.45, test: 0.47, train: 0.42 },
  { epoch: 10, validation: 0.38, test: 0.41, train: 0.35 },
  { epoch: 20, validation: 0.32, test: 0.35, train: 0.28 },
  { epoch: 30, validation: 0.28, test: 0.31, train: 0.24 },
  { epoch: 40, validation: 0.25, test: 0.28, train: 0.21 },
  { epoch: 50, validation: 0.23, test: 0.26, train: 0.19 },
  { epoch: 60, validation: 0.21, test: 0.24, train: 0.18 },
  { epoch: 70, validation: 0.20, test: 0.23, train: 0.17 },
  { epoch: 80, validation: 0.19, test: 0.22, train: 0.16 },
  { epoch: 90, validation: 0.18, test: 0.21, train: 0.15 },
  { epoch: 100, validation: 0.18, test: 0.21, train: 0.15 },
  { epoch: 110, validation: 0.17, test: 0.20, train: 0.15 },
  { epoch: 120, validation: 0.17, test: 0.20, train: 0.14 },
  { epoch: 130, validation: 0.17, test: 0.20, train: 0.14 },
  { epoch: 140, validation: 0.17, test: 0.20, train: 0.14 },
  { epoch: 150, validation: 0.17, test: 0.20, train: 0.14 }
];

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card p-3 border border-border rounded-lg shadow-lg">
        <p className="font-semibold text-card-foreground">Epoch {label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name === 'validation' ? 'Validation' :
             entry.name === 'test' ? 'Test' : 'Training'} Error: {entry.value.toFixed(3)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export function ModelPredictionErrorChart() {
  const { data: mlData, isLoading, error } = useQuery({
    queryKey: ['ml-prediction-error'],
    queryFn: async () => {
      const result = await apiClient.getMLMetrics();
      return result.data || {
        prediction_error_history: mockPredictionErrorData,
        last_updated: new Date().toISOString()
      };
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: false,
  });

  // Use API data or fallback to mock data
  // In a real implementation, you would extract prediction error history from the API response
  const chartData = mockPredictionErrorData; // Replace with real API data when available

  // Calculate metrics
  const latestValidationError = chartData[chartData.length - 1]?.validation ?? 0.17;
  const latestTestError = chartData[chartData.length - 1]?.test ?? 0.20;
  const initialValidationError = chartData[0]?.validation ?? 0.45;
  const errorReduction = ((initialValidationError - latestValidationError) / initialValidationError) * 100;

  // Check if model has converged (small change in last few epochs)
  const recentEpochs = chartData.slice(-5);
  const errorVariance = recentEpochs.reduce((acc, curr, idx, arr) => {
    if (idx === 0) return 0;
    return acc + Math.abs(curr.validation - arr[idx - 1].validation);
  }, 0) / Math.max(recentEpochs.length - 1, 1);

  const hasConverged = errorVariance < 0.005;

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingDown className="w-5 h-5" />
          Model Prediction Error
          {isLoading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary ml-2"></div>}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <p>Error loading prediction error data</p>
          </div>
        ) : (
          <>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="epoch"
                    className="text-muted-foreground"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Training Epoch', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    className="text-muted-foreground"
                    tick={{ fontSize: 12 }}
                    domain={['dataMin - 0.01', 'dataMax + 0.01']}
                    tickFormatter={(value) => value.toFixed(2)}
                    label={{ value: 'Mean Squared Error', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />

                  {/* Training error line */}
                  <Line
                    type="monotone"
                    dataKey="train"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={{ fill: '#10B981', strokeWidth: 1, r: 3 }}
                    name="Training"
                    strokeDasharray="2 2"
                  />

                  {/* Validation error line */}
                  <Line
                    type="monotone"
                    dataKey="validation"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                    name="Validation"
                    activeDot={{ r: 6, stroke: '#3B82F6', strokeWidth: 2 }}
                  />

                  {/* Test error line */}
                  <Line
                    type="monotone"
                    dataKey="test"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    dot={{ fill: '#F59E0B', strokeWidth: 1, r: 3 }}
                    name="Test"
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Error Statistics */}
            <div className="mt-4 space-y-3">
              <div className="grid grid-cols-3 gap-4 p-3 bg-muted/20 rounded-lg">
                <div className="text-center">
                  <div className="text-lg font-bold text-green-500">
                    {latestValidationError.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground">Validation Error</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-amber-500">
                    {latestTestError.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground">Test Error</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-primary">
                    {errorReduction.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Error Reduction</div>
                </div>
              </div>

              {/* Convergence Status */}
              <div className={`p-3 rounded-lg border ${
                hasConverged
                  ? 'bg-green-500/10 border-green-500/20'
                  : 'bg-amber-500/10 border-amber-500/20'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {hasConverged ? (
                      <Target className="w-4 h-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-500" />
                    )}
                    <span className="font-medium text-sm">Training Status</span>
                  </div>
                </div>
                <div className="mt-1">
                  <div className={`font-semibold ${
                    hasConverged ? 'text-green-500' : 'text-amber-500'
                  }`}>
                    {hasConverged ? 'Converged' : 'Training...'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {hasConverged
                      ? 'Model has reached optimal performance'
                      : 'Model is still improving'
                    }
                  </div>
                </div>
              </div>

              {/* Training Insights */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span>Overfitting Risk</span>
                  <span className={`font-medium ${
                    (latestTestError - latestValidationError) > 0.05
                      ? 'text-red-500' : 'text-green-500'
                  }`}>
                    {(latestTestError - latestValidationError) > 0.05 ? 'High' : 'Low'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Epochs Completed</span>
                  <span className="font-mono text-primary">
                    {chartData.length > 0 ? chartData[chartData.length - 1].epoch : 0}
                  </span>
                </div>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}