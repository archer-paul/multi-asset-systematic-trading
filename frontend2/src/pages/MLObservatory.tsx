import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ModelWeightVisualization } from "@/components/ml/ModelWeightVisualization";
import { AccuracyChart } from "@/components/charts/AccuracyChart";
import { PerformanceChart } from "@/components/charts/PerformanceChart";
import { MetaLearnerWeightsPieChart } from "@/components/charts/MetaLearnerWeightsPieChart";
import { ModelPredictionErrorChart } from "@/components/charts/ModelPredictionErrorChart";
import { Navigation } from "@/components/layout/Navigation";
import {
  Brain,
  Zap,
  Target,
  TrendingUp,
  BarChart3,
  Activity,
  Cpu,
  Database
} from "lucide-react";

export default function MLObservatory() {
  return (
    <div className="flex-1 bg-background">
      <Navigation />

      <div className="p-6">
        {/* Top section: Model Weight Visualization (left) + Prediction error chart (right) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div>
            <ModelWeightVisualization />
          </div>
          <div>
            <ModelPredictionErrorChart />
          </div>
        </div>

        {/* Individual Model Performance */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Traditional ML Models</h2>
            <div className="space-y-3">
              {[
                { name: 'XGBoost', accuracy: '76.2%', weight: '18.5%', status: 'active' },
                { name: 'Random Forest', accuracy: '74.8%', weight: '16.2%', status: 'active' },
                { name: 'Extra Trees', accuracy: '73.1%', weight: '14.1%', status: 'active' },
                { name: 'AdaBoost', accuracy: '71.9%', weight: '12.8%', status: 'active' },
                { name: 'Gradient Boosting', accuracy: '70.3%', weight: '11.4%', status: 'active' },
                { name: 'Ridge Regression', accuracy: '68.7%', weight: '9.2%', status: 'active' },
                { name: 'SVR', accuracy: '67.2%', weight: '8.1%', status: 'active' },
                { name: 'MLP', accuracy: '65.8%', weight: '6.9%', status: 'active' },
                { name: 'Lasso', accuracy: '64.4%', weight: '2.8%', status: 'warning' }
              ].map((model, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center gap-3">
                    <span className="font-medium text-sm">{model.name}</span>
                    <Badge variant={model.status === 'active' ? 'default' : 'secondary'} className="text-xs">
                      {model.status}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="font-mono text-primary">{model.accuracy}</span>
                    <span className="text-muted-foreground">Weight: {model.weight}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Transformer Architecture</h2>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold">8</div>
                  <div className="text-sm text-muted-foreground">Attention Heads</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold">6</div>
                  <div className="text-sm text-muted-foreground">Layers</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold">256</div>
                  <div className="text-sm text-muted-foreground">D-Model</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold">60</div>
                  <div className="text-sm text-muted-foreground">Sequence Length</div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">GPU Utilization</span>
                  <span className="text-sm font-mono text-green-400">78%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Memory Usage</span>
                  <span className="text-sm font-mono text-green-400">3.2GB / 8GB</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Training Speed</span>
                  <span className="text-sm font-mono text-green-400">245 samples/sec</span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Training Status and Metrics */}
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-8">
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">System Performance Metrics</h2>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="text-center p-4 bg-muted/20 rounded-lg">
                  <div className="text-3xl font-mono font-bold text-primary">84.2%</div>
                  <div className="text-sm text-muted-foreground">Overall Accuracy</div>
                </div>
                <div className="text-center p-4 bg-muted/20 rounded-lg">
                  <div className="text-3xl font-mono font-bold text-green-400">2.34</div>
                  <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
                </div>
                <div className="text-center p-4 bg-muted/20 rounded-lg">
                  <div className="text-3xl font-mono font-bold text-blue-400">95.2%</div>
                  <div className="text-sm text-muted-foreground">Cache Hit Rate</div>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <MetricCard
                  title="Transformer Model"
                  value="87.1%"
                  change="+2.3%"
                  changeType="positive"
                  icon={<Zap className="w-4 h-4" />}
                  subtitle="Attention-based accuracy"
                  compact
                />
                <MetricCard
                  title="Traditional Ensemble"
                  value="78.6%"
                  change="+1.1%"
                  changeType="positive"
                  icon={<Brain className="w-4 h-4" />}
                  subtitle="9-model average"
                  compact
                />
                <MetricCard
                  title="Meta-Learning"
                  value="82.9%"
                  change="+0.8%"
                  changeType="positive"
                  icon={<Target className="w-4 h-4" />}
                  subtitle="Stacking regressor"
                  compact
                />
              </div>
            </Card>
          </div>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Training Jobs"
              value="12"
              icon={<Activity className="w-4 h-4" />}
              subtitle="Active parallel training"
              compact
            />
            <MetricCard
              title="Model Updates"
              value="347"
              icon={<Database className="w-4 h-4" />}
              subtitle="Since deployment"
              compact
            />
            <MetricCard
              title="Inference Speed"
              value="23ms"
              icon={<Cpu className="w-4 h-4" />}
              subtitle="Average latency"
              compact
            />
            <Card className="p-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <Brain className="w-4 h-4" />
                Training Status
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Next Update</span>
                  <span className="font-mono text-muted-foreground">1h 23m</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Retrain Trigger</span>
                  <span className="font-mono text-green-400">0.2% drift</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}