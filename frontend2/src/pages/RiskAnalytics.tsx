import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Navigation } from "@/components/layout/Navigation";
import { VaRChart } from "@/components/charts/VaRChart";
import { 
  Shield, 
  TrendingDown, 
  AlertTriangle, 
  Target,
  BarChart3,
  Activity,
  Zap,
  DollarSign
} from "lucide-react";

export default function RiskAnalytics() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="p-6">
        {/* Risk Overview */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-8 p-6">
            <h2 className="text-xl font-semibold mb-4">Portfolio Risk Assessment</h2>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-yellow-400">$4,267</div>
                <div className="text-sm text-muted-foreground">1-Day VaR (95%)</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-orange-400">5.8%</div>
                <div className="text-sm text-muted-foreground">Max Drawdown</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-primary">0.73</div>
                <div className="text-sm text-muted-foreground">Portfolio Beta</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-green-400">Low</div>
                <div className="text-sm text-muted-foreground">Risk Grade</div>
              </div>
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Expected Shortfall"
              value="$6,834"
              change="-1.8%"
              changeType="positive"
              icon={<Shield className="w-4 h-4" />}
              subtitle="99% confidence, tail risk"
              compact
            />
            <MetricCard
              title="Correlation Risk"
              value="0.42"
              change="+0.05"
              changeType="negative"
              icon={<Target className="w-4 h-4" />}
              subtitle="Avg pairwise correlation"
              compact
            />
            <MetricCard
              title="Concentration Risk"
              value="12.4%"
              icon={<AlertTriangle className="w-4 h-4" />}
              subtitle="Max single position"
              compact
            />
          </div>
        </div>

        {/* Charts Section */}
        <div className="mb-6">
          <VaRChart />
        </div>

        {/* Risk Metrics Grid */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Value-at-Risk Analysis</h2>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-xl font-mono font-bold text-yellow-400">$4,267</div>
                  <div className="text-sm text-muted-foreground">Parametric VaR</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-xl font-mono font-bold text-orange-400">$4,512</div>
                  <div className="text-sm text-muted-foreground">Historical VaR</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-xl font-mono font-bold text-red-400">$4,891</div>
                  <div className="text-sm text-muted-foreground">Monte Carlo VaR</div>
                </div>
              </div>
              
              <div className="space-y-3">
                {[
                  { confidence: '90%', parametric: '$2,845', historical: '$3,012', montecarlo: '$3,234' },
                  { confidence: '95%', parametric: '$4,267', historical: '$4,512', montecarlo: '$4,891' },
                  { confidence: '99%', parametric: '$6,834', historical: '$7,123', montecarlo: '$7,456' }
                ].map((var_data, i) => (
                  <div key={i} className="grid grid-cols-4 gap-3 p-3 bg-muted/10 rounded border border-border/30">
                    <div className="font-mono text-sm text-muted-foreground">{var_data.confidence}</div>
                    <div className="font-mono text-sm text-yellow-400">{var_data.parametric}</div>
                    <div className="font-mono text-sm text-orange-400">{var_data.historical}</div>
                    <div className="font-mono text-sm text-red-400">{var_data.montecarlo}</div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Stress Testing Results</h2>
            <div className="space-y-3">
              {[
                { scenario: 'Market Crash (-20%)', impact: '-$34,567', probability: '2.1%', status: 'high' },
                { scenario: 'Tech Selloff (-30%)', impact: '-$28,234', probability: '3.4%', status: 'high' },
                { scenario: 'Interest Rate Shock', impact: '-$18,945', probability: '5.2%', status: 'medium' },
                { scenario: 'Inflation Spike', impact: '-$15,678', probability: '7.8%', status: 'medium' },
                { scenario: 'Geopolitical Crisis', impact: '-$21,456', probability: '4.1%', status: 'medium' },
                { scenario: 'Credit Crunch', impact: '-$25,789', probability: '2.8%', status: 'high' }
              ].map((stress, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      stress.status === 'high' ? 'bg-red-400' : 
                      stress.status === 'medium' ? 'bg-yellow-400' : 
                      'bg-green-400'
                    }`}></div>
                    <span className="font-medium text-sm">{stress.scenario}</span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="font-mono text-red-400">{stress.impact}</span>
                    <span className="font-mono text-muted-foreground">{stress.probability}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Position Risk & Limits */}
        <div className="grid grid-cols-12 gap-4">
          <Card className="col-span-8 p-6">
            <h2 className="text-lg font-semibold mb-4">Position Risk Analysis</h2>
            <div className="space-y-3">
              {[
                { symbol: 'AAPL', exposure: '$42,567', weight: '15.0%', var: '$1,234', beta: '1.23', risk: 'Medium' },
                { symbol: 'TSLA', exposure: '$35,234', weight: '12.4%', var: '$2,156', beta: '2.14', risk: 'High' },
                { symbol: 'NVDA', exposure: '$38,901', weight: '13.7%', var: '$1,876', beta: '1.87', risk: 'High' },
                { symbol: 'MSFT', exposure: '$28,456', weight: '10.0%', var: '$892', beta: '0.91', risk: 'Low' },
                { symbol: 'GOOGL', exposure: '$31,789', weight: '11.2%', var: '$1,045', beta: '1.08', risk: 'Medium' }
              ].map((position, i) => (
                <div key={i} className="grid grid-cols-7 gap-3 p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="font-mono font-semibold text-sm">{position.symbol}</div>
                  <div className="font-mono text-sm text-primary">{position.exposure}</div>
                  <div className="font-mono text-sm">{position.weight}</div>
                  <div className="font-mono text-sm text-yellow-400">{position.var}</div>
                  <div className="font-mono text-sm">{position.beta}</div>
                  <Badge 
                    variant={position.risk === 'High' ? 'destructive' : position.risk === 'Medium' ? 'secondary' : 'default'}
                    className="text-xs w-fit"
                  >
                    {position.risk}
                  </Badge>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all ${
                        position.risk === 'High' ? 'bg-red-400' : 
                        position.risk === 'Medium' ? 'bg-yellow-400' : 
                        'bg-green-400'
                      }`}
                      style={{ width: `${parseFloat(position.weight)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Risk Utilization"
              value="73.4%"
              icon={<Activity className="w-4 h-4" />}
              subtitle="Of maximum risk budget"
              compact
            />
            <MetricCard
              title="Volatility Regime"
              value="Normal"
              icon={<Zap className="w-4 h-4" />}
              subtitle="Market environment"
              compact
            />
            <MetricCard
              title="Liquidity Score"
              value="8.7/10"
              icon={<DollarSign className="w-4 h-4" />}
              subtitle="Portfolio liquidity"
              compact
            />
            <Card className="p-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Risk Limits
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Max Position</span>
                  <span className="font-mono text-green-400">15.0%</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Max Sector</span>
                  <span className="font-mono text-green-400">25.0%</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Max Beta</span>
                  <span className="font-mono text-yellow-400">2.5</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}