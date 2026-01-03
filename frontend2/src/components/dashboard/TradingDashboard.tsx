import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "./MetricCard";
import { 
  TrendingUp, 
  TrendingDown, 
  Shield, 
  Brain, 
  Activity, 
  DollarSign,
  Target,
  AlertTriangle,
  BarChart3,
  Globe,
  Users,
  Zap
} from "lucide-react";

export function TradingDashboard() {
  return (
    <div>
      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 gap-4">
        
        {/* Portfolio Overview - Top Row */}
        <Card className="col-span-8 p-6 bg-gradient-to-r from-card via-card to-card/80">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Portfolio Performance</h2>
            <div className="flex items-center gap-2">
              <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                +2.34% Today
              </Badge>
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-mono font-bold text-green-400">$284,756</div>
              <div className="text-sm text-muted-foreground">Total Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-mono font-bold text-green-400">+$6,672</div>
              <div className="text-sm text-muted-foreground">Today's P&L</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-mono font-bold">18.7%</div>
              <div className="text-sm text-muted-foreground">YTD Return</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-mono font-bold">2.34</div>
              <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
            </div>
          </div>

          {/* Mini chart placeholder */}
          <div className="h-32 bg-muted/20 rounded-lg flex items-center justify-center border border-border/50">
            <div className="text-muted-foreground flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              <span className="text-sm">Portfolio Performance Chart</span>
            </div>
          </div>
        </Card>

        {/* System Status */}
        <Card className="col-span-4 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            System Health
          </h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">ML Models</span>
              <div className="flex items-center gap-2">
                <div className="status-online"></div>
                <span className="text-xs text-green-400">9/9 Active</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Data Feeds</span>
              <div className="flex items-center gap-2">
                <div className="status-online"></div>
                <span className="text-xs text-green-400">12/12 Connected</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Risk Engine</span>
              <div className="flex items-center gap-2">
                <div className="status-online"></div>
                <span className="text-xs text-green-400">Operational</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">API Latency</span>
              <span className="text-xs text-green-400 font-mono">127ms</span>
            </div>
          </div>
        </Card>

        {/* Risk Metrics */}
        <div className="col-span-3 space-y-4">
          <MetricCard
            title="Value at Risk"
            value="$4,267"
            change="-2.1%"
            changeType="positive"
            icon={<Shield className="w-4 h-4" />}
            subtitle="95% confidence, 1-day"
            compact
          />
          <MetricCard
            title="Max Drawdown"
            value="5.8%"
            change="Improved"
            changeType="positive"
            icon={<TrendingDown className="w-4 h-4" />}
            compact
          />
          <MetricCard
            title="Beta"
            value="0.73"
            icon={<Target className="w-4 h-4" />}
            subtitle="vs S&P 500"
            compact
          />
        </div>

        {/* ML Performance */}
        <div className="col-span-3 space-y-4">
          <MetricCard
            title="Ensemble Accuracy"
            value="84.2%"
            change="+1.3%"
            changeType="positive"
            icon={<Brain className="w-4 h-4" />}
            subtitle="5-day prediction"
            compact
          />
          <MetricCard
            title="Transformer Model"
            value="87.1%"
            change="+0.8%"
            changeType="positive"
            icon={<Zap className="w-4 h-4" />}
            compact
          />
          <MetricCard
            title="Feature Importance"
            value="119"
            icon={<BarChart3 className="w-4 h-4" />}
            subtitle="Active features"
            compact
          />
        </div>

        {/* Sentiment Intelligence */}
        <div className="col-span-3 space-y-4">
          <MetricCard
            title="Market Sentiment"
            value="Bullish"
            change="+15 pts"
            changeType="positive"
            icon={<TrendingUp className="w-4 h-4" />}
            subtitle="Multi-source aggregate"
            compact
          />
          <MetricCard
            title="Congress Signals"
            value="Buy"
            change="3 new"
            changeType="positive"
            icon={<Users className="w-4 h-4" />}
            compact
          />
          <MetricCard
            title="Geopolitical Risk"
            value="Low"
            icon={<Globe className="w-4 h-4" />}
            subtitle="Global stability index"
            compact
          />
        </div>

        {/* Active Positions */}
        <div className="col-span-3 space-y-4">
          <MetricCard
            title="Active Positions"
            value="47"
            icon={<DollarSign className="w-4 h-4" />}
            subtitle="Across 8 sectors"
            compact
          />
          <MetricCard
            title="Cash Allocation"
            value="12.4%"
            icon={<Shield className="w-4 h-4" />}
            subtitle="$35,308 available"
            compact
          />
          <MetricCard
            title="Next Rebalance"
            value="2h 34m"
            icon={<Activity className="w-4 h-4" />}
            compact
          />
        </div>

        {/* Recent Signals */}
        <Card className="col-span-6 p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Trading Signals</h2>
          <div className="space-y-3">
            {[
              { symbol: 'AAPL', action: 'BUY', confidence: '94%', source: 'ML Ensemble', time: '09:45' },
              { symbol: 'TSLA', action: 'SELL', confidence: '87%', source: 'Congress Signal', time: '09:32' },
              { symbol: 'NVDA', action: 'HOLD', confidence: '76%', source: 'Sentiment Analysis', time: '09:18' },
              { symbol: 'MSFT', action: 'BUY', confidence: '92%', source: 'Technical Analysis', time: '09:05' }
            ].map((signal, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                <div className="flex items-center gap-3">
                  <span className="font-mono font-semibold text-sm">{signal.symbol}</span>
                  <Badge variant={signal.action === 'BUY' ? 'default' : signal.action === 'SELL' ? 'destructive' : 'secondary'} 
                         className="text-xs">
                    {signal.action}
                  </Badge>
                </div>
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <span>{signal.confidence}</span>
                  <span>•</span>
                  <span>{signal.source}</span>
                  <span>•</span>
                  <span className="font-mono">{signal.time}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* News & Events */}
        <Card className="col-span-6 p-6">
          <h2 className="text-lg font-semibold mb-4">Market Intelligence</h2>
          <div className="space-y-3">
            {[
              { 
                title: 'Fed maintains rates, dovish outlook', 
                source: 'Reuters', 
                impact: 'Medium',
                time: '2h ago',
                sentiment: 'positive'
              },
              { 
                title: 'Tech earnings beat expectations', 
                source: 'Bloomberg', 
                impact: 'High',
                time: '4h ago',
                sentiment: 'positive'
              },
              { 
                title: 'Commodity prices surge amid supply concerns', 
                source: 'MarketWatch', 
                impact: 'Medium',
                time: '6h ago',
                sentiment: 'neutral'
              }
            ].map((news, i) => (
              <div key={i} className="p-3 bg-muted/20 rounded-lg border border-border/50">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-sm leading-tight">{news.title}</h4>
                  <Badge variant="outline" className="text-xs">
                    {news.impact}
                  </Badge>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>{news.source}</span>
                  <span>•</span>
                  <span>{news.time}</span>
                  <span>•</span>
                  <div className={`w-2 h-2 rounded-full ${
                    news.sentiment === 'positive' ? 'bg-green-400' : 
                    news.sentiment === 'negative' ? 'bg-red-400' : 
                    'bg-yellow-400'
                  }`}></div>
                </div>
              </div>
            ))}
          </div>
        </Card>

      </div>
    </div>
  );
}