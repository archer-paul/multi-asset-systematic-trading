import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SentimentGauge } from "@/components/charts/SentimentGauge";
import { Navigation } from "@/components/layout/Navigation";
import { 
  TrendingUp, 
  TrendingDown, 
  Users, 
  Globe, 
  MessageCircle,
  Newspaper,
  BarChart3,
  AlertTriangle
} from "lucide-react";

export default function SentimentIntelligence() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="p-6">
        {/* Sentiment Overview */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-8 p-6">
            <h2 className="text-xl font-semibold mb-4">Multi-Source Sentiment Aggregate</h2>
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-green-400">+73</div>
                <div className="text-sm text-muted-foreground">Overall Score</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-primary">Bullish</div>
                <div className="text-sm text-muted-foreground">Market Sentiment</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold">12</div>
                <div className="text-sm text-muted-foreground">Active Sources</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-green-400">94%</div>
                <div className="text-sm text-muted-foreground">Data Quality</div>
              </div>
            </div>
            <SentimentGauge />
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="News Sentiment"
              value="+68"
              change="+12 pts"
              changeType="positive"
              icon={<Newspaper className="w-4 h-4" />}
              subtitle="12 institutional sources"
              compact
            />
            <MetricCard
              title="Social Media"
              value="+45"
              change="+8 pts"
              changeType="positive"
              icon={<MessageCircle className="w-4 h-4" />}
              subtitle="Reddit aggregation"
              compact
            />
            <MetricCard
              title="Congress Signals"
              value="Buy"
              change="3 new"
              changeType="positive"
              icon={<Users className="w-4 h-4" />}
              subtitle="Congressional activity"
              compact
            />
          </div>
        </div>

        {/* Source Breakdown */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">News Source Analysis</h2>
            <div className="space-y-3">
              {[
                { source: 'Bloomberg Markets', sentiment: 82, articles: 47, status: 'positive' },
                { source: 'Reuters Business', sentiment: 76, articles: 52, status: 'positive' },
                { source: 'Yahoo Finance', sentiment: 68, articles: 83, status: 'positive' },
                { source: 'MarketWatch', sentiment: 61, articles: 39, status: 'positive' },
                { source: 'CNBC Finance', sentiment: 58, articles: 61, status: 'neutral' },
                { source: 'Benzinga', sentiment: 54, articles: 72, status: 'neutral' },
                { source: 'Zacks', sentiment: 47, articles: 28, status: 'neutral' },
                { source: 'FinViz News', sentiment: 42, articles: 31, status: 'negative' }
              ].map((source, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      source.status === 'positive' ? 'bg-green-400' : 
                      source.status === 'negative' ? 'bg-red-400' : 
                      'bg-yellow-400'
                    }`}></div>
                    <span className="font-medium text-sm">{source.source}</span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="font-mono text-primary">{source.sentiment}</span>
                    <span className="text-muted-foreground">{source.articles} articles</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Social Media Intelligence</h2>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold text-green-400">+45</div>
                  <div className="text-sm text-muted-foreground">Reddit Score</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold">6</div>
                  <div className="text-sm text-muted-foreground">Subreddits</div>
                </div>
              </div>
              
              <div className="space-y-3">
                {[
                  { sub: 'r/investing', score: 67, posts: 234, trend: 'up' },
                  { sub: 'r/stocks', score: 52, posts: 189, trend: 'up' },
                  { sub: 'r/SecurityAnalysis', score: 48, posts: 76, trend: 'neutral' },
                  { sub: 'r/wallstreetbets', score: 34, posts: 412, trend: 'down' },
                  { sub: 'r/ValueInvesting', score: 41, posts: 93, trend: 'up' },
                  { sub: 'r/financialindependence', score: 56, posts: 67, trend: 'neutral' }
                ].map((sub, i) => (
                  <div key={i} className="flex items-center justify-between p-2 bg-muted/10 rounded border border-border/30">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm">{sub.sub}</span>
                      {sub.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-400" />}
                      {sub.trend === 'down' && <TrendingDown className="w-3 h-3 text-red-400" />}
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="font-mono text-primary">{sub.score}</span>
                      <span className="text-muted-foreground text-xs">{sub.posts}p</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>

        {/* Real-time Signals & Events */}
        <div className="grid grid-cols-12 gap-4">
          <Card className="col-span-8 p-6">
            <h2 className="text-lg font-semibold mb-4">Real-time Sentiment Signals</h2>
            <div className="space-y-3">
              {[
                { 
                  symbol: 'AAPL', 
                  sentiment: 'Bullish', 
                  score: 87, 
                  change: '+12', 
                  source: 'News Surge',
                  time: '2m ago',
                  impact: 'High'
                },
                { 
                  symbol: 'TSLA', 
                  sentiment: 'Bearish', 
                  score: 23, 
                  change: '-18', 
                  source: 'Social Media',
                  time: '5m ago',
                  impact: 'Medium'
                },
                { 
                  symbol: 'NVDA', 
                  sentiment: 'Bullish', 
                  score: 94, 
                  change: '+7', 
                  source: 'Institutional',
                  time: '8m ago',
                  impact: 'High'
                },
                { 
                  symbol: 'MSFT', 
                  sentiment: 'Neutral', 
                  score: 56, 
                  change: '+2', 
                  source: 'Mixed Sources',
                  time: '12m ago',
                  impact: 'Low'
                }
              ].map((signal, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center gap-3">
                    <span className="font-mono font-semibold text-sm">{signal.symbol}</span>
                    <Badge 
                      variant={
                        signal.sentiment === 'Bullish' ? 'default' : 
                        signal.sentiment === 'Bearish' ? 'destructive' : 'secondary'
                      } 
                      className="text-xs"
                    >
                      {signal.sentiment}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {signal.impact}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="font-mono text-primary">{signal.score}</span>
                    <span className={`font-mono ${signal.change.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
                      {signal.change}
                    </span>
                    <span className="text-muted-foreground">{signal.source}</span>
                    <span className="font-mono text-xs text-muted-foreground">{signal.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Alert Threshold"
              value="±15 pts"
              icon={<AlertTriangle className="w-4 h-4" />}
              subtitle="Sentiment change trigger"
              compact
            />
            <MetricCard
              title="Source Reliability"
              value="96.2%"
              icon={<Globe className="w-4 h-4" />}
              subtitle="Weighted accuracy"
              compact
            />
            <Card className="p-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <MessageCircle className="w-4 h-4" />
                Processing Stats
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Articles/Hour</span>
                  <span className="font-mono text-green-400">1,247</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Processing Lag</span>
                  <span className="font-mono text-green-400">45s</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Error Rate</span>
                  <span className="font-mono text-green-400">0.3%</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}