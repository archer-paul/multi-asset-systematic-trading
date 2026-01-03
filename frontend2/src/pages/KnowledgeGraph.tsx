import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Navigation } from "@/components/layout/Navigation";
import { InteractiveKnowledgeGraph } from "@/components/charts/InteractiveKnowledgeGraph";
import { 
  Network, 
  Globe, 
  Building, 
  Users,
  BarChart3,
  Activity,
  Zap,
  AlertTriangle
} from "lucide-react";

export default function KnowledgeGraph() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="p-6">
        {/* Knowledge Graph Overview */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-8 p-6">
            <h2 className="text-xl font-semibold mb-4">Economic Knowledge Graph</h2>
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-primary">1,247</div>
                <div className="text-sm text-muted-foreground">Total Entities</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-green-400">3,891</div>
                <div className="text-sm text-muted-foreground">Relationships</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-yellow-400">0.73</div>
                <div className="text-sm text-muted-foreground">Avg Clustering</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold">4.2</div>
                <div className="text-sm text-muted-foreground">Avg Degree</div>
              </div>
            </div>
            <div className="p-4">
              <InteractiveKnowledgeGraph />
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Active Nodes"
              value="847"
              change="+23"
              changeType="positive"
              icon={<Network className="w-4 h-4" />}
              subtitle="Recently updated"
              compact
            />
            <MetricCard
              title="Cascade Events"
              value="12"
              change="3 new"
              changeType="negative"
              icon={<AlertTriangle className="w-4 h-4" />}
              subtitle="Propagating effects"
              compact
            />
            <MetricCard
              title="Graph Density"
              value="0.034"
              icon={<Activity className="w-4 h-4" />}
              subtitle="Connection ratio"
              compact
            />
          </div>
        </div>

        {/* Entity Breakdown */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Entity Distribution</h2>
            <div className="space-y-3">
              {[
                { type: 'Companies', count: 427, active: 398, icon: Building, color: 'text-primary' },
                { type: 'Countries', count: 45, active: 43, icon: Globe, color: 'text-green-400' },
                { type: 'Commodities', count: 28, active: 26, icon: BarChart3, color: 'text-yellow-400' },
                { type: 'Currencies', count: 15, active: 15, icon: Zap, color: 'text-blue-400' },
                { type: 'Politicians', count: 156, active: 132, icon: Users, color: 'text-purple-400' },
                { type: 'Institutions', count: 89, active: 84, icon: Network, color: 'text-orange-400' },
                { type: 'Sectors', count: 24, active: 24, icon: BarChart3, color: 'text-cyan-400' },
                { type: 'Events', count: 463, active: 225, icon: Activity, color: 'text-red-400' }
              ].map((entity, i) => {
                const Icon = entity.icon;
                return (
                  <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                    <div className="flex items-center gap-3">
                      <Icon className={`w-4 h-4 ${entity.color}`} />
                      <span className="font-medium text-sm">{entity.type}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="font-mono text-primary">{entity.count}</span>
                      <span className="text-muted-foreground">({entity.active} active)</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Recent Relationship Updates</h2>
            <div className="space-y-3">
              {[
                { 
                  from: 'NVIDIA', 
                  to: 'Taiwan Semiconductor', 
                  type: 'Supply Chain', 
                  strength: 0.87, 
                  change: '+0.12',
                  time: '2h ago'
                },
                { 
                  from: 'USA', 
                  to: 'China', 
                  type: 'Trade Dispute', 
                  strength: 0.92, 
                  change: '+0.08',
                  time: '4h ago'
                },
                { 
                  from: 'Tesla', 
                  to: 'Lithium', 
                  type: 'Commodity Dependency', 
                  strength: 0.78, 
                  change: '-0.05',
                  time: '6h ago'
                },
                { 
                  from: 'Federal Reserve', 
                  to: 'EUR/USD', 
                  type: 'Monetary Policy', 
                  strength: 0.94, 
                  change: '+0.03',
                  time: '8h ago'
                }
              ].map((rel, i) => (
                <div key={i} className="p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold">{rel.from}</span>
                      <span className="text-muted-foreground text-xs">→</span>
                      <span className="font-mono text-sm font-semibold">{rel.to}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {rel.type}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Strength:</span>
                      <span className="font-mono text-primary">{rel.strength}</span>
                      <span className={`font-mono ${rel.change.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
                        {rel.change}
                      </span>
                    </div>
                    <span className="text-muted-foreground">{rel.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Cascade Analysis */}
        <div className="grid grid-cols-12 gap-4">
          <Card className="col-span-8 p-6">
            <h2 className="text-lg font-semibold mb-4">Cascade Effect Analysis</h2>
            <div className="space-y-3">
              {[
                { 
                  trigger: 'Fed Rate Decision', 
                  impact: 'High', 
                  affected: 847, 
                  confidence: '94%',
                  horizon: 'Medium-term',
                  sectors: ['Banking', 'Real Estate', 'Tech', 'Utilities']
                },
                { 
                  trigger: 'China Manufacturing PMI', 
                  impact: 'Medium', 
                  affected: 423, 
                  confidence: '87%',
                  horizon: 'Short-term',
                  sectors: ['Commodities', 'Shipping', 'Manufacturing']
                },
                { 
                  trigger: 'NVIDIA Earnings Beat', 
                  impact: 'Medium', 
                  affected: 298, 
                  confidence: '91%',
                  horizon: 'Immediate',
                  sectors: ['Semiconductors', 'AI', 'Gaming']
                },
                { 
                  trigger: 'Oil Price Shock', 
                  impact: 'High', 
                  affected: 734, 
                  confidence: '89%',
                  horizon: 'Long-term',
                  sectors: ['Energy', 'Transportation', 'Airlines', 'Petrochemicals']
                }
              ].map((cascade, i) => (
                <div key={i} className="p-4 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className="font-semibold text-sm">{cascade.trigger}</span>
                      <Badge 
                        variant={cascade.impact === 'High' ? 'destructive' : 'secondary'}
                        className="text-xs"
                      >
                        {cascade.impact} Impact
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="font-mono text-primary">{cascade.affected} entities</span>
                      <span className="text-muted-foreground">{cascade.confidence}</span>
                      <Badge variant="outline" className="text-xs">
                        {cascade.horizon}
                      </Badge>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {cascade.sectors.map((sector, j) => (
                      <Badge key={j} variant="outline" className="text-xs">
                        {sector}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Graph Updates"
              value="127/min"
              icon={<Activity className="w-4 h-4" />}
              subtitle="Real-time processing"
              compact
            />
            <MetricCard
              title="AI Analysis"
              value="Active"
              icon={<Zap className="w-4 h-4" />}
              subtitle="Gemini integration"
              compact
            />
            <Card className="p-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <Network className="w-4 h-4" />
                Graph Statistics
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Diameter</span>
                  <span className="font-mono text-green-400">8</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Components</span>
                  <span className="font-mono text-green-400">3</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Centrality Max</span>
                  <span className="font-mono text-primary">0.847</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}