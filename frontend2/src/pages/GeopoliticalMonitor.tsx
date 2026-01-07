import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Navigation } from "@/components/layout/Navigation";
import { 
  Globe, 
  AlertTriangle, 
  Shield, 
  Users,
  BarChart3,
  Activity,
  Target,
  TrendingUp
} from "lucide-react";

export default function GeopoliticalMonitor() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="p-6">
        {/* Global Risk Overview */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-8 p-6">
            <h2 className="text-xl font-semibold mb-4">Global Risk Assessment</h2>
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-green-400">Low</div>
                <div className="text-sm text-muted-foreground">Overall Risk</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-yellow-400">Medium</div>
                <div className="text-sm text-muted-foreground">Regional Tensions</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-primary">7.2/10</div>
                <div className="text-sm text-muted-foreground">Stability Index</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono font-bold">23</div>
                <div className="text-sm text-muted-foreground">Active Monitors</div>
              </div>
            </div>
            <div className="h-64 bg-muted/20 rounded-lg relative overflow-hidden border border-border/50 group">
              {/* Background Map Image */}
              <div className="absolute inset-0 flex items-center justify-center opacity-40 mix-blend-overlay">
                 <img 
                   src="/world-map.svg" 
                   alt="World Map" 
                   className="w-full h-full object-cover"
                   style={{ 
                     filter: 'invert(1) hue-rotate(180deg) brightness(0.6) contrast(1.2)',
                     objectPosition: 'center 60%' // Center slightly lower to focus on relevant landmasses
                   }}
                 />
              </div>

              {/* Animation Styles */}
              <style>
                {`
                  @keyframes pulse-ring {
                    0% { transform: scale(0.8); opacity: 0.5; }
                    100% { transform: scale(2.5); opacity: 0; }
                  }
                  .hotspot-container {
                    position: absolute;
                    transform: translate(-50%, -50%);
                  }
                  .hotspot-ring {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: 16px;
                    height: 16px;
                    border: 1px solid #EF4444;
                    border-radius: 50%;
                    transform: translate(-50%, -50%);
                    animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
                  }
                  .hotspot-dot {
                    width: 8px;
                    height: 8px;
                    background-color: #EF4444;
                    border-radius: 50%;
                    box-shadow: 0 0 10px rgba(239, 68, 68, 0.6);
                  }
                  .hotspot-label {
                    position: absolute;
                    top: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 10px;
                    font-weight: bold;
                    color: #EF4444;
                    white-space: nowrap;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.8);
                    opacity: 0;
                    transition: opacity 0.2s;
                    pointer-events: none;
                  }
                  .hotspot-container:hover .hotspot-label {
                    opacity: 1;
                  }
                `}
              </style>

              {/* Hotspots Overlay */}
              <div className="absolute inset-0" style={{ top: '10%' }}> {/* Adjusted top offset for projection alignment */}
                
                {/* Venezuela (approx 31.6% L, 44.4% T) */}
                <div className="hotspot-container" style={{ left: '29%', top: '52%' }}>
                  <div className="hotspot-ring"></div>
                  <div className="hotspot-dot"></div>
                  <div className="hotspot-label">Venezuela</div>
                </div>

                {/* Iran (approx 64.7% L, 32.2% T) */}
                <div className="hotspot-container" style={{ left: '62%', top: '38%' }}>
                  <div className="hotspot-ring"></div>
                  <div className="hotspot-dot"></div>
                  <div className="hotspot-label">Iran</div>
                </div>

                {/* North Korea (approx 85.3% L, 27.8% T) */}
                <div className="hotspot-container" style={{ left: '82%', top: '35%' }}>
                  <div className="hotspot-ring"></div>
                  <div className="hotspot-dot"></div>
                  <div className="hotspot-label">North Korea</div>
                </div>

                {/* India-China Border (approx 71.7% L, 31.1% T) */}
                <div className="hotspot-container" style={{ left: '72%', top: '38%' }}>
                  <div className="hotspot-ring"></div>
                  <div className="hotspot-dot"></div>
                  <div className="hotspot-label">Ind-Chi Border</div>
                </div>

              </div>

              {/* Legend */}
              <div className="absolute bottom-2 right-2 flex gap-4 text-xs bg-background/80 p-2 rounded border border-border/50 backdrop-blur-sm">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                  <span className="text-muted-foreground">Conflict Zone</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-slate-600"></span>
                  <span className="text-muted-foreground">Stable</span>
                </div>
              </div>
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Trade Conflicts"
              value="2 Active"
              change="Stable"
              changeType="neutral"
              icon={<Target className="w-4 h-4" />}
              subtitle="US-China, EU-Russia"
              compact
            />
            <MetricCard
              title="Military Tensions"
              value="Medium"
              change="+1 region"
              changeType="negative"
              icon={<AlertTriangle className="w-4 h-4" />}
              subtitle="3 flashpoints monitored"
              compact
            />
            <MetricCard
              title="Economic Sanctions"
              value="147"
              icon={<Shield className="w-4 h-4" />}
              subtitle="Active measures"
              compact
            />
          </div>
        </div>

        {/* Regional Analysis */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Regional Risk Breakdown</h2>
            <div className="space-y-3">
              {[
                { region: 'Asia-Pacific', risk: 'Medium', score: 6.2, trends: ['Taiwan Strait', 'South China Sea'], impact: 'High' },
                { region: 'Eastern Europe', risk: 'High', score: 4.1, trends: ['Ukraine Conflict', 'NATO Expansion'], impact: 'Medium' },
                { region: 'Middle East', risk: 'Medium', score: 5.8, trends: ['Iran Nuclear', 'Israel-Palestine'], impact: 'Medium' },
                { region: 'North America', risk: 'Low', score: 8.7, trends: ['USMCA Stability'], impact: 'Low' },
                { region: 'Western Europe', risk: 'Low', score: 7.9, trends: ['Energy Security'], impact: 'Medium' },
                { region: 'Latin America', risk: 'Medium', score: 6.8, trends: ['Political Instability'], impact: 'Low' },
                { region: 'Africa', risk: 'Medium', score: 5.9, trends: ['Resource Conflicts'], impact: 'Low' }
              ].map((region, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      region.risk === 'High' ? 'bg-red-400' : 
                      region.risk === 'Medium' ? 'bg-yellow-400' : 
                      'bg-green-400'
                    }`}></div>
                    <span className="font-medium text-sm">{region.region}</span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="font-mono text-primary">{region.score}</span>
                    <Badge variant="outline" className="text-xs">
                      {region.impact} Impact
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="col-span-6 p-6">
            <h2 className="text-lg font-semibold mb-4">Economic Impact Assessment</h2>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold text-yellow-400">$2.4T</div>
                  <div className="text-sm text-muted-foreground">Trade at Risk</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-2xl font-mono font-bold text-red-400">847</div>
                  <div className="text-sm text-muted-foreground">Affected Companies</div>
                </div>
              </div>
              
              <div className="space-y-3">
                {[
                  { sector: 'Technology', exposure: 'High', companies: 127, impact: '$340B' },
                  { sector: 'Energy', exposure: 'High', companies: 89, impact: '$280B' },
                  { sector: 'Manufacturing', exposure: 'Medium', companies: 234, impact: '$195B' },
                  { sector: 'Agriculture', exposure: 'Medium', companies: 156, impact: '$87B' },
                  { sector: 'Financial', exposure: 'Low', companies: 98, impact: '$45B' },
                  { sector: 'Healthcare', exposure: 'Low', companies: 143, impact: '$32B' }
                ].map((sector, i) => (
                  <div key={i} className="flex items-center justify-between p-2 bg-muted/10 rounded border border-border/30">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{sector.sector}</span>
                      <Badge 
                        variant={sector.exposure === 'High' ? 'destructive' : sector.exposure === 'Medium' ? 'secondary' : 'default'}
                        className="text-xs"
                      >
                        {sector.exposure}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="font-mono text-primary">{sector.impact}</span>
                      <span className="text-muted-foreground text-xs">{sector.companies} cos</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>

        {/* Real-time Events & Intelligence */}
        <div className="grid grid-cols-12 gap-4">
          <Card className="col-span-8 p-6">
            <h2 className="text-lg font-semibold mb-4">Real-time Intelligence Feed</h2>
            <div className="space-y-3">
              {[
                { 
                  event: 'NATO Joint Exercise in Baltic Region', 
                  severity: 'Medium', 
                  region: 'Eastern Europe',
                  impact: 'Defense stocks up 2.3%',
                  time: '1h ago',
                  confidence: '87%'
                },
                { 
                  event: 'US-China Trade Delegation Meeting', 
                  severity: 'High', 
                  region: 'Asia-Pacific',
                  impact: 'Tech sector volatility expected',
                  time: '3h ago',
                  confidence: '94%'
                },
                { 
                  event: 'Middle East Diplomatic Breakthrough', 
                  severity: 'Low', 
                  region: 'Middle East',
                  impact: 'Oil prices down 1.8%',
                  time: '5h ago',
                  confidence: '91%'
                },
                { 
                  event: 'EU Sanctions Extension Vote', 
                  severity: 'Medium', 
                  region: 'Europe',
                  impact: 'Energy sector pressure',
                  time: '7h ago',
                  confidence: '82%'
                }
              ].map((event, i) => (
                <div key={i} className="p-4 bg-muted/20 rounded-lg border border-border/50">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="font-semibold text-sm">{event.event}</span>
                      <Badge 
                        variant={
                          event.severity === 'High' ? 'destructive' : 
                          event.severity === 'Medium' ? 'secondary' : 'default'
                        }
                        className="text-xs"
                      >
                        {event.severity}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span>{event.confidence}</span>
                      <span>{event.time}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{event.region}</span>
                    <span className="font-medium text-primary">{event.impact}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <div className="col-span-4 space-y-4">
            <MetricCard
              title="Alert Level"
              value="DEFCON 3"
              icon={<AlertTriangle className="w-4 h-4" />}
              subtitle="Elevated awareness"
              compact
            />
            <MetricCard
              title="Sources Active"
              value="23/25"
              icon={<Activity className="w-4 h-4" />}
              subtitle="Think tanks & intel"
              compact
            />
            <MetricCard
              title="Prediction Accuracy"
              value="89.3%"
              icon={<TrendingUp className="w-4 h-4" />}
              subtitle="30-day average"
              compact
            />
            <Card className="p-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <Globe className="w-4 h-4" />
                Monitoring Stats
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Events/Day</span>
                  <span className="font-mono text-green-400">1,247</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Processing Lag</span>
                  <span className="font-mono text-green-400">&lt; 5min</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>False Positives</span>
                  <span className="font-mono text-green-400">3.2%</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}