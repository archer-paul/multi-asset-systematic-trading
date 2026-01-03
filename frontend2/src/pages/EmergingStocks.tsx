import { Navigation } from "@/components/layout/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Rocket, 
  TrendingUp, 
  Star, 
  BarChart3,
  Zap,
  Target,
  Globe,
  Brain
} from "lucide-react";

export default function EmergingStocks() {
  return (
    <div className="flex-1 bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Rocket className="w-5 h-5" />
                  Emerging Stock Detection
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">127</div>
                    <div className="text-sm text-muted-foreground">Stocks Screened</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">23</div>
                    <div className="text-sm text-muted-foreground">High Potential</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">8</div>
                    <div className="text-sm text-muted-foreground">Active Positions</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-400">91.3%</div>
                    <div className="text-sm text-muted-foreground">Success Rate</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top Emerging Opportunities</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { 
                      ticker: "AFRM", 
                      name: "Affirm Holdings", 
                      score: 94.2, 
                      sector: "FinTech", 
                      catalyst: "Q4 Growth", 
                      momentum: "+127%",
                      confidence: "High"
                    },
                    { 
                      ticker: "RKLB", 
                      name: "Rocket Lab", 
                      score: 89.7, 
                      sector: "Aerospace", 
                      catalyst: "New Contract", 
                      momentum: "+89%",
                      confidence: "High"
                    },
                    { 
                      ticker: "UPST", 
                      name: "Upstart Holdings", 
                      score: 86.1, 
                      sector: "AI Lending", 
                      catalyst: "Model Update", 
                      momentum: "+76%",
                      confidence: "Medium"
                    },
                    { 
                      ticker: "SOFI", 
                      name: "SoFi Technologies", 
                      score: 83.4, 
                      sector: "FinTech", 
                      catalyst: "Banking License", 
                      momentum: "+64%",
                      confidence: "Medium"
                    }
                  ].map((stock, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-muted/20 rounded-lg">
                      <div className="flex items-center gap-4">
                        <div className="text-center">
                          <div className="font-mono font-bold text-lg">{stock.ticker}</div>
                          <div className="text-xs text-muted-foreground">{stock.sector}</div>
                        </div>
                        <div>
                          <div className="font-medium">{stock.name}</div>
                          <div className="text-sm text-muted-foreground flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">{stock.catalyst}</Badge>
                            <span>Score: {stock.score}</span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-green-400 font-mono font-bold">{stock.momentum}</div>
                        <div className="text-sm">
                          <Badge variant={stock.confidence === "High" ? "default" : "secondary"} className="text-xs">
                            {stock.confidence}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Sector Heat Map</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  {[
                    { sector: "AI/ML", stocks: 34, avgScore: 87.2, color: "bg-purple-500/20 border-purple-500/30" },
                    { sector: "FinTech", stocks: 28, avgScore: 82.1, color: "bg-blue-500/20 border-blue-500/30" },
                    { sector: "Biotech", stocks: 19, avgScore: 78.9, color: "bg-green-500/20 border-green-500/30" },
                    { sector: "Clean Energy", stocks: 22, avgScore: 75.4, color: "bg-emerald-500/20 border-emerald-500/30" },
                    { sector: "Aerospace", stocks: 12, avgScore: 84.7, color: "bg-indigo-500/20 border-indigo-500/30" },
                    { sector: "Gaming", stocks: 16, avgScore: 71.3, color: "bg-pink-500/20 border-pink-500/30" }
                  ].map((sector, index) => (
                    <div key={index} className={`p-4 rounded-lg border ${sector.color}`}>
                      <div className="font-medium">{sector.sector}</div>
                      <div className="text-sm text-muted-foreground">{sector.stocks} stocks</div>
                      <div className="mt-2">
                        <div className="flex justify-between text-sm">
                          <span>Avg Score</span>
                          <span className="font-mono">{sector.avgScore}</span>
                        </div>
                        <Progress value={sector.avgScore} className="mt-1" />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Side Metrics */}
          <div className="space-y-6">
            <MetricCard
              title="Detection Accuracy"
              value="91.3%"
              icon={<Target className="w-4 h-4" />}
              change="+2.7%"
              changeType="positive"
              subtitle="12-month success rate"
              className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border-green-500/20"
            />

            <MetricCard
              title="Average Momentum"
              value="+78.4%"
              icon={<Zap className="w-4 h-4" />}
              change="+12.1%"
              changeType="positive"
              subtitle="90-day performance"
              className="bg-gradient-to-br from-yellow-500/10 to-orange-600/10 border-yellow-500/20"
            />

            <MetricCard
              title="Pipeline Value"
              value="$2.4M"
              icon={<Star className="w-4 h-4" />}
              change="+18.3%"
              changeType="positive"
              subtitle="Potential upside"
              className="bg-gradient-to-br from-blue-500/10 to-cyan-600/10 border-blue-500/20"
            />

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">ML Model Performance</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Pattern Recognition</span>
                    <span className="font-mono">94.1%</span>
                  </div>
                  <Progress value={94.1} />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Momentum Detection</span>
                    <span className="font-mono">89.7%</span>
                  </div>
                  <Progress value={89.7} />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Risk Assessment</span>
                    <span className="font-mono">92.3%</span>
                  </div>
                  <Progress value={92.3} />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Timing Prediction</span>
                    <span className="font-mono">87.6%</span>
                  </div>
                  <Progress value={87.6} />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}