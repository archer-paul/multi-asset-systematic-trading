import { Navigation } from "@/components/layout/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Users, 
  TrendingUp, 
  DollarSign, 
  FileText,
  Activity,
  Calendar,
  Award,
  AlertCircle
} from "lucide-react";

export default function CongressTrading() {
  return (
    <div className="flex-1 bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Congressional Trading Overview */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  Congressional Trading Activity
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">247</div>
                    <div className="text-sm text-muted-foreground">Active Trades</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">+18.2%</div>
                    <div className="text-sm text-muted-foreground">Avg. Performance</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">42</div>
                    <div className="text-sm text-muted-foreground">Members Tracked</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-400">$2.8M</div>
                    <div className="text-sm text-muted-foreground">Total Volume</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top Performing Members</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { name: "Nancy Pelosi", party: "D", performance: "+24.7%", trades: 18, sector: "Tech" },
                    { name: "Dan Crenshaw", party: "R", performance: "+19.3%", trades: 12, sector: "Energy" },
                    { name: "Josh Gottheimer", party: "D", performance: "+16.8%", trades: 24, sector: "Finance" },
                    { name: "Pat Fallon", party: "R", performance: "+14.2%", trades: 8, sector: "Defense" }
                  ].map((member, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center text-sm font-bold">
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-medium">{member.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {member.party} • {member.trades} trades • {member.sector}
                          </div>
                        </div>
                      </div>
                      <div className="text-green-400 font-mono font-bold">
                        {member.performance}
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
              title="Mirror Trading Performance"
              value="+15.8%"
              icon={<TrendingUp className="w-4 h-4" />}
              change="+2.3%"
              changeType="positive"
              subtitle="30-day return"
              className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border-green-500/20"
            />

            <MetricCard
              title="Disclosure Compliance"
              value="94.7%"
              icon={<FileText className="w-4 h-4" />}
              change="+1.2%"
              changeType="positive"
              subtitle="On-time filings"
              className="bg-gradient-to-br from-blue-500/10 to-cyan-600/10 border-blue-500/20"
            />

            <MetricCard
              title="Signal Confidence"
              value="87.3%"
              icon={<Activity className="w-4 h-4" />}
              change="+3.1%"
              changeType="positive"
              subtitle="Model accuracy"
              className="bg-gradient-to-br from-purple-500/10 to-violet-600/10 border-purple-500/20"
            />

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Recent Filings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {[
                  { ticker: "NVDA", action: "BUY", amount: "$50K-$100K", member: "Pelosi", date: "2h ago" },
                  { ticker: "TSLA", action: "SELL", amount: "$15K-$50K", member: "Crenshaw", date: "4h ago" },
                  { ticker: "MSFT", action: "BUY", amount: "$1K-$15K", member: "AOC", date: "6h ago" }
                ].map((filing, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Badge variant={filing.action === "BUY" ? "default" : "destructive"} className="text-xs">
                        {filing.action}
                      </Badge>
                      <span className="font-mono font-bold">{filing.ticker}</span>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">{filing.amount}</div>
                      <div className="text-muted-foreground text-xs">{filing.date}</div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}