import { Card } from "@/components/ui/card";
import { Navigation } from "@/components/layout/Navigation";
import { Link } from "react-router-dom";
import { 
  Users, 
  TrendingUp, 
  BarChart3,
  ArrowRight,
  Globe,
  Building,
  Target,
  Calendar
} from "lucide-react";

const additionalPages = [
  {
    title: "Congress Trading",
    description: "Monitor and analyze congressional trading patterns and disclosure filings",
    path: "/congress-trading",
    icon: <Users className="w-6 h-6" />,
    metrics: ["47 Active Positions", "12% Weekly Change", "8.2% Avg Return"],
    category: "Political Intelligence"
  },
  {
    title: "Emerging Stocks",
    description: "AI-powered discovery of emerging market opportunities and growth stocks",
    path: "/emerging-stocks", 
    icon: <TrendingUp className="w-6 h-6" />,
    metrics: ["234 Tracked Stocks", "15 New Additions", "23% Avg Growth"],
    category: "Market Discovery"
  },
  {
    title: "Multi-Frame Analysis",
    description: "Cross-timeframe technical analysis with adaptive algorithms",
    path: "/multi-frame-analysis",
    icon: <BarChart3 className="w-6 h-6" />,
    metrics: ["9 Timeframes", "Real-time Sync", "95% Accuracy"],
    category: "Technical Analysis"
  }
];

export default function More() {
  return (
    <div className="flex-1 bg-background">
      <Navigation />
      
      <div className="p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-semibold mb-2">Additional Analysis Tools</h1>
          <p className="text-muted-foreground">
            Specialized modules for advanced market intelligence and systematic trading strategies
          </p>
        </div>

        {/* Tools Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {additionalPages.map((page, index) => (
            <Link 
              key={index} 
              to={page.path}
              className="group block transition-all duration-300 hover:translate-y-[-4px]"
            >
              <Card className="p-6 h-full border border-border/50 bg-card/50 backdrop-blur-sm hover:border-primary/30 hover:shadow-lg transition-all duration-300">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary/20 transition-colors">
                      {page.icon}
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg group-hover:text-primary transition-colors">
                        {page.title}
                      </h3>
                      <span className="text-xs text-muted-foreground font-mono tracking-wide uppercase">
                        {page.category}
                      </span>
                    </div>
                  </div>
                  <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all duration-300" />
                </div>
                
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                  {page.description}
                </p>
                
                <div className="space-y-2">
                  {page.metrics.map((metric, i) => (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">•</span>
                      <span className="font-mono text-primary/80">{metric}</span>
                    </div>
                  ))}
                </div>
              </Card>
            </Link>
          ))}
        </div>

        {/* Quick Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <Card className="p-4 text-center bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
            <div className="text-2xl font-mono font-bold text-primary mb-1">47</div>
            <div className="text-xs text-muted-foreground">Congress Positions</div>
          </Card>
          <Card className="p-4 text-center bg-gradient-to-br from-green-500/5 to-green-500/10 border-green-500/20">
            <div className="text-2xl font-mono font-bold text-green-400 mb-1">234</div>
            <div className="text-xs text-muted-foreground">Emerging Stocks</div>
          </Card>
          <Card className="p-4 text-center bg-gradient-to-br from-blue-500/5 to-blue-500/10 border-blue-500/20">
            <div className="text-2xl font-mono font-bold text-blue-400 mb-1">9</div>
            <div className="text-xs text-muted-foreground">Timeframes</div>
          </Card>
          <Card className="p-4 text-center bg-gradient-to-br from-purple-500/5 to-purple-500/10 border-purple-500/20">
            <div className="text-2xl font-mono font-bold text-purple-400 mb-1">95%</div>
            <div className="text-xs text-muted-foreground">Avg Accuracy</div>
          </Card>
        </div>

        {/* Integration Status */}
        <Card className="p-6 bg-muted/20 border-border/50">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Globe className="w-5 h-5" />
            System Integration Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Building className="w-4 h-4 text-green-400" />
                <span className="text-sm">Data Pipeline</span>
              </div>
              <span className="text-sm font-mono text-green-400">Online</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-green-400" />
                <span className="text-sm">ML Services</span>
              </div>
              <span className="text-sm font-mono text-green-400">Active</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-yellow-400" />
                <span className="text-sm">Market Data</span>
              </div>
              <span className="text-sm font-mono text-yellow-400">Delayed</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}