import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  BarChart3,
  Brain,
  TrendingUp,
  Shield,
  Globe,
  GitBranch,
  Activity,
  AlertTriangle
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";

const navigationItems = [
  {
    href: "/",
    label: "Dashboard",
    icon: BarChart3,
    description: "Portfolio Overview",
    primary: true
  },
  {
    href: "/ml-observatory",
    label: "ML Observatory", 
    icon: Brain,
    description: "Model Performance",
    primary: true
  },
  {
    href: "/sentiment-intelligence",
    label: "Sentiment Intelligence",
    icon: TrendingUp,
    description: "Multi-source Analysis",
    primary: true
  },
  {
    href: "/risk-analytics",
    label: "Risk Analytics",
    icon: Shield,
    description: "VaR & Stress Testing",
    primary: true
  },
  {
    href: "/knowledge-graph",
    label: "Knowledge Graph",
    icon: GitBranch,
    description: "Economic Relationships",
    primary: true
  },
  {
    href: "/geopolitical-monitor",
    label: "Geopolitical Monitor",
    icon: Globe,
    description: "Global Risk Assessment",
    primary: true
  },
  {
    href: "/technical-analysis",
    label: "Technical Analysis",
    icon: Activity,
    description: "Ichimoku & Multi-Timeframe",
    primary: true
  },
  {
    href: "/congress-trading",
    label: "Congress Trading",
    icon: AlertTriangle,
    description: "Congressional Analysis",
    primary: false
  },
  {
    href: "/emerging-stocks",
    label: "Emerging Stocks",
    icon: TrendingUp,
    description: "Stock Detection",
    primary: false
  },
];

export function Navigation() {
  const location = useLocation();

  return (
    <div className="border-b border-border bg-card/50 backdrop-blur-sm">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Quantitative Alpha Engine</h1>
            <p className="text-sm text-muted-foreground">Advanced Multi-Asset Systematic Trading</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="status-online"></div>
              <span className="text-sm text-muted-foreground">System Active</span>
            </div>
            <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
              Live Trading
            </Badge>
          </div>
        </div>
        
        <nav className="relative">
          <div className="flex items-center gap-1 overflow-x-auto scrollbar-hide pb-2">
            {navigationItems.filter(item => item.primary).map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.href;
              
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all hover:bg-muted/50 whitespace-nowrap",
                    isActive ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
            
            {/* Multi-Frame Analysis as primary nav item */}
            <Link
              to="/multi-frame-analysis"
              className={cn(
                "flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all hover:bg-muted/50 whitespace-nowrap",
                location.pathname === "/multi-frame-analysis" ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:text-foreground"
              )}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Multi-Frame Analysis</span>
            </Link>

            {/* Secondary pages - Only show if there are non-primary items */}
            {navigationItems.filter(item => !item.primary).length > 0 && (
              <div className="relative group">
                <button className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all hover:bg-muted/50 text-muted-foreground hover:text-foreground whitespace-nowrap">
                  <Activity className="w-4 h-4" />
                  <span>More</span>
                </button>

                <div className="absolute top-full right-0 mt-1 w-48 bg-card border border-border rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                  {navigationItems.filter(item => !item.primary).map((item) => {
                    const Icon = item.icon;
                    const isActive = location.pathname === item.href;

                    return (
                      <Link
                        key={item.href}
                        to={item.href}
                        className={cn(
                          "flex items-center gap-2 px-3 py-2 text-sm transition-all hover:bg-muted/50 first:rounded-t-lg last:rounded-b-lg",
                          isActive ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:text-foreground"
                        )}
                      >
                        <Icon className="w-4 h-4" />
                        <div>
                          <div>{item.label}</div>
                          <div className="text-xs text-muted-foreground">{item.description}</div>
                        </div>
                      </Link>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </nav>
      </div>
    </div>
  );
}