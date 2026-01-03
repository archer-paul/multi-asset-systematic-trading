import { ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon?: ReactNode;
  subtitle?: string;
  className?: string;
  compact?: boolean;
}

export function MetricCard({ 
  title, 
  value, 
  change, 
  changeType = 'neutral', 
  icon, 
  subtitle,
  className,
  compact = false
}: MetricCardProps) {
  const changeColors = {
    positive: 'text-green-400',
    negative: 'text-red-400',
    neutral: 'text-muted-foreground'
  };

  if (compact) {
    return (
      <Card className={cn("p-3 bg-card/50 border-border/50", className)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {icon && <div className="text-muted-foreground">{icon}</div>}
            <span className="text-xs text-muted-foreground uppercase tracking-wide">
              {title}
            </span>
          </div>
          <div className="text-right">
            <div className="font-mono font-semibold text-sm">{value}</div>
            {change && (
              <div className={cn("text-xs", changeColors[changeType])}>{change}</div>
            )}
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className={cn("p-4 bg-card/50 border-border/50", className)}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon && <div className="text-muted-foreground">{icon}</div>}
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
            {title}
          </h3>
        </div>
      </div>
      
      <div className="space-y-1">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-mono font-bold">{value}</span>
          {change && (
            <span className={cn("text-sm font-medium", changeColors[changeType])}>
              {change}
            </span>
          )}
        </div>
        {subtitle && (
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
    </Card>
  );
}