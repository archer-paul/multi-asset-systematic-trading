import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from 'lucide-react';

interface DataPoint {
  timestamp: string;
  value: number;
  volume?: number;
}

interface PerformanceChartProps {
  data: DataPoint[];
  title: string;
  type?: 'line' | 'area';
  color?: string;
  showVolume?: boolean;
}

export function PerformanceChart({ 
  data, 
  title, 
  type = 'line', 
  color = 'hsl(var(--primary))',
  showVolume = false 
}: PerformanceChartProps) {
  const Chart = type === 'area' ? AreaChart : LineChart;

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <TrendingUp className="w-5 h-5 text-primary" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <Chart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="hsl(var(--border))" 
                opacity={0.3}
              />
              <XAxis 
                dataKey="timestamp" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={11}
                fontFamily="var(--font-mono)"
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={11}
                fontFamily="var(--font-mono)"
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: 'var(--radius)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '12px'
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              
              {type === 'area' ? (
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke={color}
                  fill={`${color}20`}
                  strokeWidth={2}
                />
              ) : (
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: color, strokeWidth: 2 }}
                />
              )}
            </Chart>
          </ResponsiveContainer>
        </div>
        
        {showVolume && (
          <div className="mt-4 h-16">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <Area 
                  type="monotone" 
                  dataKey="volume" 
                  stroke="hsl(var(--muted-foreground))"
                  fill="hsl(var(--muted))"
                  opacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}