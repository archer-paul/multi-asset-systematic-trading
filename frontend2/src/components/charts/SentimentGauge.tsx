import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity } from 'lucide-react';

interface SentimentData {
  name: string;
  value: number;
  color: string;
}

const sentimentData: SentimentData[] = [
  { name: 'Bullish', value: 67, color: 'hsl(140 60% 55%)' },
  { name: 'Neutral', value: 23, color: 'hsl(45 85% 55%)' },
  { name: 'Bearish', value: 10, color: 'hsl(0 75% 60%)' }
];

const sourceData = [
  { source: 'News', bullish: 45, bearish: 12, neutral: 18 },
  { source: 'Social', bullish: 32, bearish: 8, neutral: 15 },
  { source: 'Institutional', bullish: 28, bearish: 5, neutral: 12 },
  { source: 'Technical', bullish: 22, bearish: 7, neutral: 8 }
];

export function SentimentGauge() {
  const totalSentiment = sentimentData.reduce((sum, item) => sum + item.value, 0);
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Sentiment Distribution */}
      <Card className="bg-card/50 backdrop-blur-sm border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Activity className="w-5 h-5 text-primary" />
            Market Sentiment Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-6">
            <div className="h-32 w-32">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sentimentData}
                    cx="50%"
                    cy="50%"
                    innerRadius={20}
                    outerRadius={60}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {sentimentData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="flex-1 space-y-2">
              {sentimentData.map((item, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm font-medium">{item.name}</span>
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-bold text-sm">{item.value}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-border/50">
            <div className="text-center">
              <div className="text-2xl font-mono font-bold text-primary">+{totalSentiment}</div>
              <div className="text-xs text-muted-foreground">Composite Score</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Source Breakdown */}
      <Card className="bg-card/50 backdrop-blur-sm border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">
            Sentiment by Source
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sourceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <XAxis 
                  dataKey="source" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={11}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={11}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: 'var(--radius)'
                  }}
                />
                <Bar dataKey="bullish" stackId="a" fill="hsl(140 60% 55%)" />
                <Bar dataKey="neutral" stackId="a" fill="hsl(45 85% 55%)" />
                <Bar dataKey="bearish" stackId="a" fill="hsl(0 75% 60%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}