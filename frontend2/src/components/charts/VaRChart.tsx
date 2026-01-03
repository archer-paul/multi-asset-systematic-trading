import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area } from 'recharts';
import { Shield } from "lucide-react";

const varData = [
  { date: 'Jan 1', var95: 4267, var99: 6834, realized: 3890 },
  { date: 'Jan 8', var95: 4512, var99: 7123, realized: 4234 },
  { date: 'Jan 15', var95: 4891, var99: 7456, realized: 5120 },
  { date: 'Jan 22', var95: 4234, var99: 6890, realized: 3950 },
  { date: 'Jan 29', var95: 4678, var99: 7234, realized: 4567 },
  { date: 'Feb 5', var95: 4456, var99: 7012, realized: 4123 },
  { date: 'Feb 12', var95: 4789, var99: 7345, realized: 4890 },
  { date: 'Feb 19', var95: 4623, var99: 7190, realized: 4456 },
  { date: 'Feb 26', var95: 4834, var99: 7456, realized: 5023 },
  { date: 'Mar 5', var95: 4567, var99: 7123, realized: 4234 }
];

const stressData = [
  { scenario: 'Market Crash', impact: -34567, probability: 2.1 },
  { scenario: 'Tech Selloff', impact: -28234, probability: 3.4 },
  { scenario: 'Rate Shock', impact: -18945, probability: 5.2 },
  { scenario: 'Inflation', impact: -15678, probability: 7.8 },
  { scenario: 'Geopolitical', impact: -21456, probability: 4.1 },
  { scenario: 'Credit Crunch', impact: -25789, probability: 2.8 }
];

export function VaRChart() {
  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            VaR Evolution Over Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={varData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="date" 
                  className="text-muted-foreground"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  className="text-muted-foreground"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `$${(value/1000).toFixed(0)}k`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string) => [
                    `$${value.toLocaleString()}`,
                    name === 'var95' ? 'VaR 95%' : name === 'var99' ? 'VaR 99%' : 'Realized Loss'
                  ]}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="var95" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  name="VaR 95%"
                />
                <Line 
                  type="monotone" 
                  dataKey="var99" 
                  stroke="hsl(var(--chart-4))" 
                  strokeWidth={2}
                  name="VaR 99%"
                />
                <Line 
                  type="monotone" 
                  dataKey="realized" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={3}
                  name="Realized Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Stress Test Impact vs Probability
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stressData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="scenario" 
                  className="text-muted-foreground"
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  yAxisId="left"
                  className="text-muted-foreground"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `$${Math.abs(value/1000).toFixed(0)}k`}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  className="text-muted-foreground"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string) => [
                    name === 'impact' ? `$${Math.abs(value).toLocaleString()}` : `${value}%`,
                    name === 'impact' ? 'Potential Loss' : 'Probability'
                  ]}
                />
                <Area 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="impact" 
                  stroke="hsl(var(--chart-4))" 
                  fill="hsl(var(--chart-4))" 
                  fillOpacity={0.3}
                  name="impact"
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="probability" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={3}
                  name="probability"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}