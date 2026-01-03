import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Brain } from "lucide-react";

const accuracyData = [
  { epoch: 1, training: 65.2, validation: 62.8 },
  { epoch: 5, training: 72.4, validation: 68.9 },
  { epoch: 10, training: 78.6, validation: 74.2 },
  { epoch: 15, training: 83.1, validation: 78.5 },
  { epoch: 20, training: 86.7, validation: 81.9 },
  { epoch: 25, training: 89.2, validation: 84.6 },
  { epoch: 30, training: 91.5, validation: 86.8 },
  { epoch: 35, training: 93.1, validation: 88.4 },
  { epoch: 40, training: 94.2, validation: 89.7 },
  { epoch: 45, training: 95.1, validation: 90.8 },
  { epoch: 50, training: 95.8, validation: 91.6 }
];

export function AccuracyChart() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Model Training Progress
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={accuracyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="epoch" 
                className="text-muted-foreground"
                tick={{ fontSize: 12 }}
                label={{ value: 'Training Epoch', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                className="text-muted-foreground"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${value}%`}
                domain={[60, 100]}
                label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
                labelFormatter={(label) => `Epoch: ${label}`}
                formatter={(value: number, name: string) => [
                  `${value.toFixed(1)}%`,
                  name === 'training' ? 'Training Set' : 'Validation Set'
                ]}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="training" 
                stroke="hsl(var(--chart-1))" 
                strokeWidth={3}
                dot={{ fill: 'hsl(var(--chart-1))', strokeWidth: 2, r: 4 }}
                name="Training Set"
              />
              <Line 
                type="monotone" 
                dataKey="validation" 
                stroke="hsl(var(--chart-3))" 
                strokeWidth={3}
                dot={{ fill: 'hsl(var(--chart-3))', strokeWidth: 2, r: 4 }}
                name="Validation Set"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="grid grid-cols-3 gap-4 mt-4 p-4 bg-muted/20 rounded-lg">
          <div className="text-center">
            <div className="text-2xl font-bold text-chart-1">95.8%</div>
            <div className="text-sm text-muted-foreground">Training Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-chart-3">91.6%</div>
            <div className="text-sm text-muted-foreground">Validation Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">4.2%</div>
            <div className="text-sm text-muted-foreground">Generalization Gap</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}