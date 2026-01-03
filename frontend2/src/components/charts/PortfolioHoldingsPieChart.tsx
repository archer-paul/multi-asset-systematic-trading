import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { PieChart as PieChartIcon, TrendingUp, TrendingDown } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient, type PortfolioHoldings } from "@/lib/api";

// Mock data as fallback
const mockHoldingsData = [
  { sector: 'Technology', value: 450000, weight: 45, color: '#3B82F6', stocks: ['AAPL', 'MSFT', 'GOOGL'] },
  { sector: 'Healthcare', value: 200000, weight: 20, color: '#10B981', stocks: ['JNJ', 'PFE', 'UNH'] },
  { sector: 'Financial', value: 150000, weight: 15, color: '#F59E0B', stocks: ['JPM', 'BAC', 'GS'] },
  { sector: 'Consumer Disc.', value: 100000, weight: 10, color: '#EF4444', stocks: ['TSLA', 'AMZN', 'NFLX'] },
  { sector: 'Energy', value: 50000, weight: 5, color: '#8B5CF6', stocks: ['XOM', 'CVX', 'COP'] },
  { sector: 'Other', value: 50000, weight: 5, color: '#6B7280', stocks: ['Various'] }
];

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280'];

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-card p-3 border border-border rounded-lg shadow-lg">
        <p className="font-semibold text-card-foreground">{data.sector}</p>
        <p className="text-sm text-muted-foreground">
          Value: ${data.value.toLocaleString()}
        </p>
        <p className="text-sm text-muted-foreground">
          Weight: {data.weight.toFixed(1)}%
        </p>
        <div className="mt-2">
          <p className="text-xs text-muted-foreground font-medium">Top Holdings:</p>
          <p className="text-xs text-muted-foreground">
            {data.stocks?.slice(0, 3).join(', ')}
          </p>
        </div>
      </div>
    );
  }
  return null;
};

const CustomLabelContent = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  // Only show label if percentage is significant enough
  if (percent < 0.05) return null;

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      className="text-xs font-medium"
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export function PortfolioHoldingsPieChart() {
  const { data: holdingsData, isLoading, error } = useQuery({
    queryKey: ['portfolio-holdings'],
    queryFn: async () => {
      const result = await apiClient.getPortfolioHoldings();
      return result.data || {
        holdings: [],
        sector_allocation: {
          'Technology': { value: 450000, weight: 0.45 },
          'Healthcare': { value: 200000, weight: 0.20 },
          'Financial': { value: 150000, weight: 0.15 },
          'Consumer Disc.': { value: 100000, weight: 0.10 },
          'Energy': { value: 50000, weight: 0.05 },
          'Other': { value: 50000, weight: 0.05 }
        },
        total_value: 1000000,
        last_updated: new Date().toISOString()
      } as PortfolioHoldings;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: false,
  });

  // Transform API data or use mock data
  const chartData = holdingsData?.sector_allocation
    ? Object.entries(holdingsData.sector_allocation).map(([sector, data], index) => ({
        sector,
        value: data.value,
        weight: data.weight * 100,
        color: COLORS[index % COLORS.length],
        stocks: holdingsData.holdings
          .filter(holding => holding.sector === sector)
          .map(holding => holding.symbol)
          .slice(0, 3)
      }))
    : mockHoldingsData;

  const totalValue = chartData.reduce((sum, item) => sum + item.value, 0);
  const topSector = chartData.reduce((max, item) => item.value > max.value ? item : max, chartData[0]);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PieChartIcon className="w-5 h-5" />
          Portfolio Holdings by Sector
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-64 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : error ? (
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <p>Error loading portfolio data</p>
          </div>
        ) : (
          <>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={CustomLabelContent}
                    outerRadius={80}
                    innerRadius={25} // Modern donut chart style
                    fill="#8884d8"
                    dataKey="value"
                    stroke="none"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Summary Statistics */}
            <div className="mt-4 space-y-3">
              <div className="grid grid-cols-2 gap-4 p-3 bg-muted/20 rounded-lg">
                <div className="text-center">
                  <div className="text-lg font-bold text-primary">
                    ${totalValue.toLocaleString()}
                  </div>
                  <div className="text-xs text-muted-foreground">Total Value</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-chart-1">
                    {topSector.sector}
                  </div>
                  <div className="text-xs text-muted-foreground">Largest Sector</div>
                </div>
              </div>

              {/* Sector breakdown list */}
              <div className="space-y-2">
                {chartData.slice(0, 4).map((item, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-foreground">{item.sector}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{item.weight.toFixed(1)}%</span>
                      {item.weight > 20 ?
                        <TrendingUp className="w-3 h-3 text-green-500" /> :
                        <TrendingDown className="w-3 h-3 text-muted-foreground" />
                      }
                    </div>
                  </div>
                ))}
                {chartData.length > 4 && (
                  <div className="text-xs text-muted-foreground text-center pt-1">
                    +{chartData.length - 4} more sectors
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}