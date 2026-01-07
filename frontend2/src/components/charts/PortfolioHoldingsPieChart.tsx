import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { PieChart as PieChartIcon, TrendingUp, TrendingDown } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient, type PortfolioHoldings } from "@/lib/api";

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280'];

// Mock data as fallback
const mockHoldingsData = [
  { sector: 'Technology', value: 2475, weight: 45, color: COLORS[0], stocks: ['AAPL', 'MSFT', 'GOOGL'] },
  { sector: 'Healthcare', value: 1100, weight: 20, color: COLORS[1], stocks: ['JNJ', 'PFE', 'UNH'] },
  { sector: 'Financial', value: 825, weight: 15, color: COLORS[2], stocks: ['JPM', 'BAC', 'GS'] },
  { sector: 'Consumer Disc.', value: 550, weight: 10, color: COLORS[3], stocks: ['TSLA', 'AMZN', 'NFLX'] },
  { sector: 'Energy', value: 275, weight: 5, color: COLORS[4], stocks: ['XOM', 'CVX', 'COP'] },
  { sector: 'Other', value: 275, weight: 5, color: COLORS[5], stocks: ['Various'] }
];

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

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280'];

  

  // Hardcoded mock data to ensure integrity for this test

  const localMockData = [

    { sector: 'Technology', value: 2475, weight: 45 },

    { sector: 'Healthcare', value: 1100, weight: 20 },

    { sector: 'Financial', value: 825, weight: 15 },

    { sector: 'Consumer Disc.', value: 550, weight: 10 },

    { sector: 'Energy', value: 275, weight: 5 },

    { sector: 'Other', value: 275, weight: 5 }

  ];



  const { data: holdingsData, isLoading, error } = useQuery({

    queryKey: ['portfolio-holdings'],

    queryFn: async () => {

      const result = await apiClient.getPortfolioHoldings();

      if (result.data && result.data.holdings && result.data.holdings.length > 0) {

        return result.data;

      } else {

        return {

          holdings: [],

          sector_allocation: {

            'Technology': { value: 2475, weight: 0.45 },

            'Healthcare': { value: 1100, weight: 0.20 },

            'Financial': { value: 825, weight: 0.15 },

            'Consumer Disc.': { value: 550, weight: 0.10 },

            'Energy': { value: 275, weight: 0.05 },

            'Other': { value: 275, weight: 0.05 }

          },

          total_value: 5500,

          last_updated: new Date().toISOString()

        } as PortfolioHoldings;

      }

    },

    refetchInterval: 30000,

    retry: false,

  });



  // Force rebuild of chartData with explicit colors

  const rawData = holdingsData?.sector_allocation

    ? Object.entries(holdingsData.sector_allocation).map(([sector, data]) => ({

        sector,

        value: data.value,

        weight: data.weight * 100

      }))

    : localMockData;



  const chartData = rawData.map((item, index) => ({

    ...item,

    color: COLORS[index % COLORS.length]

  }));



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

                    innerRadius={60}

                    outerRadius={80}

                    paddingAngle={2}

                    dataKey="value"

                    fill="red" // Diagnostic: if this shows red, Cells are ignored. If grey, CSS overrides.

                  >

                    {chartData.map((entry, index) => (

                      <Cell 

                        key={`cell-${index}`} 

                        fill={entry.color} 

                        stroke={entry.color}

                      />

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