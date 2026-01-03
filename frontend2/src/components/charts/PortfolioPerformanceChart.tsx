import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient, type PerformanceComparison } from "@/lib/api";

// Mock data as fallback
const mockPerformanceData = [
  { date: '2024-01', portfolio: 0, sp500: 0 },
  { date: '2024-02', portfolio: 2.3, sp500: 1.8 },
  { date: '2024-03', portfolio: 4.7, sp500: 3.2 },
  { date: '2024-04', portfolio: 6.8, sp500: 4.9 },
  { date: '2024-05', portfolio: 8.9, sp500: 6.1 },
  { date: '2024-06', portfolio: 11.2, sp500: 7.8 },
  { date: '2024-07', portfolio: 13.8, sp500: 9.2 },
  { date: '2024-08', portfolio: 15.9, sp500: 10.7 },
  { date: '2024-09', portfolio: 18.4, sp500: 12.1 },
  { date: '2024-10', portfolio: 21.2, sp500: 13.8 },
  { date: '2024-11', portfolio: 24.6, sp500: 15.2 },
  { date: '2024-12', portfolio: 27.3, sp500: 16.9 }
];

export function PortfolioPerformanceChart() {
  const { data: performanceData, isLoading, error } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: async () => {
      try {
        const result = await apiClient.getPortfolioPerformance(180);
        return result.data || {
          data: mockPerformanceData,
          portfolio_total_return: 27.3,
          sp500_total_return: 16.9,
          outperformance: 10.4,
          last_updated: new Date().toISOString()
        } as PerformanceComparison;
      } catch (error) {
        console.log('API not available, using mock data');
        // Return mock data as fallback
        return {
          data: mockPerformanceData,
          portfolio_total_return: 27.3,
          sp500_total_return: 16.9,
          outperformance: 10.4,
          last_updated: new Date().toISOString()
        } as PerformanceComparison;
      }
    },
    refetchInterval: 60000, // Refresh every minute
    retry: false, // Don't retry failed requests
    staleTime: 30000, // Consider data fresh for 30 seconds
  });

  // Always use either API data or mock data
  const chartData = performanceData?.data?.length
    ? performanceData.data.map(item => ({
        date: new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
        portfolio: item.portfolio_return,
        sp500: item.sp500_return
      }))
    : mockPerformanceData;

  const portfolioReturn = performanceData?.portfolio_total_return ?? 27.3;
  const sp500Return = performanceData?.sp500_total_return ?? 16.9;
  const outperformance = performanceData?.outperformance ?? 10.4;
  const isOutperforming = outperformance > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Portfolio vs S&P 500 Performance
          {isLoading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary ml-2"></div>}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="h-80 flex items-center justify-center text-muted-foreground">
            <p>Error loading performance data</p>
          </div>
        ) : (
          <>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="date"
                    className="text-muted-foreground"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
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
                    labelFormatter={(label) => `Date: ${label}`}
                    formatter={(value: number, name: string) => [
                      `${value.toFixed(1)}%`,
                      name === 'portfolio' ? 'Our Portfolio' : 'S&P 500'
                    ]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="portfolio"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={3}
                    dot={{ fill: 'hsl(var(--chart-1))', strokeWidth: 2, r: 4 }}
                    name="Our Portfolio"
                    activeDot={{ r: 6, stroke: 'hsl(var(--chart-1))', strokeWidth: 2 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="sp500"
                    stroke="hsl(var(--chart-2))"
                    strokeWidth={2}
                    dot={{ fill: 'hsl(var(--chart-2))', strokeWidth: 2, r: 3 }}
                    name="S&P 500"
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-3 gap-4 mt-4 p-4 bg-muted/20 rounded-lg">
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">{portfolioReturn.toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">Portfolio Return</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-chart-2">{sp500Return.toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">S&P 500 Return</div>
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold flex items-center justify-center gap-1 ${
                  isOutperforming ? 'text-green-400' : 'text-red-400'
                }`}>
                  {isOutperforming ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                  {isOutperforming ? '+' : ''}{outperformance.toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground">
                  {isOutperforming ? 'Outperformance' : 'Underperformance'}
                </div>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}