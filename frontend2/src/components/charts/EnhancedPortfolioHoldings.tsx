import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { PieChart as PieChartIcon, ChevronDown, TrendingUp, TrendingDown, Building2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";

// Couleurs par secteur avec variations pour les entreprises
const sectorColors = {
  'Technology': {
    base: '#3B82F6',
    variations: ['#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#1E3A8A']
  },
  'Healthcare': {
    base: '#10B981',
    variations: ['#10B981', '#059669', '#047857', '#065F46', '#064E3B']
  },
  'Financial': {
    base: '#F59E0B',
    variations: ['#F59E0B', '#D97706', '#B45309', '#92400E', '#78350F']
  },
  'Consumer Discretionary': {
    base: '#EF4444',
    variations: ['#EF4444', '#DC2626', '#B91C1C', '#991B1B', '#7F1D1D']
  },
  'Energy': {
    base: '#8B5CF6',
    variations: ['#8B5CF6', '#7C3AED', '#6D28D9', '#5B21B6', '#4C1D95']
  },
  'Communication Services': {
    base: '#06B6D4',
    variations: ['#06B6D4', '#0891B2', '#0E7490', '#155E75', '#164E63']
  },
  'Industrials': {
    base: '#84CC16',
    variations: ['#84CC16', '#65A30D', '#4D7C0F', '#365314', '#1A2E05']
  },
  'Real Estate': {
    base: '#F97316',
    variations: ['#F97316', '#EA580C', '#C2410C', '#9A3412', '#7C2D12']
  },
  'Materials': {
    base: '#EC4899',
    variations: ['#EC4899', '#DB2777', '#BE185D', '#9D174D', '#831843']
  },
  'Utilities': {
    base: '#6B7280',
    variations: ['#6B7280', '#4B5563', '#374151', '#1F2937', '#111827']
  },
  'Consumer Staples': {
    base: '#14B8A6',
    variations: ['#14B8A6', '#0D9488', '#0F766E', '#115E59', '#134E4A']
  }
};

// Mock data détaillé avec holdings individuelles
const mockDetailedHoldings = [
  // Technology
  { symbol: 'AAPL', company: 'Apple Inc.', sector: 'Technology', shares: 1200, price: 185.50, value: 222600, weight: 8.9 },
  { symbol: 'MSFT', company: 'Microsoft Corp.', sector: 'Technology', shares: 800, price: 375.25, value: 300200, weight: 12.0 },
  { symbol: 'GOOGL', company: 'Alphabet Inc.', sector: 'Technology', shares: 600, price: 145.80, value: 87480, weight: 3.5 },
  { symbol: 'NVDA', company: 'NVIDIA Corp.', sector: 'Technology', shares: 400, price: 875.30, value: 350120, weight: 14.0 },
  { symbol: 'TSLA', company: 'Tesla Inc.', sector: 'Technology', shares: 300, price: 248.75, value: 74625, weight: 3.0 },

  // Healthcare
  { symbol: 'JNJ', company: 'Johnson & Johnson', sector: 'Healthcare', shares: 1500, price: 165.40, value: 248100, weight: 9.9 },
  { symbol: 'PFE', company: 'Pfizer Inc.', sector: 'Healthcare', shares: 2000, price: 28.90, value: 57800, weight: 2.3 },
  { symbol: 'UNH', company: 'UnitedHealth Group', sector: 'Healthcare', shares: 400, price: 520.75, value: 208300, weight: 8.3 },

  // Financial
  { symbol: 'JPM', company: 'JPMorgan Chase', sector: 'Financial', shares: 800, price: 155.20, value: 124160, weight: 5.0 },
  { symbol: 'BAC', company: 'Bank of America', sector: 'Financial', shares: 1200, price: 35.80, value: 42960, weight: 1.7 },
  { symbol: 'BRK.B', company: 'Berkshire Hathaway', sector: 'Financial', shares: 300, price: 445.60, value: 133680, weight: 5.3 },

  // Consumer Discretionary
  { symbol: 'AMZN', company: 'Amazon.com Inc.', sector: 'Consumer Discretionary', shares: 500, price: 155.40, value: 77700, weight: 3.1 },
  { symbol: 'HD', company: 'Home Depot Inc.', sector: 'Consumer Discretionary', shares: 400, price: 385.20, value: 154080, weight: 6.2 },

  // Energy
  { symbol: 'XOM', company: 'Exxon Mobil Corp.', sector: 'Energy', shares: 1000, price: 115.75, value: 115750, weight: 4.6 },
  { symbol: 'CVX', company: 'Chevron Corp.', sector: 'Energy', shares: 600, price: 158.90, value: 95340, weight: 3.8 }
];

interface HoldingData {
  symbol: string;
  company: string;
  sector: string;
  shares: number;
  price: number;
  value: number;
  weight: number;
}

interface PieChartData {
  name: string;
  value: number;
  weight: number;
  color: string;
  sector?: string;
  company?: string;
  shares?: number;
  price?: number;
}

export function EnhancedPortfolioHoldings() {
  const [viewMode, setViewMode] = useState<'sector' | 'individual'>('sector');
  const [selectedDetail, setSelectedDetail] = useState<string>('overview');
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);

  const { data: holdingsData, isLoading, error } = useQuery({
    queryKey: ['detailed-portfolio-holdings'],
    queryFn: async () => {
      try {
        const result = await apiClient.getPortfolioHoldings();
        return result.data?.holdings || mockDetailedHoldings;
      } catch (error) {
        console.log('API not available, using mock data');
        return mockDetailedHoldings;
      }
    },
    refetchInterval: 30000,
    retry: false,
  });

  const holdings: HoldingData[] = holdingsData || mockDetailedHoldings;

  // Agrégation par secteur
  const sectorData = holdings.reduce((acc, holding) => {
    if (!acc[holding.sector]) {
      acc[holding.sector] = { value: 0, weight: 0, holdings: [] };
    }
    acc[holding.sector].value += holding.value;
    acc[holding.sector].weight += holding.weight;
    acc[holding.sector].holdings.push(holding);
    return acc;
  }, {} as Record<string, { value: number; weight: number; holdings: HoldingData[] }>);

  // Données pour le graphique en secteurs
  const sectorChartData: PieChartData[] = Object.entries(sectorData).map(([sector, data]) => ({
    name: sector,
    value: data.value,
    weight: data.weight,
    color: sectorColors[sector as keyof typeof sectorColors]?.base || '#6B7280'
  }));

  // Données pour le graphique individuel avec couleurs par secteur
  const individualChartData: PieChartData[] = holdings.map((holding) => {
    const sectorColorPalette = sectorColors[holding.sector as keyof typeof sectorColors];
    const sectorHoldings = sectorData[holding.sector].holdings;
    const indexInSector = sectorHoldings.findIndex(h => h.symbol === holding.symbol);
    const colorIndex = indexInSector % 5;

    return {
      name: holding.symbol,
      value: holding.value,
      weight: holding.weight,
      color: sectorColorPalette?.variations[colorIndex] || '#6B7280',
      sector: holding.sector,
      company: holding.company,
      shares: holding.shares,
      price: holding.price
    };
  });

  const currentChartData = viewMode === 'sector' ? sectorChartData : individualChartData;
  const totalValue = holdings.reduce((sum, holding) => sum + holding.value, 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card p-4 border border-border rounded-lg shadow-lg max-w-xs">
          <p className="font-semibold text-card-foreground text-lg">{data.name}</p>
          {data.company && <p className="text-sm text-muted-foreground">{data.company}</p>}
          {data.sector && viewMode === 'individual' && (
            <p className="text-xs text-muted-foreground mb-2">{data.sector}</p>
          )}
          <div className="space-y-1">
            <p className="text-sm">
              <span className="text-muted-foreground">Value:</span>{' '}
              <span className="font-semibold text-green-400">${data.value.toLocaleString()}</span>
            </p>
            <p className="text-sm">
              <span className="text-muted-foreground">Weight:</span>{' '}
              <span className="font-semibold">{data.weight.toFixed(1)}%</span>
            </p>
            {data.shares && (
              <p className="text-sm">
                <span className="text-muted-foreground">Shares:</span>{' '}
                <span className="font-semibold">{data.shares.toLocaleString()}</span>
              </p>
            )}
            {data.price && (
              <p className="text-sm">
                <span className="text-muted-foreground">Price:</span>{' '}
                <span className="font-semibold">${data.price.toFixed(2)}</span>
              </p>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomPieCell = (props: any) => {
    const { payload } = props;
    const isHovered = hoveredSegment === payload.name;
    const scale = isHovered ? 1.05 : 1;
    const opacity = hoveredSegment && !isHovered ? 0.6 : 1;

    return (
      <Cell
        {...props}
        fill={payload.color}
        style={{
          transform: `scale(${scale})`,
          transformOrigin: 'center',
          opacity: opacity,
          transition: 'all 0.2s ease-in-out',
          filter: isHovered ? 'brightness(1.1)' : 'brightness(1)'
        }}
        onMouseEnter={() => setHoveredSegment(payload.name)}
        onMouseLeave={() => setHoveredSegment(null)}
      />
    );
  };

  const getDetailedInfo = () => {
    if (selectedDetail === 'overview') {
      return (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <div className="text-lg font-bold text-blue-400">{holdings.length}</div>
              <div className="text-xs text-muted-foreground">Total Holdings</div>
            </div>
            <div className="p-3 bg-green-500/10 rounded-lg">
              <div className="text-lg font-bold text-green-400">{Object.keys(sectorData).length}</div>
              <div className="text-xs text-muted-foreground">Sectors</div>
            </div>
          </div>
          <div className="text-sm text-muted-foreground">
            Largest holding: {holdings.sort((a, b) => b.value - a.value)[0]?.symbol} (${holdings.sort((a, b) => b.value - a.value)[0]?.value.toLocaleString()})
          </div>
        </div>
      );
    }

    if (selectedDetail in sectorData) {
      const sector = sectorData[selectedDetail];
      return (
        <div className="space-y-2">
          <div className="text-sm font-semibold text-foreground">{selectedDetail}</div>
          {sector.holdings.slice(0, 5).map((holding, index) => (
            <div key={index} className="flex justify-between items-center text-sm">
              <div className="flex items-center gap-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: sectorColors[selectedDetail as keyof typeof sectorColors]?.variations[index % 5] || '#6B7280'
                  }}
                />
                <span className="font-medium">{holding.symbol}</span>
              </div>
              <div className="text-right">
                <div className="font-medium">${holding.value.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">{holding.shares} shares</div>
              </div>
            </div>
          ))}
        </div>
      );
    }

    return null;
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PieChartIcon className="w-5 h-5" />
          Portfolio Holdings
        </CardTitle>
        <div className="flex gap-2">
          <Select value={viewMode} onValueChange={(value: 'sector' | 'individual') => setViewMode(value)}>
            <SelectTrigger className="w-32 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sector">By Sector</SelectItem>
              <SelectItem value="individual">Individual</SelectItem>
            </SelectContent>
          </Select>
        </div>
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
            <div className="h-64 relative">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={currentChartData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    innerRadius={30}
                    dataKey="value"
                    stroke="none"
                  >
                    {currentChartData.map((entry, index) => (
                      <CustomPieCell key={`cell-${index}`} payload={entry} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Summary */}
            <div className="mt-4 p-3 bg-muted/20 rounded-lg">
              <div className="text-center">
                <div className="text-xl font-bold text-primary">
                  ${totalValue.toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Total Portfolio Value</div>
              </div>
            </div>

            {/* Detailed Information Dropdown */}
            <div className="mt-4">
              <Select value={selectedDetail} onValueChange={setSelectedDetail}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select details to view" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="overview">Portfolio Overview</SelectItem>
                  {Object.keys(sectorData).map((sector) => (
                    <SelectItem key={sector} value={sector}>
                      {sector} Details
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Detail Content */}
            <div className="mt-3">
              {getDetailedInfo()}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}