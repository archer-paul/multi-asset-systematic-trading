import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Navigation } from "@/components/layout/Navigation";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  RefreshCw,
  Eye,
  Target,
  Layers,
  Gauge
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine } from "recharts";

interface TechnicalData {
  symbol: string;
  timeframes: {
    [key: string]: {
      trend_score: number;
      momentum_score: number;
      volatility_score: number;
      volume_score: number;
      timeframe_score: number;
    };
  };
  overall_signal: string;
  confidence: number;
  composite_score: number;
  market_regime: string;
  ichimoku_signals: {
    [key: string]: {
      tenkan_kijun_cross: string;
      price_vs_cloud: string;
      cloud_color: string;
      chikou_confirmation: boolean;
    };
  };
}

interface TechnicalAnalysisData {
  technical_analysis: { [symbol: string]: TechnicalData };
  summary: {
    total_symbols: number;
    bullish_symbols: number;
    bearish_symbols: number;
    neutral_symbols: number;
    avg_confidence: number;
    market_regimes: { [regime: string]: number };
  };
  timestamp: string;
}

interface IchimokuCloudProps {
  data: Array<{
    time: string;
    price: number;
    tenkan: number;
    kijun: number;
    senkou_a: number;
    senkou_b: number;
    chikou: number;
  }>;
}

const IchimokuCloudChart: React.FC<IchimokuCloudProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="time" stroke="#9CA3AF" />
        <YAxis stroke="#9CA3AF" />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1F2937',
            border: '1px solid #374151',
            borderRadius: '6px'
          }}
        />

        {/* Ichimoku Cloud */}
        <Area
          dataKey="senkou_a"
          stroke="#10B981"
          fill="url(#cloudGradient)"
          fillOpacity={0.3}
          strokeWidth={1}
        />
        <Area
          dataKey="senkou_b"
          stroke="#EF4444"
          fill="url(#cloudGradient)"
          fillOpacity={0.3}
          strokeWidth={1}
        />

        {/* Price Line */}
        <Line
          type="monotone"
          dataKey="price"
          stroke="#F59E0B"
          strokeWidth={2}
          dot={false}
        />

        {/* Tenkan-sen (Conversion Line) */}
        <Line
          type="monotone"
          dataKey="tenkan"
          stroke="#3B82F6"
          strokeWidth={1.5}
          strokeDasharray="5 5"
          dot={false}
        />

        {/* Kijun-sen (Base Line) */}
        <Line
          type="monotone"
          dataKey="kijun"
          stroke="#8B5CF6"
          strokeWidth={1.5}
          strokeDasharray="10 5"
          dot={false}
        />

        {/* Chikou Span (Lagging Line) */}
        <Line
          type="monotone"
          dataKey="chikou"
          stroke="#EC4899"
          strokeWidth={1}
          strokeDasharray="2 2"
          dot={false}
        />

        <defs>
          <linearGradient id="cloudGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#EF4444" stopOpacity={0.1}/>
          </linearGradient>
        </defs>
      </AreaChart>
    </ResponsiveContainer>
  );
};

const TechnicalAnalysis: React.FC = () => {
  const [data, setData] = useState<TechnicalAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('daily');

  // Mock data pour les tests
  const mockTechnicalData: TechnicalAnalysisData = {
    technical_analysis: {
      'AAPL': {
        timeframes: {
          'short_term': { trend_score: 0.73, momentum_score: 0.68, volatility_score: 0.45, volume_score: 0.82, timeframe_score: 0.67 },
          'medium_term': { trend_score: 0.84, momentum_score: 0.71, volatility_score: 0.52, volume_score: 0.78, timeframe_score: 0.71 },
          'long_term': { trend_score: 0.91, momentum_score: 0.65, volatility_score: 0.38, volume_score: 0.69, timeframe_score: 0.66 }
        },
        overall_signal: 'BUY',
        confidence: 0.78,
        composite_score: 0.68,
        market_regime: 'uptrending',
        ichimoku_signals: {
          'short_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'medium_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'long_term': { tenkan_kijun_cross: 'neutral', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: false }
        }
      },
      'MSFT': {
        timeframes: {
          'short_term': { trend_score: 0.81, momentum_score: 0.75, volatility_score: 0.42, volume_score: 0.88, timeframe_score: 0.72 },
          'medium_term': { trend_score: 0.87, momentum_score: 0.69, volatility_score: 0.48, volume_score: 0.83, timeframe_score: 0.72 },
          'long_term': { trend_score: 0.93, momentum_score: 0.71, volatility_score: 0.35, volume_score: 0.75, timeframe_score: 0.69 }
        },
        overall_signal: 'STRONG_BUY',
        confidence: 0.85,
        composite_score: 0.71,
        market_regime: 'uptrending',
        ichimoku_signals: {
          'short_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'medium_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'long_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true }
        }
      },
      'TSLA': {
        timeframes: {
          'short_term': { trend_score: 0.45, momentum_score: 0.38, volatility_score: 0.72, volume_score: 0.91, timeframe_score: 0.62 },
          'medium_term': { trend_score: 0.52, momentum_score: 0.44, volatility_score: 0.68, volume_score: 0.85, timeframe_score: 0.62 },
          'long_term': { trend_score: 0.38, momentum_score: 0.31, volatility_score: 0.75, volume_score: 0.67, timeframe_score: 0.53 }
        },
        overall_signal: 'HOLD',
        confidence: 0.62,
        composite_score: 0.59,
        market_regime: 'volatile',
        ichimoku_signals: {
          'short_term': { tenkan_kijun_cross: 'neutral', price_vs_cloud: 'inside', cloud_color: 'red', chikou_confirmation: false },
          'medium_term': { tenkan_kijun_cross: 'bearish', price_vs_cloud: 'inside', cloud_color: 'red', chikou_confirmation: false },
          'long_term': { tenkan_kijun_cross: 'bearish', price_vs_cloud: 'below', cloud_color: 'red', chikou_confirmation: false }
        }
      },
      'GOOGL': {
        timeframes: {
          'short_term': { trend_score: 0.76, momentum_score: 0.82, volatility_score: 0.55, volume_score: 0.73, timeframe_score: 0.72 },
          'medium_term': { trend_score: 0.83, momentum_score: 0.78, volatility_score: 0.49, volume_score: 0.71, timeframe_score: 0.70 },
          'long_term': { trend_score: 0.89, momentum_score: 0.74, volatility_score: 0.41, volume_score: 0.68, timeframe_score: 0.68 }
        },
        overall_signal: 'BUY',
        confidence: 0.81,
        composite_score: 0.70,
        market_regime: 'uptrending',
        ichimoku_signals: {
          'short_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'medium_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'long_term': { tenkan_kijun_cross: 'neutral', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true }
        }
      },
      'NVDA': {
        timeframes: {
          'short_term': { trend_score: 0.95, momentum_score: 0.91, volatility_score: 0.68, volume_score: 0.94, timeframe_score: 0.87 },
          'medium_term': { trend_score: 0.92, momentum_score: 0.88, volatility_score: 0.65, volume_score: 0.89, timeframe_score: 0.84 },
          'long_term': { trend_score: 0.87, momentum_score: 0.83, volatility_score: 0.58, volume_score: 0.81, timeframe_score: 0.77 }
        },
        overall_signal: 'STRONG_BUY',
        confidence: 0.94,
        composite_score: 0.83,
        market_regime: 'uptrending',
        ichimoku_signals: {
          'short_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'medium_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true },
          'long_term': { tenkan_kijun_cross: 'bullish', price_vs_cloud: 'above', cloud_color: 'green', chikou_confirmation: true }
        }
      }
    },
    summary: {
      total_symbols: 5,
      bullish_symbols: 3,
      bearish_symbols: 0,
      neutral_symbols: 2,
      avg_confidence: 0.80,
      market_regimes: {
        uptrending: 4,
        downtrending: 0,
        ranging: 0,
        volatile: 1,
        unknown: 0
      }
    },
    timestamp: new Date().toISOString()
  };

  // Mock Ichimoku data for demonstration
  const mockIchimokuData = Array.from({ length: 50 }, (_, i) => ({
    time: new Date(Date.now() - (49 - i) * 24 * 60 * 60 * 1000).toLocaleDateString(),
    price: 180 + Math.random() * 20 + Math.sin(i * 0.1) * 10,
    tenkan: 175 + Math.random() * 15 + Math.sin(i * 0.1) * 8,
    kijun: 170 + Math.random() * 18 + Math.sin(i * 0.15) * 12,
    senkou_a: 172 + Math.random() * 16 + Math.sin(i * 0.12) * 10,
    senkou_b: 168 + Math.random() * 20 + Math.sin(i * 0.08) * 15,
    chikou: 175 + Math.random() * 18 + Math.sin(i * 0.09) * 11,
  }));

  useEffect(() => {
    fetchTechnicalAnalysis();
  }, []);

  const fetchTechnicalAnalysis = async () => {
    setLoading(true);
    try {
      // Essaie d'abord l'API backend
      const response = await fetch('/api/technical-analysis');
      if (response.ok) {
        const result = await response.json();
        setData(result);
        if (result.technical_analysis && Object.keys(result.technical_analysis).length > 0) {
          setSelectedSymbol(Object.keys(result.technical_analysis)[0]);
        }
      } else {
        throw new Error('API not available');
      }
    } catch (error) {
      console.log('API not available, using mock data:', error);
      // Si l'API n'est pas disponible, utilise les données mock
      setData(mockTechnicalData);
      setSelectedSymbol('AAPL');
    } finally {
      setLoading(false);
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'STRONG_BUY': return 'bg-green-500';
      case 'BUY': return 'bg-green-400';
      case 'HOLD': return 'bg-yellow-500';
      case 'SELL': return 'bg-red-400';
      case 'STRONG_SELL': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'uptrending': return 'text-green-400';
      case 'downtrending': return 'text-red-400';
      case 'volatile': return 'text-yellow-400';
      case 'ranging': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  if (loading) {
    return (
      <div className="flex-1 bg-background">
        <Navigation />
        <div className="p-6 flex items-center justify-center">
          <RefreshCw className="w-8 h-8 animate-spin text-primary" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-background">
      <Navigation />

      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-semibold mb-2">Technical Analysis Dashboard</h1>
            <p className="text-muted-foreground">
              Multi-timeframe analysis with Ichimoku clouds, MACD, RSI, and comprehensive indicators
            </p>
          </div>
          <Button onClick={fetchTechnicalAnalysis} variant="outline" size="sm" className="gap-2">
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
        </div>

        {data && (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-muted-foreground">Bullish</span>
                  </div>
                  <div className="text-2xl font-bold text-green-400">{data.summary.bullish_symbols}</div>
                  <div className="text-xs text-muted-foreground">
                    {((data.summary.bullish_symbols / data.summary.total_symbols) * 100).toFixed(1)}%
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingDown className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-muted-foreground">Bearish</span>
                  </div>
                  <div className="text-2xl font-bold text-red-400">{data.summary.bearish_symbols}</div>
                  <div className="text-xs text-muted-foreground">
                    {((data.summary.bearish_symbols / data.summary.total_symbols) * 100).toFixed(1)}%
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm text-muted-foreground">Neutral</span>
                  </div>
                  <div className="text-2xl font-bold text-yellow-400">{data.summary.neutral_symbols}</div>
                  <div className="text-xs text-muted-foreground">
                    {((data.summary.neutral_symbols / data.summary.total_symbols) * 100).toFixed(1)}%
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Gauge className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-muted-foreground">Avg Confidence</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-400">
                    {(data.summary.avg_confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">System confidence</div>
                </CardContent>
              </Card>
            </div>

            {/* Symbol Selection */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5" />
                    Symbols Overview
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {Object.entries(data.technical_analysis).map(([symbol, analysis]) => (
                      <div
                        key={symbol}
                        className={`p-3 rounded-lg border cursor-pointer transition-all ${
                          selectedSymbol === symbol
                            ? 'border-primary bg-primary/10'
                            : 'border-border hover:border-primary/50'
                        }`}
                        onClick={() => setSelectedSymbol(symbol)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{symbol}</span>
                          <Badge className={getSignalColor(analysis.overall_signal)}>
                            {analysis.overall_signal}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Score:</span>
                          <span className={`font-mono ${
                            analysis.composite_score > 0 ? 'text-green-400' :
                            analysis.composite_score < 0 ? 'text-red-400' : 'text-yellow-400'
                          }`}>
                            {formatScore(analysis.composite_score)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Confidence:</span>
                          <span className="text-blue-400 font-mono">
                            {(analysis.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Regime:</span>
                          <span className={`capitalize ${getRegimeColor(analysis.market_regime)}`}>
                            {analysis.market_regime}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Ichimoku Cloud Chart */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Layers className="w-5 h-5" />
                    Ichimoku Cloud Analysis - {selectedSymbol}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <IchimokuCloudChart data={mockIchimokuData} />
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-4 text-xs">
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-0.5 bg-yellow-500"></div>
                      <span>Price</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-0.5 bg-blue-500 opacity-75" style={{borderTop: '1px dashed'}}></div>
                      <span>Tenkan-sen</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-0.5 bg-purple-500 opacity-75"></div>
                      <span>Kijun-sen</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-2 bg-gradient-to-b from-green-500 to-red-500 opacity-30"></div>
                      <span>Cloud</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-0.5 bg-pink-500 opacity-75" style={{borderTop: '1px dotted'}}></div>
                      <span>Chikou</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Detailed Analysis for Selected Symbol */}
            {selectedSymbol && data.technical_analysis[selectedSymbol] && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Multi-Timeframe Analysis - {selectedSymbol}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {Object.entries(data.technical_analysis[selectedSymbol].timeframes).map(([timeframe, signals]) => (
                      <div key={timeframe} className="p-4 border border-border rounded-lg">
                        <h4 className="font-medium mb-3 capitalize">{timeframe.replace('_', ' ')}</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Trend:</span>
                            <span className={`font-mono ${
                              signals.trend_score > 0 ? 'text-green-400' :
                              signals.trend_score < 0 ? 'text-red-400' : 'text-yellow-400'
                            }`}>
                              {formatScore(signals.trend_score)}%
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Momentum:</span>
                            <span className={`font-mono ${
                              signals.momentum_score > 0 ? 'text-green-400' :
                              signals.momentum_score < 0 ? 'text-red-400' : 'text-yellow-400'
                            }`}>
                              {formatScore(signals.momentum_score)}%
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Volatility:</span>
                            <span className={`font-mono ${
                              signals.volatility_score > 0 ? 'text-green-400' :
                              signals.volatility_score < 0 ? 'text-red-400' : 'text-yellow-400'
                            }`}>
                              {formatScore(signals.volatility_score)}%
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Volume:</span>
                            <span className={`font-mono ${
                              signals.volume_score > 0 ? 'text-green-400' :
                              signals.volume_score < 0 ? 'text-red-400' : 'text-yellow-400'
                            }`}>
                              {formatScore(signals.volume_score)}%
                            </span>
                          </div>
                          <div className="border-t border-border pt-2 mt-2">
                            <div className="flex justify-between text-sm font-medium">
                              <span>Overall:</span>
                              <span className={`font-mono ${
                                signals.timeframe_score > 0 ? 'text-green-400' :
                                signals.timeframe_score < 0 ? 'text-red-400' : 'text-yellow-400'
                              }`}>
                                {formatScore(signals.timeframe_score)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Market Regime Distribution */}
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  Market Regime Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  {Object.entries(data.summary.market_regimes).map(([regime, count]) => (
                    <div key={regime} className="text-center">
                      <div className={`text-2xl font-bold ${getRegimeColor(regime)}`}>
                        {count}
                      </div>
                      <div className="text-sm text-muted-foreground capitalize">
                        {regime}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {((count / data.summary.total_symbols) * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
};

export default TechnicalAnalysis;