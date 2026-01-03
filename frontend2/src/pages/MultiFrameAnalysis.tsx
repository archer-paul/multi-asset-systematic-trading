import React, { useState, useEffect } from "react";
import { Navigation } from "@/components/layout/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Clock,
  BarChart3,
  Activity,
  Layers,
  Eye,
  Zap,
  Target,
  TrendingUp,
  RefreshCw,
  Brain,
  Lightbulb,
  Star
} from "lucide-react";

interface EmergingStock {
  symbol: string;
  company_name: string;
  score: number;
  growth_potential: string;
  timeframe: string;
  key_drivers: string[];
  risk_factors: string[];
  market_cap: number;
  sector: string;
  confidence: number;
}

interface LongTermRecommendation {
  symbol: string;
  company_name: string;
  recommendation: string;
  target_price_3y: number;
  target_price_5y: number;
  current_price: number;
  dcf_valuation: number;
  esg_score: number;
  sector_outlook: string;
  key_catalysts: string[];
  risks: string[];
  confidence: number;
}

export default function MultiFrameAnalysis() {
  const [emergingStocks, setEmergingStocks] = useState<EmergingStock[]>([]);
  const [longTermRecommendations, setLongTermRecommendations] = useState<LongTermRecommendation[]>([]);
  const [loading, setLoading] = useState(true);

  // Mock data pour les stocks émergents
  const mockEmergingStocks: EmergingStock[] = [
    {
      symbol: 'PLTR',
      company_name: 'Palantir Technologies Inc.',
      score: 87.3,
      growth_potential: 'high',
      timeframe: 'medium-term',
      key_drivers: ['AI Growth', 'Government Contracts', 'Commercial Expansion'],
      risk_factors: ['Regulatory Risk', 'Competition'],
      market_cap: 45.2,
      sector: 'Technology',
      confidence: 0.84
    },
    {
      symbol: 'SOFI',
      company_name: 'SoFi Technologies Inc.',
      score: 82.1,
      growth_potential: 'high',
      timeframe: 'medium-term',
      key_drivers: ['Banking Charter', 'Fintech Growth', 'Student Loans'],
      risk_factors: ['Interest Rate Risk', 'Credit Risk'],
      market_cap: 8.7,
      sector: 'Financial Services',
      confidence: 0.78
    },
    {
      symbol: 'RBLX',
      company_name: 'Roblox Corporation',
      score: 79.5,
      growth_potential: 'high',
      timeframe: 'long-term',
      key_drivers: ['Metaverse Growth', 'Young Demographics', 'Platform Expansion'],
      risk_factors: ['User Safety', 'Content Moderation'],
      market_cap: 22.1,
      sector: 'Communication Services',
      confidence: 0.71
    },
    {
      symbol: 'CRWD',
      company_name: 'CrowdStrike Holdings Inc.',
      score: 85.7,
      growth_potential: 'high',
      timeframe: 'medium-term',
      key_drivers: ['Cybersecurity Growth', 'Cloud Adoption', 'Threat Intelligence'],
      risk_factors: ['Competition', 'Valuation'],
      market_cap: 78.9,
      sector: 'Technology',
      confidence: 0.88
    },
    {
      symbol: 'NET',
      company_name: 'Cloudflare Inc.',
      score: 83.2,
      growth_potential: 'high',
      timeframe: 'long-term',
      key_drivers: ['Edge Computing', 'Security Growth', 'Network Expansion'],
      risk_factors: ['Competition', 'Market Saturation'],
      market_cap: 32.4,
      sector: 'Technology',
      confidence: 0.82
    },
    {
      symbol: 'ABNB',
      company_name: 'Airbnb Inc.',
      score: 76.8,
      growth_potential: 'medium',
      timeframe: 'medium-term',
      key_drivers: ['Travel Recovery', 'International Growth', 'Experiences'],
      risk_factors: ['Regulatory Changes', 'Economic Sensitivity'],
      market_cap: 89.3,
      sector: 'Consumer Discretionary',
      confidence: 0.73
    }
  ];

  // Mock data pour les recommandations long terme
  const mockLongTermRecommendations: LongTermRecommendation[] = [
    {
      symbol: 'NVDA',
      company_name: 'NVIDIA Corporation',
      recommendation: 'Strong Buy',
      target_price_3y: 2200.0,
      target_price_5y: 3500.0,
      current_price: 875.0,
      dcf_valuation: 1850.0,
      esg_score: 7.2,
      sector_outlook: 'Very Positive',
      key_catalysts: ['AI Revolution', 'Data Center Growth', 'Autonomous Vehicles', 'Gaming Evolution'],
      risks: ['Semiconductor Cyclicality', 'Geopolitical Tensions', 'Competition'],
      confidence: 0.91
    },
    {
      symbol: 'MSFT',
      company_name: 'Microsoft Corporation',
      recommendation: 'Strong Buy',
      target_price_3y: 650.0,
      target_price_5y: 950.0,
      current_price: 420.0,
      dcf_valuation: 580.0,
      esg_score: 8.5,
      sector_outlook: 'Positive',
      key_catalysts: ['Cloud Dominance', 'AI Integration', 'Productivity Suite', 'Gaming Growth'],
      risks: ['Cloud Competition', 'Regulatory Scrutiny', 'Economic Slowdown'],
      confidence: 0.88
    },
    {
      symbol: 'GOOGL',
      company_name: 'Alphabet Inc.',
      recommendation: 'Buy',
      target_price_3y: 220.0,
      target_price_5y: 320.0,
      current_price: 145.0,
      dcf_valuation: 185.0,
      esg_score: 7.8,
      sector_outlook: 'Positive',
      key_catalysts: ['Search Moat', 'Cloud Growth', 'AI Leadership', 'YouTube Expansion'],
      risks: ['Regulatory Pressure', 'Competition', 'Ad Market Cycles'],
      confidence: 0.85
    },
    {
      symbol: 'TSLA',
      company_name: 'Tesla Inc.',
      recommendation: 'Hold',
      target_price_3y: 350.0,
      target_price_5y: 500.0,
      current_price: 248.0,
      dcf_valuation: 320.0,
      esg_score: 8.9,
      sector_outlook: 'Positive',
      key_catalysts: ['EV Adoption', 'Autonomous Driving', 'Energy Storage', 'Manufacturing Scale'],
      risks: ['EV Competition', 'Execution Risk', 'Valuation Premium'],
      confidence: 0.72
    }
  ];

  useEffect(() => {
    fetchAnalysisData();
  }, []);

  const fetchAnalysisData = async () => {
    setLoading(true);
    try {
      // Essaie d'abord l'API backend pour les stocks émergents
      try {
        const emergingResponse = await fetch('/api/emerging-stocks');
        if (emergingResponse.ok) {
          const emergingData = await emergingResponse.json();
          setEmergingStocks(emergingData.emerging_stocks || []);
        } else {
          throw new Error('Emerging stocks API not available');
        }
      } catch (emergingError) {
        console.log('Emerging stocks API not available, using mock data');
        setEmergingStocks(mockEmergingStocks);
      }

      // Essaie d'abord l'API backend pour les recommandations long terme
      try {
        const longTermResponse = await fetch('/api/long-term-analysis');
        if (longTermResponse.ok) {
          const longTermData = await longTermResponse.json();
          setLongTermRecommendations(longTermData.recommendations || []);
        } else {
          throw new Error('Long-term analysis API not available');
        }
      } catch (longTermError) {
        console.log('Long-term analysis API not available, using mock data');
        setLongTermRecommendations(mockLongTermRecommendations);
      }
    } catch (error) {
      console.error('Error fetching analysis data:', error);
      // Utilise les données mock en cas d'erreur générale
      setEmergingStocks(mockEmergingStocks);
      setLongTermRecommendations(mockLongTermRecommendations);
    } finally {
      setLoading(false);
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'Strong Buy': return 'bg-green-600 text-white';
      case 'Buy': return 'bg-green-500 text-white';
      case 'Hold': return 'bg-yellow-500 text-white';
      case 'Sell': return 'bg-red-500 text-white';
      case 'Strong Sell': return 'bg-red-600 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  return (
    <div className="flex-1 bg-background">
      <Navigation />

      <main className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-semibold mb-2">Multi-Frame Analysis Dashboard</h1>
            <p className="text-muted-foreground">
              Court terme, moyen terme et long terme avec recommandations AI
            </p>
          </div>
          <Button onClick={fetchAnalysisData} variant="outline" size="sm" className="gap-2">
            <RefreshCw className="w-4 h-4" />
            Actualiser
          </Button>
        </div>

        <Tabs defaultValue="short-term" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="short-term">Court Terme</TabsTrigger>
            <TabsTrigger value="medium-term">Moyen Terme</TabsTrigger>
            <TabsTrigger value="long-term">Long Terme</TabsTrigger>
          </TabsList>

          {/* Short Term Analysis */}
          <TabsContent value="short-term" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <MetricCard
                title="Signaux actifs"
                value="23"
                icon={<Activity className="w-4 h-4" />}
                change="+5"
                changeType="positive"
                subtitle="Scalping & Intraday"
                className="bg-gradient-to-br from-blue-500/10 to-cyan-600/10 border-blue-500/20"
              />
              <MetricCard
                title="Précision"
                value="94.7%"
                icon={<Target className="w-4 h-4" />}
                change="+2.3%"
                changeType="positive"
                subtitle="1m - 1h"
                className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border-green-500/20"
              />
              <MetricCard
                title="Latence"
                value="2.3s"
                icon={<Zap className="w-4 h-4" />}
                change="-0.4s"
                changeType="positive"
                subtitle="Temps de réponse"
                className="bg-gradient-to-br from-purple-500/10 to-violet-600/10 border-purple-500/20"
              />
              <MetricCard
                title="Volumes"
                value="187%"
                icon={<BarChart3 className="w-4 h-4" />}
                change="+34%"
                changeType="positive"
                subtitle="vs moyenne 30j"
                className="bg-gradient-to-br from-yellow-500/10 to-orange-600/10 border-yellow-500/20"
              />
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  Analyse Court Terme (1m - 1h)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { timeframe: "1m", name: "Scalping", signal: "BUY", strength: 87.3, confidence: 92.1 },
                    { timeframe: "5m", name: "Court terme", signal: "BUY", strength: 79.2, confidence: 88.7 },
                    { timeframe: "15m", name: "Intraday", signal: "HOLD", strength: 65.4, confidence: 75.3 },
                    { timeframe: "1h", name: "Horaire", signal: "BUY", strength: 82.7, confidence: 90.2 }
                  ].map((tf, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="text-center">
                          <div className="font-mono font-bold">{tf.timeframe}</div>
                          <div className="text-xs text-muted-foreground">{tf.name}</div>
                        </div>
                        <Badge variant={tf.signal === "BUY" ? "default" : tf.signal === "SELL" ? "destructive" : "secondary"}>
                          {tf.signal}
                        </Badge>
                      </div>
                      <div className="text-right text-sm">
                        <div className="font-mono text-green-400">{tf.strength}%</div>
                        <div className="text-xs text-muted-foreground">Force</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Medium Term Analysis */}
          <TabsContent value="medium-term" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <MetricCard
                title="Opportunités"
                value={emergingStocks.length.toString()}
                icon={<Lightbulb className="w-4 h-4" />}
                change="+3"
                changeType="positive"
                subtitle="Stocks émergents"
                className="bg-gradient-to-br from-orange-500/10 to-yellow-600/10 border-orange-500/20"
              />
              <MetricCard
                title="Score moyen"
                value={emergingStocks.length > 0 ? `${(emergingStocks.reduce((sum, stock) => sum + stock.score, 0) / emergingStocks.length).toFixed(1)}%` : "0%"}
                icon={<Star className="w-4 h-4" />}
                change="+5.2%"
                changeType="positive"
                subtitle="Performance IA"
                className="bg-gradient-to-br from-purple-500/10 to-pink-600/10 border-purple-500/20"
              />
              <MetricCard
                title="Confiance IA"
                value={emergingStocks.length > 0 ? `${(emergingStocks.reduce((sum, stock) => sum + stock.confidence * 100, 0) / emergingStocks.length).toFixed(1)}%` : "0%"}
                icon={<Brain className="w-4 h-4" />}
                change="+8.1%"
                changeType="positive"
                subtitle="Système expert"
                className="bg-gradient-to-br from-blue-500/10 to-indigo-600/10 border-blue-500/20"
              />
              <MetricCard
                title="Secteurs actifs"
                value={emergingStocks.length > 0 ? new Set(emergingStocks.map(s => s.sector)).size.toString() : "0"}
                icon={<Layers className="w-4 h-4" />}
                change="+2"
                changeType="positive"
                subtitle="Diversification"
                className="bg-gradient-to-br from-green-500/10 to-teal-600/10 border-green-500/20"
              />
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Recommandations AI Emerging Stock Detection
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">
                    <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-primary" />
                    <p className="text-muted-foreground">Chargement des recommandations AI...</p>
                  </div>
                ) : emergingStocks.length > 0 ? (
                  <div className="space-y-4">
                    {emergingStocks.slice(0, 6).map((stock, index) => (
                      <div key={index} className="p-4 border border-border rounded-lg">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <h4 className="font-semibold text-lg">{stock.symbol}</h4>
                            <p className="text-sm text-muted-foreground">{stock.company_name}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-green-400">{stock.score.toFixed(1)}%</div>
                            <Badge className={
                              stock.growth_potential === 'high' ? 'bg-green-500' :
                              stock.growth_potential === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                            }>
                              {stock.growth_potential} potential
                            </Badge>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Secteur</p>
                            <p className="text-sm font-medium">{stock.sector}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Cap. marché</p>
                            <p className="text-sm font-medium">${stock.market_cap}B</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Horizon</p>
                            <p className="text-sm font-medium capitalize">{stock.timeframe}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Confiance IA</p>
                            <p className="text-sm font-medium text-blue-400">{(stock.confidence * 100).toFixed(1)}%</p>
                          </div>
                        </div>

                        <div className="mb-3">
                          <p className="text-xs text-muted-foreground mb-1">Catalyseurs clés</p>
                          <div className="flex flex-wrap gap-1">
                            {stock.key_drivers.slice(0, 3).map((driver, i) => (
                              <Badge key={i} variant="outline" className="text-xs">{driver}</Badge>
                            ))}
                          </div>
                        </div>

                        <div>
                          <p className="text-xs text-muted-foreground mb-1">Facteurs de risque</p>
                          <div className="flex flex-wrap gap-1">
                            {stock.risk_factors.slice(0, 2).map((risk, i) => (
                              <Badge key={i} variant="outline" className="text-xs border-red-500/30 text-red-400">{risk}</Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">Aucune recommandation disponible</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Long Term Analysis */}
          <TabsContent value="long-term" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <MetricCard
                title="Recommandations"
                value={longTermRecommendations.length.toString()}
                icon={<Target className="w-4 h-4" />}
                change="+2"
                changeType="positive"
                subtitle="Investissement LT"
                className="bg-gradient-to-br from-blue-500/10 to-indigo-600/10 border-blue-500/20"
              />
              <MetricCard
                title="Rendement 3Y"
                value={longTermRecommendations.length > 0 ? `${(longTermRecommendations.reduce((sum, rec) => sum + ((rec.target_price_3y - rec.current_price) / rec.current_price * 100), 0) / longTermRecommendations.length).toFixed(1)}%` : "0%"}
                icon={<TrendingUp className="w-4 h-4" />}
                change="+12.4%"
                changeType="positive"
                subtitle="Potentiel moyen"
                className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border-green-500/20"
              />
              <MetricCard
                title="Score ESG"
                value={longTermRecommendations.length > 0 ? `${(longTermRecommendations.reduce((sum, rec) => sum + rec.esg_score, 0) / longTermRecommendations.length).toFixed(1)}` : "0"}
                icon={<Layers className="w-4 h-4" />}
                change="+0.3"
                changeType="positive"
                subtitle="Durabilité"
                className="bg-gradient-to-br from-teal-500/10 to-cyan-600/10 border-teal-500/20"
              />
              <MetricCard
                title="Confiance DCF"
                value={longTermRecommendations.length > 0 ? `${(longTermRecommendations.reduce((sum, rec) => sum + rec.confidence * 100, 0) / longTermRecommendations.length).toFixed(1)}%` : "0%"}
                icon={<Eye className="w-4 h-4" />}
                change="+4.7%"
                changeType="positive"
                subtitle="Modèle financier"
                className="bg-gradient-to-br from-purple-500/10 to-violet-600/10 border-purple-500/20"
              />
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="w-5 h-5" />
                  Recommandations Long Terme (3-5 ans)
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">
                    <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-primary" />
                    <p className="text-muted-foreground">Chargement des analyses long terme...</p>
                  </div>
                ) : longTermRecommendations.length > 0 ? (
                  <div className="space-y-6">
                    {longTermRecommendations.map((rec, index) => (
                      <div key={index} className="p-6 border border-border rounded-lg">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h4 className="font-semibold text-xl">{rec.symbol}</h4>
                            <p className="text-muted-foreground">{rec.company_name}</p>
                          </div>
                          <div className="text-right">
                            <Badge className={getRecommendationColor(rec.recommendation)}>
                              {rec.recommendation}
                            </Badge>
                            <div className="text-sm text-muted-foreground mt-1">
                              Confiance: {(rec.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Prix actuel</p>
                            <p className="text-lg font-bold">${rec.current_price.toFixed(2)}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Objectif 3 ans</p>
                            <p className="text-lg font-bold text-green-400">${rec.target_price_3y.toFixed(2)}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Objectif 5 ans</p>
                            <p className="text-lg font-bold text-blue-400">${rec.target_price_5y.toFixed(2)}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Valorisation DCF</p>
                            <p className="text-lg font-bold text-purple-400">${rec.dcf_valuation.toFixed(2)}</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Potentiel 3 ans</p>
                            <p className="text-sm font-bold text-green-400">
                              +{(((rec.target_price_3y - rec.current_price) / rec.current_price) * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Score ESG</p>
                            <p className="text-sm font-bold text-teal-400">{rec.esg_score}/10</p>
                          </div>
                        </div>

                        <div className="mb-4">
                          <p className="text-xs text-muted-foreground mb-2">Catalyseurs principaux</p>
                          <div className="flex flex-wrap gap-1">
                            {rec.key_catalysts.map((catalyst, i) => (
                              <Badge key={i} variant="outline" className="text-xs border-green-500/30 text-green-400">
                                {catalyst}
                              </Badge>
                            ))}
                          </div>
                        </div>

                        <div>
                          <p className="text-xs text-muted-foreground mb-2">Risques identifiés</p>
                          <div className="flex flex-wrap gap-1">
                            {rec.risks.map((risk, i) => (
                              <Badge key={i} variant="outline" className="text-xs border-red-500/30 text-red-400">
                                {risk}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">Aucune recommandation long terme disponible</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}