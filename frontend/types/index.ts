// Types for API responses and data structures

export interface CacheStats {
  overall_hit_rate: number;
  memory_usage: {
    total_mb: number;
    avg_entry_kb: number;
    efficiency: number;
  };
  daily_stats: {
    total_misses: number;
  };
  active_entries: number;
  avg_ttl_seconds: number;
  hit_rates: Record<string, number>;
  ttl_seconds: Record<string, number>;
}

export interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  network_io: number;
  components: Array<{
    component: string;
    status: string;
    uptime: string;
  }>;
  api_performance: Record<string, {
    avg_response_time: number;
    requests_per_minute: number;
  }>;
}

export interface BatchResults {
  correlation_improvements: Record<string, number>;
  total_symbols: number;
  training_duration_hours: number;
  avg_improvement: number;
  cv_score: number;
  total_features: number;
  feature_importance_score: number;
  dimensionality_reduction: string;
}

export interface StressTestResults {
  historical_scenarios: Record<string, number>;
  custom_scenarios: Record<string, {
    impact: number;
    confidence: number;
    duration: string;
  }>;
  summary: {
    worst_case_scenario: string;
    average_impact: string;
    recovery_time: string;
    scenarios_tested: string;
  };
}

export interface MonteCarloResults {
  parameters: {
    num_simulations: string;
    time_horizon: string;
    confidence_level: string;
    correlation_model: string;
  };
  distribution: {
    expected_return: number;
    std_deviation: number;
    skewness: number;
    kurtosis: number;
  };
  percentiles: Record<string, number>;
  tail_risk: {
    var_95: number;
    var_99: number;
    expected_shortfall: number;
    max_drawdown: number;
  };
}

export interface MacroSentiment {
  economic_indicators: Record<string, number>;
  central_bank_communications: Array<{
    bank: string;
    latest_statement: string;
    sentiment_score: number;
    impact_on_markets: string;
  }>;
  global_risks: Record<string, {
    score: number;
    trend: string;
  }>;
  sector_impacts: Record<string, number>;
}

export interface CongressTrade {
  politician: string;
  symbol: string;
  transaction_type: string;
  amount_range: string;
  date: string;
  sentiment_score: number;
  market_performance: number;
}

export interface CongressData {
  trades: CongressTrade[];
  summary: {
    total_trades: number;
    net_activity: string;
    top_symbols: string[];
    performance_vs_market: number;
  };
}

export interface EmergingStock {
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

export interface EmergingStocksData {
  emerging_stocks: EmergingStock[];
  summary: {
    total_opportunities: number;
    avg_score: number;
    high_potential_count: number;
    sectors_represented: string[];
  };
}

export interface LongTermRecommendation {
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

export interface MarketOutlook {
  overall_sentiment: string;
  sector_rotations: string[];
  macro_trends: string[];
  risk_factors: string[];
}

export interface LongTermData {
  recommendations: LongTermRecommendation[];
  market_outlook: MarketOutlook;
}

export interface TimeframeData {
  trend: string;
  rsi: number;
  macd: string;
  volume: string;
}

export interface SymbolAnalysis {
  timeframes: {
    [key: string]: TimeframeData;
  };
  overall_signal: string;
  confidence: number;
  support_levels: number[];
  resistance_levels: number[];
}

export interface TechnicalData {
  technical_analysis: {
    [symbol: string]: SymbolAnalysis;
  };
  summary: {
    bullish_symbols: number;
    bearish_symbols: number;
    neutral_symbols: number;
    high_volume_symbols: string[];
  };
}