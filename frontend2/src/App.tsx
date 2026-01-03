import React from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Footer } from "@/components/layout/Footer";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import MLObservatory from "./pages/MLObservatory";
import SentimentIntelligence from "./pages/SentimentIntelligence";
import RiskAnalytics from "./pages/RiskAnalytics";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import GeopoliticalMonitor from "./pages/GeopoliticalMonitor";
import CongressTrading from "./pages/CongressTrading";
import EmergingStocks from "./pages/EmergingStocks";
import MultiFrameAnalysis from "./pages/MultiFrameAnalysis";
import TechnicalAnalysis from "@/pages/TechnicalAnalysis";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <div className="min-h-screen flex flex-col">
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/ml-observatory" element={<MLObservatory />} />
            <Route path="/sentiment-intelligence" element={<SentimentIntelligence />} />
            <Route path="/risk-analytics" element={<RiskAnalytics />} />
            <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
            <Route path="/geopolitical-monitor" element={<GeopoliticalMonitor />} />
            <Route path="/congress-trading" element={<CongressTrading />} />
            <Route path="/emerging-stocks" element={<EmergingStocks />} />
            <Route path="/multi-frame-analysis" element={<MultiFrameAnalysis />} />
            <Route path="/technical-analysis" element={<TechnicalAnalysis />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
          <Footer />
        </div>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
