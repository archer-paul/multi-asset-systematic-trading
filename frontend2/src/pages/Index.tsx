import { TradingDashboard } from "@/components/dashboard/TradingDashboard";
import { Navigation } from "@/components/layout/Navigation";
import { PortfolioPerformanceChart } from "@/components/charts/PortfolioPerformanceChart";
import { EnhancedPortfolioHoldings } from "@/components/charts/EnhancedPortfolioHoldings";

const Index = () => {
  return (
    <div className="flex-1 bg-background">
      <Navigation />
      <div className="p-6">
        {/* Main dashboard section with 2/3 - 1/3 layout */}
        <div className="mb-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Performance chart takes 2/3 of the width */}
          <div className="lg:col-span-2">
            <PortfolioPerformanceChart />
          </div>
          {/* Holdings pie chart takes 1/3 of the width */}
          <div className="lg:col-span-1">
            <EnhancedPortfolioHoldings />
          </div>
        </div>

        {/* Rest of the dashboard components */}
        <TradingDashboard />
      </div>
    </div>
  );
};

export default Index;
