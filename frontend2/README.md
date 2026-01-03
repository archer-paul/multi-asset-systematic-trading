# Trading Bot Frontend

Advanced React frontend for the Quantitative Alpha Engine trading system.

## Project info

**Author**: Paul Archer - Imperial College London
**Backend**: Advanced Quantitative Trading System with Economic Intelligence

## Technology Stack

This project is built with:

- **React 18** - Modern UI library
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **ShadCN/UI** - Beautiful, accessible component library
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization for financial charts
- **React Query** - Server state management
- **React Router** - Client-side routing

## Features

- **Portfolio Dashboard** - Real-time portfolio performance vs S&P 500
- **ML Observatory** - Machine learning model performance tracking
- **Risk Analytics** - VaR analysis and stress testing
- **Sentiment Intelligence** - Multi-source sentiment analysis
- **Knowledge Graph** - Economic relationship visualization
- **Congressional Trading** - Institutional trading signal analysis
- **Emerging Stocks** - AI-powered growth stock detection

## Development Setup

### Prerequisites

- Node.js 18+
- npm or yarn package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Navigate to frontend directory
cd frontend2

# Install dependencies
npm install

# Start development server
npm run dev
```

### Configuration

Create a `.env` file in the root directory:

```bash
REACT_APP_API_URL=http://localhost:8080
VITE_API_URL=http://localhost:8080
```

### Build and Deploy

```bash
# Development build
npm run build:dev

# Production build
npm run build

# Preview production build
npm run preview
```

## API Integration

The frontend connects to the Python Flask backend running on port 8080. See `src/lib/api.ts` for API client implementation.

### WebSocket Support

Real-time updates are supported via WebSocket connections for:
- Portfolio performance updates
- Trading signals
- ML model metrics
- Risk alerts

## Component Architecture

```
src/
├── components/
│   ├── charts/          # Financial data visualizations
│   ├── dashboard/       # Dashboard components
│   ├── layout/          # Navigation and layout
│   └── ui/              # ShadCN UI components
├── pages/               # Route pages
├── lib/                 # Utilities and API client
└── hooks/               # Custom React hooks
```

## Trading Dashboard Features

### Portfolio Analytics
- Real-time P&L tracking
- Performance vs benchmarks
- Sector allocation pie charts
- Holdings management

### Machine Learning Observatory
- Model ensemble weights visualization
- Prediction confidence tracking
- Training progress monitoring
- Feature importance analysis

### Risk Management
- Value at Risk (VaR) calculations
- Stress test scenarios
- Portfolio exposure analysis
- Risk alert monitoring

## Contributing

1. Follow TypeScript best practices
2. Use ShadCN components when possible
3. Implement responsive design
4. Add proper error handling
5. Write meaningful component documentation

## License

This project is part of the Quantitative Alpha Engine research system at Imperial College London.