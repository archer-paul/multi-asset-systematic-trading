# Quantitative Alpha Engine - Frontend

Modern React/Next.js dashboard for the Quantitative Alpha Engine trading bot.

## Features

- **Modern React Architecture**: Built with Next.js 14 and TypeScript
- **Responsive Design**: Tailwind CSS with dark theme
- **Animated Components**: Framer Motion animations
- **Real-time Data**: Socket.IO integration
- **Interactive Charts**: Recharts for data visualization
- **4 Main Pages**:
  - Dashboard: Portfolio overview and performance
  - Machine Learning: Model analytics and performance
  - Risk Management: Risk analysis and alerts
  - Portfolio: Holdings and allocation details

## Tech Stack

- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Heroicons
- **State Management**: React Hooks
- **Real-time**: Socket.IO Client

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Export static files for Firebase
npm run export
```

## Deployment

### Firebase Hosting

1. Install Firebase CLI:
```bash
npm install -g firebase-tools
```

2. Login to Firebase:
```bash
firebase login
```

3. Initialize project (if not already done):
```bash
firebase init hosting
```

4. Build and deploy:
```bash
npm run build
npm run export
firebase deploy
```

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Dashboard page
│   ├── machine-learning/  # ML analytics page
│   ├── risk-management/   # Risk management page
│   ├── portfolio/         # Portfolio page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # Reusable components
│   ├── Layout/           # Layout components
│   │   ├── Sidebar.tsx   # Animated sidebar
│   │   ├── Header.tsx    # Header with status
│   │   ├── Footer.tsx    # Footer with links
│   │   └── Layout.tsx    # Main layout wrapper
│   ├── Charts/           # Chart components
│   │   └── PerformanceChart.tsx
│   └── MetricCard.tsx    # Metric display card
├── lib/                  # Utilities and configs
├── types/                # TypeScript type definitions
└── public/               # Static assets
```

## Design System

### Colors
- **Dark Theme**: Primary background colors
- **Accent Colors**: Blue, Green, Red, Yellow, Purple
- **Trading Colors**: Profit green, Loss red, Neutral gray

### Components
- **MetricCard**: Reusable metric display with animations
- **PerformanceChart**: Interactive performance charts
- **Sidebar**: Animated navigation menu
- **Layout**: Responsive layout wrapper

## Data Integration

The frontend connects to the Flask backend API running on port 5000:

- Real-time market data via WebSocket
- Portfolio analytics via REST API
- ML model performance metrics
- Risk management alerts

## Configuration

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=ws://localhost:5000
```

### Firebase Configuration

The project is configured to deploy to Firebase Hosting with the project ID `quantitative-alpha-engine`.

## Author

**Paul Archer**
Imperial College London

- Email: paul.erwan.archer@gmail.com
- GitHub: [archer-paul](https://github.com/archer-paul)
- LinkedIn: [p-archer](https://www.linkedin.com/in/p-archer/)
- Website: [paul-archer.vercel.app](https://paul-archer.vercel.app/)

## License

Private project for portfolio demonstration.