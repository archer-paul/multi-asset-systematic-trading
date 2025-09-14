import './globals.css'
import { Inter, JetBrains_Mono } from 'next/font/google'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-jetbrains-mono',
})

export const metadata = {
  title: 'Quantitative Alpha Engine - Trading Dashboard',
  description: 'Advanced quantitative trading platform with machine learning and risk management',
  keywords: ['trading', 'quantitative', 'machine learning', 'finance', 'alpha'],
  authors: [{ name: 'Paul Archer', url: 'https://paul-archer.vercel.app' }],
  creator: 'Paul Archer',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://quantitative-alpha-engine.web.app',
    title: 'Quantitative Alpha Engine',
    description: 'Advanced quantitative trading platform with machine learning and risk management',
    siteName: 'QAE Dashboard',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Quantitative Alpha Engine',
    description: 'Advanced quantitative trading platform',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="theme-color" content="#0f1629" />
      </head>
      <body className="font-sans bg-dark-100 text-white antialiased">
        {children}
      </body>
    </html>
  )
}