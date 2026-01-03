import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

// Créons des composants temporaires simples
const SimpleIndex = () => (
  <div style={{ padding: "20px" }}>
    <h1>🚀 Trading Bot Dashboard</h1>
    <p>Dashboard principal - En cours de développement</p>
    <div style={{ marginTop: "20px", padding: "10px", backgroundColor: "#f8f9fa", borderRadius: "5px" }}>
      <h2>Fonctionnalités à venir :</h2>
      <ul>
        <li>📊 Graphique de performance vs S&P 500</li>
        <li>🥧 Camembert des holdings par secteur</li>
        <li>📈 Métriques en temps réel</li>
      </ul>
    </div>
  </div>
);

const SimpleMLObservatory = () => (
  <div style={{ padding: "20px" }}>
    <h1>🧠 ML Observatory</h1>
    <p>Observatoire des modèles de machine learning</p>
    <div style={{ marginTop: "20px", padding: "10px", backgroundColor: "#f8f9fa", borderRadius: "5px" }}>
      <h2>Fonctionnalités à venir :</h2>
      <ul>
        <li>🥧 Camembert des poids du meta-learner</li>
        <li>📉 Graphique d'erreur de prédiction</li>
        <li>📊 Métriques de convergence</li>
      </ul>
    </div>
  </div>
);

const SimpleNavigation = () => (
  <nav style={{ padding: "10px", backgroundColor: "#000", color: "#fff" }}>
    <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
      <h2 style={{ margin: 0 }}>Trading Bot</h2>
      <a href="/" style={{ color: "#fff", textDecoration: "none" }}>Dashboard</a>
      <a href="/ml-observatory" style={{ color: "#fff", textDecoration: "none" }}>ML Observatory</a>
    </div>
  </nav>
);

const NotFound = () => (
  <div style={{ padding: "20px", textAlign: "center" }}>
    <h1>404 - Page non trouvée</h1>
    <p><a href="/">Retour au dashboard</a></p>
  </div>
);

const App = () => (
  <BrowserRouter>
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <SimpleNavigation />
      <div style={{ flex: 1 }}>
        <Routes>
          <Route path="/" element={<SimpleIndex />} />
          <Route path="/ml-observatory" element={<SimpleMLObservatory />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
      <footer style={{ padding: "10px", backgroundColor: "#f8f9fa", textAlign: "center" }}>
        <p>Trading Bot Frontend - Powered by React + Vite</p>
      </footer>
    </div>
  </BrowserRouter>
);

export default App;