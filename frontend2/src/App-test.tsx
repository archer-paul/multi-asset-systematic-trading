import React from "react";

const App = () => {
  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>🚀 Trading Bot Frontend Test</h1>
      <p>Si vous voyez ce message, React fonctionne !</p>
      <div style={{ marginTop: "20px", padding: "10px", backgroundColor: "#f0f0f0", borderRadius: "5px" }}>
        <h2>Statut :</h2>
        <ul>
          <li>✅ React chargé</li>
          <li>✅ Vite serveur actif</li>
          <li>✅ Composant rendu</li>
        </ul>
      </div>
    </div>
  );
};

export default App;