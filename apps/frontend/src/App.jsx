import ChatPanel from "./components/ChatPanel.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Fresh Flow</p>
          <h1>Operational Command Assistant</h1>
          <p className="subhead">
            Translate requests into inventory actions, forecasts, and promotions with built-in confirmations.
          </p>
        </div>
        <div className="status-card">
          <span className="status-dot" />
          <div>
            <p className="status-title">Assistant Status</p>
            <p className="status-subtitle">Secure API connected</p>
          </div>
        </div>
      </header>
      <main className="app-main">
        <ChatPanel />
      </main>
    </div>
  );
}
