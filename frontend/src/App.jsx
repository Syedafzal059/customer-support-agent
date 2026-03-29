import { useCallback, useMemo, useState } from "react";
import "./App.css";

const USER_ID_KEY = "csa_user_id";

function loadOrCreateUserId() {
  try {
    let id = localStorage.getItem(USER_ID_KEY);
    if (!id || !id.trim()) {
      id = `web-${crypto.randomUUID?.() ?? Date.now().toString(36)}`;
      localStorage.setItem(USER_ID_KEY, id);
    }
    return id.trim();
  } catch {
    return `web-${Date.now().toString(36)}`;
  }
}

function apiBaseUrl() {
  const raw = import.meta.env.VITE_API_URL;
  if (raw && String(raw).trim()) {
    return String(raw).replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

export default function App() {
  const [userId] = useState(loadOrCreateUserId);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const baseUrl = useMemo(() => apiBaseUrl(), []);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setError(null);
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);
    try {
      const res = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: text }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d) => d.msg || d).join(" ")
              : res.statusText;
        throw new Error(detail || `Request failed (${res.status})`);
      }
      setMessages((prev) => [...prev, { role: "assistant", text: data.response ?? "" }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Network error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [baseUrl, input, loading, userId]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Support chat</h1>
      </header>

      <div className="messages" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty-state">Send a message to get started.</div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              {m.text}
            </div>
          ))
        )}
      </div>

      <div className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your message… (Enter to send, Shift+Enter for newline)"
          disabled={loading}
          aria-label="Message"
        />
        <div className="composer-actions">
          {error && <p className="error">{error}</p>}
          <button type="button" className="send" onClick={send} disabled={loading || !input.trim()}>
            {loading ? "Sending…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
