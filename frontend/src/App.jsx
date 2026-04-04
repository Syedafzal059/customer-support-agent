import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
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

  const messagesRef = useRef(null);
  const baseUrl = useMemo(() => apiBaseUrl(), []);

  useLayoutEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "auto" });
  }, [messages, loading]);

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
      const reqId =
        res.headers.get("X-Request-ID")?.trim() ||
        res.headers.get("x-request-id")?.trim() ||
        null;
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
      const rid = (data.request_id && String(data.request_id).trim()) || reqId;
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.response ?? "", requestId: rid || undefined, feedbackSent: false },
      ]);
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

  const sendFeedback = useCallback(
    async (messageIndex, thumbs) => {
      const m = messages[messageIndex];
      if (!m || m.role !== "assistant" || !m.requestId || m.feedbackSent) return;
      setError(null);
      try {
        const res = await fetch(`${baseUrl}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            request_id: m.requestId,
            user_id: userId,
            thumbs,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const detail =
            typeof data.detail === "string"
              ? data.detail
              : Array.isArray(data.detail)
                ? data.detail.map((d) => d.msg || d).join(" ")
                : res.statusText;
          throw new Error(detail || `Feedback failed (${res.status})`);
        }
        setMessages((prev) =>
          prev.map((msg, i) => (i === messageIndex ? { ...msg, feedbackSent: true } : msg)),
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Feedback failed";
        setError(msg);
      }
    },
    [baseUrl, messages, userId],
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Support</h1>
      </header>

      <div className="messages" ref={messagesRef} aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty-state">Send a message to get started.</div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`message-block ${m.role}`}>
              <div className={`bubble ${m.role}`}>{m.text}</div>
              {m.role === "assistant" && m.requestId && (
                <div className="feedback-row" aria-label="Rate this reply">
                  <span className="feedback-label">Helpful?</span>
                  <button
                    type="button"
                    className="thumb"
                    disabled={!!m.feedbackSent}
                    onClick={() => sendFeedback(i, "up")}
                    aria-label="Thumbs up"
                  >
                    👍
                  </button>
                  <button
                    type="button"
                    className="thumb"
                    disabled={!!m.feedbackSent}
                    onClick={() => sendFeedback(i, "down")}
                    aria-label="Thumbs down"
                  >
                    👎
                  </button>
                  {m.feedbackSent && <span className="feedback-thanks">Thanks for the feedback.</span>}
                </div>
              )}
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
