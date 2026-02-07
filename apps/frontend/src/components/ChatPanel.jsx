import { useMemo, useState } from "react";
import MessageList from "./MessageList.jsx";
import ActionPreview from "./ActionPreview.jsx";
import FileUpload from "./FileUpload.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:4000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

const initialAssistant = {
  id: "intro",
  role: "assistant",
  content:
    "I can ingest data, update inventory, run forecasts, and plan promotions. Tell me what you need or upload a CSV.",
  timestamp: new Date().toISOString(),
};

export default function ChatPanel() {
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([initialAssistant]);
  const [input, setInput] = useState("");
  const [attachment, setAttachment] = useState(null);
  const [pendingConfirmation, setPendingConfirmation] = useState(null);
  const [actionPreview, setActionPreview] = useState(null);
  const [structuredOutput, setStructuredOutput] = useState(null);
  const [loading, setLoading] = useState(false);

  const canSend = useMemo(() => input.trim().length > 0 || attachment, [input, attachment]);

  const callApi = async (path, payload) => {
    const headers = { "Content-Type": "application/json" };
    if (API_KEY) {
      headers.Authorization = `Bearer ${API_KEY}`;
    }
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || error.error || "Request failed");
    }
    return response.json();
  };

  const handleSend = async () => {
    if (!canSend || loading) return;
    const text = input.trim();
    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text || "[File uploaded]",
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const payload = {
        conversationId,
        message: text,
        attachment,
      };
      const result = await callApi("/api/chat/interpret", payload);
      setConversationId(result.conversationId);
      setActionPreview(result.actionPreview || null);
      setStructuredOutput(result.structuredOutput || null);

      if (result.requiresConfirmation) {
        setPendingConfirmation({
          confirmationId: result.confirmationId,
          message: result.assistantMessage,
        });
      } else {
        setPendingConfirmation(null);
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.assistantMessage,
          timestamp: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setAttachment(null);
      setLoading(false);
    }
  };

  const handleConfirm = async (approve) => {
    if (!pendingConfirmation) return;
    setLoading(true);
    try {
      const result = await callApi("/api/chat/confirm", {
        confirmationId: pendingConfirmation.confirmationId,
        approve,
      });
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.assistantMessage,
          timestamp: new Date().toISOString(),
        },
      ]);
      setStructuredOutput(result.structuredOutput || null);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setPendingConfirmation(null);
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <section className="chat-shell">
      <div className="chat-panel">
        <div className="chat-header">
          <div>
            <h2>Assistant Console</h2>
            <p>Natural language commands to structured operations.</p>
          </div>
          <span className={loading ? "pill busy" : "pill"}>{loading ? "Working" : "Ready"}</span>
        </div>

        <MessageList messages={messages} />

        <div className="chat-input">
          <textarea
            placeholder="Ask for an ingest, update, forecast, or promotion plan..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
          />
          <div className="chat-actions">
            <FileUpload attachment={attachment} onAttachment={setAttachment} />
            <button type="button" disabled={!canSend || loading} onClick={handleSend}>
              Send
            </button>
          </div>
        </div>

        {pendingConfirmation && (
          <div className="confirm-bar">
            <p>{pendingConfirmation.message}</p>
            <div>
              <button type="button" className="secondary" onClick={() => handleConfirm(false)}>
                Cancel
              </button>
              <button type="button" onClick={() => handleConfirm(true)}>
                Confirm
              </button>
            </div>
          </div>
        )}
      </div>

      <aside className="preview-panel">
        <ActionPreview title="Action Preview" data={actionPreview} emptyLabel="No action planned yet." />
        <ActionPreview title="Structured Output" data={structuredOutput} emptyLabel="No output yet." />
      </aside>
    </section>
  );
}
