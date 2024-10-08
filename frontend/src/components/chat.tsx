// src/components/Chat.tsx
import React, { useState, useEffect } from "react";
import { WebSocketService } from "../services/websocket";
import "./chat.css"

const Chat: React.FC = () => {
  const [message, setMessage] = useState<string>("");
  const [maxTokens, setMaxTokens] = useState<number>(100);
  const [responses, setResponses] = useState<string[]>([]);
  const [wsService, setWsService] = useState<WebSocketService | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const webSocketUrl = "ws://localhost:8080/ws";
    const ws = new WebSocketService(webSocketUrl, handleNewMessage);
    setWsService(ws);

    // Clean up WebSocket connection on component unmount
    return () => {
      ws.closeConnection();
    };
  }, []);

  const handleNewMessage = (message: string) => {
    setResponses((prevResponses) => [...prevResponses, message]);
    setLoading(false);
  };

  const handleSendMessage = () => {
    if (wsService && message) {
      setLoading(true);
      wsService.sendMessage(message, maxTokens);
      setMessage("");
    }
  };

  return (
    <div className="chat-container">
      <h1>MyGPT trained on literature novels</h1>
      <div className="chat-box">
        {responses.map((response, index) => (
          <div key={index} className="chat-response">{response}</div>
        ))}
        {loading && <div className="loader"></div>}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          className="message-input"
        />
        <select
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          className="dropdown"
        >
          <option value={100}>Short</option>
          <option value={200}>Mid</option>
          <option value={400}>Long</option>
        </select>
        <button onClick={handleSendMessage} className="send-button">Send</button>
      </div>
    </div>
  );
};

export default Chat;
