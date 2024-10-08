export class WebSocketService {
    private socket: WebSocket;
    private messageHandler: (message: string) => void;
  
    constructor(url: string, messageHandler: (message: string) => void) {
      this.socket = new WebSocket(url);
      this.messageHandler = messageHandler;
  
      this.socket.onmessage = (event) => {
        this.messageHandler(event.data);
      };
  
      this.socket.onopen = () => {
        console.log("WebSocket connection established");
      };
  
      this.socket.onclose = () => {
        console.log("WebSocket connection closed");
      };
    }
  
    public sendMessage(message: string, maxTokens: number) {
      console.log('send message')
      
      const payload = {
        message: message,
        max_tokens: maxTokens
      };
      this.socket.send(JSON.stringify(payload));
    }
  
    public closeConnection() {
      this.socket.close();
    }
  }
  