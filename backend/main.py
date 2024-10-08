import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
from model_utils import load_model, generate_response

app = FastAPI()

# Load the model once at startup
model = load_model('model/mygpt_model.pth')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that handles the real-time interaction with the model.
    """
    await websocket.accept()


    try:
        while True:
            data = await websocket.receive_text()
            print(data)
            
            try:
                data_dict = json.loads(data)
                user_message = data_dict.get("message", "")
                max_tokens = data_dict.get("max_tokens", 100)
            except json.JSONDecodeError:
                await websocket.send_text("Invalid input format. Expected JSON.")
                continue

            model_response = generate_response(model, user_message, max_tokens=max_tokens)
            await websocket.send_text(model_response)

    except WebSocketDisconnect:
        print("Client disconnected")