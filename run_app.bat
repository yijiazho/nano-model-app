@echo off

cd backend
echo Starting FastAPI backend...
start cmd /k "venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8080 --reload"

cd ../frontend
echo Starting React frontend...
start cmd /k "npm start"

cd ..
