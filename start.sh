#!/bin/bash
set -e

# Function to handle shutdown
shutdown() {
    echo "Shutting down..."
    kill $API_PID $STREAMLIT_PID 2>/dev/null
    wait $API_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Trap signals
trap shutdown SIGINT SIGTERM

# Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a bit for the API to start
sleep 3

# Start Streamlit frontend
echo "Starting Streamlit frontend on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo "Both services started:"
echo "  - API: http://localhost:8000"
echo "  - UI:  http://localhost:8501"

# Wait for either process to exit
wait -n $API_PID $STREAMLIT_PID 2>/dev/null

# If we get here, one process exited, so shut down both
shutdown