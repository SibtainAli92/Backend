#!/bin/bash
# Startup script for FastAPI backend

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the server
python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --reload
