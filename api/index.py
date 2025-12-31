"""
Vercel serverless function entry point for FastAPI backend.

This file handles the Vercel serverless function interface.
"""

import os
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set environment variables for serverless environment BEFORE any imports
os.environ.setdefault("DEBUG", "False")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Import the ASGI adapter and FastAPI app
from mangum import Mangum
from app import app

# Wrap the FastAPI app with Mangum for Vercel compatibility
handler = Mangum(app)

# Export the handler for Vercel
__all__ = ["handler"]
