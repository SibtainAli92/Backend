"""
Vercel entry point for FastAPI backend.

This file serves as the adapter between Vercel's serverless functions
and the FastAPI application using Mangum as an ASGI adapter.
"""

import os
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set environment variables for serverless environment
os.environ.setdefault("DEBUG", "False")

# Import the FastAPI app and Mangum for serverless compatibility
from app import app as application
from mangum import Mangum

# Wrap the FastAPI app with Mangum for Vercel compatibility
# Use lifespan='off' to prevent startup errors from crashing the function
handler = Mangum(application, lifespan="off")

# Export the handler for Vercel
__all__ = ["handler", "application"]
