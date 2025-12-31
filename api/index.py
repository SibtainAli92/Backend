"""
Vercel entry point for FastAPI backend.

This file serves as the adapter between Vercel's serverless functions
and the FastAPI application.
"""

import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import the FastAPI app
from app import app as application

# Vercel will call this function
ASGI_APP = application
