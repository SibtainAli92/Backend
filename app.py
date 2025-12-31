"""
FastAPI application for Book RAG Agent.

Serverless-compatible version - handles startup gracefully without crashes.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional

# Only load .env file in development (not on Vercel)
# Vercel sets environment variables directly
if os.getenv("VERCEL") != "1":
    from dotenv import load_dotenv
    load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import get_agent, AgentError, reset_agent
from models import ChatRequest, ChatResponse, Session
from connection import ensure_qdrant_collection, health_check as connection_health_check, QdrantConnectionError, validate_qdrant_connection
from config import validate_config_on_startup, get_config
from logger import get_logger

logger = get_logger(__name__)

# Check if running in serverless environment
IS_SERVERLESS = os.getenv("VERCEL") == "1" or os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None

# In-memory session storage (use Redis/database for production)
sessions: Dict[str, Session] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan with graceful error handling for serverless.

    In serverless mode, we try to initialize but don't fail hard.
    This allows the function to start and return errors gracefully.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting Book RAG Agent API (Serverless: {IS_SERVERLESS})")
    logger.info("=" * 60)

    # Print env vars to prove they're loaded
    qdrant_url = os.getenv('QDRANT_URL', 'NOT SET')
    api_key = os.getenv('QDRANT_API_KEY', '')
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'book_chunks')

    logger.info(f"[ENV CHECK] QDRANT_URL = {qdrant_url}")
    logger.info(f"[ENV CHECK] QDRANT_API_KEY = {api_key[:4]}...{api_key[-4:] if api_key else 'EMPTY'} (len={len(api_key)})")
    logger.info(f"[ENV CHECK] QDRANT_COLLECTION_NAME = {collection_name}")

    # Store startup status for health checks
    app.state.qdrant_status = {"healthy": False, "error": None, "vector_count": 0}

    try:
        # Try to validate config, but don't fail in serverless
        try:
            config = validate_config_on_startup()
            logger.info("Configuration validated")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            app.state.qdrant_status = {
                "healthy": False,
                "error": f"Configuration error: {str(e)}",
                "vector_count": 0
            }
            yield
            return

        # STEP 1: Test raw Qdrant connection (with timeout for serverless)
        logger.info("[STARTUP] Step 1: Testing Qdrant connection...")
        try:
            qdrant_result = validate_qdrant_connection()

            if not qdrant_result.get("success"):
                error_msg = qdrant_result.get('exception_message', 'Unknown error')
                logger.error(f"[STARTUP] CRITICAL: Qdrant connection FAILED!")
                logger.error(f"[STARTUP] Error: {error_msg}")
                logger.error(f"[STARTUP] Full result: {qdrant_result}")

                # Store error for health endpoint
                app.state.qdrant_status = {
                    "healthy": False,
                    "error": error_msg,
                    "vector_count": 0,
                    "details": qdrant_result
                }

                # In serverless, we allow degraded mode but log it
                logger.warning("[STARTUP] Starting in DEGRADED MODE - RAG queries will fail!")
            else:
                logger.info(f"[STARTUP] Qdrant connected! Collections: {qdrant_result.get('collections')}")

                # STEP 2: Validate collection exists and has data
                logger.info(f"[STARTUP] Step 2: Validating collection '{collection_name}'...")
                try:
                    collection_info = ensure_qdrant_collection(
                        collection_name=collection_name,
                        vector_size=768
                    )
                    vector_count = collection_info.get('points_count', 0)
                    logger.info(f"[STARTUP] SUCCESS! Collection '{collection_name}' ready")
                    logger.info(f"[STARTUP] Vector count: {vector_count}")

                    app.state.qdrant_status = {
                        "healthy": True,
                        "error": None,
                        "vector_count": vector_count,
                        "collection": collection_name
                    }

                except QdrantConnectionError as e:
                    logger.error(f"[STARTUP] Collection validation FAILED: {e}")
                    app.state.qdrant_status = {
                        "healthy": False,
                        "error": str(e),
                        "vector_count": 0
                    }
                    logger.warning("[STARTUP] Starting in DEGRADED MODE - RAG queries will fail!")

                except Exception as e:
                    logger.error(f"[STARTUP] Unexpected collection error: {type(e).__name__}: {e}")
                    app.state.qdrant_status = {
                        "healthy": False,
                        "error": f"{type(e).__name__}: {e}",
                        "vector_count": 0
                    }
                    logger.warning("[STARTUP] Starting in DEGRADED MODE - RAG queries will fail!")

        except Exception as e:
            logger.error(f"[STARTUP] Qdrant connection attempt failed: {type(e).__name__}: {e}")
            app.state.qdrant_status = {
                "healthy": False,
                "error": f"Connection failed: {type(e).__name__}: {e}",
                "vector_count": 0
            }

        logger.info("=" * 60)
        logger.info(f"[STARTUP] Final status: Qdrant healthy={app.state.qdrant_status['healthy']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"[STARTUP] Fatal error: {type(e).__name__}: {e}", exc_info=True)
        app.state.qdrant_status = {
            "healthy": False,
            "error": f"Startup failed: {type(e).__name__}: {e}",
            "vector_count": 0
        }
        # In serverless, we don't crash - we just note the error
        if not IS_SERVERLESS:
            raise

    yield

    # Shutdown
    logger.info("Shutting down")
    try:
        reset_agent()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    sessions.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Book RAG Agent API",
    description="API for querying robotics book content using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models with validation
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    services: Optional[Dict] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "").lower() == "true" else None,
            "status_code": 500
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Log request
    logger.info(
        f"Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path
    )

    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000

        logger.request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration,
            request_id=request_id
        )

        return response

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(
            f"Request failed: {str(e)}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration
        )
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Book RAG Agent API",
        "version": "1.0.0",
        "status": "running",
        "serverless": IS_SERVERLESS,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(request: Request):
    """
    Health check - returns FULL diagnostic info.

    Includes:
    - Startup status from app.state
    - Live connection test
    - Vector count
    - All errors exposed
    """
    try:
        # Get startup status
        startup_status = getattr(request.app.state, 'qdrant_status', {
            "healthy": False,
            "error": "Status not initialized",
            "vector_count": 0
        })

        # Run live health check
        services = connection_health_check()
        all_healthy = all(s.get("healthy", False) for s in services.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "serverless": IS_SERVERLESS,
            "startup_validation": startup_status,
            "live_check": services,
            "env": {
                "QDRANT_URL": os.getenv("QDRANT_URL", "NOT SET"),
                "QDRANT_API_KEY_LENGTH": len(os.getenv("QDRANT_API_KEY", "")),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "book_chunks"),
                "GEMINI_API_KEY_LENGTH": len(os.getenv("GEMINI_API_KEY", "")),
            },
            "rag_ready": startup_status.get("healthy", False) and startup_status.get("vector_count", 0) > 0
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "serverless": IS_SERVERLESS,
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "rag_ready": False
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for querying the Book RAG Agent.

    Args:
        request: Chat request with message, optional selected_text, and session_id

    Returns:
        ChatResponse with agent's answer and sources

    Raises:
        HTTPException: On validation or processing errors
    """
    start_time = time.time()

    # Validate message
    if not request.message or not request.message.strip():
        logger.warning("Empty message received")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )

    message = request.message.strip()

    # Log request
    logger.info(
        "Chat request received",
        message_length=len(message),
        has_selected_text=bool(request.selected_text),
        session_id=request.session_id
    )

    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in sessions:
            sessions[session_id] = Session(id=session_id)
            logger.info(f"New session created", session_id=session_id)

        session = sessions[session_id]

        # Check if session is expired
        if session.is_expired():
            logger.info(f"Session expired, creating new", old_session_id=session_id)
            sessions[session_id] = Session(id=session_id)
            session = sessions[session_id]

        # Determine mode
        mode = "selected" if request.selected_text else "rag"

        # Get agent (may raise AgentError)
        try:
            agent = get_agent()
        except AgentError as e:
            logger.error(f"Failed to get agent: {str(e)}")
            return ChatResponse(
                response="The AI agent is currently unavailable. Please try again later.",
                session_id=session_id,
                sources=[],
                error=str(e)
            )

        # Execute agent
        result = agent.run(
            message=message,
            mode=mode,
            selected_text=request.selected_text,
            session_context={
                "history": [turn.dict() for turn in session.history]
            },
            top_k=request.top_k
        )

        # Handle agent result
        response_text = result.get("response", "")
        sources = result.get("sources", []) or []
        success = result.get("success", False)
        error = result.get("error")

        # If successful, add to session history
        if success and response_text:
            session.add_turn(
                query=message,
                response=response_text,
                sources=sources
            )

        duration = (time.time() - start_time) * 1000
        logger.info(
            "Chat request completed",
            session_id=session_id,
            success=success,
            sources_count=len(sources),
            duration_ms=round(duration, 2)
        )

        return ChatResponse(
            response=response_text or "I couldn't generate a response. Please try again.",
            session_id=session_id,
            sources=sources,
            error=error
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)

        # Return error response instead of raising 500
        return ChatResponse(
            response=f"An error occurred while processing your request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            sources=[],
            error=str(e)
        )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get session information and conversation history.

    Args:
        session_id: Session identifier

    Returns:
        Session information with conversation history

    Raises:
        HTTPException: If session not found
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty"
        )

    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    session = sessions[session_id]

    return {
        "session_id": session.id,
        "created_at": session.created_at.isoformat(),
        "last_active": session.last_active.isoformat(),
        "history": [turn.dict() for turn in session.history],
        "message_count": len(session.history),
        "is_expired": session.is_expired()
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clear its history.

    Args:
        session_id: Session identifier

    Returns:
        Confirmation message

    Raises:
        HTTPException: If session not found
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty"
        )

    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    del sessions[session_id]
    logger.info(f"Session deleted", session_id=session_id)

    return {
        "message": "Session deleted successfully",
        "session_id": session_id
    }


@app.post("/sessions/cleanup")
async def cleanup_expired_sessions():
    """
    Clean up expired sessions.

    Returns:
        Number of sessions cleaned up
    """
    expired_ids = [
        session_id
        for session_id, session in sessions.items()
        if session.is_expired()
    ]

    for session_id in expired_ids:
        del sessions[session_id]

    logger.info(f"Cleaned up expired sessions", count=len(expired_ids))

    return {
        "message": "Expired sessions cleaned up",
        "cleaned_count": len(expired_ids),
        "remaining_sessions": len(sessions)
    }


@app.get("/sessions/stats")
async def get_session_stats():
    """
    Get session statistics.

    Returns:
        Statistics about active sessions
    """
    active_count = sum(1 for s in sessions.values() if not s.is_expired())
    expired_count = len(sessions) - active_count
    total_messages = sum(len(s.history) for s in sessions.values())

    return {
        "total_sessions": len(sessions),
        "active_sessions": active_count,
        "expired_sessions": expired_count,
        "total_messages": total_messages
    }


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )
