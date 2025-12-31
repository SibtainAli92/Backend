"""
Connection utilities for Qdrant, Gemini, and OpenAI.

SIMPLIFIED VERSION - Returns raw SDK exceptions only.
"""

import os
import time
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load dotenv at module level FIRST
load_dotenv(override=True)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import google.generativeai as genai
from openai import OpenAI

from logger import get_logger

logger = get_logger(__name__)


class QdrantConnectionError(Exception):
    """Qdrant connection error - wraps raw SDK exceptions."""
    pass


class QdrantEndpointError(QdrantConnectionError):
    """Invalid Qdrant endpoint."""
    pass


class QdrantAuthError(QdrantConnectionError):
    """Invalid Qdrant API key."""
    pass


class GeminiConnectionError(Exception):
    """Gemini connection error."""
    pass


class Connections:
    """Manages connections to external services."""

    def __init__(self):
        self._qdrant_client: Optional[QdrantClient] = None
        self._openai_client: Optional[OpenAI] = None
        self._gemini_configured: bool = False
        self._qdrant_healthy: bool = False

    def get_qdrant_client(self) -> QdrantClient:
        """
        Get or create Qdrant client.

        Creates client with url and api_key from environment.
        Does NOT cache failed clients.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client

        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()

        logger.info(f"[QDRANT] Creating client with url={url[:50]}...")
        logger.info(f"[QDRANT] API key present: {bool(api_key)}, length: {len(api_key)}")

        if not url:
            raise QdrantConnectionError("QDRANT_URL environment variable is not set")

        # Create client - let SDK exceptions propagate
        self._qdrant_client = QdrantClient(
            url=url,
            api_key=api_key if api_key else None,
            timeout=30
        )

        logger.info("[QDRANT] Client instance created")
        return self._qdrant_client

    def test_qdrant_connection(self) -> dict:
        """
        Test Qdrant connection by calling get_collections().

        Returns raw result or raw exception - NO custom messages.
        """
        try:
            client = self.get_qdrant_client()

            logger.info("[QDRANT] Testing connection with get_collections()...")
            start = time.time()
            result = client.get_collections()
            duration = (time.time() - start) * 1000

            collections = [c.name for c in result.collections]
            logger.info(f"[QDRANT] SUCCESS! Collections: {collections}, duration: {duration:.2f}ms")

            self._qdrant_healthy = True
            return {
                "success": True,
                "collections": collections,
                "duration_ms": duration
            }

        except Exception as e:
            # Return RAW exception info - no masking
            self._qdrant_healthy = False
            error_info = {
                "success": False,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "raw_error": repr(e)
            }
            logger.error(f"[QDRANT] FAILED: {error_info}")
            return error_info

    def ensure_qdrant_collection(
        self,
        collection_name: str = "book_chunks",
        vector_size: int = 768,
        distance: Distance = Distance.COSINE
    ) -> dict:
        """
        Ensure collection exists and validate it has data.

        Returns dict with collection info including vector count.
        Raises on failure - never silently fails.
        """
        client = self.get_qdrant_client()

        logger.info(f"[QDRANT] Validating collection: {collection_name}")

        # This will raise raw SDK exception on failure
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        exists = collection_name in collection_names

        if not exists:
            logger.warning(f"[QDRANT] Collection '{collection_name}' NOT FOUND!")
            logger.warning(f"[QDRANT] Available collections: {collection_names}")
            raise QdrantConnectionError(
                f"Collection '{collection_name}' does not exist. "
                f"Available collections: {collection_names}. "
                f"Please run the embedding pipeline to create and populate the collection."
            )

        # Get collection info including vector count
        info = client.get_collection(collection_name)
        points_count = info.points_count or 0

        logger.info(f"[QDRANT] Collection validated: {collection_name}")
        logger.info(f"[QDRANT] Vector count: {points_count}")

        if points_count == 0:
            logger.warning(f"[QDRANT] Collection '{collection_name}' is EMPTY (0 vectors)!")
            raise QdrantConnectionError(
                f"Collection '{collection_name}' exists but has 0 vectors. "
                f"Please run the embedding pipeline to populate the collection."
            )

        return {
            "collection_name": collection_name,
            "exists": True,
            "points_count": points_count,
            "vector_size": info.config.params.vectors.size if info.config else vector_size
        }

    def configure_gemini(self) -> bool:
        """Configure Gemini API."""
        if self._gemini_configured:
            return True

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise GeminiConnectionError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        self._gemini_configured = True
        logger.info("[GEMINI] Configured successfully")
        return True

    def get_openai_client(self) -> OpenAI:
        """Get OpenAI client."""
        if self._openai_client is not None:
            return self._openai_client

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self._openai_client = OpenAI(api_key=api_key)
        logger.info("[OPENAI] Client created")
        return self._openai_client

    def health_check(self) -> dict:
        """Health check - returns raw results."""
        status = {
            "qdrant": {"healthy": False, "details": None},
            "gemini": {"healthy": False, "details": None},
        }

        # Test Qdrant
        qdrant_result = self.test_qdrant_connection()
        status["qdrant"] = {
            "healthy": qdrant_result.get("success", False),
            "details": qdrant_result
        }

        # Test Gemini
        try:
            self.configure_gemini()
            status["gemini"] = {"healthy": True, "details": "Configured"}
        except Exception as e:
            status["gemini"] = {
                "healthy": False,
                "details": {"exception_type": type(e).__name__, "message": str(e)}
            }

        return status


# Global instance
connections = Connections()


def get_qdrant_client() -> QdrantClient:
    """Get global Qdrant client."""
    return connections.get_qdrant_client()


def get_openai_client() -> OpenAI:
    """Get global OpenAI client."""
    return connections.get_openai_client()


def configure_gemini() -> bool:
    """Configure Gemini."""
    return connections.configure_gemini()


def ensure_qdrant_collection(**kwargs) -> bool:
    """Ensure collection exists."""
    return connections.ensure_qdrant_collection(**kwargs)


def validate_qdrant_connection():
    """Test Qdrant connection."""
    return connections.test_qdrant_connection()


def health_check() -> dict:
    """Health check."""
    return connections.health_check()
