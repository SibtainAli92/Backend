"""
RAG query tool for retrieving relevant information from book content.

This tool enables the agent to search the vector database for relevant chunks
based on user queries, supporting both normal RAG mode and selected text mode.
"""

from typing import List, Dict, Any, Optional
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
import google.generativeai as genai
import cohere

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import get_qdrant_client, QdrantConnectionError, QdrantEndpointError, QdrantAuthError
from models import Source
from logger import get_logger

logger = get_logger(__name__)


class RAGToolError(Exception):
    """Custom exception for RAG tool errors."""
    pass


class EmbeddingError(RAGToolError):
    """Error during embedding generation."""
    pass


class VectorSearchError(RAGToolError):
    """Error during vector search."""
    pass


class RAGTool:
    """Handles RAG queries against the vector database with robust error handling."""

    def __init__(self, collection_name: str = None):
        """
        Initialize RAG tool.

        Args:
            collection_name: Name of the Qdrant collection (defaults to env var)
        """
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "book_chunks")
        self._qdrant_client: Optional[QdrantClient] = None
        self._cohere_client: Optional[cohere.Client] = None

        # Initialize Cohere for embeddings
        self._configure_cohere()

        # Embedding model configuration - using Cohere to match Qdrant collection
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")
        self.embedding_dimensions = 1024  # Cohere embed-english-v3.0 dimensions

        logger.info(
            "RAGTool initialized",
            collection=self.collection_name,
            embedding_model=self.embedding_model_name
        )

    def _configure_cohere(self) -> None:
        """Configure Cohere API for embeddings."""
        if self._cohere_client is not None:
            return

        cohere_api_key = os.getenv("COHERE_API_KEY", "").strip()
        if not cohere_api_key:
            logger.error("COHERE_API_KEY not configured")
            raise EmbeddingError("COHERE_API_KEY environment variable is required for embeddings")

        try:
            self._cohere_client = cohere.Client(api_key=cohere_api_key)
            logger.info("Cohere configured for embeddings")
        except Exception as e:
            logger.error(f"Failed to configure Cohere: {str(e)}")
            raise EmbeddingError(f"Failed to configure Cohere: {str(e)}")

    @property
    def qdrant_client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._qdrant_client is None:
            try:
                self._qdrant_client = get_qdrant_client()
            except Exception as e:
                logger.error(f"Failed to get Qdrant client: {str(e)}")
                raise VectorSearchError(f"Qdrant connection failed: {str(e)}")
        return self._qdrant_client

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for text using Cohere.

        Args:
            text: Text to embed (will be truncated if too long)

        Returns:
            List of floats representing the embedding vector, or None if failed
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for embedding", text_type=type(text).__name__)
            return None

        # Truncate very long text to avoid API limits
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated for embedding", original_length=len(text), max_chars=max_chars)

        try:
            start_time = time.time()
            logger.debug(f"Generating embedding", text_length=len(text))

            # Use Cohere embeddings
            response = self._cohere_client.embed(
                texts=[text],
                model=self.embedding_model_name,
                input_type="search_query"
            )

            duration = (time.time() - start_time) * 1000

            if not response.embeddings or len(response.embeddings) == 0:
                logger.error("Embedding result is empty")
                return None

            embedding = response.embeddings[0]

            logger.debug(
                "Embedding generated",
                vector_size=len(embedding),
                duration_ms=round(duration, 2)
            )
            return embedding

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Embedding generation failed: {error_msg}", exc_info=True)

            # Check for quota errors
            if "quota" in error_msg.lower() or "429" in error_msg:
                logger.warning("Embedding quota exceeded")
                return None

            return None

    def rag_query(
        self,
        query: str,
        mode: str = "rag",
        top_k: int = 5,
        selected_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant information from book content or selected text.

        Args:
            query: The search query (required, non-empty string)
            mode: Operation mode ("rag" or "selected")
            top_k: Number of results to retrieve (1-20)
            selected_text: Text to search against in selected mode

        Returns:
            Dictionary with chunks, sources, success flag, and optional error
        """
        # Input validation
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided", query_type=type(query).__name__)
            return self._error_response("Invalid query: must be a non-empty string")

        query = query.strip()
        if not query:
            return self._error_response("Invalid query: cannot be empty or whitespace only")

        # Validate mode
        if mode not in ("rag", "selected"):
            logger.warning(f"Invalid mode: {mode}")
            return self._error_response(f"Invalid mode '{mode}': must be 'rag' or 'selected'")

        # Validate top_k
        top_k = max(1, min(20, top_k))  # Clamp to valid range

        logger.info(
            "RAG query started",
            mode=mode,
            query_length=len(query),
            top_k=top_k,
            has_selected_text=bool(selected_text)
        )

        try:
            if mode == "selected":
                if not selected_text or not isinstance(selected_text, str):
                    return self._error_response("Selected mode requires non-empty selected_text")
                return self._query_selected_text(query, selected_text.strip(), top_k)
            else:
                return self._query_book_content(query, top_k)

        except QdrantEndpointError as e:
            error_msg = str(e)
            logger.error(f"Invalid Qdrant cluster endpoint: {error_msg}")
            return self._error_response(f"Invalid Qdrant cluster endpoint: {error_msg}")

        except QdrantAuthError as e:
            error_msg = str(e)
            logger.error(f"Invalid Qdrant cluster API key: {error_msg}")
            return self._error_response(f"Invalid Qdrant cluster API key: {error_msg}")

        except QdrantConnectionError as e:
            error_msg = str(e)
            logger.error(f"Qdrant connection error in RAG query: {error_msg}")
            return self._error_response(f"Vector database connection failed: {error_msg}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"RAG query failed: {error_msg}", exc_info=True)
            return self._error_response(f"Query failed: {error_msg}")

    def _error_response(self, error: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "chunks": [],
            "sources": [],
            "success": False,
            "error": error
        }

    def _success_response(self, chunks: List[Dict], sources: List[Dict]) -> Dict[str, Any]:
        """Create a standardized success response."""
        return {
            "chunks": chunks,
            "sources": sources,
            "success": True,
            "error": None
        }

    def _query_book_content(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        Query the full book content using vector similarity search.

        ALWAYS attempts retrieval - never skips or returns early without trying.

        Args:
            query: The search query
            top_k: Number of results to retrieve

        Returns:
            Dictionary with retrieved chunks and sources
        """
        start_time = time.time()

        # STEP 1: Generate query embedding
        logger.info(f"[RAG PIPELINE] Step 1: Generating embedding for query")
        logger.info(f"[RAG PIPELINE] Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        query_embedding = self.generate_embedding(query)

        if query_embedding is None:
            logger.error("[RAG PIPELINE] FAILED at Step 1: Embedding generation returned None")
            return self._error_response(
                "Embedding generation failed. Check Gemini API key configuration and quota."
            )

        logger.info(f"[RAG PIPELINE] Step 1 SUCCESS: Embedding generated (dim={len(query_embedding)})")

        # STEP 2: Search Qdrant
        logger.info(f"[RAG PIPELINE] Step 2: Searching Qdrant collection '{self.collection_name}'")
        logger.info(f"[RAG PIPELINE] Parameters: top_k={top_k}, score_threshold=0.3")

        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=0.3  # Lower threshold for better recall
            )

            duration = (time.time() - start_time) * 1000
            results_count = len(search_results) if search_results else 0

            logger.info(f"[RAG PIPELINE] Step 2 SUCCESS: Retrieved {results_count} results in {duration:.2f}ms")
            logger.qdrant_query(
                collection=self.collection_name,
                results_count=results_count,
                duration_ms=duration
            )

        except UnexpectedResponse as e:
            error_str = str(e)
            logger.error(f"[RAG PIPELINE] FAILED at Step 2: Qdrant UnexpectedResponse")
            logger.error(f"[RAG PIPELINE] Error details: {error_str}")

            # Check for specific error types
            if '403' in error_str or 'forbidden' in error_str.lower():
                return self._error_response(
                    f"Qdrant authentication failed (403 Forbidden). "
                    f"Please verify QDRANT_API_KEY is correct. Error: {error_str}"
                )
            elif '404' in error_str:
                return self._error_response(
                    f"Qdrant collection '{self.collection_name}' not found. "
                    f"Please run the embedding pipeline first. Error: {error_str}"
                )
            else:
                return self._error_response(f"Qdrant search failed: {error_str}")

        except Exception as e:
            logger.error(f"[RAG PIPELINE] FAILED at Step 2: {type(e).__name__}: {str(e)}", exc_info=True)
            return self._error_response(f"Vector search failed: {type(e).__name__}: {str(e)}")

        # STEP 3: Process results
        logger.info(f"[RAG PIPELINE] Step 3: Processing search results")

        if not search_results:
            logger.warning("[RAG PIPELINE] No results found - search returned empty")
            return self._success_response([], [])

        # Process results with defensive coding
        chunks, sources = self._process_search_results(search_results)

        total_duration = (time.time() - start_time) * 1000
        logger.info(
            f"[RAG PIPELINE] COMPLETE: Retrieved {len(chunks)} chunks, {len(sources)} sources",
            results_count=len(chunks),
            duration_ms=round(total_duration, 2)
        )

        # Log top result scores for debugging
        if sources:
            top_scores = [s.get('score', 0) for s in sources[:3]]
            logger.info(f"[RAG PIPELINE] Top 3 scores: {top_scores}")

        return self._success_response(chunks, sources)

    def _query_selected_text(
        self,
        query: str,
        selected_text: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Query against user-provided selected text.

        Args:
            query: The search query
            selected_text: User-selected text to search within
            top_k: Number of results to retrieve

        Returns:
            Dictionary with relevant chunks and sources
        """
        start_time = time.time()

        # Generate embedding for selected text
        selected_embedding = self.generate_embedding(selected_text)
        if selected_embedding is None:
            return self._error_response(
                "Failed to generate embedding for selected text. "
                "Check Gemini API key and quota."
            )

        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return self._error_response(
                "Failed to generate embedding for query. "
                "Check Gemini API key and quota."
            )

        try:
            # Search for chunks similar to selected text
            selected_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=selected_embedding,
                limit=top_k * 2,  # Get more results to filter
                with_payload=True,
                score_threshold=0.5  # Higher threshold for selected text
            )

            # Also search using the query for additional context
            query_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=0.3
            )

        except UnexpectedResponse as e:
            error_str = str(e)
            logger.error(f"Qdrant selected text search failed: {error_str}")
            return self._error_response(f"Vector search failed: {error_str}")

        except Exception as e:
            logger.error(f"Selected text search failed: {str(e)}", exc_info=True)
            return self._error_response(f"Vector search failed: {str(e)}")

        # Combine and deduplicate results safely
        all_results = {}

        for result in (selected_results or []):
            if result is None:
                continue
            point_id = getattr(result, 'id', None) or id(result)
            score = getattr(result, 'score', 0) or 0
            existing_score = getattr(all_results.get(point_id), 'score', 0) if point_id in all_results else 0
            if point_id not in all_results or score > existing_score:
                all_results[point_id] = result

        for result in (query_results or []):
            if result is None:
                continue
            point_id = getattr(result, 'id', None) or id(result)
            score = getattr(result, 'score', 0) or 0
            existing_score = getattr(all_results.get(point_id), 'score', 0) if point_id in all_results else 0
            if point_id not in all_results or score > existing_score:
                all_results[point_id] = result

        # Sort by score and take top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: getattr(x, 'score', 0) or 0,
            reverse=True
        )[:top_k]

        # Process results
        chunks, sources = self._process_search_results(sorted_results)

        # Add selected text as primary source
        selected_preview = selected_text[:200] + "..." if len(selected_text) > 200 else selected_text
        sources.insert(0, Source(
            text=selected_preview,
            book_id="selected",
            chapter_id=None,
            section_id=None,
            source_url=None,
            score=1.0
        ).dict())

        duration = (time.time() - start_time) * 1000
        logger.info(
            "Selected text query completed",
            results_count=len(chunks),
            duration_ms=round(duration, 2)
        )

        return self._success_response(chunks, sources)

    def _process_search_results(self, results: List) -> tuple:
        """
        Process search results into chunks and sources with defensive coding.

        Args:
            results: List of search results from Qdrant

        Returns:
            Tuple of (chunks, sources)
        """
        chunks = []
        sources = []

        if not results:
            return chunks, sources

        for result in results:
            if result is None:
                continue

            # Safely extract payload
            payload = getattr(result, 'payload', None)
            if payload is None:
                payload = {}

            # Safely extract score
            score = getattr(result, 'score', 0)
            if score is None:
                score = 0.0

            # Extract text with fallback
            text = payload.get("text", "") or ""

            # Build chunk
            chunk = {
                "text": text,
                "score": float(score),
                "metadata": {
                    "book_id": payload.get("book_id"),
                    "chapter_id": payload.get("chapter_id"),
                    "section_id": payload.get("section_id"),
                    "source_url": payload.get("source_url")
                }
            }
            chunks.append(chunk)

            # Build source with preview
            text_preview = text[:200] + "..." if len(text) > 200 else text

            try:
                source = Source(
                    text=text_preview,
                    book_id=payload.get("book_id", "") or "unknown",
                    chapter_id=payload.get("chapter_id"),
                    section_id=payload.get("section_id"),
                    source_url=payload.get("source_url"),
                    score=float(score)
                )
                sources.append(source.dict())
            except Exception as e:
                logger.warning(f"Failed to create Source object: {str(e)}")
                # Create minimal source dict as fallback
                sources.append({
                    "text": text_preview,
                    "book_id": "unknown",
                    "score": float(score)
                })

        return chunks, sources


# Function tool schema for OpenAI Agents SDK
RAG_QUERY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_query",
        "description": "Retrieve relevant information from the robotics book content or user-selected text. Use this tool to find relevant passages before answering user questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to find relevant information for"
                },
                "mode": {
                    "type": "string",
                    "enum": ["rag", "selected"],
                    "description": "Operation mode: 'rag' for searching the full book content, 'selected' for searching within user-provided selected text"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of relevant passages to retrieve (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query", "mode"]
        }
    }
}
