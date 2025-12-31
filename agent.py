"""
Book RAG Agent using Gemini 2.0 Flash - Simplified approach without function calling.

This module creates an intelligent agent that answers questions about robotics
content from the book using RAG (Retrieval Augmented Generation).
"""

import os
import time
from typing import Dict, Any, Optional, List
import google.generativeai as genai

from tools import RAGTool
from logger import get_logger

logger = get_logger(__name__)


class AgentError(Exception):
    """Custom exception for agent errors."""
    pass


class BookRAGAgent:
    """
    Intelligent agent for answering questions about robotics book content.

    Uses Gemini 2.0 Flash with direct RAG integration (no function calling).
    """

    def __init__(self, model: str = None):
        """
        Initialize the Book RAG Agent.

        Args:
            model: Model name to use (default: from env GEMINI_MODEL_NAME or gemini-2.5-flash)
        """
        # Use model from environment variable or default
        self.model = model or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self._model_instance = None
        self._rag_tool = None

        # Configure Gemini
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not configured")
            raise AgentError("GEMINI_API_KEY environment variable is required")

        try:
            genai.configure(api_key=gemini_api_key)
            logger.info(f"Agent initialized", model=self.model)
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {str(e)}")
            raise AgentError(f"Failed to configure Gemini API: {str(e)}")

    @property
    def model_instance(self):
        """Lazy-load the model instance."""
        if self._model_instance is None:
            try:
                self._model_instance = genai.GenerativeModel(model_name=self.model)
                logger.info(f"Gemini model loaded: {self.model}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise AgentError(f"Failed to load Gemini model: {str(e)}")
        return self._model_instance

    @property
    def rag_tool(self) -> RAGTool:
        """Lazy-load the RAG tool."""
        if self._rag_tool is None:
            try:
                self._rag_tool = RAGTool()
                logger.info("RAG tool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RAG tool: {str(e)}")
                raise AgentError(f"Failed to initialize RAG tool: {str(e)}")
        return self._rag_tool

    def run(
        self,
        message: str,
        mode: str = "rag",
        selected_text: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run the agent to answer a question.

        Args:
            message: User's question (required, non-empty)
            mode: Query mode ('rag' or 'selected')
            selected_text: Selected text for context (if mode='selected')
            session_context: Conversation history and context
            top_k: Number of results to retrieve (1-20)

        Returns:
            Dict containing response, sources, success status, and error (if any)
        """
        start_time = time.time()

        # Input validation
        if not message or not isinstance(message, str):
            logger.warning("Invalid message received", message_type=type(message).__name__)
            return self._error_response("Invalid message: must be a non-empty string")

        message = message.strip()
        if not message:
            return self._error_response("Invalid message: cannot be empty")

        # Validate mode
        if mode not in ("rag", "selected"):
            logger.warning(f"Invalid mode: {mode}")
            mode = "rag"  # Default to rag mode

        # Validate top_k
        top_k = max(1, min(20, top_k))

        logger.info(
            "Processing query",
            message_preview=message[:50] + "..." if len(message) > 50 else message,
            mode=mode,
            top_k=top_k,
            has_selected_text=bool(selected_text)
        )

        try:
            # Step 1: Execute RAG search to get relevant context
            rag_result = self._execute_rag_query(message, mode, top_k, selected_text)

            if not rag_result['success']:
                return self._handle_rag_failure(rag_result, message)

            sources = rag_result.get('sources') or []
            logger.info(f"Retrieved {len(sources)} sources from RAG")

            # Step 2: Handle case of no sources found - EXPLICIT response
            if not sources:
                logger.warning(
                    "[RAG] No relevant chunks found",
                    query_preview=message[:50] if message else "empty",
                    collection=self.rag_tool.collection_name
                )
                return self._create_response(
                    "**No relevant information was found in the book.**\n\n"
                    f"Your query: \"{message[:100]}{'...' if len(message) > 100 else ''}\"\n\n"
                    "The knowledge base was searched but returned no matching content. "
                    "Please try:\n"
                    "- Rephrasing your question\n"
                    "- Using different keywords\n"
                    "- Asking about a topic covered in the robotics book",
                    sources=[],
                    success=True  # Query succeeded, just no results
                )

            # Step 3: Build context and generate response
            context = self._build_context(sources)
            history = self._format_history(session_context)
            prompt = self._build_prompt(message, context, history)

            # Step 4: Generate response with Gemini
            response_text = self._generate_response(prompt)

            if not response_text:
                return self._error_response("Empty response from model")

            duration = (time.time() - start_time) * 1000
            logger.info(
                "Query completed successfully",
                response_length=len(response_text),
                sources_count=len(sources),
                duration_ms=round(duration, 2)
            )

            return self._create_response(response_text, sources, success=True)

        except AgentError as e:
            logger.error(f"Agent error: {str(e)}")
            return self._error_response(str(e))

        except Exception as e:
            logger.error(f"Unexpected error in agent: {str(e)}", exc_info=True)
            return self._error_response(f"An unexpected error occurred: {str(e)}")

    def _execute_rag_query(
        self,
        message: str,
        mode: str,
        top_k: int,
        selected_text: Optional[str]
    ) -> Dict[str, Any]:
        """Execute RAG query with error handling."""
        try:
            return self.rag_tool.rag_query(
                query=message,
                mode=mode,
                top_k=top_k,
                selected_text=selected_text
            )
        except Exception as e:
            logger.error(f"RAG query execution failed: {str(e)}")
            return {
                'success': False,
                'sources': [],
                'error': str(e)
            }

    def _handle_rag_failure(self, rag_result: Dict[str, Any], message: str) -> Dict[str, Any]:
        """
        Handle RAG query failure with EXPLICIT error messages.

        NO GENERIC FALLBACKS - always expose the real error.
        """
        error_msg = rag_result.get('error') or 'Unknown RAG error'

        # Log full error details
        logger.error(
            f"[RAG FAILURE] Query failed",
            error=error_msg,
            query_preview=message[:50] if message else "empty"
        )

        # Check for quota/rate limit errors - still expose the real error
        if 'quota' in error_msg.lower() or '429' in str(error_msg):
            logger.error(f"[RAG FAILURE] API quota exceeded: {error_msg}")
            return self._create_response(
                f"**RAG Query Failed:** API rate limit exceeded.\n\n"
                f"**Error Details:** {error_msg}\n\n"
                f"Please wait a few minutes and try again.",
                sources=[],
                success=False,
                error=f"API quota exceeded: {error_msg}"
            )

        # Check for authentication errors (403)
        if '403' in error_msg or 'forbidden' in error_msg.lower() or 'unauthorized' in error_msg.lower():
            logger.error(f"[RAG FAILURE] Authentication error: {error_msg}")
            return self._create_response(
                f"**RAG Query Failed:** Vector database authentication error.\n\n"
                f"**Error Details:** {error_msg}\n\n"
                f"This is a configuration issue. Please check the Qdrant API key.",
                sources=[],
                success=False,
                error=f"Qdrant authentication failed: {error_msg}"
            )

        # Check for connection errors
        if 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
            logger.error(f"[RAG FAILURE] Connection error: {error_msg}")
            return self._create_response(
                f"**RAG Query Failed:** Cannot connect to vector database.\n\n"
                f"**Error Details:** {error_msg}\n\n"
                f"Please verify the Qdrant URL and network connectivity.",
                sources=[],
                success=False,
                error=f"Qdrant connection failed: {error_msg}"
            )

        # Check for embedding errors
        if 'embedding' in error_msg.lower():
            logger.error(f"[RAG FAILURE] Embedding error: {error_msg}")
            return self._create_response(
                f"**RAG Query Failed:** Could not generate query embedding.\n\n"
                f"**Error Details:** {error_msg}\n\n"
                f"Please check the Gemini API key configuration.",
                sources=[],
                success=False,
                error=f"Embedding generation failed: {error_msg}"
            )

        # All other failures - NEVER hide the real error
        logger.error(f"[RAG FAILURE] Unhandled error type: {error_msg}")
        return self._create_response(
            f"**RAG Query Failed:** {error_msg}\n\n"
            f"The knowledge base could not be queried. This error has been logged.",
            sources=[],
            success=False,
            error=error_msg
        )

    # NOTE: Fallback response generation REMOVED
    # RAG-first principle: Never answer without attempting knowledge base retrieval
    # All failures are now explicitly reported with real error messages

    def _build_context(self, sources: List[Dict[str, Any]]) -> str:
        """Build context string from sources."""
        if not sources:
            return ""

        context_parts = []
        for idx, source in enumerate(sources[:5], 1):  # Use top 5 sources
            if not source or not isinstance(source, dict):
                continue

            source_text = source.get('text', '') or ''
            book_id = source.get('book_id', 'unknown') or 'unknown'
            score = source.get('score', 0) or 0

            if source_text:
                context_parts.append(
                    f"[Source {idx} - {book_id}, Score: {score:.2f}]\n{source_text}"
                )

        return "\n\n".join(context_parts)

    def _format_history(self, session_context: Optional[Dict[str, Any]]) -> str:
        """Format conversation history from session context."""
        if not session_context or not isinstance(session_context, dict):
            return ""

        history = session_context.get('history')
        if not history or not isinstance(history, list):
            return ""

        # Take last 3 turns
        recent_history = history[-3:]
        history_parts = []

        for turn in recent_history:
            if not turn or not isinstance(turn, dict):
                continue

            query = turn.get('query', '') or ''
            response = turn.get('response', '') or ''

            if query:
                history_parts.append(f"User: {query}")
            if response:
                # Truncate long responses
                response_preview = response[:500] + "..." if len(response) > 500 else response
                history_parts.append(f"Assistant: {response_preview}")

        if history_parts:
            return "\n\nPrevious conversation:\n" + "\n".join(history_parts)
        return ""

    def _build_prompt(self, message: str, context: str, history: str) -> str:
        """Build the complete prompt for Gemini."""
        system_instructions = """You are an expert robotics and physical AI educator helping students understand complex technical concepts.

YOUR RESPONSE PHILOSOPHY:
- **Explanation-First**: When users ask "What is X?", "Explain Y", or "Tell me about Z", provide COMPREHENSIVE, DETAILED explanations
- **Synthesize Multiple Sources**: Combine information from all provided sources to give a complete picture
- **Depth Over Brevity**: Prefer thorough explanations with examples, definitions, and context over short summaries
- **Assume Learning Intent**: Users are here to learn deeply, not get one-line definitions

GROUNDING RULES (MUST FOLLOW):
1. Use ONLY information from the provided context sources
2. DO NOT add information not present in the sources
3. DO NOT hallucinate facts, examples, or technical details
4. If sources lack information on a specific aspect, acknowledge it briefly but still explain what IS available

EXPLANATION GUIDELINES:
- For "What is..." questions: Define thoroughly, explain purpose/function, provide context from sources
- For "Explain..." questions: Give detailed multi-paragraph explanations covering all aspects mentioned in sources
- For concept questions: Break down complexity, explain components, show relationships
- For technical questions: Provide technical depth while remaining clear

FORMATTING:
- Use markdown for structure (headers, bullet points, bold for emphasis)
- Organize longer explanations with clear sections
- Always cite sources at the end: [Source 1], [Source 2], etc.

Your goal: Transform retrieved context into educational, comprehensive explanations that help users deeply understand the topic."""

        return f"""{system_instructions}

CONTEXT FROM BOOK:
{context}
{history}

USER QUESTION: {message}

Provide a detailed, thorough explanation using ALL relevant information from the sources above. Cite your sources."""

    def _generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using Gemini with error handling."""
        try:
            logger.debug("Generating response with Gemini")
            start_time = time.time()

            response = self.model_instance.generate_content(prompt)

            duration = (time.time() - start_time) * 1000
            logger.llm_call(
                model=self.model,
                duration_ms=round(duration, 2)
            )

            if not response:
                logger.warning("Empty response object from Gemini")
                return None

            # Safely extract text
            response_text = getattr(response, 'text', None)
            if not response_text:
                logger.warning("Response has no text attribute or text is empty")
                return None

            logger.debug(f"Generated response", length=len(response_text))
            return response_text

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            return None

    def _create_response(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        success: bool = True,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized response dictionary."""
        return {
            'response': response,
            'sources': sources or [],
            'success': success,
            'error': error
        }

    def _error_response(self, error: str) -> Dict[str, Any]:
        """Create an error response."""
        return self._create_response(
            f"I encountered an error while processing your question: {error}",
            sources=[],
            success=False,
            error=error
        )


# Global agent instance (singleton pattern)
_agent_instance: Optional[BookRAGAgent] = None


def get_agent() -> BookRAGAgent:
    """
    Get or create the global Book RAG Agent instance.

    Returns:
        BookRAGAgent instance

    Raises:
        AgentError: If agent initialization fails
    """
    global _agent_instance

    if _agent_instance is None:
        logger.info("Initializing BookRAGAgent singleton")
        try:
            _agent_instance = BookRAGAgent()
            logger.info("BookRAGAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BookRAGAgent: {str(e)}")
            raise AgentError(f"Agent initialization failed: {str(e)}")

    return _agent_instance


def reset_agent() -> None:
    """Reset the global agent instance (useful for testing or reloading config)."""
    global _agent_instance
    _agent_instance = None
    logger.info("Agent instance reset")
