"""
Unit tests for all data models - Validation and behavior.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Chapter, Section, BookContent, Chunk,
    Query, Source, Response,
    ChatRequest, ChatResponse,
    RAGToolRequest, RAGToolResponse,
    Session, ConversationTurn
)
from tests.test_logger import test_logger


class TestBookModels:
    """Test suite for book-related models."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: models/book_models.py")

    def test_chapter_creation(self):
        """Test Chapter model creation."""
        test_logger.log_test_start("book_models.py", "Chapter", "valid_creation")

        try:
            chapter = Chapter(
                id="ch1",
                title="Introduction",
                content="Chapter content",
                section_ids=["s1", "s2"],
                book_id="book1"
            )

            assert chapter.id == "ch1"
            assert chapter.title == "Introduction"
            assert len(chapter.section_ids) == 2

            test_logger.log_test_pass("book_models.py", "Chapter", "valid_creation", "Chapter created successfully")
        except Exception as e:
            test_logger.log_test_fail("book_models.py", "Chapter", "valid_creation", str(e))
            raise

    def test_chapter_missing_required_fields(self):
        """Test Chapter fails without required fields."""
        test_logger.log_test_start("book_models.py", "Chapter", "missing_required_fields")

        try:
            with pytest.raises(ValidationError):
                Chapter(id="ch1", title="Test")

            test_logger.log_test_pass(
                "book_models.py",
                "Chapter",
                "missing_required_fields",
                "Correctly validates required fields"
            )
        except AssertionError:
            test_logger.log_test_fail("book_models.py", "Chapter", "missing_required_fields", "Failed to validate")
            raise
        except Exception as e:
            test_logger.log_test_fail("book_models.py", "Chapter", "missing_required_fields", str(e))
            raise

    def test_section_creation(self):
        """Test Section model creation."""
        test_logger.log_test_start("book_models.py", "Section", "valid_creation")

        try:
            section = Section(
                id="s1",
                title="Section 1",
                content="Section content",
                chapter_id="ch1",
                book_id="book1"
            )

            assert section.id == "s1"
            assert section.chapter_id == "ch1"

            test_logger.log_test_pass("book_models.py", "Section", "valid_creation", "Section created successfully")
        except Exception as e:
            test_logger.log_test_fail("book_models.py", "Section", "valid_creation", str(e))
            raise

    def test_book_content_with_defaults(self):
        """Test BookContent with default values."""
        test_logger.log_test_start("book_models.py", "BookContent", "default_values")

        try:
            book = BookContent(
                id="book1",
                title="Test Book",
                content="Book content"
            )

            assert book.chapters == []
            assert book.sections == []
            assert book.metadata == {}
            assert isinstance(book.last_modified, datetime)

            test_logger.log_test_pass("book_models.py", "BookContent", "default_values", "Defaults set correctly")
        except Exception as e:
            test_logger.log_test_fail("book_models.py", "BookContent", "default_values", str(e))
            raise

    def test_chunk_creation(self):
        """Test Chunk model creation."""
        test_logger.log_test_start("book_models.py", "Chunk", "valid_creation")

        try:
            chunk = Chunk(
                id="chunk1",
                text="Chunk text",
                book_id="book1",
                chapter_id="ch1",
                section_id="s1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"page": 1}
            )

            assert chunk.id == "chunk1"
            assert len(chunk.embedding) == 3
            assert chunk.metadata["page"] == 1

            test_logger.log_test_pass("book_models.py", "Chunk", "valid_creation", "Chunk created successfully")
        except Exception as e:
            test_logger.log_test_fail("book_models.py", "Chunk", "valid_creation", str(e))
            raise


class TestChatModels:
    """Test suite for chat-related models."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: models/chat_models.py")

    def test_query_valid_mode_rag(self):
        """Test Query with valid RAG mode."""
        test_logger.log_test_start("chat_models.py", "Query", "valid_rag_mode")

        try:
            query = Query(
                id="q1",
                text="What is kinematics?",
                mode="rag",
                session_id="session1"
            )

            assert query.mode == "rag"
            assert query.top_k == 5  # Default

            test_logger.log_test_pass("chat_models.py", "Query", "valid_rag_mode", "RAG mode validated")
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "Query", "valid_rag_mode", str(e))
            raise

    def test_query_valid_mode_selected(self):
        """Test Query with valid selected mode."""
        test_logger.log_test_start("chat_models.py", "Query", "valid_selected_mode")

        try:
            query = Query(
                id="q2",
                text="Explain this",
                mode="selected",
                session_id="session1",
                selected_text="Selected content"
            )

            assert query.mode == "selected"
            assert query.selected_text is not None

            test_logger.log_test_pass("chat_models.py", "Query", "valid_selected_mode", "Selected mode validated")
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "Query", "valid_selected_mode", str(e))
            raise

    def test_query_invalid_mode(self):
        """Test Query rejects invalid mode."""
        test_logger.log_test_start("chat_models.py", "Query", "invalid_mode_rejection")

        try:
            with pytest.raises(ValidationError):
                Query(
                    id="q3",
                    text="Test",
                    mode="invalid",
                    session_id="session1"
                )

            test_logger.log_test_pass(
                "chat_models.py",
                "Query",
                "invalid_mode_rejection",
                "Invalid mode rejected correctly"
            )
        except AssertionError:
            test_logger.log_test_fail("chat_models.py", "Query", "invalid_mode_rejection", "Failed to reject invalid mode")
            raise
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "Query", "invalid_mode_rejection", str(e))
            raise

    def test_query_top_k_bounds(self):
        """Test Query top_k validation."""
        test_logger.log_test_start("chat_models.py", "Query", "top_k_bounds")

        try:
            # Valid bounds
            q1 = Query(id="q1", text="test", mode="rag", session_id="s1", top_k=1)
            q2 = Query(id="q2", text="test", mode="rag", session_id="s1", top_k=20)
            assert q1.top_k == 1
            assert q2.top_k == 20

            # Invalid bounds
            with pytest.raises(ValidationError):
                Query(id="q3", text="test", mode="rag", session_id="s1", top_k=0)

            with pytest.raises(ValidationError):
                Query(id="q4", text="test", mode="rag", session_id="s1", top_k=21)

            test_logger.log_test_pass("chat_models.py", "Query", "top_k_bounds", "top_k bounds validated (1-20)")
        except AssertionError:
            test_logger.log_test_fail("chat_models.py", "Query", "top_k_bounds", "Failed to validate top_k bounds")
            raise
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "Query", "top_k_bounds", str(e))
            raise

    def test_source_creation(self):
        """Test Source model creation."""
        test_logger.log_test_start("chat_models.py", "Source", "creation")

        try:
            source = Source(
                text="Source text",
                book_id="book1",
                chapter_id="ch1",
                score=0.95
            )

            assert source.text == "Source text"
            assert source.score == 0.95

            test_logger.log_test_pass("chat_models.py", "Source", "creation", "Source created successfully")
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "Source", "creation", str(e))
            raise

    def test_chat_request_validation(self):
        """Test ChatRequest validation."""
        test_logger.log_test_start("chat_models.py", "ChatRequest", "validation")

        try:
            # Valid request
            req = ChatRequest(message="Test question")
            assert req.message == "Test question"
            assert req.top_k == 5
            assert req.temperature == 0.7

            # Empty message should fail
            with pytest.raises(ValidationError):
                ChatRequest(message="")

            test_logger.log_test_pass("chat_models.py", "ChatRequest", "validation", "Validation working correctly")
        except AssertionError:
            test_logger.log_test_fail("chat_models.py", "ChatRequest", "validation", "Validation failed")
            raise
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "ChatRequest", "validation", str(e))
            raise

    def test_chat_request_selected_text_limit(self):
        """Test ChatRequest selected_text length limit."""
        test_logger.log_test_start("chat_models.py", "ChatRequest", "selected_text_limit")

        try:
            # Valid selected text
            req = ChatRequest(message="Test", selected_text="a" * 10000)
            assert len(req.selected_text) == 10000

            # Too long selected text should fail
            with pytest.raises(ValidationError):
                ChatRequest(message="Test", selected_text="a" * 10001)

            test_logger.log_test_pass(
                "chat_models.py",
                "ChatRequest",
                "selected_text_limit",
                "Selected text length validated (max 10000)"
            )
        except AssertionError:
            test_logger.log_test_fail("chat_models.py", "ChatRequest", "selected_text_limit", "Failed to validate length")
            raise
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "ChatRequest", "selected_text_limit", str(e))
            raise

    def test_chat_response_creation(self):
        """Test ChatResponse creation."""
        test_logger.log_test_start("chat_models.py", "ChatResponse", "creation")

        try:
            response = ChatResponse(
                response="Answer text",
                session_id="session1",
                sources=[],
                error=None
            )

            assert response.response == "Answer text"
            assert response.session_id == "session1"
            assert response.sources == []

            test_logger.log_test_pass("chat_models.py", "ChatResponse", "creation", "Response created successfully")
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "ChatResponse", "creation", str(e))
            raise

    def test_rag_tool_request_mode_validation(self):
        """Test RAGToolRequest mode validation."""
        test_logger.log_test_start("chat_models.py", "RAGToolRequest", "mode_validation")

        try:
            # Valid modes
            req1 = RAGToolRequest(query="test", mode="rag", top_k=5)
            req2 = RAGToolRequest(query="test", mode="selected", top_k=5)
            assert req1.mode == "rag"
            assert req2.mode == "selected"

            # Invalid mode
            with pytest.raises(ValidationError):
                RAGToolRequest(query="test", mode="invalid", top_k=5)

            test_logger.log_test_pass(
                "chat_models.py",
                "RAGToolRequest",
                "mode_validation",
                "Mode validation working (rag/selected only)"
            )
        except AssertionError:
            test_logger.log_test_fail("chat_models.py", "RAGToolRequest", "mode_validation", "Failed to validate mode")
            raise
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "RAGToolRequest", "mode_validation", str(e))
            raise

    def test_rag_tool_response_creation(self):
        """Test RAGToolResponse creation."""
        test_logger.log_test_start("chat_models.py", "RAGToolResponse", "creation")

        try:
            response = RAGToolResponse(
                chunks=[{"text": "chunk1"}],
                sources=[],
                success=True,
                error=None
            )

            assert response.success is True
            assert len(response.chunks) == 1

            test_logger.log_test_pass("chat_models.py", "RAGToolResponse", "creation", "Response created successfully")
        except Exception as e:
            test_logger.log_test_fail("chat_models.py", "RAGToolResponse", "creation", str(e))
            raise


class TestSessionModel:
    """Test suite for session model."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: models/session_model.py")

    def test_session_creation(self):
        """Test Session model creation."""
        test_logger.log_test_start("session_model.py", "Session", "creation")

        try:
            session = Session(id="s1")

            assert session.id == "s1"
            assert session.history == []
            assert session.context == {}
            assert isinstance(session.created_at, datetime)

            test_logger.log_test_pass("session_model.py", "Session", "creation", "Session created with defaults")
        except Exception as e:
            test_logger.log_test_fail("session_model.py", "Session", "creation", str(e))
            raise

    def test_session_add_turn(self):
        """Test adding conversation turn to session."""
        test_logger.log_test_start("session_model.py", "Session.add_turn", "add_conversation")

        try:
            session = Session(id="s1")
            initial_time = session.last_active

            session.add_turn("Question", "Answer", [])

            assert len(session.history) == 1
            assert session.history[0].query == "Question"
            assert session.history[0].response == "Answer"
            assert session.last_active > initial_time

            test_logger.log_test_pass("session_model.py", "Session.add_turn", "add_conversation", "Turn added successfully")
        except Exception as e:
            test_logger.log_test_fail("session_model.py", "Session.add_turn", "add_conversation", str(e))
            raise

    def test_session_history_limit(self):
        """Test session history limit (max 50)."""
        test_logger.log_test_start("session_model.py", "Session.add_turn", "history_limit")

        try:
            session = Session(id="s1")

            # Add 60 turns
            for i in range(60):
                session.add_turn(f"Q{i}", f"A{i}", [])

            # Should keep only last 50
            assert len(session.history) == 50
            assert session.history[0].query == "Q10"  # First 10 removed
            assert session.history[-1].query == "Q59"

            test_logger.log_test_pass(
                "session_model.py",
                "Session.add_turn",
                "history_limit",
                "History limited to 50 exchanges"
            )
        except Exception as e:
            test_logger.log_test_fail("session_model.py", "Session.add_turn", "history_limit", str(e))
            raise

    def test_session_is_expired(self):
        """Test session expiration check."""
        test_logger.log_test_start("session_model.py", "Session.is_expired", "expiration_check")

        try:
            from datetime import timedelta
            session = Session(id="s1")

            # Fresh session should not be expired
            assert session.is_expired(hours=24) is False

            # Simulate old session
            session.last_active = datetime.utcnow() - timedelta(hours=25)
            assert session.is_expired(hours=24) is True

            test_logger.log_test_pass("session_model.py", "Session.is_expired", "expiration_check", "Expiration logic correct")
        except Exception as e:
            test_logger.log_test_fail("session_model.py", "Session.is_expired", "expiration_check", str(e))
            raise

    def test_conversation_turn_creation(self):
        """Test ConversationTurn model."""
        test_logger.log_test_start("session_model.py", "ConversationTurn", "creation")

        try:
            turn = ConversationTurn(
                query="Test question",
                response="Test answer",
                sources=[{"text": "source"}]
            )

            assert turn.query == "Test question"
            assert turn.response == "Test answer"
            assert len(turn.sources) == 1
            assert isinstance(turn.timestamp, datetime)

            test_logger.log_test_pass("session_model.py", "ConversationTurn", "creation", "Turn created successfully")
        except Exception as e:
            test_logger.log_test_fail("session_model.py", "ConversationTurn", "creation", str(e))
            raise
