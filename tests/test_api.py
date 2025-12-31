"""
Integration tests for API endpoints - Complete end-to-end testing.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from tests.test_logger import test_logger


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: app.py - API Endpoints - Root")

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        test_logger.log_test_start("app.py", "root()", "response")

        try:
            response = client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert "message" in data
            assert "version" in data
            assert "status" in data

            test_logger.log_test_pass("app.py", "root()", "response", "Root endpoint returns correct data")
        except Exception as e:
            test_logger.log_test_fail("app.py", "root()", "response", str(e))
            raise


class TestHealthEndpoint:
    """Test health check endpoint."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: app.py - API Endpoints - Health")

    def test_health_check(self, client):
        """Test health check endpoint."""
        test_logger.log_test_start("app.py", "health_check()", "status")

        try:
            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data

            test_logger.log_test_pass("app.py", "health_check()", "status", "Health check returns healthy status")
        except Exception as e:
            test_logger.log_test_fail("app.py", "health_check()", "status", str(e))
            raise


class TestChatEndpoint:
    """Test chat endpoint - most critical."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: app.py - API Endpoints - Chat")

    @patch('app.get_agent')
    def test_chat_valid_request(self, mock_get_agent, client):
        """Test chat endpoint with valid request."""
        test_logger.log_test_start("app.py", "chat()", "valid_request")

        try:
            # Mock agent
            mock_agent = Mock()
            mock_agent.run.return_value = {
                'response': 'Test answer',
                'sources': [{'text': 'source', 'book_id': 'book1', 'score': 0.9}],
                'success': True,
                'error': None
            }
            mock_get_agent.return_value = mock_agent

            # Make request
            response = client.post("/chat", json={
                "message": "What is kinematics?",
                "session_id": "test_session"
            })

            assert response.status_code == 200
            data = response.json()

            assert "response" in data
            assert "session_id" in data
            assert "sources" in data
            assert data["response"] == "Test answer"

            test_logger.log_test_pass("app.py", "chat()", "valid_request", "Chat endpoint handles valid requests")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "valid_request", str(e))
            raise

    @patch('app.get_agent')
    def test_chat_creates_session(self, mock_get_agent, client):
        """Test chat endpoint creates new session if not provided."""
        test_logger.log_test_start("app.py", "chat()", "auto_create_session")

        try:
            mock_agent = Mock()
            mock_agent.run.return_value = {
                'response': 'Answer',
                'sources': [],
                'success': True,
                'error': None
            }
            mock_get_agent.return_value = mock_agent

            response = client.post("/chat", json={
                "message": "Test question"
                # No session_id provided
            })

            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["session_id"] is not None

            test_logger.log_test_pass("app.py", "chat()", "auto_create_session", "Auto-creates session when not provided")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "auto_create_session", str(e))
            raise

    @patch('app.get_agent')
    def test_chat_rag_mode(self, mock_get_agent, client):
        """Test chat endpoint in RAG mode (no selected_text)."""
        test_logger.log_test_start("app.py", "chat()", "rag_mode")

        try:
            mock_agent = Mock()
            mock_agent.run.return_value = {
                'response': 'RAG answer',
                'sources': [],
                'success': True,
                'error': None
            }
            mock_get_agent.return_value = mock_agent

            response = client.post("/chat", json={
                "message": "Test",
                "session_id": "test"
            })

            # Verify agent was called with mode='rag'
            call_args = mock_agent.run.call_args
            assert call_args[1]['mode'] == 'rag'

            test_logger.log_test_pass("app.py", "chat()", "rag_mode", "RAG mode selected when no selected_text")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "rag_mode", str(e))
            raise

    @patch('app.get_agent')
    def test_chat_selected_mode(self, mock_get_agent, client):
        """Test chat endpoint in selected text mode."""
        test_logger.log_test_start("app.py", "chat()", "selected_mode")

        try:
            mock_agent = Mock()
            mock_agent.run.return_value = {
                'response': 'Selected text answer',
                'sources': [],
                'success': True,
                'error': None
            }
            mock_get_agent.return_value = mock_agent

            response = client.post("/chat", json={
                "message": "Explain this",
                "selected_text": "The Denavit-Hartenberg convention...",
                "session_id": "test"
            })

            # Verify agent was called with mode='selected'
            call_args = mock_agent.run.call_args
            assert call_args[1]['mode'] == 'selected'
            assert call_args[1]['selected_text'] == "The Denavit-Hartenberg convention..."

            test_logger.log_test_pass("app.py", "chat()", "selected_mode", "Selected mode used when selected_text provided")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "selected_mode", str(e))
            raise

    def test_chat_invalid_request_empty_message(self, client):
        """Test chat endpoint rejects empty message."""
        test_logger.log_test_start("app.py", "chat()", "empty_message_validation")

        try:
            response = client.post("/chat", json={
                "message": "",
                "session_id": "test"
            })

            assert response.status_code == 422  # Validation error

            test_logger.log_test_pass("app.py", "chat()", "empty_message_validation", "Empty message rejected")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "empty_message_validation", str(e))
            raise

    def test_chat_invalid_request_missing_message(self, client):
        """Test chat endpoint requires message field."""
        test_logger.log_test_start("app.py", "chat()", "missing_message_validation")

        try:
            response = client.post("/chat", json={
                "session_id": "test"
            })

            assert response.status_code == 422

            test_logger.log_test_pass("app.py", "chat()", "missing_message_validation", "Missing message field rejected")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "missing_message_validation", str(e))
            raise

    @patch('app.get_agent')
    def test_chat_agent_failure(self, mock_get_agent, client):
        """Test chat endpoint handles agent failures."""
        test_logger.log_test_start("app.py", "chat()", "agent_failure_handling")

        try:
            mock_agent = Mock()
            mock_agent.run.return_value = {
                'response': 'Error message',
                'sources': [],
                'success': False,
                'error': 'Agent failed'
            }
            mock_get_agent.return_value = mock_agent

            response = client.post("/chat", json={
                "message": "Test",
                "session_id": "test"
            })

            assert response.status_code == 500

            test_logger.log_test_pass("app.py", "chat()", "agent_failure_handling", "Agent failures handled with 500 error")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "agent_failure_handling", str(e))
            raise

    @patch('app.get_agent')
    def test_chat_session_history(self, mock_get_agent, client):
        """Test chat endpoint maintains session history."""
        test_logger.log_test_start("app.py", "chat()", "session_history")

        try:
            mock_agent = Mock()
            mock_agent.run.return_value = {
                'response': 'Answer',
                'sources': [],
                'success': True,
                'error': None
            }
            mock_get_agent.return_value = mock_agent

            # First message
            response1 = client.post("/chat", json={
                "message": "First question",
                "session_id": "history_test"
            })
            assert response1.status_code == 200

            # Second message in same session
            response2 = client.post("/chat", json={
                "message": "Second question",
                "session_id": "history_test"
            })
            assert response2.status_code == 200

            # Verify session context includes history
            second_call_args = mock_agent.run.call_args_list[1]
            session_context = second_call_args[1]['session_context']
            assert 'history' in session_context
            assert len(session_context['history']) > 0

            test_logger.log_test_pass("app.py", "chat()", "session_history", "Session history maintained across requests")
        except Exception as e:
            test_logger.log_test_fail("app.py", "chat()", "session_history", str(e))
            raise


class TestSessionEndpoints:
    """Test session management endpoints."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: app.py - API Endpoints - Session Management")

    @patch('app.get_agent')
    def test_get_session(self, mock_get_agent, client):
        """Test get session endpoint."""
        test_logger.log_test_start("app.py", "get_session()", "retrieve_session")

        try:
            # Create a session first
            mock_agent = Mock()
            mock_agent.run.return_value = {'response': 'Test', 'sources': [], 'success': True, 'error': None}
            mock_get_agent.return_value = mock_agent

            client.post("/chat", json={"message": "Test", "session_id": "get_test"})

            # Get session
            response = client.get("/session/get_test")
            assert response.status_code == 200

            data = response.json()
            assert "session_id" in data
            assert "created_at" in data
            assert "history" in data

            test_logger.log_test_pass("app.py", "get_session()", "retrieve_session", "Session retrieved successfully")
        except Exception as e:
            test_logger.log_test_fail("app.py", "get_session()", "retrieve_session", str(e))
            raise

    def test_get_session_not_found(self, client):
        """Test get session returns 404 for non-existent session."""
        test_logger.log_test_start("app.py", "get_session()", "not_found")

        try:
            response = client.get("/session/nonexistent")
            assert response.status_code == 404

            test_logger.log_test_pass("app.py", "get_session()", "not_found", "Returns 404 for non-existent session")
        except Exception as e:
            test_logger.log_test_fail("app.py", "get_session()", "not_found", str(e))
            raise

    @patch('app.get_agent')
    def test_delete_session(self, mock_get_agent, client):
        """Test delete session endpoint."""
        test_logger.log_test_start("app.py", "delete_session()", "delete_session")

        try:
            # Create session
            mock_agent = Mock()
            mock_agent.run.return_value = {'response': 'Test', 'sources': [], 'success': True, 'error': None}
            mock_get_agent.return_value = mock_agent

            client.post("/chat", json={"message": "Test", "session_id": "delete_test"})

            # Delete session
            response = client.delete("/session/delete_test")
            assert response.status_code == 200

            # Verify it's deleted
            get_response = client.get("/session/delete_test")
            assert get_response.status_code == 404

            test_logger.log_test_pass("app.py", "delete_session()", "delete_session", "Session deleted successfully")
        except Exception as e:
            test_logger.log_test_fail("app.py", "delete_session()", "delete_session", str(e))
            raise

    def test_delete_session_not_found(self, client):
        """Test delete session returns 404 for non-existent session."""
        test_logger.log_test_start("app.py", "delete_session()", "not_found")

        try:
            response = client.delete("/session/nonexistent")
            assert response.status_code == 404

            test_logger.log_test_pass("app.py", "delete_session()", "not_found", "Returns 404 for non-existent session")
        except Exception as e:
            test_logger.log_test_fail("app.py", "delete_session()", "not_found", str(e))
            raise

    def test_cleanup_expired_sessions(self, client):
        """Test cleanup expired sessions endpoint."""
        test_logger.log_test_start("app.py", "cleanup_expired_sessions()", "cleanup")

        try:
            response = client.post("/sessions/cleanup")
            assert response.status_code == 200

            data = response.json()
            assert "message" in data
            assert "cleaned_count" in data

            test_logger.log_test_pass("app.py", "cleanup_expired_sessions()", "cleanup", "Cleanup endpoint works")
        except Exception as e:
            test_logger.log_test_fail("app.py", "cleanup_expired_sessions()", "cleanup", str(e))
            raise
