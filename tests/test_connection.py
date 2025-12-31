"""
Unit tests for connection.py - Service client initialization and management.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from qdrant_client import QdrantClient
from openai import OpenAI

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import Connections, get_qdrant_client, get_openai_client, configure_gemini
from tests.test_logger import test_logger


class TestConnections:
    """Test suite for Connections class."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: connection.py - Connections Class")

    def test_connections_init(self):
        """Test Connections initialization."""
        test_logger.log_test_start("connection.py", "Connections.__init__", "initialization")

        try:
            conn = Connections()
            assert conn._qdrant_client is None
            assert conn._openai_client is None
            assert conn._gemini_configured is False

            test_logger.log_test_pass(
                "connection.py",
                "Connections.__init__",
                "initialization",
                "All attributes initialized correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "Connections.__init__", "initialization", str(e))
            raise

    @patch.dict(os.environ, {'QDRANT_URL': 'http://test:6333', 'QDRANT_API_KEY': 'test_key'})
    @patch('connection.QdrantClient')
    def test_get_qdrant_client_with_api_key(self, mock_client):
        """Test Qdrant client creation with API key."""
        test_logger.log_test_start("connection.py", "get_qdrant_client", "with_api_key")

        try:
            conn = Connections()
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            client = conn.get_qdrant_client()

            assert client is not None
            mock_client.assert_called_once_with(
                url='http://test:6333',
                api_key='test_key'
            )

            test_logger.log_test_pass(
                "connection.py",
                "get_qdrant_client",
                "with_api_key",
                "Client created with correct parameters"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_qdrant_client", "with_api_key", str(e))
            raise

    @patch.dict(os.environ, {'QDRANT_URL': 'http://test:6333'}, clear=True)
    @patch('connection.QdrantClient')
    def test_get_qdrant_client_without_api_key(self, mock_client):
        """Test Qdrant client creation without API key."""
        test_logger.log_test_start("connection.py", "get_qdrant_client", "without_api_key")

        try:
            conn = Connections()
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Remove QDRANT_API_KEY if exists
            os.environ.pop('QDRANT_API_KEY', None)

            client = conn.get_qdrant_client()

            assert client is not None
            mock_client.assert_called_once_with(url='http://test:6333')

            test_logger.log_test_pass(
                "connection.py",
                "get_qdrant_client",
                "without_api_key",
                "Client created without API key"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_qdrant_client", "without_api_key", str(e))
            raise

    def test_get_qdrant_client_singleton(self):
        """Test Qdrant client singleton pattern."""
        test_logger.log_test_start("connection.py", "get_qdrant_client", "singleton_pattern")

        try:
            with patch('connection.QdrantClient') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance

                conn = Connections()
                client1 = conn.get_qdrant_client()
                client2 = conn.get_qdrant_client()

                assert client1 is client2
                assert mock_client.call_count == 1

                test_logger.log_test_pass(
                    "connection.py",
                    "get_qdrant_client",
                    "singleton_pattern",
                    "Same instance returned on multiple calls"
                )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_qdrant_client", "singleton_pattern", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_openai_key'})
    @patch('connection.OpenAI')
    def test_get_openai_client(self, mock_openai):
        """Test OpenAI client creation."""
        test_logger.log_test_start("connection.py", "get_openai_client", "creation")

        try:
            conn = Connections()
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            client = conn.get_openai_client()

            assert client is not None
            mock_openai.assert_called_once_with(api_key='test_openai_key')

            test_logger.log_test_pass(
                "connection.py",
                "get_openai_client",
                "creation",
                "OpenAI client created successfully"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_openai_client", "creation", str(e))
            raise

    @patch.dict(os.environ, {}, clear=True)
    def test_get_openai_client_missing_key(self):
        """Test OpenAI client fails without API key."""
        test_logger.log_test_start("connection.py", "get_openai_client", "missing_key_error")

        try:
            conn = Connections()
            os.environ.pop('OPENAI_API_KEY', None)

            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                conn.get_openai_client()

            test_logger.log_test_pass(
                "connection.py",
                "get_openai_client",
                "missing_key_error",
                "Correctly raises ValueError when API key missing"
            )
        except AssertionError:
            test_logger.log_test_fail(
                "connection.py",
                "get_openai_client",
                "missing_key_error",
                "Failed to raise ValueError for missing API key"
            )
            raise
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_openai_client", "missing_key_error", str(e))
            raise

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_gemini_key'})
    @patch('connection.genai.configure')
    def test_configure_gemini(self, mock_configure):
        """Test Gemini configuration."""
        test_logger.log_test_start("connection.py", "configure_gemini", "configuration")

        try:
            conn = Connections()
            conn.configure_gemini()

            mock_configure.assert_called_once_with(api_key='test_gemini_key')
            assert conn._gemini_configured is True

            test_logger.log_test_pass(
                "connection.py",
                "configure_gemini",
                "configuration",
                "Gemini configured successfully"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "configure_gemini", "configuration", str(e))
            raise

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_gemini_missing_key(self):
        """Test Gemini configuration fails without API key."""
        test_logger.log_test_start("connection.py", "configure_gemini", "missing_key_error")

        try:
            conn = Connections()
            os.environ.pop('GEMINI_API_KEY', None)

            with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
                conn.configure_gemini()

            test_logger.log_test_pass(
                "connection.py",
                "configure_gemini",
                "missing_key_error",
                "Correctly raises ValueError when API key missing"
            )
        except AssertionError:
            test_logger.log_test_fail(
                "connection.py",
                "configure_gemini",
                "missing_key_error",
                "Failed to raise ValueError for missing API key"
            )
            raise
        except Exception as e:
            test_logger.log_test_fail("connection.py", "configure_gemini", "missing_key_error", str(e))
            raise

    @patch('connection.QdrantClient')
    def test_ensure_qdrant_collection_creates_new(self, mock_client_class):
        """Test Qdrant collection creation when it doesn't exist."""
        test_logger.log_test_start("connection.py", "ensure_qdrant_collection", "create_new_collection")

        try:
            mock_client = Mock()
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            conn = Connections()
            conn._qdrant_client = mock_client

            conn.ensure_qdrant_collection("test_collection", 768)

            mock_client.create_collection.assert_called_once()
            call_args = mock_client.create_collection.call_args
            assert call_args[1]['collection_name'] == 'test_collection'

            test_logger.log_test_pass(
                "connection.py",
                "ensure_qdrant_collection",
                "create_new_collection",
                "Collection created when it doesn't exist"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "ensure_qdrant_collection", "create_new_collection", str(e))
            raise

    @patch('connection.QdrantClient')
    def test_ensure_qdrant_collection_exists(self, mock_client_class):
        """Test Qdrant collection check when collection exists."""
        test_logger.log_test_start("connection.py", "ensure_qdrant_collection", "collection_exists")

        try:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.name = 'test_collection'
            mock_collections = Mock()
            mock_collections.collections = [mock_collection]
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            conn = Connections()
            conn._qdrant_client = mock_client

            conn.ensure_qdrant_collection("test_collection", 768)

            mock_client.create_collection.assert_not_called()

            test_logger.log_test_pass(
                "connection.py",
                "ensure_qdrant_collection",
                "collection_exists",
                "Skips creation when collection exists"
            )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "ensure_qdrant_collection", "collection_exists", str(e))
            raise

    def test_global_get_qdrant_client(self):
        """Test global get_qdrant_client function."""
        test_logger.log_test_start("connection.py", "get_qdrant_client (global)", "global_function")

        try:
            with patch('connection.connections.get_qdrant_client') as mock_method:
                mock_client = Mock()
                mock_method.return_value = mock_client

                client = get_qdrant_client()

                assert client is mock_client
                mock_method.assert_called_once()

                test_logger.log_test_pass(
                    "connection.py",
                    "get_qdrant_client (global)",
                    "global_function",
                    "Global function delegates correctly"
                )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_qdrant_client (global)", "global_function", str(e))
            raise

    def test_global_get_openai_client(self):
        """Test global get_openai_client function."""
        test_logger.log_test_start("connection.py", "get_openai_client (global)", "global_function")

        try:
            with patch('connection.connections.get_openai_client') as mock_method:
                mock_client = Mock()
                mock_method.return_value = mock_client

                client = get_openai_client()

                assert client is mock_client
                mock_method.assert_called_once()

                test_logger.log_test_pass(
                    "connection.py",
                    "get_openai_client (global)",
                    "global_function",
                    "Global function delegates correctly"
                )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "get_openai_client (global)", "global_function", str(e))
            raise

    def test_global_configure_gemini(self):
        """Test global configure_gemini function."""
        test_logger.log_test_start("connection.py", "configure_gemini (global)", "global_function")

        try:
            with patch('connection.connections.configure_gemini') as mock_method:
                configure_gemini()

                mock_method.assert_called_once()

                test_logger.log_test_pass(
                    "connection.py",
                    "configure_gemini (global)",
                    "global_function",
                    "Global function delegates correctly"
                )
        except Exception as e:
            test_logger.log_test_fail("connection.py", "configure_gemini (global)", "global_function", str(e))
            raise
