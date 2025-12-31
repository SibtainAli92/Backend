"""
Unit tests for RAG tool - Vector search and embedding generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rag_tool import RAGTool, RAG_QUERY_TOOL_SCHEMA
from tests.test_logger import test_logger


class TestRAGTool:
    """Test suite for RAGTool class."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: tools/rag_tool.py - RAGTool")

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.GenerativeModel')
    def test_rag_tool_initialization(self, mock_model, mock_configure, mock_qdrant):
        """Test RAGTool initialization."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.__init__", "initialization")

        try:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client

            tool = RAGTool("test_collection")

            assert tool.collection_name == "test_collection"
            assert tool.qdrant_client is not None
            mock_configure.assert_called_once()

            test_logger.log_test_pass("rag_tool.py", "RAGTool.__init__", "initialization", "Initialized correctly")
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.__init__", "initialization", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.embed_content')
    def test_generate_embedding(self, mock_embed, mock_configure, mock_qdrant):
        """Test embedding generation."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.generate_embedding", "embedding_generation")

        try:
            mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}

            tool = RAGTool()
            embedding = tool.generate_embedding("test text")

            assert embedding == [0.1, 0.2, 0.3]
            mock_embed.assert_called_once()
            call_args = mock_embed.call_args
            assert call_args[1]['content'] == "test text"
            assert call_args[1]['task_type'] == "retrieval_query"

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.generate_embedding",
                "embedding_generation",
                "Embedding generated successfully"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.generate_embedding", "embedding_generation", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.embed_content')
    def test_rag_query_rag_mode_success(self, mock_embed, mock_configure, mock_qdrant):
        """Test RAG query in RAG mode with successful results."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "rag_mode_success")

        try:
            # Setup mocks
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            mock_embed.return_value = {'embedding': [0.1] * 768}

            # Mock search results
            mock_result = Mock()
            mock_result.score = 0.9
            mock_result.payload = {
                'text': 'Test chunk text',
                'book_id': 'book1',
                'chapter_id': 'ch1',
                'section_id': 's1',
                'source_url': 'http://test.com'
            }
            mock_client.search.return_value = [mock_result]

            tool = RAGTool()
            result = tool.rag_query("test query", mode="rag", top_k=5)

            assert result['success'] is True
            assert len(result['chunks']) == 1
            assert len(result['sources']) == 1
            assert result['chunks'][0]['text'] == 'Test chunk text'
            assert result['chunks'][0]['score'] == 0.9
            assert result['error'] is None

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "rag_mode_success",
                "RAG mode query successful with results"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "rag_mode_success", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.embed_content')
    def test_rag_query_rag_mode_empty_results(self, mock_embed, mock_configure, mock_qdrant):
        """Test RAG query with no results found."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "rag_mode_empty_results")

        try:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            mock_embed.return_value = {'embedding': [0.1] * 768}

            # Empty search results
            mock_client.search.return_value = []

            tool = RAGTool()
            result = tool.rag_query("test query", mode="rag", top_k=5)

            assert result['success'] is True
            assert len(result['chunks']) == 0
            assert len(result['sources']) == 0

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "rag_mode_empty_results",
                "Handles empty results correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "rag_mode_empty_results", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.embed_content')
    def test_rag_query_selected_mode(self, mock_embed, mock_configure, mock_qdrant):
        """Test RAG query in selected text mode."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "selected_mode")

        try:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            mock_embed.return_value = {'embedding': [0.1] * 768}

            # Mock search results
            mock_result = Mock()
            mock_result.id = 1
            mock_result.score = 0.85
            mock_result.payload = {
                'text': 'Relevant chunk',
                'book_id': 'book1',
                'chapter_id': 'ch1'
            }
            mock_client.search.return_value = [mock_result]

            tool = RAGTool()
            result = tool.rag_query(
                "test query",
                mode="selected",
                top_k=5,
                selected_text="Selected text from book"
            )

            assert result['success'] is True
            assert len(result['sources']) >= 1
            # First source should be the selected text
            assert result['sources'][0]['book_id'] == 'selected'

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "selected_mode",
                "Selected mode includes selected text as primary source"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "selected_mode", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    def test_rag_query_invalid_mode(self, mock_configure, mock_qdrant):
        """Test RAG query with invalid mode."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "invalid_mode")

        try:
            tool = RAGTool()
            result = tool.rag_query("test", mode="invalid", top_k=5)

            assert result['success'] is False
            assert result['error'] is not None
            assert 'Invalid mode' in result['error']

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "invalid_mode",
                "Invalid mode returns error correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "invalid_mode", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    def test_rag_query_selected_without_text(self, mock_configure, mock_qdrant):
        """Test RAG query selected mode without selected_text."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "selected_without_text")

        try:
            tool = RAGTool()
            result = tool.rag_query("test", mode="selected", top_k=5, selected_text=None)

            assert result['success'] is False
            assert result['error'] is not None

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "selected_without_text",
                "Selected mode requires selected_text"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "selected_without_text", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.embed_content')
    def test_rag_query_exception_handling(self, mock_embed, mock_configure, mock_qdrant):
        """Test RAG query exception handling."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "exception_handling")

        try:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            mock_embed.side_effect = Exception("Embedding failed")

            tool = RAGTool()
            result = tool.rag_query("test", mode="rag", top_k=5)

            assert result['success'] is False
            assert result['error'] is not None
            assert 'Embedding failed' in result['error']

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "exception_handling",
                "Exceptions handled gracefully"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "exception_handling", str(e))
            raise

    @patch('tools.rag_tool.get_qdrant_client')
    @patch('tools.rag_tool.configure_gemini')
    @patch('tools.rag_tool.genai.embed_content')
    def test_rag_query_score_threshold(self, mock_embed, mock_configure, mock_qdrant):
        """Test RAG query applies score threshold correctly."""
        test_logger.log_test_start("rag_tool.py", "RAGTool.rag_query", "score_threshold")

        try:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            mock_embed.return_value = {'embedding': [0.1] * 768}

            tool = RAGTool()
            tool.rag_query("test", mode="rag", top_k=5)

            # Verify search was called with score_threshold
            call_args = mock_client.search.call_args
            assert 'score_threshold' in call_args[1]
            assert call_args[1]['score_threshold'] == 0.5

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAGTool.rag_query",
                "score_threshold",
                "Score threshold (0.5) applied correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAGTool.rag_query", "score_threshold", str(e))
            raise


class TestRAGToolSchema:
    """Test RAG tool schema definition."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: RAG_QUERY_TOOL_SCHEMA")

    def test_tool_schema_structure(self):
        """Test RAG tool schema has correct structure."""
        test_logger.log_test_start("rag_tool.py", "RAG_QUERY_TOOL_SCHEMA", "structure")

        try:
            assert 'type' in RAG_QUERY_TOOL_SCHEMA
            assert RAG_QUERY_TOOL_SCHEMA['type'] == 'function'
            assert 'function' in RAG_QUERY_TOOL_SCHEMA

            function = RAG_QUERY_TOOL_SCHEMA['function']
            assert 'name' in function
            assert function['name'] == 'rag_query'
            assert 'description' in function
            assert 'parameters' in function

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAG_QUERY_TOOL_SCHEMA",
                "structure",
                "Schema structure is correct"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAG_QUERY_TOOL_SCHEMA", "structure", str(e))
            raise

    def test_tool_schema_parameters(self):
        """Test RAG tool schema parameters."""
        test_logger.log_test_start("rag_tool.py", "RAG_QUERY_TOOL_SCHEMA", "parameters")

        try:
            params = RAG_QUERY_TOOL_SCHEMA['function']['parameters']
            props = params['properties']

            # Check required parameters
            assert 'query' in props
            assert 'mode' in props
            assert 'top_k' in props

            # Check mode enum
            assert 'enum' in props['mode']
            assert set(props['mode']['enum']) == {'rag', 'selected'}

            # Check top_k bounds
            assert props['top_k']['minimum'] == 1
            assert props['top_k']['maximum'] == 20

            # Check required fields
            assert set(params['required']) == {'query', 'mode'}

            test_logger.log_test_pass(
                "rag_tool.py",
                "RAG_QUERY_TOOL_SCHEMA",
                "parameters",
                "Parameters defined correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("rag_tool.py", "RAG_QUERY_TOOL_SCHEMA", "parameters", str(e))
            raise
