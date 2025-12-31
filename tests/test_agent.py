"""
Unit tests for agent.py - OpenAI Agent with RAG integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import BookRAGAgent, get_agent
from tests.test_logger import test_logger


class TestBookRAGAgent:
    """Test suite for BookRAGAgent class."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: agent.py - BookRAGAgent")

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_initialization(self, mock_rag_tool, mock_openai):
        """Test BookRAGAgent initialization."""
        test_logger.log_test_start("agent.py", "BookRAGAgent.__init__", "initialization")

        try:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            agent = BookRAGAgent(model="gemini-2.5-flash")

            assert agent.model == "gemini-2.5-flash"
            assert agent.rag_tool is not None
            assert agent.client is not None
            mock_openai.assert_called_once()

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent.__init__",
                "initialization",
                "Agent initialized successfully"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent.__init__", "initialization", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_system_prompt_no_hallucination(self, mock_rag_tool, mock_openai):
        """Test agent system prompt enforces no hallucination."""
        test_logger.log_test_start("agent.py", "BookRAGAgent", "system_prompt_no_hallucination")

        try:
            agent = BookRAGAgent()

            # Check critical rules in system prompt
            prompt = agent.system_prompt.lower()
            assert 'only answer' in prompt
            assert 'never hallucinate' in prompt or 'do not hallucinate' in prompt or 'not hallucinate' in prompt
            assert 'citations' in prompt or 'sources' in prompt
            assert 'rag_query' in prompt

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent",
                "system_prompt_no_hallucination",
                "System prompt enforces no hallucination and requires citations"
            )
        except AssertionError:
            test_logger.log_test_fail(
                "agent.py",
                "BookRAGAgent",
                "system_prompt_no_hallucination",
                "System prompt missing critical rules"
            )
            raise
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent", "system_prompt_no_hallucination", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_run_rag_mode(self, mock_rag_tool, mock_openai):
        """Test agent execution in RAG mode."""
        test_logger.log_test_start("agent.py", "BookRAGAgent.run", "rag_mode")

        try:
            # Setup mocks
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_tool_instance = Mock()
            mock_rag_tool.return_value = mock_tool_instance
            mock_tool_instance.rag_query.return_value = {
                'chunks': [{'text': 'Test chunk', 'score': 0.9}],
                'sources': [{'text': 'Source text', 'book_id': 'book1', 'score': 0.9}],
                'success': True,
                'error': None
            }

            # Mock OpenAI responses
            mock_tool_call = Mock()
            mock_tool_call.id = 'tool_123'
            mock_tool_call.function.name = 'rag_query'
            mock_tool_call.function.arguments = json.dumps({'query': 'test', 'mode': 'rag', 'top_k': 5})

            mock_message_with_tools = Mock()
            mock_message_with_tools.tool_calls = [mock_tool_call]
            mock_message_with_tools.model_dump.return_value = {'role': 'assistant', 'content': None}

            mock_first_response = Mock()
            mock_first_response.choices = [Mock(message=mock_message_with_tools)]

            mock_final_message = Mock()
            mock_final_message.content = 'This is the answer with citations.'
            mock_final_response = Mock()
            mock_final_response.choices = [Mock(message=mock_final_message)]

            mock_client.chat.completions.create.side_effect = [mock_first_response, mock_final_response]

            # Run agent
            agent = BookRAGAgent()
            result = agent.run("What is kinematics?", mode="rag")

            assert result['success'] is True
            assert result['response'] == 'This is the answer with citations.'
            assert len(result['sources']) == 1

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent.run",
                "rag_mode",
                "Agent executed successfully in RAG mode"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent.run", "rag_mode", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_run_selected_mode(self, mock_rag_tool, mock_openai):
        """Test agent execution in selected text mode."""
        test_logger.log_test_start("agent.py", "BookRAGAgent.run", "selected_mode")

        try:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_tool_instance = Mock()
            mock_rag_tool.return_value = mock_tool_instance
            mock_tool_instance.rag_query.return_value = {
                'chunks': [],
                'sources': [{'text': 'Selected text', 'book_id': 'selected', 'score': 1.0}],
                'success': True,
                'error': None
            }

            # Mock no tool call response
            mock_message = Mock()
            mock_message.content = 'Direct answer based on selected text'
            mock_message.tool_calls = None

            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]

            mock_client.chat.completions.create.return_value = mock_response

            agent = BookRAGAgent()
            result = agent.run(
                "Explain this",
                mode="selected",
                selected_text="The Denavit-Hartenberg convention..."
            )

            # Should work even without tool calls
            assert result is not None

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent.run",
                "selected_mode",
                "Agent handles selected mode"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent.run", "selected_mode", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_run_with_session_context(self, mock_rag_tool, mock_openai):
        """Test agent execution with session context."""
        test_logger.log_test_start("agent.py", "BookRAGAgent.run", "with_session_context")

        try:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock response
            mock_message = Mock()
            mock_message.content = 'Answer'
            mock_message.tool_calls = None
            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]
            mock_client.chat.completions.create.return_value = mock_response

            agent = BookRAGAgent()
            session_context = {
                'history': [
                    {'query': 'Previous question', 'response': 'Previous answer'}
                ]
            }

            result = agent.run("Follow up question", mode="rag", session_context=session_context)

            # Verify history was included in messages
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']

            # Should have system + history + current message
            assert len(messages) >= 3

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent.run",
                "with_session_context",
                "Session context included in conversation"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent.run", "with_session_context", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_run_exception_handling(self, mock_rag_tool, mock_openai):
        """Test agent exception handling."""
        test_logger.log_test_start("agent.py", "BookRAGAgent.run", "exception_handling")

        try:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            agent = BookRAGAgent()
            result = agent.run("test question", mode="rag")

            assert result['success'] is False
            assert result['error'] is not None
            assert 'API Error' in result['error']

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent.run",
                "exception_handling",
                "Exceptions handled gracefully"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent.run", "exception_handling", str(e))
            raise

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_agent_run_tool_parameters(self, mock_rag_tool, mock_openai):
        """Test agent passes correct parameters to tools."""
        test_logger.log_test_start("agent.py", "BookRAGAgent.run", "tool_parameters")

        try:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_tool_instance = Mock()
            mock_rag_tool.return_value = mock_tool_instance
            mock_tool_instance.rag_query.return_value = {
                'chunks': [],
                'sources': [],
                'success': True,
                'error': None
            }

            # Mock tool call
            mock_tool_call = Mock()
            mock_tool_call.id = 'tool_123'
            mock_tool_call.function.arguments = json.dumps({
                'query': 'test query',
                'mode': 'rag',
                'top_k': 5
            })

            mock_message = Mock()
            mock_message.tool_calls = [mock_tool_call]
            mock_message.model_dump.return_value = {}

            mock_first_response = Mock()
            mock_first_response.choices = [Mock(message=mock_message)]

            mock_final_message = Mock()
            mock_final_message.content = 'Answer'
            mock_final_response = Mock()
            mock_final_response.choices = [Mock(message=mock_final_message)]

            mock_client.chat.completions.create.side_effect = [mock_first_response, mock_final_response]

            agent = BookRAGAgent()
            agent.run("test", mode="rag")

            # Verify RAG tool was called correctly
            mock_tool_instance.rag_query.assert_called_once()
            call_kwargs = mock_tool_instance.rag_query.call_args[1]
            assert call_kwargs['query'] == 'test query'
            assert call_kwargs['mode'] == 'rag'
            assert call_kwargs['top_k'] == 5

            test_logger.log_test_pass(
                "agent.py",
                "BookRAGAgent.run",
                "tool_parameters",
                "Tool parameters passed correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "BookRAGAgent.run", "tool_parameters", str(e))
            raise


class TestGetAgent:
    """Test get_agent singleton function."""

    def setup_method(self):
        """Setup test environment."""
        test_logger.log_section("TESTING: agent.py - get_agent()")

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('agent.OpenAI')
    @patch('agent.RAGTool')
    def test_get_agent_singleton(self, mock_rag_tool, mock_openai):
        """Test get_agent returns same instance."""
        test_logger.log_test_start("agent.py", "get_agent", "singleton_pattern")

        try:
            # Reset global instance
            import agent as agent_module
            agent_module._agent_instance = None

            agent1 = get_agent()
            agent2 = get_agent()

            assert agent1 is agent2

            test_logger.log_test_pass(
                "agent.py",
                "get_agent",
                "singleton_pattern",
                "Singleton pattern working correctly"
            )
        except Exception as e:
            test_logger.log_test_fail("agent.py", "get_agent", "singleton_pattern", str(e))
            raise
