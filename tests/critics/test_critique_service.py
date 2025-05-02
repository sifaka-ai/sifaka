"""
Comprehensive tests for the CritiqueService class.

This module contains tests for the CritiqueService class, including edge cases,
error handling, and async methods.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

from sifaka.critics.services.critique import CritiqueService
from sifaka.critics.managers.prompt import PromptManager
from sifaka.critics.managers.response import ResponseParser
from sifaka.critics.managers.memory import MemoryManager


class MockLanguageModel:
    """Mock language model for testing."""

    def __init__(self, response=None, error=False):
        """Initialize with configurable response and error state."""
        self.response = response or {"valid": True}
        self.error = error
        self.invoke_calls = []
        self.ainvoke_calls = []

    def invoke(self, prompt):
        """Mock invoke method."""
        self.invoke_calls.append(prompt)
        if self.error:
            raise Exception("Test exception")
        return self.response

    async def ainvoke(self, prompt):
        """Mock async invoke method."""
        self.ainvoke_calls.append(prompt)
        if self.error:
            raise Exception("Test async exception")
        return self.response


class SyncOnlyModel:
    """Mock model that only supports synchronous calls."""

    def __init__(self, response=None):
        """Initialize with a configurable response."""
        self.response = response or {"valid": True}
        self.invoke_calls = []

    def invoke(self, prompt):
        """Mock invoke method."""
        self.invoke_calls.append(prompt)
        return self.response


class TestCritiqueService(unittest.TestCase):
    """Comprehensive tests for the CritiqueService class."""

    def setUp(self):
        """Set up the test environment."""
        self.prompt_manager = Mock(spec=PromptManager)
        self.response_parser = Mock(spec=ResponseParser)
        self.memory_manager = Mock(spec=MemoryManager)

        # Set up default return values
        self.prompt_manager.create_validation_prompt.return_value = "validation prompt"
        self.prompt_manager.create_critique_prompt.return_value = "critique prompt"
        self.prompt_manager.create_improvement_prompt.return_value = "improvement prompt"
        self.prompt_manager.create_reflection_prompt.return_value = "reflection prompt"

        self.response_parser.parse_validation_response.return_value = True
        self.response_parser.parse_critique_response.return_value = {
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Minor issue"],
            "suggestions": ["Minor suggestion"]
        }
        self.response_parser.parse_improvement_response.return_value = "Improved content"
        self.response_parser.parse_reflection_response.return_value = "Reflection content"

        self.memory_manager.get_memory.return_value = ["Previous reflection"]

        # Create a default model
        self.model = MockLanguageModel()

        # Create the service
        self.service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )

    def test_init(self):
        """Test initialization."""
        # Test initialization without memory manager
        service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )
        self.assertIsNone(service._memory_manager)

        # Test initialization with memory manager
        service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )
        self.assertEqual(service._memory_manager, self.memory_manager)

    def test_validate_success(self):
        """Test successful validation."""
        result = self.service.validate("Test text")
        self.assertTrue(result)
        self.prompt_manager.create_validation_prompt.assert_called_once_with("Test text")
        self.response_parser.parse_validation_response.assert_called_once()

    def test_validate_empty_text(self):
        """Test validation with empty text."""
        # Test with empty string
        with self.assertRaises(ValueError):
            self.service.validate("")

        # Test with None
        with self.assertRaises(ValueError):
            self.service.validate(None)

        # Test with whitespace
        with self.assertRaises(ValueError):
            self.service.validate("   ")

    def test_validate_error(self):
        """Test validation with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # The service should handle the exception and return False
        result = service.validate("Test text")
        self.assertFalse(result)

    def test_critique_success(self):
        """Test successful critique."""
        result = self.service.critique("Test text")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good content")
        self.assertEqual(result["issues"], ["Minor issue"])
        self.assertEqual(result["suggestions"], ["Minor suggestion"])
        self.prompt_manager.create_critique_prompt.assert_called_once_with("Test text")
        self.response_parser.parse_critique_response.assert_called_once()

    def test_critique_empty_text(self):
        """Test critique with empty text."""
        # Test with empty string
        with self.assertRaises(ValueError):
            self.service.critique("")

        # Test with None
        with self.assertRaises(ValueError):
            self.service.critique(None)

        # Test with whitespace
        with self.assertRaises(ValueError):
            self.service.critique("   ")

    def test_critique_error(self):
        """Test critique with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # The service should handle the exception and return a default response
        result = service.critique("Test text")
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Failed to critique text", result["feedback"])
        self.assertEqual(result["issues"], ["Failed to parse model response"])

    def test_improve_with_string_feedback(self):
        """Test improve with string feedback."""
        result = self.service.improve("Test text", "Test feedback")
        self.assertEqual(result, "Improved content")
        self.prompt_manager.create_improvement_prompt.assert_called_once()
        self.response_parser.parse_improvement_response.assert_called_once()

    def test_improve_with_violations(self):
        """Test improve with violations list."""
        violations = [
            {"rule_name": "Rule 1", "message": "Message 1"},
            {"rule_name": "Rule 2", "message": "Message 2"},
        ]
        result = self.service.improve("Test text", violations)
        self.assertEqual(result, "Improved content")

        # Check that the violations were converted to a string
        self.prompt_manager.create_improvement_prompt.assert_called_once()
        call_args = self.prompt_manager.create_improvement_prompt.call_args[0]
        self.assertEqual(call_args[0], "Test text")
        self.assertIn("Rule 1: Message 1", call_args[1])
        self.assertIn("Rule 2: Message 2", call_args[1])

    def test_improve_empty_text(self):
        """Test improve with empty text."""
        # Test with empty string
        with self.assertRaises(ValueError):
            self.service.improve("", "Test feedback")

        # Test with None
        with self.assertRaises(ValueError):
            self.service.improve(None, "Test feedback")

        # Test with whitespace
        with self.assertRaises(ValueError):
            self.service.improve("   ", "Test feedback")

    def test_improve_error(self):
        """Test improve with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # The service should propagate the exception
        with self.assertRaises(ValueError):
            service.improve("Test text", "Test feedback")

    def test_violations_to_feedback(self):
        """Test violations to feedback conversion."""
        # Test with violations
        violations = [
            {"rule_name": "Rule 1", "message": "Message 1"},
            {"rule_name": "Rule 2", "message": "Message 2"},
        ]
        result = self.service._violations_to_feedback(violations)
        self.assertIn("The following issues were found:", result)
        self.assertIn("- Rule 1: Message 1", result)
        self.assertIn("- Rule 2: Message 2", result)

        # Test with empty violations
        result = self.service._violations_to_feedback([])
        self.assertEqual(result, "No issues found.")

        # Test with missing rule_name or message
        violations = [
            {"message": "Message without rule"},
            {"rule_name": "Rule without message"},
            {},
        ]
        result = self.service._violations_to_feedback(violations)
        self.assertIn("- Rule 1: Message without rule", result)
        self.assertIn("- Rule without message: Unknown issue", result)
        self.assertIn("- Rule 3: Unknown issue", result)

    def test_generate_reflection(self):
        """Test reflection generation."""
        self.service._generate_reflection("Original", "Feedback", "Improved")
        self.prompt_manager.create_reflection_prompt.assert_called_once_with(
            "Original", "Feedback", "Improved"
        )
        self.response_parser.parse_reflection_response.assert_called_once()
        self.memory_manager.add_to_memory.assert_called_once_with("Reflection content")

    def test_generate_reflection_no_memory_manager(self):
        """Test reflection generation with no memory manager."""
        service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # Should not raise an exception
        service._generate_reflection("Original", "Feedback", "Improved")

        # Prompt manager should not be called
        self.prompt_manager.create_reflection_prompt.assert_not_called()

    def test_generate_reflection_error(self):
        """Test reflection generation with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )

        # Should not raise an exception
        service._generate_reflection("Original", "Feedback", "Improved")

        # Memory manager should not be called
        self.memory_manager.add_to_memory.assert_not_called()

    def test_generate_reflection_empty_response(self):
        """Test reflection generation with empty response."""
        # Set the parser to return None for the reflection
        self.response_parser.parse_reflection_response.return_value = None

        self.service._generate_reflection("Original", "Feedback", "Improved")

        # Memory manager should not be called
        self.memory_manager.add_to_memory.assert_not_called()


class TestCritiqueServiceAsync:
    """Tests for async methods of CritiqueService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.prompt_manager = Mock(spec=PromptManager)
        self.response_parser = Mock(spec=ResponseParser)
        self.memory_manager = Mock(spec=MemoryManager)

        # Set up default return values
        self.prompt_manager.create_validation_prompt.return_value = "validation prompt"
        self.prompt_manager.create_critique_prompt.return_value = "critique prompt"
        self.prompt_manager.create_improvement_prompt.return_value = "improvement prompt"
        self.prompt_manager.create_reflection_prompt.return_value = "reflection prompt"

        self.response_parser.parse_validation_response.return_value = True
        self.response_parser.parse_critique_response.return_value = {
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Minor issue"],
            "suggestions": ["Minor suggestion"]
        }
        self.response_parser.parse_improvement_response.return_value = "Improved content"
        self.response_parser.parse_reflection_response.return_value = "Reflection content"

        self.memory_manager.get_memory.return_value = ["Previous reflection"]

        # Create a model with both sync and async methods
        self.model = MockLanguageModel()

        # Create the service
        self.service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )

    @pytest.mark.asyncio
    async def test_avalidate_success(self):
        """Test successful async validation."""
        result = await self.service.avalidate("Test text")
        assert result is True
        self.prompt_manager.create_validation_prompt.assert_called_once_with("Test text")
        self.response_parser.parse_validation_response.assert_called_once()
        # Check that the async method was called
        assert len(self.model.ainvoke_calls) == 1

    @pytest.mark.asyncio
    async def test_avalidate_empty_text(self):
        """Test async validation with empty text."""
        # Test with empty string
        with pytest.raises(ValueError):
            await self.service.avalidate("")

        # Test with None
        with pytest.raises(ValueError):
            await self.service.avalidate(None)

        # Test with whitespace
        with pytest.raises(ValueError):
            await self.service.avalidate("   ")

    @pytest.mark.asyncio
    async def test_avalidate_error(self):
        """Test async validation with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # The service should handle the exception and return False
        result = await service.avalidate("Test text")
        assert result is False

    @pytest.mark.asyncio
    async def test_avalidate_fallback(self):
        """Test async validation fallback to sync."""
        # Create a model that only has sync invoke
        model = SyncOnlyModel({"valid": True})

        service = CritiqueService(
            llm_provider=model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        result = await service.avalidate("Test text")
        assert result is True
        assert len(model.invoke_calls) == 1

    @pytest.mark.asyncio
    async def test_acritique_success(self):
        """Test successful async critique."""
        result = await self.service.acritique("Test text")
        assert result["score"] == 0.8
        assert result["feedback"] == "Good content"
        assert result["issues"] == ["Minor issue"]
        assert result["suggestions"] == ["Minor suggestion"]
        self.prompt_manager.create_critique_prompt.assert_called_once_with("Test text")
        self.response_parser.parse_critique_response.assert_called_once()
        # Check that the async method was called
        assert len(self.model.ainvoke_calls) == 1

    @pytest.mark.asyncio
    async def test_acritique_empty_text(self):
        """Test async critique with empty text."""
        # Test with empty string
        with pytest.raises(ValueError):
            await self.service.acritique("")

        # Test with None
        with pytest.raises(ValueError):
            await self.service.acritique(None)

        # Test with whitespace
        with pytest.raises(ValueError):
            await self.service.acritique("   ")

    @pytest.mark.asyncio
    async def test_acritique_error(self):
        """Test async critique with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # The service should handle the exception and return a default response
        result = await service.acritique("Test text")
        assert result["score"] == 0.0
        assert "Failed to critique text" in result["feedback"]
        assert result["issues"] == ["Failed to parse model response"]

    @pytest.mark.asyncio
    async def test_acritique_fallback(self):
        """Test async critique fallback to sync."""
        # Create a model that only has sync invoke
        model = SyncOnlyModel({"score": 0.8, "feedback": "Good content"})

        service = CritiqueService(
            llm_provider=model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        result = await service.acritique("Test text")
        assert result["score"] == 0.8
        assert len(model.invoke_calls) == 1

    @pytest.mark.asyncio
    async def test_aimprove_success(self):
        """Test successful async improve."""
        # Reset mock to clear previous calls
        self.model = MockLanguageModel({"improved_text": "Improved content"})

        # Create a new service with the fresh model
        service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            # No memory manager to avoid reflection call
        )

        result = await service.aimprove("Test text", "Test feedback")
        assert result == "Improved content"
        self.prompt_manager.create_improvement_prompt.assert_called_once()
        self.response_parser.parse_improvement_response.assert_called_once()
        # Check that the async method was called exactly once
        assert len(self.model.ainvoke_calls) == 1

    @pytest.mark.asyncio
    async def test_aimprove_with_violations(self):
        """Test async improve with violations list."""
        violations = [
            {"rule_name": "Rule 1", "message": "Message 1"},
            {"rule_name": "Rule 2", "message": "Message 2"},
        ]
        result = await self.service.aimprove("Test text", violations)
        assert result == "Improved content"

        # Check that the violations were converted to a string
        self.prompt_manager.create_improvement_prompt.assert_called_once()
        call_args = self.prompt_manager.create_improvement_prompt.call_args[0]
        assert call_args[0] == "Test text"
        assert "Rule 1: Message 1" in call_args[1]
        assert "Rule 2: Message 2" in call_args[1]

    @pytest.mark.asyncio
    async def test_aimprove_empty_text(self):
        """Test async improve with empty text."""
        # Test with empty string
        with pytest.raises(ValueError):
            await self.service.aimprove("", "Test feedback")

        # Test with None
        with pytest.raises(ValueError):
            await self.service.aimprove(None, "Test feedback")

        # Test with whitespace
        with pytest.raises(ValueError):
            await self.service.aimprove("   ", "Test feedback")

    @pytest.mark.asyncio
    async def test_aimprove_error(self):
        """Test async improve with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # The service should propagate the exception
        with pytest.raises(ValueError):
            await service.aimprove("Test text", "Test feedback")

    @pytest.mark.asyncio
    async def test_aimprove_fallback(self):
        """Test async improve fallback to sync."""
        # Create a model that only has sync invoke
        model = SyncOnlyModel({"improved_text": "Improved content"})

        service = CritiqueService(
            llm_provider=model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            # No memory manager to avoid reflection call
        )

        result = await service.aimprove("Test text", "Test feedback")
        assert result == "Improved content"
        assert len(model.invoke_calls) == 1

    @pytest.mark.asyncio
    async def test_generate_reflection_async(self):
        """Test async reflection generation."""
        await self.service._generate_reflection_async("Original", "Feedback", "Improved")
        self.prompt_manager.create_reflection_prompt.assert_called_once_with(
            "Original", "Feedback", "Improved"
        )
        self.response_parser.parse_reflection_response.assert_called_once()
        self.memory_manager.add_to_memory.assert_called_once_with("Reflection content")
        # Check that the async method was called
        assert len(self.model.ainvoke_calls) == 1

    @pytest.mark.asyncio
    async def test_generate_reflection_async_no_memory(self):
        """Test async reflection generation with no memory manager."""
        service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
        )

        # Should not raise an exception
        await service._generate_reflection_async("Original", "Feedback", "Improved")

        # Prompt manager should not be called
        self.prompt_manager.create_reflection_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_reflection_async_error(self):
        """Test async reflection generation with error."""
        # Create a model that raises an exception
        error_model = MockLanguageModel(error=True)
        service = CritiqueService(
            llm_provider=error_model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )

        # Should not raise an exception
        await service._generate_reflection_async("Original", "Feedback", "Improved")

        # Memory manager should not be called
        self.memory_manager.add_to_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_reflection_async_fallback(self):
        """Test async reflection generation fallback to sync."""
        # Create a model that only has sync invoke
        model = SyncOnlyModel({"reflection": "Reflection content"})

        service = CritiqueService(
            llm_provider=model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )

        await service._generate_reflection_async("Original", "Feedback", "Improved")
        assert len(model.invoke_calls) == 1
        self.memory_manager.add_to_memory.assert_called_once_with("Reflection content")


if __name__ == "__main__":
    unittest.main()