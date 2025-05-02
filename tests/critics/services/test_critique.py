"""
Tests for the critics CritiqueService class.

This module contains comprehensive tests for the CritiqueService class that
is responsible for critiquing, validating, and improving text using language models.

Tests cover all public methods including validate, critique, improve, and their
asynchronous counterparts, as well as the private helper methods. Both normal
and edge cases are tested.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from typing import Dict, List, Any, Optional

from sifaka.critics.services.critique import CritiqueService
from sifaka.critics.managers.memory import MemoryManager
from sifaka.critics.managers.prompt import PromptManager
from sifaka.critics.managers.response import ResponseParser


class TestCritiqueService(unittest.TestCase):
    """Tests for the CritiqueService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.llm_provider = MagicMock()
        self.prompt_manager = MagicMock(spec=PromptManager)
        self.response_parser = MagicMock(spec=ResponseParser)
        self.memory_manager = MagicMock(spec=MemoryManager)

        # Create the service
        self.service = CritiqueService(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager
        )

        # Set up common prompt responses
        self.prompt_manager.create_validation_prompt.return_value = "validation prompt"
        self.prompt_manager.create_critique_prompt.return_value = "critique prompt"
        self.prompt_manager.create_improvement_prompt.return_value = "improvement prompt"
        self.prompt_manager.create_reflection_prompt.return_value = "reflection prompt"

        # Set up common model responses
        self.llm_provider.invoke.return_value = "model response"

        # Set up common parser responses
        self.response_parser.parse_validation_response.return_value = True
        self.response_parser.parse_critique_response.return_value = {
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Minor issue"],
            "suggestions": ["Suggestion"]
        }
        self.response_parser.parse_improvement_response.return_value = "Improved text"
        self.response_parser.parse_reflection_response.return_value = "Reflection"

    def test_initialization(self):
        """Test initialization with all parameters."""
        self.assertEqual(self.service._model, self.llm_provider)
        self.assertEqual(self.service._prompt_manager, self.prompt_manager)
        self.assertEqual(self.service._response_parser, self.response_parser)
        self.assertEqual(self.service._memory_manager, self.memory_manager)

    def test_initialization_without_memory_manager(self):
        """Test initialization without memory manager."""
        service = CritiqueService(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser
        )
        self.assertEqual(service._model, self.llm_provider)
        self.assertEqual(service._prompt_manager, self.prompt_manager)
        self.assertEqual(service._response_parser, self.response_parser)
        self.assertIsNone(service._memory_manager)

    def test_validate_success(self):
        """Test validate method with successful validation."""
        result = self.service.validate("Test text")

        # Verify result
        self.assertTrue(result)

        # Verify interactions
        self.prompt_manager.create_validation_prompt.assert_called_once_with("Test text")
        self.llm_provider.invoke.assert_called_once_with("validation prompt")
        self.response_parser.parse_validation_response.assert_called_once_with("model response")

    def test_validate_empty_text(self):
        """Test validate method with empty text."""
        with self.assertRaises(ValueError):
            self.service.validate("")

        with self.assertRaises(ValueError):
            self.service.validate("   ")

        with self.assertRaises(ValueError):
            self.service.validate(None)

    def test_validate_model_exception(self):
        """Test validate method when model raises an exception."""
        # Set up model to raise an exception
        self.llm_provider.invoke.side_effect = Exception("Model error")

        # Validate should not propagate the exception
        result = self.service.validate("Test text")

        # Should return False on error
        self.assertFalse(result)

    def test_critique_success(self):
        """Test critique method with successful critique."""
        result = self.service.critique("Test text")

        # Verify result
        self.assertEqual(result, {
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Minor issue"],
            "suggestions": ["Suggestion"]
        })

        # Verify interactions
        self.prompt_manager.create_critique_prompt.assert_called_once_with("Test text")
        self.llm_provider.invoke.assert_called_once_with("critique prompt")
        self.response_parser.parse_critique_response.assert_called_once_with("model response")

    def test_critique_empty_text(self):
        """Test critique method with empty text."""
        with self.assertRaises(ValueError):
            self.service.critique("")

        with self.assertRaises(ValueError):
            self.service.critique("   ")

        with self.assertRaises(ValueError):
            self.service.critique(None)

    def test_critique_model_exception(self):
        """Test critique method when model raises an exception."""
        # Set up model to raise an exception
        self.llm_provider.invoke.side_effect = Exception("Model error")

        # Critique should not propagate the exception
        result = self.service.critique("Test text")

        # Should return an error response
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Failed to critique text", result["feedback"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(len(result["suggestions"]), 1)

    def test_improve_with_string_feedback(self):
        """Test improve method with string feedback."""
        # Note: The improve method also generates a reflection, so multiple calls to invoke are expected
        result = self.service.improve("Test text", "Feedback")

        # Verify result
        self.assertEqual(result, "Improved text")

        # Verify interactions - don't check exact number of calls since reflection generation is done
        self.memory_manager.get_memory.assert_called_once()
        self.prompt_manager.create_improvement_prompt.assert_called_once()

        # Check that invoke was called with the improvement prompt (among other calls)
        self.llm_provider.invoke.assert_any_call("improvement prompt")
        self.response_parser.parse_improvement_response.assert_called_once_with("model response")

        # Verify reflection was also generated
        self.prompt_manager.create_reflection_prompt.assert_called_once()
        self.llm_provider.invoke.assert_any_call("reflection prompt")
        self.response_parser.parse_reflection_response.assert_called_once()
        self.memory_manager.add_to_memory.assert_called_once_with("Reflection")

    def test_improve_with_violations_feedback(self):
        """Test improve method with violations list as feedback."""
        # Create a test violations list
        violations = [
            {"rule_name": "Grammar", "message": "Fix grammar"},
            {"rule_name": "Style", "message": "Improve style"}
        ]

        # Mock the _violations_to_feedback method
        with patch.object(self.service, '_violations_to_feedback', return_value="Formatted feedback") as mock_format:
            result = self.service.improve("Test text", violations)

            # Verify _violations_to_feedback was called
            mock_format.assert_called_once_with(violations)

            # Verify other interactions
            self.memory_manager.get_memory.assert_called_once()
            self.prompt_manager.create_improvement_prompt.assert_called_once()

            # Verify the result
            self.assertEqual(result, "Improved text")

    def test_improve_no_memory_manager(self):
        """Test improve method without memory manager."""
        # Create a service without memory manager
        service = CritiqueService(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser
        )

        # Should work without memory manager
        result = service.improve("Test text", "Feedback")

        # Verify result
        self.assertEqual(result, "Improved text")

        # Create improvement prompt should be called without reflections
        self.prompt_manager.create_improvement_prompt.assert_called_with("Test text", "Feedback", None)

    def test_improve_empty_text(self):
        """Test improve method with empty text."""
        with self.assertRaises(ValueError):
            self.service.improve("", "Feedback")

        with self.assertRaises(ValueError):
            self.service.improve("   ", "Feedback")

        with self.assertRaises(ValueError):
            self.service.improve(None, "Feedback")

    def test_improve_model_exception(self):
        """Test improve method when model raises an exception."""
        # Set up model to raise an exception
        self.llm_provider.invoke.side_effect = Exception("Model error")

        # Improve should propagate the exception
        with self.assertRaises(ValueError) as context:
            self.service.improve("Test text", "Feedback")

        # Should include the original error
        self.assertIn("Failed to improve text", str(context.exception))
        self.assertIn("Model error", str(context.exception))

    def test_violations_to_feedback(self):
        """Test _violations_to_feedback method."""
        # Empty violations
        self.assertEqual(
            self.service._violations_to_feedback([]),
            "No issues found."
        )

        # Single violation
        violations = [{"rule_name": "Grammar", "message": "Fix grammar"}]
        self.assertEqual(
            self.service._violations_to_feedback(violations),
            "The following issues were found:\n- Grammar: Fix grammar\n"
        )

        # Multiple violations
        violations = [
            {"rule_name": "Grammar", "message": "Fix grammar"},
            {"rule_name": "Style", "message": "Improve style"}
        ]
        expected = "The following issues were found:\n- Grammar: Fix grammar\n- Style: Improve style\n"
        self.assertEqual(self.service._violations_to_feedback(violations), expected)

        # Violations with missing keys
        violations = [
            {"rule_name": "Grammar", "other_key": "value"},  # Missing message
            {"message": "Improve style"}  # Missing rule_name
        ]
        expected = "The following issues were found:\n- Grammar: Unknown issue\n- Rule 2: Improve style\n"
        self.assertEqual(self.service._violations_to_feedback(violations), expected)

    def test_generate_reflection(self):
        """Test _generate_reflection method."""
        # Call the method
        self.service._generate_reflection(
            original_text="Original",
            feedback="Feedback",
            improved_text="Improved"
        )

        # Verify interactions
        self.prompt_manager.create_reflection_prompt.assert_called_once_with(
            "Original", "Feedback", "Improved"
        )
        self.llm_provider.invoke.assert_called_once_with("reflection prompt")
        self.response_parser.parse_reflection_response.assert_called_once_with("model response")
        self.memory_manager.add_to_memory.assert_called_once_with("Reflection")

    def test_generate_reflection_no_memory_manager(self):
        """Test _generate_reflection method without memory manager."""
        # Create a service without memory manager
        service = CritiqueService(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser
        )

        # Should do nothing without memory manager
        service._generate_reflection(
            original_text="Original",
            feedback="Feedback",
            improved_text="Improved"
        )

        # No interactions should happen
        self.prompt_manager.create_reflection_prompt.assert_not_called()
        self.llm_provider.invoke.assert_not_called()

    def test_generate_reflection_model_exception(self):
        """Test _generate_reflection method when model raises an exception."""
        # Set up model to raise an exception
        self.llm_provider.invoke.side_effect = Exception("Model error")

        # Should not propagate the exception
        self.service._generate_reflection(
            original_text="Original",
            feedback="Feedback",
            improved_text="Improved"
        )

        # Verify interactions
        self.prompt_manager.create_reflection_prompt.assert_called_once()
        self.llm_provider.invoke.assert_called_once()

        # Should not add anything to memory
        self.memory_manager.add_to_memory.assert_not_called()

    def test_generate_reflection_empty_result(self):
        """Test _generate_reflection method when parser returns None."""
        # Set up parser to return None
        self.response_parser.parse_reflection_response.return_value = None

        # Should not add anything to memory
        self.service._generate_reflection(
            original_text="Original",
            feedback="Feedback",
            improved_text="Improved"
        )

        # Verify interactions
        self.prompt_manager.create_reflection_prompt.assert_called_once()
        self.llm_provider.invoke.assert_called_once()
        self.response_parser.parse_reflection_response.assert_called_once()

        # Should not add anything to memory
        self.memory_manager.add_to_memory.assert_not_called()


class TestCritiqueServiceAsync:
    """Tests for asynchronous methods of CritiqueService."""

    @pytest.fixture
    async def service(self):
        """Create a service for testing."""
        # Create mock objects
        llm_provider = MagicMock()
        llm_provider.ainvoke = AsyncMock(return_value="model response")
        llm_provider.invoke = MagicMock(return_value="sync model response")

        prompt_manager = MagicMock(spec=PromptManager)
        prompt_manager.create_validation_prompt = MagicMock(return_value="validation prompt")
        prompt_manager.create_critique_prompt = MagicMock(return_value="critique prompt")
        prompt_manager.create_improvement_prompt = MagicMock(return_value="improvement prompt")
        prompt_manager.create_reflection_prompt = MagicMock(return_value="reflection prompt")

        response_parser = MagicMock(spec=ResponseParser)
        response_parser.parse_validation_response = MagicMock(return_value=True)
        response_parser.parse_critique_response = MagicMock(return_value={
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Minor issue"],
            "suggestions": ["Suggestion"]
        })
        response_parser.parse_improvement_response = MagicMock(return_value="Improved text")
        response_parser.parse_reflection_response = MagicMock(return_value="Reflection")

        memory_manager = MagicMock(spec=MemoryManager)
        memory_manager.get_memory = MagicMock(return_value=["Previous reflection"])
        memory_manager.add_to_memory = MagicMock()

        service = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            response_parser=response_parser,
            memory_manager=memory_manager
        )

        return service

    @pytest.mark.asyncio
    async def test_avalidate_success(self, service):
        """Test avalidate method with successful validation."""
        result = await service.avalidate("Test text")

        # Verify result
        assert result is True

        # Verify interactions
        service._prompt_manager.create_validation_prompt.assert_called_once_with("Test text")
        service._model.ainvoke.assert_called_once_with("validation prompt")
        service._response_parser.parse_validation_response.assert_called_once_with("model response")

    @pytest.mark.asyncio
    async def test_avalidate_empty_text(self, service):
        """Test avalidate method with empty text."""
        with pytest.raises(ValueError):
            await service.avalidate("")

        with pytest.raises(ValueError):
            await service.avalidate("   ")

        with pytest.raises(ValueError):
            await service.avalidate(None)

    @pytest.mark.asyncio
    async def test_avalidate_model_exception(self, service):
        """Test avalidate method when model raises an exception."""
        # Set up model to raise an exception
        service._model.ainvoke.side_effect = Exception("Model error")

        # avalidate should not propagate the exception
        result = await service.avalidate("Test text")

        # Should return False on error
        assert result is False

    @pytest.mark.asyncio
    async def test_avalidate_fallback_to_sync(self, service):
        """Test avalidate falling back to synchronous invoke if ainvoke is not available."""
        # Remove the ainvoke method
        del service._model.ainvoke

        # Should fall back to synchronous invoke
        result = await service.avalidate("Test text")

        # Verify result
        assert result is True

        # Verify that synchronous invoke was called
        service._model.invoke.assert_called_once_with("validation prompt")

    @pytest.mark.asyncio
    async def test_acritique_success(self, service):
        """Test acritique method with successful critique."""
        result = await service.acritique("Test text")

        # Verify result
        assert result == {
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Minor issue"],
            "suggestions": ["Suggestion"]
        }

        # Verify interactions
        service._prompt_manager.create_critique_prompt.assert_called_once_with("Test text")
        service._model.ainvoke.assert_called_once_with("critique prompt")
        service._response_parser.parse_critique_response.assert_called_once_with("model response")

    @pytest.mark.asyncio
    async def test_acritique_empty_text(self, service):
        """Test acritique method with empty text."""
        with pytest.raises(ValueError):
            await service.acritique("")

        with pytest.raises(ValueError):
            await service.acritique("   ")

        with pytest.raises(ValueError):
            await service.acritique(None)

    @pytest.mark.asyncio
    async def test_acritique_model_exception(self, service):
        """Test acritique method when model raises an exception."""
        # Set up model to raise an exception
        service._model.ainvoke.side_effect = Exception("Model error")

        # acritique should not propagate the exception
        result = await service.acritique("Test text")

        # Should return an error response
        assert result["score"] == 0.0
        assert "Failed to critique text" in result["feedback"]
        assert len(result["issues"]) == 1
        assert len(result["suggestions"]) == 1

    @pytest.mark.asyncio
    async def test_aimprove_with_string_feedback(self, service):
        """Test aimprove method with string feedback."""
        result = await service.aimprove("Test text", "Feedback")

        # Verify result
        assert result == "Improved text"

        # Verify interactions
        service._memory_manager.get_memory.assert_called_once()
        service._prompt_manager.create_improvement_prompt.assert_called_once()
        service._model.ainvoke.assert_any_call("improvement prompt")
        service._response_parser.parse_improvement_response.assert_called_once_with("model response")

    @pytest.mark.asyncio
    async def test_aimprove_with_violations_feedback(self, service):
        """Test aimprove method with violations list as feedback."""
        # Create a test violations list
        violations = [
            {"rule_name": "Grammar", "message": "Fix grammar"},
            {"rule_name": "Style", "message": "Improve style"}
        ]

        # Mock the _violations_to_feedback method
        with patch.object(service, '_violations_to_feedback', return_value="Formatted feedback") as mock_format:
            result = await service.aimprove("Test text", violations)

            # Verify _violations_to_feedback was called
            mock_format.assert_called_once_with(violations)

            # Verify other interactions
            service._memory_manager.get_memory.assert_called_once()
            service._prompt_manager.create_improvement_prompt.assert_called_once()

            # Verify the result
            assert result == "Improved text"

    @pytest.mark.asyncio
    async def test_generate_reflection_async(self, service):
        """Test _generate_reflection_async method."""
        # Call the method
        await service._generate_reflection_async(
            original_text="Original",
            feedback="Feedback",
            improved_text="Improved"
        )

        # Verify interactions
        service._prompt_manager.create_reflection_prompt.assert_called_once_with(
            "Original", "Feedback", "Improved"
        )
        service._model.ainvoke.assert_called_once_with("reflection prompt")
        service._response_parser.parse_reflection_response.assert_called_once_with("model response")
        service._memory_manager.add_to_memory.assert_called_once_with("Reflection")

    @pytest.mark.asyncio
    async def test_generate_reflection_async_model_exception(self, service):
        """Test _generate_reflection_async method when model raises an exception."""
        # Set up model to raise an exception
        service._model.ainvoke.side_effect = Exception("Model error")

        # Should not propagate the exception
        await service._generate_reflection_async(
            original_text="Original",
            feedback="Feedback",
            improved_text="Improved"
        )

        # Verify interactions
        service._prompt_manager.create_reflection_prompt.assert_called_once()
        service._model.ainvoke.assert_called_once()

        # Should not add anything to memory
        service._memory_manager.add_to_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()