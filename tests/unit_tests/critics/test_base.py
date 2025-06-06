"""Comprehensive unit tests for critic base classes.

This module tests the base critic infrastructure:
- BaseCritic abstract base class
- Critic protocol compliance
- Critique result handling
- Error handling and logging
- Async/sync method coordination

Tests cover:
- BaseCritic implementation requirements
- Critique and improve method contracts
- Error handling scenarios
- Logging and performance tracking
- Protocol compliance verification
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from abc import ABC

from sifaka.critics.base import BaseCritic
from sifaka.core.thought import SifakaThought, CritiqueResult
from sifaka.core.interfaces import Critic
from sifaka.utils.errors import CritiqueError


class TestBaseCritic:
    """Test the BaseCritic abstract base class."""

    def test_base_critic_is_abstract(self):
        """Test that BaseCritic cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCritic("test", "description")

    def test_base_critic_subclass_requirements(self):
        """Test that BaseCritic subclasses must implement abstract methods."""
        class IncompleteCritic(BaseCritic):
            # Missing critique_async and improve_async
            pass
        
        with pytest.raises(TypeError):
            IncompleteCritic("test", "description")

    def test_base_critic_partial_implementation(self):
        """Test BaseCritic with only one abstract method implemented."""
        class PartialCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"feedback": "test", "suggestions": [], "needs_improvement": False}
            # Missing improve_async
        
        with pytest.raises(TypeError):
            PartialCritic("test", "description")

    def test_base_critic_complete_implementation(self):
        """Test a complete BaseCritic implementation."""
        class CompleteCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {
                    "feedback": "Test feedback",
                    "suggestions": ["suggestion1", "suggestion2"],
                    "needs_improvement": True
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return "Improved text based on critique"
        
        critic = CompleteCritic("test-critic", "Test description")
        
        assert critic.name == "test-critic"
        assert critic.description == "Test description"
        assert hasattr(critic, 'critique_async')
        assert hasattr(critic, 'improve_async')

    def test_base_critic_initialization(self):
        """Test BaseCritic initialization parameters."""
        class TestCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"feedback": "test", "suggestions": [], "needs_improvement": False}
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return "improved"
        
        # Test with description
        critic_with_desc = TestCritic("critic-name", "Detailed description")
        assert critic_with_desc.name == "critic-name"
        assert critic_with_desc.description == "Detailed description"
        
        # Test without description
        critic_no_desc = TestCritic("critic-name")
        assert critic_no_desc.name == "critic-name"
        assert critic_no_desc.description == ""

    @pytest.mark.asyncio
    async def test_base_critic_critique_contract(self):
        """Test that critique_async follows the expected contract."""
        class TestCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {
                    "feedback": "The text needs improvement",
                    "suggestions": ["Add more examples", "Improve clarity"],
                    "needs_improvement": True
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return "improved"
        
        critic = TestCritic("test-critic")
        thought = SifakaThought(prompt="Test", current_text="Test content")
        
        result = await critic.critique_async(thought)
        
        # Verify contract
        assert isinstance(result, dict)
        assert "feedback" in result
        assert "suggestions" in result
        assert "needs_improvement" in result
        assert isinstance(result["feedback"], str)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["needs_improvement"], bool)

    @pytest.mark.asyncio
    async def test_base_critic_improve_contract(self):
        """Test that improve_async follows the expected contract."""
        class TestCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"feedback": "test", "suggestions": [], "needs_improvement": False}
            
            async def improve_async(self, thought: SifakaThought) -> str:
                original_text = thought.current_text or thought.prompt
                return f"Improved: {original_text}"
        
        critic = TestCritic("test-critic")
        thought = SifakaThought(prompt="Test", current_text="Original text")
        
        result = await critic.improve_async(thought)
        
        # Verify contract
        assert isinstance(result, str)
        assert "Improved: Original text" == result

    @pytest.mark.asyncio
    async def test_base_critic_error_handling(self):
        """Test error handling in BaseCritic methods."""
        class ErrorCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                raise ValueError("Critique failed")
            
            async def improve_async(self, thought: SifakaThought) -> str:
                raise RuntimeError("Improvement failed")
        
        critic = ErrorCritic("error-critic")
        thought = SifakaThought(prompt="Test")
        
        # Errors should propagate
        with pytest.raises(ValueError, match="Critique failed"):
            await critic.critique_async(thought)
        
        with pytest.raises(RuntimeError, match="Improvement failed"):
            await critic.improve_async(thought)

    def test_base_critic_protocol_compliance(self):
        """Test that BaseCritic implementations satisfy Critic protocol."""
        class TestCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                return {"feedback": "test", "suggestions": [], "needs_improvement": False}
            
            async def improve_async(self, thought: SifakaThought) -> str:
                return "improved"
        
        critic = TestCritic("test-critic")
        
        # Should satisfy Critic protocol
        assert hasattr(critic, 'critique_async')
        assert hasattr(critic, 'improve_async')
        assert hasattr(critic, 'name')
        assert callable(critic.critique_async)
        assert callable(critic.improve_async)
        assert isinstance(critic.name, str)


class TestCriticIntegration:
    """Test integration scenarios with critics."""

    @pytest.mark.asyncio
    async def test_critic_with_thought_evolution(self):
        """Test critic behavior with evolving thoughts."""
        class EvolutionCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                # Analyze thought evolution
                generation_count = len(thought.generations)
                has_improvements = thought.iteration > 0
                
                if generation_count == 0:
                    return {
                        "feedback": "No generations yet",
                        "suggestions": ["Generate initial content"],
                        "needs_improvement": True
                    }
                elif not has_improvements:
                    return {
                        "feedback": "Initial generation looks good but could be improved",
                        "suggestions": ["Add more detail", "Improve structure"],
                        "needs_improvement": True
                    }
                else:
                    return {
                        "feedback": "Good evolution through iterations",
                        "suggestions": [],
                        "needs_improvement": False
                    }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                current = thought.current_text or thought.prompt
                return f"Enhanced: {current} [with additional detail and structure]"
        
        critic = EvolutionCritic("evolution-critic")
        
        # Test with new thought
        new_thought = SifakaThought(prompt="Test prompt")
        result = await critic.critique_async(new_thought)
        assert result["needs_improvement"] is True
        assert "No generations yet" in result["feedback"]
        
        # Test with evolved thought
        evolved_thought = SifakaThought(prompt="Test", iteration=1)
        evolved_thought.generations.append(Mock())  # Simulate generation
        result = await critic.critique_async(evolved_thought)
        assert "Good evolution" in result["feedback"]

    @pytest.mark.asyncio
    async def test_multiple_critics_workflow(self):
        """Test using multiple critics in sequence."""
        class GrammarCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                text = thought.current_text or thought.prompt
                has_errors = "teh" in text.lower()  # Simple grammar check
                
                return {
                    "feedback": "Grammar issues detected" if has_errors else "Grammar looks good",
                    "suggestions": ["Fix spelling errors"] if has_errors else [],
                    "needs_improvement": has_errors
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                text = thought.current_text or thought.prompt
                return text.replace("teh", "the")
        
        class ContentCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                text = thought.current_text or thought.prompt
                is_short = len(text) < 50
                
                return {
                    "feedback": "Content is too brief" if is_short else "Content length is good",
                    "suggestions": ["Add more detail", "Include examples"] if is_short else [],
                    "needs_improvement": is_short
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                text = thought.current_text or thought.prompt
                if len(text) < 50:
                    return f"{text} This has been expanded with additional detail and examples."
                return text
        
        grammar_critic = GrammarCritic("grammar-critic")
        content_critic = ContentCritic("content-critic")
        
        # Test with problematic text
        thought = SifakaThought(prompt="Test", current_text="Teh quick brown fox")
        
        # Grammar critique
        grammar_result = await grammar_critic.critique_async(thought)
        assert grammar_result["needs_improvement"] is True
        assert "Grammar issues" in grammar_result["feedback"]
        
        # Content critique
        content_result = await content_critic.critique_async(thought)
        assert content_result["needs_improvement"] is True
        assert "too brief" in content_result["feedback"]
        
        # Apply improvements
        improved_grammar = await grammar_critic.improve_async(thought)
        assert "The quick brown fox" in improved_grammar
        
        # Update thought and test content critic
        thought.current_text = improved_grammar
        improved_content = await content_critic.improve_async(thought)
        assert len(improved_content) > len(improved_grammar)

    @pytest.mark.asyncio
    async def test_critic_performance_tracking(self):
        """Test performance tracking in critic operations."""
        class SlowCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                await asyncio.sleep(0.01)  # Simulate processing time
                return {"feedback": "Slow critique", "suggestions": [], "needs_improvement": False}
            
            async def improve_async(self, thought: SifakaThought) -> str:
                await asyncio.sleep(0.01)  # Simulate processing time
                return "Slowly improved text"
        
        critic = SlowCritic("slow-critic")
        thought = SifakaThought(prompt="Test")
        
        # Time the critique operation
        start_time = datetime.now()
        await critic.critique_async(thought)
        critique_duration = (datetime.now() - start_time).total_seconds()
        
        # Time the improve operation
        start_time = datetime.now()
        await critic.improve_async(thought)
        improve_duration = (datetime.now() - start_time).total_seconds()
        
        # Verify operations took some time
        assert critique_duration > 0.005  # At least 5ms
        assert improve_duration > 0.005   # At least 5ms

    @pytest.mark.asyncio
    async def test_critic_with_complex_thought(self):
        """Test critic with a complex thought containing full audit trail."""
        class ComprehensiveCritic(BaseCritic):
            async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
                # Analyze complete thought state
                has_generations = len(thought.generations) > 0
                has_validations = len(thought.validations) > 0
                has_previous_critiques = len(thought.critiques) > 0
                
                feedback_parts = []
                suggestions = []
                
                if not has_generations:
                    feedback_parts.append("No generations found")
                    suggestions.append("Generate initial content")
                
                if not has_validations:
                    feedback_parts.append("No validation results")
                    suggestions.append("Run validators")
                
                if has_previous_critiques:
                    feedback_parts.append(f"Found {len(thought.critiques)} previous critiques")
                
                feedback = "; ".join(feedback_parts) if feedback_parts else "Comprehensive analysis complete"
                needs_improvement = len(suggestions) > 0
                
                return {
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "needs_improvement": needs_improvement
                }
            
            async def improve_async(self, thought: SifakaThought) -> str:
                base_text = thought.current_text or thought.prompt
                
                # Consider previous critiques
                if thought.critiques:
                    return f"[Revised based on {len(thought.critiques)} critiques] {base_text}"
                else:
                    return f"[Initial improvement] {base_text}"
        
        critic = ComprehensiveCritic("comprehensive-critic")
        
        # Create complex thought
        thought = SifakaThought(prompt="Complex test", current_text="Test content")
        
        # Add audit trail data
        thought.generations.append(Mock())
        thought.validations.append(Mock())
        thought.critiques.append(Mock())
        
        # Test critique
        result = await critic.critique_async(thought)
        assert "Found 1 previous critiques" in result["feedback"]
        assert result["needs_improvement"] is False  # Has all components
        
        # Test improvement
        improved = await critic.improve_async(thought)
        assert "[Revised based on 1 critiques]" in improved
