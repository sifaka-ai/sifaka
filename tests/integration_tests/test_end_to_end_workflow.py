"""End-to-end integration tests for Sifaka workflow.

This module provides comprehensive integration testing of the complete Sifaka
workflow from prompt to final output:
- Complete workflow execution with real components
- Multi-iteration improvement cycles
- Conversation continuity and management
- Error handling and recovery scenarios
- Performance characteristics under load
- Real-world usage patterns

Tests cover:
- Single thought processing
- Conversation chains
- Batch processing
- Error scenarios and recovery
- Performance under various conditions
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from sifaka import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.utils.errors import GraphExecutionError


class MockPydanticAgent:
    """Mock PydanticAI agent for testing."""

    def __init__(self, model_name: str = "mock:gpt-4"):
        self.model = model_name
        self.call_count = 0
        self.responses = [
            "This is a first draft response that might need improvement.",
            "This is an improved response with better structure and more detail.",
            "This is the final polished response with excellent quality.",
        ]

    async def run_async(self, prompt: str, message_history: List[Dict] = None) -> Mock:
        """Mock agent execution."""
        self.call_count += 1

        # Simulate some processing time
        await asyncio.sleep(0.01)

        # Return different responses for iterations
        response_index = min(self.call_count - 1, len(self.responses) - 1)
        response_text = self.responses[response_index]

        # Create mock result
        result = Mock()
        result.data = response_text
        result.all_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]

        return result


class MockValidator:
    """Mock validator for testing."""

    def __init__(self, name: str, pass_threshold: int = 2):
        self.name = name
        self.pass_threshold = pass_threshold
        self.call_count = 0

    async def validate_async(self, text: str) -> Dict[str, Any]:
        """Mock validation that passes after threshold iterations."""
        self.call_count += 1

        # Simulate validation logic
        passed = self.call_count >= self.pass_threshold
        score = 0.9 if passed else 0.5

        return {
            "passed": passed,
            "score": score,
            "details": {"validation_count": self.call_count, "threshold": self.pass_threshold},
        }


class MockCritic:
    """Mock critic for testing."""

    def __init__(self, name: str, improvement_threshold: int = 2):
        self.name = name
        self.improvement_threshold = improvement_threshold
        self.call_count = 0

    async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
        """Mock critique that suggests improvement until threshold."""
        self.call_count += 1

        needs_improvement = self.call_count < self.improvement_threshold

        if needs_improvement:
            return {
                "feedback": f"Iteration {self.call_count}: Needs more work",
                "suggestions": ["Add more detail", "Improve structure"],
                "needs_improvement": True,
            }
        else:
            return {
                "feedback": f"Iteration {self.call_count}: Excellent work",
                "suggestions": [],
                "needs_improvement": False,
            }

    async def improve_async(self, thought: SifakaThought) -> str:
        """Mock improvement generation."""
        return f"Improved text based on {self.name} critique"


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    # Create mock agent
    mock_agent = MockPydanticAgent()

    # Create mock validators
    validators = [
        MockValidator("length", pass_threshold=2),
        MockValidator("coherence", pass_threshold=1),
    ]

    # Create mock critics
    critics = {
        "constitutional": MockCritic("constitutional", improvement_threshold=2),
        "reflexion": MockCritic("reflexion", improvement_threshold=3),
    }

    # Create dependencies
    deps = SifakaDependencies(
        generator_agent=mock_agent, validators=validators, critics=critics, retrievers={}
    )

    return deps


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow execution."""

    @pytest.mark.asyncio
    async def test_single_thought_complete_workflow(self, mock_dependencies):
        """Test complete workflow for a single thought."""
        engine = SifakaEngine(dependencies=mock_dependencies)

        # Process a thought
        thought = await engine.think("Explain the benefits of renewable energy", max_iterations=3)

        # Verify workflow completion
        assert thought is not None
        assert thought.prompt == "Explain the benefits of renewable energy"
        assert thought.final_text is not None
        assert len(thought.final_text) > 0

        # Verify audit trail
        assert len(thought.generations) > 0
        assert len(thought.validations) > 0
        assert len(thought.critiques) > 0

        # Verify iterations occurred
        assert thought.iteration > 0
        assert thought.iteration <= 3

        # Verify final state
        assert thought.validation_passed() is True
        assert thought.needs_improvement() is False

    @pytest.mark.asyncio
    async def test_conversation_workflow(self, mock_dependencies):
        """Test conversation workflow with multiple related thoughts."""
        engine = SifakaEngine(dependencies=mock_dependencies)

        # Start conversation
        first_thought = await engine.think("What is renewable energy?")

        # Continue conversation
        second_thought = await engine.continue_thought(
            first_thought, "How does solar power work specifically?"
        )

        # Continue again
        third_thought = await engine.continue_thought(second_thought, "What are the cost benefits?")

        # Verify conversation structure
        assert first_thought.parent_thought_id is None
        assert second_thought.parent_thought_id == first_thought.id
        assert third_thought.parent_thought_id == second_thought.id

        # Verify conversation continuity
        conv_id = first_thought.conversation_id or first_thought.id
        assert second_thought.conversation_id == conv_id
        assert third_thought.conversation_id == conv_id

        # All thoughts should be complete
        for thought in [first_thought, second_thought, third_thought]:
            assert thought.final_text is not None
            assert len(thought.generations) > 0

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, mock_dependencies):
        """Test batch processing of multiple thoughts."""
        engine = SifakaEngine(dependencies=mock_dependencies)

        prompts = [
            "Explain solar energy",
            "Describe wind power",
            "What is hydroelectric energy?",
            "How does geothermal energy work?",
            "What are the benefits of nuclear power?",
        ]

        # Process batch
        start_time = time.time()
        thoughts = await engine.batch_think(prompts, max_iterations=2)
        end_time = time.time()

        # Verify results
        assert len(thoughts) == len(prompts)

        # All thoughts should be complete
        for i, thought in enumerate(thoughts):
            assert thought.prompt == prompts[i]
            assert thought.final_text is not None
            assert len(thought.generations) > 0

        # Should complete in reasonable time (parallel processing)
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_workflow_with_validation_failures(self, mock_dependencies):
        """Test workflow behavior when validations initially fail."""
        # Create validator that fails first few times
        strict_validator = MockValidator("strict", pass_threshold=3)
        mock_dependencies.validators = [strict_validator]

        engine = SifakaEngine(dependencies=mock_dependencies)

        thought = await engine.think("Write a brief explanation", max_iterations=5)

        # Should eventually pass after iterations
        assert thought.validation_passed() is True
        assert thought.iteration >= 3  # Should take at least 3 iterations
        assert len(thought.validations) >= 3

    @pytest.mark.asyncio
    async def test_workflow_with_critic_feedback(self, mock_dependencies):
        """Test workflow behavior with critic feedback driving iterations."""
        # Create critic that provides feedback for several iterations
        demanding_critic = MockCritic("demanding", improvement_threshold=4)
        mock_dependencies.critics = {"demanding": demanding_critic}

        engine = SifakaEngine(dependencies=mock_dependencies)

        thought = await engine.think("Explain a complex topic", max_iterations=5)

        # Should iterate based on critic feedback
        assert thought.needs_improvement() is False
        assert thought.iteration >= 3  # Should take multiple iterations
        assert len(thought.critiques) >= 3

    @pytest.mark.asyncio
    async def test_workflow_max_iterations_limit(self, mock_dependencies):
        """Test that workflow respects max iterations limit."""
        # Create validators and critics that never pass
        never_pass_validator = MockValidator("never_pass", pass_threshold=100)
        never_satisfied_critic = MockCritic("never_satisfied", improvement_threshold=100)

        mock_dependencies.validators = [never_pass_validator]
        mock_dependencies.critics = {"never_satisfied": never_satisfied_critic}

        engine = SifakaEngine(dependencies=mock_dependencies)

        thought = await engine.think("This will hit max iterations", max_iterations=3)

        # Should stop at max iterations even if not perfect
        assert thought.iteration == 3
        assert thought.final_text is not None  # Should still produce output

        # May not pass all validations due to limit
        assert len(thought.generations) == 3  # One per iteration

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, mock_dependencies):
        """Test workflow error handling and recovery."""
        # Create agent that fails on first call
        failing_agent = MockPydanticAgent()
        original_run = failing_agent.run_async

        async def failing_run(*args, **kwargs):
            if failing_agent.call_count == 0:
                failing_agent.call_count += 1
                raise Exception("Simulated generation failure")
            return await original_run(*args, **kwargs)

        failing_agent.run_async = failing_run
        mock_dependencies.generator_agent = failing_agent

        engine = SifakaEngine(dependencies=mock_dependencies)

        # Should handle error gracefully
        with pytest.raises(GraphExecutionError):
            await engine.think("This will fail initially")

    @pytest.mark.asyncio
    async def test_workflow_performance_characteristics(self, mock_dependencies):
        """Test workflow performance under various conditions."""
        engine = SifakaEngine(dependencies=mock_dependencies)

        # Test with different prompt lengths
        short_prompt = "Brief explanation"
        long_prompt = "Provide a comprehensive, detailed explanation " * 20

        # Process both
        start_time = time.time()
        short_thought = await engine.think(short_prompt, max_iterations=2)
        short_time = time.time() - start_time

        start_time = time.time()
        long_thought = await engine.think(long_prompt, max_iterations=2)
        long_time = time.time() - start_time

        # Both should complete successfully
        assert short_thought.final_text is not None
        assert long_thought.final_text is not None

        # Performance should be reasonable
        assert short_time < 5.0
        assert long_time < 10.0

    @pytest.mark.asyncio
    async def test_workflow_memory_efficiency(self, mock_dependencies):
        """Test workflow memory efficiency with multiple thoughts."""
        import gc
        import sys

        engine = SifakaEngine(dependencies=mock_dependencies)

        # Process many thoughts to test memory usage
        thoughts = []
        for i in range(20):
            thought = await engine.think(f"Prompt {i}", max_iterations=1)
            thoughts.append(thought)

            # Force garbage collection periodically
            if i % 5 == 0:
                gc.collect()

        # All thoughts should be complete
        assert len(thoughts) == 20
        for thought in thoughts:
            assert thought.final_text is not None

        # Memory usage should be reasonable
        # (This is a basic check - more sophisticated memory profiling could be added)
        total_size = sum(sys.getsizeof(thought) for thought in thoughts)
        assert total_size < 1024 * 1024  # Less than 1MB for 20 thoughts

    @pytest.mark.asyncio
    async def test_workflow_state_consistency(self, mock_dependencies):
        """Test that workflow maintains consistent state throughout execution."""
        engine = SifakaEngine(dependencies=mock_dependencies)

        thought = await engine.think("Test state consistency", max_iterations=3)

        # Verify state consistency
        assert thought.id is not None
        assert len(thought.id) > 0

        # Timestamps should be consistent
        assert thought.created_at <= thought.updated_at

        # Audit trail should be complete and ordered
        for i, generation in enumerate(thought.generations):
            assert generation.iteration == i
            if i > 0:
                assert generation.timestamp >= thought.generations[i - 1].timestamp

        # Validation and critique counts should make sense
        assert len(thought.validations) >= len(thought.generations)
        assert len(thought.critiques) >= len(thought.generations)

        # Final state should be consistent
        if thought.final_text:
            assert thought.final_text == thought.generations[-1].text
