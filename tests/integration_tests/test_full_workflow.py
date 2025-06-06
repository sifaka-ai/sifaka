"""Comprehensive integration tests for full Sifaka workflow.

This module tests the complete Sifaka workflow end-to-end:
- SifakaEngine with real components
- Full thought processing pipeline
- Integration between all components
- Error handling and recovery
- Performance characteristics

Tests cover:
- Complete thought workflows
- Multi-iteration processing
- Conversation continuity
- Storage integration
- Mock-based testing without external APIs
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from sifaka import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.storage.memory import MemoryStorage


class TestFullWorkflow:
    """Test complete Sifaka workflows."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create comprehensive mock dependencies."""
        deps = Mock(spec=SifakaDependencies)
        
        # Mock generator agent
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.data = "Generated text about renewable energy benefits and challenges."
        mock_result.new_messages.return_value = []
        mock_result.cost = 0.05
        mock_result.usage = Mock()
        mock_result.usage.model_dump.return_value = {"total_tokens": 50}
        mock_agent.run.return_value = mock_result
        deps.generator = mock_agent
        
        # Mock validators
        mock_validator1 = AsyncMock()
        mock_validator1.name = "length-validator"
        mock_validator1.validate_async.return_value = {
            "passed": True,
            "details": {"word_count": 45, "character_count": 250},
            "score": 0.9
        }
        
        mock_validator2 = AsyncMock()
        mock_validator2.name = "content-validator"
        mock_validator2.validate_async.return_value = {
            "passed": False,
            "details": {"issue": "lacks specific examples"},
            "score": 0.6
        }
        
        deps.validators = [mock_validator1, mock_validator2]
        
        # Mock critics
        mock_critic = AsyncMock()
        mock_critic.name = "constitutional-critic"
        mock_critic.critique_async.return_value = {
            "feedback": "The text needs more balanced perspective and specific examples",
            "suggestions": ["Add counterarguments", "Include specific examples", "Cite sources"],
            "needs_improvement": True,
            "confidence": 0.8
        }
        mock_critic.improve_async.return_value = "Improved text with balanced perspective, specific examples, and proper citations."
        
        deps.critics = {"constitutional": mock_critic}
        deps.always_apply_critics = False
        deps.never_apply_critics = False
        
        # Mock retrievers (empty for basic test)
        deps.retrievers = {}
        
        return deps

    @pytest.fixture
    def memory_storage(self):
        """Create memory storage for testing."""
        return MemoryStorage()

    @pytest.mark.asyncio
    async def test_basic_thought_processing(self, mock_dependencies):
        """Test basic thought processing workflow."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Process a thought
        thought = await engine.think("Explain the benefits of renewable energy")
        
        # Verify thought structure
        assert isinstance(thought, SifakaThought)
        assert thought.prompt == "Explain the benefits of renewable energy"
        assert thought.current_text is not None
        assert len(thought.current_text) > 0
        
        # Verify audit trail
        assert len(thought.generations) >= 1
        assert len(thought.validations) >= 1
        assert len(thought.critiques) >= 0  # Depends on validation results
        
        # Verify final state
        assert thought.final_text is not None
        assert thought.iteration >= 0

    @pytest.mark.asyncio
    async def test_multi_iteration_workflow(self, mock_dependencies):
        """Test multi-iteration improvement workflow."""
        # Configure for multiple iterations
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Process with multiple iterations allowed
        thought = await engine.think("Explain renewable energy", max_iterations=3)
        
        # Should have processed multiple iterations due to failed validation
        assert thought.iteration >= 1
        assert len(thought.generations) >= 2
        assert len(thought.validations) >= 2
        assert len(thought.critiques) >= 1
        
        # Verify improvement over iterations
        first_generation = thought.generations[0].text
        final_generation = thought.generations[-1].text
        
        # Final should be different (improved)
        assert first_generation != final_generation
        assert len(final_generation) >= len(first_generation)

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, mock_dependencies):
        """Test conversation continuity between thoughts."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # First thought
        thought1 = await engine.think("What are the main types of renewable energy?")
        
        # Continue conversation
        thought2 = await engine.continue_thought(
            thought1,
            "Tell me more about solar energy specifically"
        )
        
        # Verify conversation connectivity
        assert thought2.parent_thought_id == thought1.id
        assert thought2.conversation_id == thought1.id  # First thought sets conversation ID
        assert thought1.child_thought_ids == [thought2.id]
        
        # Third thought in conversation
        thought3 = await engine.continue_thought(
            thought2,
            "What about the costs of solar installation?"
        )
        
        # Verify conversation chain
        assert thought3.parent_thought_id == thought2.id
        assert thought3.conversation_id == thought1.id  # Same conversation
        assert thought2.child_thought_ids == [thought3.id]

    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_dependencies):
        """Test batch processing of multiple thoughts."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        prompts = [
            "Explain solar energy",
            "Describe wind power",
            "What is hydroelectric energy?",
            "How does geothermal energy work?"
        ]
        
        # Process batch
        thoughts = await engine.batch_think(prompts, max_iterations=2)
        
        # Verify all thoughts processed
        assert len(thoughts) == len(prompts)
        
        for i, thought in enumerate(thoughts):
            assert thought.prompt == prompts[i]
            assert thought.final_text is not None
            assert len(thought.generations) >= 1
            
        # Verify thoughts are independent
        thought_ids = {t.id for t in thoughts}
        assert len(thought_ids) == len(thoughts)  # All unique

    @pytest.mark.asyncio
    async def test_storage_integration(self, mock_dependencies, memory_storage):
        """Test integration with storage backend."""
        # Create engine with storage
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Process thought
        thought = await engine.think("Test storage integration")
        
        # Store thought
        await memory_storage.store_thought(thought)
        
        # Retrieve and verify
        retrieved = await memory_storage.retrieve_thought(thought.id)
        
        assert retrieved is not None
        assert retrieved.id == thought.id
        assert retrieved.prompt == thought.prompt
        assert retrieved.final_text == thought.final_text
        assert len(retrieved.generations) == len(thought.generations)
        assert len(retrieved.validations) == len(thought.validations)

    @pytest.mark.asyncio
    async def test_error_handling_generator_failure(self, mock_dependencies):
        """Test error handling when generator fails."""
        # Make generator fail
        mock_dependencies.generator.run.side_effect = Exception("Generator API error")
        
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await engine.think("Test generator failure")

    @pytest.mark.asyncio
    async def test_error_handling_validator_failure(self, mock_dependencies):
        """Test error handling when validator fails."""
        # Make one validator fail
        mock_dependencies.validators[0].validate_async.side_effect = Exception("Validator error")
        
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Should continue with other validators
        thought = await engine.think("Test validator failure")
        
        # Should still complete successfully
        assert thought.final_text is not None
        # May have fewer validation results due to failure

    @pytest.mark.asyncio
    async def test_error_handling_critic_failure(self, mock_dependencies):
        """Test error handling when critic fails."""
        # Make critic fail
        mock_dependencies.critics["constitutional"].critique_async.side_effect = Exception("Critic error")
        
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Should handle critic failure gracefully
        thought = await engine.think("Test critic failure")
        
        # Should still complete successfully
        assert thought.final_text is not None
        # May have no critique results due to failure

    @pytest.mark.asyncio
    async def test_performance_characteristics(self, mock_dependencies):
        """Test performance characteristics of full workflow."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        import time
        
        # Single thought performance
        start_time = time.time()
        thought = await engine.think("Test performance")
        single_duration = time.time() - start_time
        
        # Should complete reasonably quickly (mocked)
        assert single_duration < 5.0  # Less than 5 seconds
        assert thought.final_text is not None
        
        # Batch performance
        prompts = ["Test prompt " + str(i) for i in range(5)]
        
        start_time = time.time()
        thoughts = await engine.batch_think(prompts)
        batch_duration = time.time() - start_time
        
        # Batch should be efficient
        assert batch_duration < 10.0  # Less than 10 seconds for 5 thoughts
        assert len(thoughts) == 5

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_dependencies):
        """Test concurrent thought processing."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Process multiple thoughts concurrently
        tasks = [
            engine.think("Concurrent test 1"),
            engine.think("Concurrent test 2"),
            engine.think("Concurrent test 3")
        ]
        
        thoughts = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(thoughts) == 3
        for thought in thoughts:
            assert thought.final_text is not None
            assert len(thought.generations) >= 1
        
        # Should have unique IDs
        thought_ids = {t.id for t in thoughts}
        assert len(thought_ids) == 3

    @pytest.mark.asyncio
    async def test_complex_workflow_with_tools(self, mock_dependencies):
        """Test complex workflow with tool usage."""
        # Add mock tool to dependencies
        mock_tool = AsyncMock()
        mock_tool.name = "search-tool"
        mock_tool.search_async.return_value = [
            {"title": "Renewable Energy Guide", "url": "https://example.com", "snippet": "Comprehensive guide"}
        ]
        
        mock_dependencies.retrievers = {"search": mock_tool}
        
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Process thought that might use tools
        thought = await engine.think("Research the latest developments in solar energy technology")
        
        # Verify thought completed
        assert thought.final_text is not None
        assert len(thought.generations) >= 1
        
        # May have tool calls recorded
        # (Depends on implementation details)

    @pytest.mark.asyncio
    async def test_workflow_with_custom_configuration(self):
        """Test workflow with custom configuration."""
        # Create custom dependencies
        deps = Mock(spec=SifakaDependencies)
        
        # Minimal mock setup
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.data = "Custom configured response"
        mock_result.new_messages.return_value = []
        mock_result.cost = None
        mock_result.usage = None
        mock_agent.run.return_value = mock_result
        deps.generator = mock_agent
        
        # No validators or critics for this test
        deps.validators = []
        deps.critics = {}
        deps.retrievers = {}
        deps.always_apply_critics = False
        deps.never_apply_critics = True
        
        engine = SifakaEngine(dependencies=deps)
        
        # Process with custom configuration
        thought = await engine.think("Test custom configuration", max_iterations=1)
        
        # Should complete with minimal processing
        assert thought.final_text == "Custom configured response"
        assert len(thought.generations) == 1
        assert len(thought.validations) == 0  # No validators
        assert len(thought.critiques) == 0   # No critics

    @pytest.mark.asyncio
    async def test_thought_evolution_tracking(self, mock_dependencies):
        """Test tracking of thought evolution through iterations."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        # Process thought with multiple iterations
        thought = await engine.think("Analyze renewable energy policies", max_iterations=3)
        
        # Verify evolution tracking
        assert len(thought.generations) >= 1
        
        # Check iteration progression
        for i, generation in enumerate(thought.generations):
            assert generation.iteration == i
        
        # Check validation progression
        for validation in thought.validations:
            assert validation.iteration >= 0
            assert validation.iteration < len(thought.generations)
        
        # Check critique progression
        for critique in thought.critiques:
            assert critique.iteration >= 0
            assert critique.iteration < len(thought.generations)
        
        # Verify techniques applied
        if len(thought.critiques) > 0:
            assert len(thought.techniques_applied) > 0
            assert "constitutional-critic" in thought.techniques_applied

    @pytest.mark.asyncio
    async def test_workflow_state_consistency(self, mock_dependencies):
        """Test that workflow maintains consistent state throughout processing."""
        engine = SifakaEngine(dependencies=mock_dependencies)
        
        thought = await engine.think("Test state consistency")
        
        # Verify state consistency
        assert thought.prompt is not None
        assert thought.current_text is not None
        assert thought.final_text is not None
        assert thought.current_text == thought.final_text  # Should be finalized
        
        # Verify timestamps are consistent
        assert thought.created_at <= thought.updated_at
        
        # Verify audit trail consistency
        for generation in thought.generations:
            assert generation.iteration <= thought.iteration
            assert generation.timestamp >= thought.created_at
        
        for validation in thought.validations:
            assert validation.iteration <= thought.iteration
            assert validation.timestamp >= thought.created_at
        
        for critique in thought.critiques:
            assert critique.iteration <= thought.iteration
            assert critique.timestamp >= thought.created_at
