"""Performance benchmarks for Sifaka core components.

These benchmarks track the performance of key Sifaka operations to detect
regressions and measure improvements over time.
"""

import pytest
from datetime import datetime
import sys
import os

# Add sifaka to path for benchmarking
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sifaka'))

try:
    from core.thought import SifakaThought
    from core.models.generation import Generation
    from core.models.validation import ValidationResult
    from core.models.critique import CritiqueResult
    from core.models.tool_call import ToolCall
    from core.container import SifakaContainer
    from core.registry import SifakaRegistry
    SIFAKA_AVAILABLE = True
except ImportError:
    SIFAKA_AVAILABLE = False


@pytest.mark.skipif(not SIFAKA_AVAILABLE, reason="Sifaka not available")
class TestSifakaPerformance:
    """Performance benchmarks for Sifaka components."""

    def test_thought_creation_performance(self, benchmark):
        """Benchmark SifakaThought creation."""
        def create_thought():
            return SifakaThought(prompt="Test prompt for benchmarking")
        
        result = benchmark(create_thought)
        assert result.prompt == "Test prompt for benchmarking"

    def test_generation_model_performance(self, benchmark):
        """Benchmark Generation model creation and operations."""
        def create_and_analyze_generation():
            generation = Generation(
                iteration=1,
                text="This is a test generation for performance benchmarking",
                model="test:model",
                timestamp=datetime.now(),
                conversation_history=[
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"}
                ],
                cost=0.001,
                usage={"prompt_tokens": 10, "completion_tokens": 20}
            )
            
            # Perform operations that would be common in real usage
            token_count = generation.get_token_count()
            summary = generation.get_conversation_summary()
            memory_usage = generation.get_memory_usage()
            
            return generation, token_count, summary, memory_usage
        
        result = benchmark(create_and_analyze_generation)
        generation, token_count, summary, memory_usage = result
        
        assert generation.text == "This is a test generation for performance benchmarking"
        assert "prompt_tokens" in token_count
        assert summary["message_count"] == 2

    def test_validation_model_performance(self, benchmark):
        """Benchmark ValidationResult model creation and operations."""
        def create_and_analyze_validation():
            validation = ValidationResult(
                iteration=1,
                validator="test_validator",
                passed=True,
                details={
                    "score": 0.95,
                    "word_count": 50,
                    "min_required": 10,
                    "max_allowed": 100,
                    "suggestions": ["Great work!", "Keep it up!"]
                },
                timestamp=datetime.now()
            )
            
            # Perform operations
            score = validation.get_validation_score()
            metrics = validation.get_numeric_metrics()
            suggestions = validation.get_suggestions()
            summary = validation.get_summary()
            
            return validation, score, metrics, suggestions, summary
        
        result = benchmark(create_and_analyze_validation)
        validation, score, metrics, suggestions, summary = result
        
        assert validation.passed is True
        assert score == 1.0  # Passed validation should return 1.0
        assert "score" in metrics

    def test_critique_model_performance(self, benchmark):
        """Benchmark CritiqueResult model creation and operations."""
        def create_and_analyze_critique():
            critique = CritiqueResult(
                iteration=1,
                critic="test_critic",
                feedback="This is comprehensive feedback for testing",
                suggestions=["Add more detail", "Improve clarity", "Include examples"],
                timestamp=datetime.now(),
                confidence=0.85,
                reasoning="Based on clarity and completeness analysis",
                needs_improvement=True,
                processing_time_ms=1250.5,
                model_name="test:model"
            )
            
            # Perform operations
            confidence_level = critique.get_confidence_level()
            categories = critique.get_suggestion_categories()
            priority = critique.get_improvement_priority()
            summary = critique.get_summary()
            
            return critique, confidence_level, categories, priority, summary
        
        result = benchmark(create_and_analyze_critique)
        critique, confidence_level, categories, priority, summary = result
        
        assert critique.confidence == 0.85
        assert confidence_level == "high"
        assert len(categories) > 0

    def test_tool_call_performance(self, benchmark):
        """Benchmark ToolCall model creation and operations."""
        def create_and_analyze_tool_call():
            tool_call = ToolCall(
                iteration=1,
                tool_name="web_search",
                args={"query": "test query", "max_results": 5},
                result={
                    "results": [f"result_{i}" for i in range(10)],
                    "total_found": 1250,
                    "execution_time": 2.5
                },
                execution_time=2.35,
                timestamp=datetime.now()
            )
            
            # Perform operations
            category = tool_call.get_tool_category()
            performance_cat = tool_call.get_performance_category()
            result_size = tool_call.get_result_size()
            summary = tool_call.get_summary()
            
            return tool_call, category, performance_cat, result_size, summary
        
        result = benchmark(create_and_analyze_tool_call)
        tool_call, category, performance_cat, result_size, summary = result
        
        assert tool_call.tool_name == "web_search"
        assert category == "search"
        assert performance_cat == "normal"

    def test_thought_operations_performance(self, benchmark):
        """Benchmark common SifakaThought operations."""
        def perform_thought_operations():
            thought = SifakaThought(prompt="Performance test prompt")
            
            # Add multiple generations
            for i in range(3):
                thought.add_generation(
                    f"Generation {i} text content",
                    "test:model",
                    None  # No pydantic_result for benchmark
                )
            
            # Add multiple validations
            for i in range(5):
                thought.add_validation(
                    f"validator_{i}",
                    i % 2 == 0,  # Alternate pass/fail
                    {"score": 0.8 + (i * 0.05), "iteration": i}
                )
            
            # Add multiple critiques
            for i in range(3):
                thought.add_critique(
                    f"critic_{i}",
                    f"Feedback {i}",
                    [f"Suggestion {i}.1", f"Suggestion {i}.2"],
                    confidence=0.7 + (i * 0.1)
                )
            
            # Perform analysis operations
            summary = thought.get_summary()
            memory_usage = thought.get_memory_usage()
            current_validations = thought.get_current_iteration_validations()
            current_critiques = thought.get_current_iteration_critiques()
            
            return thought, summary, memory_usage, current_validations, current_critiques
        
        result = benchmark(perform_thought_operations)
        thought, summary, memory_usage, current_validations, current_critiques = result
        
        assert summary["generations_count"] == 3
        assert summary["validations_count"] == 5
        assert summary["critiques_count"] == 3
        assert "total_size_bytes" in memory_usage

    def test_container_performance(self, benchmark):
        """Benchmark SifakaContainer operations."""
        def container_operations():
            container = SifakaContainer()
            
            # Test node registration and retrieval
            stats = container.get_container_stats()
            all_nodes = container.get_all_nodes()
            
            # Test plugin operations
            container.register_plugin("test_plugin", {"name": "test"})
            plugin = container.get_plugin("test_plugin")
            
            # Test singleton operations
            container.register_singleton("test_singleton", "singleton_value")
            singleton = container.get_singleton("test_singleton")
            
            return container, stats, all_nodes, plugin, singleton
        
        result = benchmark(container_operations)
        container, stats, all_nodes, plugin, singleton = result
        
        assert "registered_nodes" in stats
        assert len(all_nodes) > 0
        assert plugin["name"] == "test"
        assert singleton == "singleton_value"

    def test_registry_performance(self, benchmark):
        """Benchmark SifakaRegistry operations."""
        def registry_operations():
            registry = SifakaRegistry()
            
            # Test plugin listing
            plugins = registry.list_plugins()
            critic_plugins = registry.list_plugins("critic")
            
            # Test stats
            stats = registry.get_registry_stats()
            
            return registry, plugins, critic_plugins, stats
        
        result = benchmark(registry_operations)
        registry, plugins, critic_plugins, stats = result
        
        assert len(plugins) > 0
        assert len(critic_plugins) > 0
        assert "total_plugins" in stats

    def test_memory_optimization_performance(self, benchmark):
        """Benchmark memory optimization operations."""
        def memory_optimization():
            thought = SifakaThought(prompt="Memory optimization test")
            
            # Create a large thought with lots of data
            for i in range(10):
                thought.add_generation(
                    f"Large generation {i} " + "x" * 1000,  # Large text
                    "test:model",
                    None
                )
                
                thought.add_validation(
                    f"validator_{i}",
                    True,
                    {"large_data": list(range(100)), "iteration": i}
                )
            
            # Test memory operations
            initial_memory = thought.get_memory_usage()
            
            # Clean up history
            cleanup_stats = thought.cleanup_history(keep_last_n=3)
            
            final_memory = thought.get_memory_usage()
            
            return thought, initial_memory, cleanup_stats, final_memory
        
        result = benchmark(memory_optimization)
        thought, initial_memory, cleanup_stats, final_memory = result
        
        assert initial_memory["total_size_bytes"] > final_memory["total_size_bytes"]
        assert cleanup_stats["generations"] > 0
        assert cleanup_stats["validations"] > 0
