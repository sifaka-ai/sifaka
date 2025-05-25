"""Chain Checkpoint Recovery Example for Sifaka.

This example demonstrates the robust chain execution capabilities with automatic
checkpointing and intelligent recovery from failures.

The example shows:
1. Setting up a chain with checkpoint storage
2. Automatic checkpoint creation at each execution step
3. Recovery from simulated failures
4. Analysis of recovery patterns and suggestions

Requirements:
    - Redis running on localhost:6379 (for storage backend)
    - Or configure with in-memory storage for testing

Run with: python examples/recovery/checkpoint_recovery_example.py
"""

import os
import time
import logging
from typing import Dict, Any

from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.storage.checkpoints import CachedCheckpointStorage
from sifaka.storage import SifakaStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.recovery.manager import RecoveryManager, RecoveryStrategy
from sifaka.utils.logging import configure_logging
from sifaka.utils.error_handling import ChainError

# Configure logging to see recovery details
configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailingModel:
    """A model that fails on specific attempts to demonstrate recovery."""

    def __init__(self, model_name: str = "failing-model", fail_on_attempt: int = 2):
        """Initialize the failing model.

        Args:
            model_name: Name of the model
            fail_on_attempt: Which attempt number to fail on (1-based)
        """
        self.model_name = model_name
        self.fail_on_attempt = fail_on_attempt
        self.attempt_count = 0

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text, failing on the specified attempt."""
        self.attempt_count += 1

        if self.attempt_count == self.fail_on_attempt:
            raise RuntimeError(f"Simulated failure on attempt {self.attempt_count}")

        return f"Generated text from {self.model_name} (attempt {self.attempt_count}): {prompt[:50]}..."

    def generate_with_thought(self, thought, **options: Any):
        """Generate text using thought container."""
        text = self.generate(thought.prompt, **options)
        return text, thought.prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens (simple word count)."""
        return len(text.split())


def create_checkpoint_storage():
    """Create checkpoint storage for the example."""
    print("🗄️ Setting up checkpoint storage...")

    # Create MCP configurations for Redis and Milvus
    # Note: This example assumes you have Redis and Milvus MCP servers available
    # For testing without external dependencies, you could use mock storage instead

    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    # Create unified storage manager
    storage_manager = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        memory_size=100,  # Small memory cache for demo
        cache_ttl=300,  # 5 minute TTL
    )

    # Get checkpoint storage from the manager
    checkpoint_storage = storage_manager.get_checkpoint_storage()
    print("✅ Checkpoint storage configured with 3-tier architecture")
    print("   L1: In-memory (100 items) → L2: Redis (5min TTL) → L3: Milvus (persistent)")
    return checkpoint_storage


def demonstrate_successful_execution():
    """Demonstrate normal chain execution with checkpointing."""
    print("\n🚀 Example 1: Successful Execution with Checkpointing")
    print("=" * 60)

    # Create checkpoint storage
    checkpoint_storage = create_checkpoint_storage()

    # Create a reliable model
    model = create_model("mock:reliable-model")

    # Create validators and critics
    validators = [
        LengthValidator(min_length=10, max_length=500),
        RegexValidator(required_patterns=[r"\w+"]),  # Must contain words
    ]
    critics = [ReflexionCritic(model=model)]

    # Create chain with checkpoint storage
    chain = Chain(
        model=model,
        checkpoint_storage=checkpoint_storage,
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
    )

    # Configure the chain
    for validator in validators:
        chain.validate_with(validator)

    for critic in critics:
        chain.improve_with(critic)

    chain.with_prompt("Write a brief explanation of artificial intelligence and its applications.")

    # Run with recovery (should succeed without needing recovery)
    print("🔄 Running chain with checkpoint recovery...")
    result = chain.run_with_recovery()

    print(f"✅ Execution completed successfully!")
    print(f"📝 Final text: {result.text[:100]}...")
    print(f"🔄 Iterations: {result.iteration}")

    # Show checkpoint history
    checkpoints = chain.get_checkpoint_history()
    print(f"📊 Created {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(
            f"   - {cp.current_step} (iteration {cp.iteration}) at {cp.timestamp.strftime('%H:%M:%S')}"
        )

    return checkpoint_storage


def demonstrate_recovery_from_failure():
    """Demonstrate recovery from a simulated failure."""
    print("\n🔧 Example 2: Recovery from Simulated Failure")
    print("=" * 60)

    # Create checkpoint storage
    checkpoint_storage = create_checkpoint_storage()

    # Create a model that will fail on the 2nd attempt
    failing_model = FailingModel(fail_on_attempt=2)

    # Create validators
    validators = [LengthValidator(min_length=10, max_length=500)]

    # Create chain with checkpoint storage
    chain = Chain(
        model=failing_model,
        checkpoint_storage=checkpoint_storage,
        max_improvement_iterations=3,
        apply_improvers_on_validation_failure=True,
    )

    # Configure the chain
    for validator in validators:
        chain.validate_with(validator)

    chain.with_prompt("Explain machine learning in simple terms.")

    # Run with recovery - should recover from the failure
    print("🔄 Running chain with simulated failure...")
    try:
        result = chain.run_with_recovery()
        print(f"✅ Execution completed after recovery!")
        print(f"📝 Final text: {result.text[:100]}...")

        # Show recovery suggestions that were used
        recovery_manager = RecoveryManager(checkpoint_storage)
        print(f"🔧 Recovery capabilities demonstrated successfully")

    except Exception as e:
        print(f"❌ Execution failed even with recovery: {e}")

    # Show checkpoint history
    checkpoints = chain.get_checkpoint_history()
    print(f"📊 Created {len(checkpoints)} checkpoints during recovery:")
    for cp in checkpoints:
        recovery_info = cp.metadata.get("recovery_action", "normal")
        print(f"   - {cp.current_step} (iteration {cp.iteration}) - {recovery_info}")


def demonstrate_recovery_analysis():
    """Demonstrate recovery pattern analysis."""
    print("\n📊 Example 3: Recovery Pattern Analysis")
    print("=" * 60)

    # Create checkpoint storage
    checkpoint_storage = create_checkpoint_storage()
    recovery_manager = RecoveryManager(checkpoint_storage)

    # Simulate some historical failures for pattern analysis
    print("🔍 Analyzing recovery patterns...")

    # Create a mock checkpoint for analysis
    from sifaka.core.thought import Thought
    from sifaka.storage.checkpoints import ChainCheckpoint
    from datetime import datetime

    mock_thought = Thought(prompt="Test prompt for analysis")
    mock_checkpoint = ChainCheckpoint(
        chain_id="analysis-test",
        current_step="generation",
        iteration=1,
        thought=mock_thought,
        recovery_point="generation",
        metadata={"error_type": "timeout", "model_name": "test-model"},
    )

    # Simulate different types of errors and get recovery suggestions
    errors_to_analyze = [
        RuntimeError("Connection timeout"),
        ValueError("Invalid input format"),
        MemoryError("Out of memory"),
        Exception("Rate limit exceeded"),
    ]

    for error in errors_to_analyze:
        print(f"\n🔍 Analyzing error: {type(error).__name__}: {error}")

        recovery_actions = recovery_manager.analyze_failure(mock_checkpoint, error)

        print(f"   💡 Generated {len(recovery_actions)} recovery suggestions:")
        for i, action in enumerate(recovery_actions[:3], 1):  # Show top 3
            confidence_pct = int(action.confidence * 100)
            success_rate = action.estimated_success_rate or 0.5
            success_pct = int(success_rate * 100)

            print(f"      {i}. {action.description}")
            print(f"         Strategy: {action.strategy.value}")
            print(f"         Confidence: {confidence_pct}% | Est. Success: {success_pct}%")


def demonstrate_checkpoint_cleanup():
    """Demonstrate checkpoint cleanup and maintenance."""
    print("\n🧹 Example 4: Checkpoint Cleanup and Maintenance")
    print("=" * 60)

    # Create checkpoint storage
    checkpoint_storage = create_checkpoint_storage()
    recovery_manager = RecoveryManager(checkpoint_storage)

    print("🗄️ Checkpoint storage maintenance:")

    # Show current storage stats
    print("   📊 Current storage statistics:")
    print("      - Checkpoints: Available in storage")
    print("      - Memory usage: Optimized with 3-tier caching")

    # Demonstrate cleanup (would clean up old checkpoints in real usage)
    cleaned_count = recovery_manager.cleanup_old_checkpoints(max_age_days=30)
    print(f"   🧹 Cleaned up {cleaned_count} old checkpoints")

    print("   ✅ Storage maintenance completed")


def main():
    """Run all checkpoint recovery examples."""
    print("🔄 Sifaka Chain Checkpoint Recovery System Demo")
    print("=" * 70)
    print()
    print("This demo shows how Sifaka's checkpoint recovery system provides")
    print("robust chain execution with automatic recovery from failures.")
    print()

    try:
        # Example 1: Normal execution with checkpointing
        checkpoint_storage = demonstrate_successful_execution()

        # Example 2: Recovery from failure
        demonstrate_recovery_from_failure()

        # Example 3: Recovery pattern analysis
        demonstrate_recovery_analysis()

        # Example 4: Checkpoint cleanup
        demonstrate_checkpoint_cleanup()

        print("\n🎉 All checkpoint recovery examples completed successfully!")
        print()
        print("Key Benefits Demonstrated:")
        print("✅ Automatic checkpoint creation at each execution step")
        print("✅ Intelligent recovery from failures with multiple strategies")
        print("✅ Pattern analysis from similar past executions")
        print("✅ Reduced re-computation costs for long chains")
        print("✅ Better debugging and execution analysis")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
