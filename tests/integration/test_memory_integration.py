"""Integration tests for memory-bounded operations."""

import pytest
import psutil
import os
from sifaka import improve
from sifaka import Config


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_bounded_large_text():
    """Test memory bounds with large text inputs."""
    # Create a large text that might cause memory issues
    large_text = "This needs improvement. " * 1000  # ~25KB of text

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Configure strict memory limits
    config = Config(
        max_history_size=3,  # Only keep 3 iterations in history
        max_text_length=50000,  # 50KB max text
    )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("No API key available")

    result = await improve(
        large_text,
        max_iterations=10,  # Try many iterations
        api_key=api_key,
        config=config,
    )

    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Assertions
    assert result.final_text != large_text
    # Check critiques length instead of history
    assert len(result.critiques) >= 1
    assert memory_increase < 100  # Less than 100MB increase

    # Verify that we actually did multiple iterations
    assert result.iteration > 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_improvements():
    """Test multiple concurrent improvement operations."""
    import asyncio
    from sifaka import improve

    texts = [
        "First text that needs improvement.",
        "Second text with different content.",
        "Third text for concurrent testing.",
    ]

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("No API key available")

    async def run_concurrent():
        tasks = [
            improve(
                text,
                max_iterations=2,
                api_key=api_key,
            )
            for text in texts
        ]
        return await asyncio.gather(*tasks)

    results = await run_concurrent()

    # Verify all completed successfully
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.final_text != texts[i]
        assert result.iteration > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_cleanup():
    """Test that memory is properly cleaned up after operations."""
    import gc

    process = psutil.Process(os.getpid())

    # Run multiple improvements
    for i in range(5):
        initial_memory = process.memory_info().rss / 1024 / 1024

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        result = await improve(
            f"Text iteration {i} that needs improvement.",
            max_iterations=3,
            api_key=api_key,
        )

        # Delete result and force garbage collection
        del result
        gc.collect()

        # Check memory didn't grow significantly
        final_memory = process.memory_info().rss / 1024 / 1024
        assert (final_memory - initial_memory) < 50  # Less than 50MB growth
