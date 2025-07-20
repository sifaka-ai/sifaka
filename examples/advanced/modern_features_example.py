"""Modern Sifaka Features Example.

This example showcases the latest features including:
- Structured configuration with sub-configs
- Parallel batch processing
- Connection pooling
- Performance monitoring
- Plugin system
- Advanced middleware
"""

import asyncio
import time

from sifaka import improve
from sifaka.core.config import Config, CriticConfig, EngineConfig, LLMConfig
from sifaka.core.llm_client_pool import get_global_pool
from sifaka.core.monitoring import get_global_monitor
from sifaka.core.types import CriticType


async def batch_processing_demo():
    """Demonstrate high-performance batch processing."""
    print("🚀 Batch Processing with Connection Pooling")
    print("=" * 50)

    # Large batch of content to process
    content_batch = [
        "AI will change everything.",
        "Machine learning is the future.",
        "Data science drives decisions.",
        "Automation improves efficiency.",
        "Technology enables innovation.",
        "Digital transformation is key.",
        "Cloud computing scales globally.",
        "Mobile apps connect people.",
        "IoT sensors gather data.",
        "Blockchain ensures trust.",
    ]

    # Optimized config for batch processing
    batch_config = Config(
        llm=LLMConfig(model="gpt-4o-mini", temperature=0.7, timeout_seconds=30),
        critic=CriticConfig(critics=[CriticType.SELF_REFINE]),
        engine=EngineConfig(
            max_iterations=2,
            parallel_critics=True,  # Enable parallel processing
        ),
    )

    # Start performance monitoring
    monitor = get_global_monitor()
    monitor.start_monitoring(max_iterations=2)

    start_time = time.time()

    # Process all content in parallel batches for optimal performance
    batch_size = 5
    all_results = []

    for i in range(0, len(content_batch), batch_size):
        batch = content_batch[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} items")

        # Create tasks for this batch
        tasks = [
            improve(
                text,
                critics=[CriticType.SELF_REFINE],
                max_iterations=2,
                config=batch_config,
            )
            for text in batch
        ]

        # Process batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results.extend(batch_results)

    end_time = time.time()

    # Stop monitoring and get metrics
    metrics = monitor.end_monitoring()

    # Display results
    print(
        f"\n✅ Processed {len(content_batch)} items in {end_time - start_time:.2f} seconds"
    )
    print("📊 Performance Metrics:")
    print(f"   - Total LLM calls: {metrics.llm_calls}")
    print(f"   - Total tokens used: {metrics.tokens_used}")
    print(
        f"   - Average time per item: {(end_time - start_time) / len(content_batch):.2f}s"
    )

    successful_results = [r for r in all_results if not isinstance(r, Exception)]
    print(
        f"   - Success rate: {len(successful_results)}/{len(content_batch)} ({len(successful_results)/len(content_batch)*100:.1f}%)"
    )

    # Show some examples
    print("\n📝 Sample Results:")
    for i, (original, result) in enumerate(
        zip(content_batch[:3], successful_results[:3])
    ):
        if not isinstance(result, Exception):
            print(f"   {i+1}. '{original}' → '{result.final_text}'")


async def config_patterns_demo():
    """Demonstrate different configuration patterns."""
    print("\n\n⚙️ Configuration Patterns")
    print("=" * 50)

    # Pattern 1: Minimal config
    minimal_config = Config()  # Uses all defaults
    print("1. Minimal config (all defaults)")

    # Pattern 2: Development config
    dev_config = Config(
        llm=LLMConfig(
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for consistent testing
        ),
        engine=EngineConfig(
            max_iterations=1  # Fast iterations for development
        ),
    )
    print("2. Development config (fast, consistent)")

    # Pattern 3: Production config
    prod_config = Config(
        llm=LLMConfig(
            model="gpt-4o",  # Higher quality model
            temperature=0.7,
            timeout_seconds=60,
        ),
        critic=CriticConfig(
            critics=[CriticType.CONSTITUTIONAL, CriticType.SELF_RAG]  # Multiple critics
        ),
        engine=EngineConfig(
            max_iterations=5,  # More thorough processing
            parallel_critics=True,
        ),
    )
    print("3. Production config (high quality, thorough)")

    # Test text
    test_text = "This is a simple test."

    # Compare processing with different configs
    configs = [
        ("Minimal", minimal_config),
        ("Development", dev_config),
        ("Production", prod_config),
    ]

    print(f"\n🧪 Testing configs with: '{test_text}'")

    for name, config in configs:
        start = time.time()
        result = await improve(
            test_text,
            critics=[CriticType.SELF_REFINE],
            max_iterations=config.engine.max_iterations,
            config=config,
        )
        duration = time.time() - start

        print(f"   {name}: {duration:.2f}s → '{result.final_text}'")


async def performance_monitoring_demo():
    """Demonstrate built-in performance monitoring."""
    print("\n\n📊 Performance Monitoring")
    print("=" * 50)

    # Configure with monitoring enabled
    config = Config(
        llm=LLMConfig(model="gpt-4o-mini", temperature=0.7),
        critic=CriticConfig(critics=[CriticType.REFLEXION, CriticType.SELF_REFINE]),
    )

    # Process with monitoring
    monitor = get_global_monitor()
    monitor.start_monitoring(max_iterations=3)

    await improve(
        "Write a better version of this sentence.",
        critics=[CriticType.REFLEXION, CriticType.SELF_REFINE],
        max_iterations=3,
        config=config,
    )

    metrics = monitor.end_monitoring()

    print("📈 Detailed Performance Metrics:")
    print(f"   - Duration: {metrics.total_duration:.2f}s")
    print(f"   - LLM calls: {metrics.llm_calls}")
    print(f"   - Critic calls: {metrics.critic_calls}")
    print(f"   - Tokens used: {metrics.tokens_used}")
    print(f"   - Iterations: {metrics.iterations_completed}")
    print(f"   - Critics used: {', '.join(metrics.critics_used)}")
    print(f"   - Final confidence: {metrics.final_confidence:.3f}")

    if metrics.errors:
        print(f"   - Errors: {len(metrics.errors)}")


async def connection_pool_demo():
    """Demonstrate connection pooling for better performance."""
    print("\n\n🔗 Connection Pooling")
    print("=" * 50)

    # Get global connection pool stats
    pool = get_global_pool()
    initial_stats = await pool.get_pool_status()

    print("Initial pool status:")
    if initial_stats:
        for pool_key, stats in initial_stats.items():
            print(f"   Pool {pool_key}: {stats['total_connections']} total")
    else:
        print("   - No connections yet")

    # Make multiple requests to see pool usage
    config = Config(llm=LLMConfig(model="gpt-4o-mini"))

    tasks = [
        improve(f"Improve this text: sentence {i}", config=config) for i in range(5)
    ]

    # Process concurrently to see pool behavior
    await asyncio.gather(*tasks)

    final_stats = await pool.get_pool_status()
    print("\nFinal pool status:")
    for pool_key, stats in final_stats.items():
        print(f"   Pool {pool_key}:")
        print(f"      - Total: {stats['total_connections']}")
        print(f"      - Active: {stats['active_connections']}")
        print(f"      - Idle: {stats['idle_connections']}")
    # Get pool metrics if available
    pool_metrics = pool.get_metrics()
    print("\n   Pool Metrics:")
    print(f"   - Pool hits: {pool_metrics.pool_hits}")
    print(f"   - Pool misses: {pool_metrics.pool_misses}")


async def main():
    """Run all modern feature demonstrations."""
    print("🌟 Modern Sifaka Features Showcase")
    print("=" * 60)

    await batch_processing_demo()
    await config_patterns_demo()
    await performance_monitoring_demo()
    await connection_pool_demo()

    print("\n" + "=" * 60)
    print("✨ All demonstrations completed!")
    print("💡 These features showcase Sifaka's enterprise-ready capabilities:")
    print("   - High-performance batch processing")
    print("   - Flexible configuration patterns")
    print("   - Built-in performance monitoring")
    print("   - Automatic connection pooling")


if __name__ == "__main__":
    import os

    if not any(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
        ]
    ):
        print("❌ No API keys found. Please set at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GEMINI_API_KEY")
    else:
        asyncio.run(main())
