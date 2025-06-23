"""Examples demonstrating advanced Sifaka features."""

import asyncio
import logging
from sifaka import (
    improve,
    MiddlewarePipeline,
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    monitor,
    Validator,
)
from sifaka.validators import LengthValidator, ContentValidator

# Set up logging
logging.basicConfig(level=logging.INFO)


async def middleware_example():
    """Example: Using middleware for cross-cutting concerns."""
    print("\n=== Middleware Example ===")
    
    # Create middleware pipeline
    pipeline = MiddlewarePipeline()
    pipeline.add(LoggingMiddleware(log_level="INFO"))
    pipeline.add(MetricsMiddleware())
    pipeline.add(CachingMiddleware(max_size=10))
    
    # Get metrics middleware reference
    metrics = pipeline.middleware[1]  # MetricsMiddleware is second
    
    # Run improvement with middleware
    text = "Write about the benefits of renewable energy"
    
    result = await improve(
        text,
        critics=["reflexion", "constitutional"],
        max_iterations=2,
        middleware=pipeline
    )
    
    print(f"Final text: {result.final_text[:100]}...")
    
    # Show metrics
    if isinstance(metrics, MetricsMiddleware):
        print("\nMetrics collected:")
        for key, value in metrics.get_metrics().items():
            print(f"  {key}: {value}")


async def composable_validators_example():
    """Example: Using composable validators."""
    print("\n=== Composable Validators Example ===")
    
    # Create validators using the fluent interface
    essay_validator = (
        Validator.create("essay")
        .length(500, 2000)
        .sentences(10, 50)
        .words(100, 400)
        .contains(["introduction", "conclusion", "thesis"])
        .build()
    )
    
    # Compose validators with operators
    technical_validator = (
        Validator.length(200, 1000) & 
        Validator.contains(["algorithm", "performance", "complexity"], mode="any")
    )
    
    # Either email OR phone number
    contact_validator = (
        Validator.matches(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email') |
        Validator.matches(r'\d{3}-\d{3}-\d{4}', 'phone')
    )
    
    # Use composed validators
    result = await improve(
        "Write a technical essay about machine learning",
        validators=[essay_validator, technical_validator],
        max_iterations=3
    )
    
    print("Validation results:")
    for v in result.validations:
        print(f"  {v.validator}: {'✓' if v.passed else '✗'} (score: {v.score:.2f})")


async def performance_monitoring_example():
    """Example: Performance monitoring."""
    print("\n=== Performance Monitoring Example ===")
    
    # Use monitor context manager
    async with monitor(print_summary=False) as mon:
        # Start monitoring
        metrics = mon.start_monitoring(max_iterations=3)
        
        # Simulate tracked operations
        async def mock_llm_call():
            await asyncio.sleep(0.1)
            return "Mock response"
        
        # Track LLM call
        await mon.track_llm_call(mock_llm_call)
        
        # Track critic calls
        for critic in ["reflexion", "constitutional"]:
            await mon.track_critic_call(critic, mock_llm_call)
        
        # Run actual improvement
        result = await improve(
            "Explain quantum computing in simple terms",
            critics=["reflexion"],
            max_iterations=2
        )
        
        # Update metrics from result
        mon.update_from_result(result)
        
        # End monitoring
        final_metrics = mon.end_monitoring()
    
    # Show detailed metrics
    print("Performance Metrics:")
    print(final_metrics)
    
    # Show as dictionary
    print("\nDetailed metrics:")
    metrics_dict = final_metrics.to_dict()
    for category, data in metrics_dict.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")


async def middleware_with_monitoring():
    """Example: Combining middleware with monitoring."""
    print("\n=== Combined Middleware + Monitoring Example ===")
    
    # Create middleware with metrics
    pipeline = MiddlewarePipeline()
    metrics_middleware = MetricsMiddleware()
    pipeline.add(LoggingMiddleware())
    pipeline.add(metrics_middleware)
    
    # Run multiple improvements
    texts = [
        "Write about artificial intelligence",
        "Explain machine learning",
        "Describe neural networks"
    ]
    
    for text in texts:
        result = await improve(
            text,
            critics=["reflexion"],
            max_iterations=1,
            middleware=pipeline
        )
        print(f"Improved: {text[:30]}... -> {result.final_text[:50]}...")
    
    # Show aggregated metrics
    print("\nAggregated Metrics:")
    metrics = metrics_middleware.get_metrics()
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Average time: {metrics.get('avg_time_per_request', 0):.2f}s")
    print(f"  Average confidence: {metrics['average_confidence']:.2%}")
    print(f"  Total LLM calls: {metrics['llm_calls']}")


async def custom_validator_example():
    """Example: Creating custom validators with the builder."""
    print("\n=== Custom Validator Example ===")
    
    # Create a custom SEO-optimized content validator
    seo_validator = (
        Validator.create("seo_content")
        .length(300, 2000)  # Good length for SEO
        .words(50, 400)
        .contains(["keyword", "content", "value"], mode="any")  # SEO keywords
        .matches(r'<h[1-6]>', 'headers')  # Has headers
        .custom(
            name="keyword_density",
            check=lambda text: 0.01 <= text.lower().count("keyword") / len(text.split()) <= 0.03,
            detail=lambda text: f"Keyword density: {text.lower().count('keyword') / len(text.split()):.2%}"
        )
        .build()
    )
    
    result = await improve(
        "Write SEO-optimized content about keyword optimization",
        validators=[seo_validator],
        max_iterations=2
    )
    
    print("SEO Validation Results:")
    for v in result.validations:
        if v.validator == "seo_content":
            print(f"  {v.details}")


async def main():
    """Run all examples."""
    examples = [
        middleware_example,
        composable_validators_example,
        performance_monitoring_example,
        middleware_with_monitoring,
        custom_validator_example,
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())