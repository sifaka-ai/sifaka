"""DataDog integration example for Sifaka.

This example shows how to send Sifaka metrics and traces to DataDog
for monitoring and observability.
"""

import asyncio
import time
from typing import Optional, Dict, Any

# DataDog imports
from datadog import initialize, api
from datadog import DogStatsd
from ddtrace import tracer, patch_all

from sifaka import improve_sync, improve_async
from sifaka.core.middleware import BaseMiddleware
from sifaka.core.models import SifakaResult


class DataDogMiddleware(BaseMiddleware):
    """Middleware to send metrics and traces to DataDog."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "sifaka",
        tags: Optional[Dict[str, str]] = None,
        enable_tracing: bool = True,
    ):
        """Initialize DataDog middleware.

        Args:
            api_key: DataDog API key (uses DD_API_KEY env var if None)
            app_key: DataDog app key (uses DD_APP_KEY env var if None)
            host: DogStatsD host
            port: DogStatsD port
            prefix: Metric name prefix
            tags: Default tags to add to all metrics
            enable_tracing: Whether to enable APM tracing
        """
        # Initialize DataDog
        options = {
            "statsd_host": host,
            "statsd_port": port,
            "statsd_constant_tags": self._format_tags(tags or {}),
        }

        if api_key:
            options["api_key"] = api_key
        if app_key:
            options["app_key"] = app_key

        initialize(**options)

        # Create custom statsd client with prefix
        self.statsd = DogStatsd(
            host=host,
            port=port,
            namespace=prefix,
            constant_tags=self._format_tags(tags or {}),
        )

        self.prefix = prefix
        self.enable_tracing = enable_tracing

        if enable_tracing:
            # Configure tracer
            tracer.configure(
                hostname=host,
                port=8126,  # Default APM port
                service=prefix,
                env=tags.get("env", "development") if tags else "development",
            )

    def _format_tags(self, tags: Dict[str, str]) -> list:
        """Format tags dict to DataDog format."""
        return [f"{k}:{v}" for k, v in tags.items()]

    async def pre_improve(self, text: str, **kwargs) -> Dict[str, Any]:
        """Start tracking metrics and trace for improvement."""
        context = {
            "start_time": time.time(),
            "critic": kwargs.get("critics", ["reflexion"])[0],
            "model": kwargs.get("model", "default"),
        }

        # Start trace span if enabled
        if self.enable_tracing:
            span = tracer.start_span(
                "sifaka.improve", service=self.prefix, resource=context["critic"]
            )
            span.set_tags(
                {
                    "critic": context["critic"],
                    "model": context["model"],
                    "text.length": len(text),
                    "max_iterations": kwargs.get("max_iterations", 3),
                }
            )
            context["span"] = span

        # Increment active improvements gauge
        self.statsd.gauge(
            "active_improvements", 1, tags=[f"critic:{context['critic']}"]
        )

        # Track input text length
        self.statsd.histogram(
            "text.length.input", len(text), tags=[f"critic:{context['critic']}"]
        )

        return context

    async def post_improve(
        self,
        result: SifakaResult,
        context: Dict[str, Any],
        error: Optional[Exception] = None,
    ):
        """Send metrics and complete trace after improvement."""
        # Calculate duration
        duration = time.time() - context["start_time"]
        tags = [f"critic:{context['critic']}", f"model:{context['model']}"]

        # Decrement active improvements
        self.statsd.gauge(
            "active_improvements", -1, tags=[f"critic:{context['critic']}"]
        )

        if error:
            # Track error
            self.statsd.increment(
                "errors",
                tags=[
                    f"critic:{context['critic']}",
                    f"error_type:{type(error).__name__}",
                ],
            )

            # Track failed improvement
            self.statsd.increment("improvements", tags=tags + ["status:failed"])

            # Mark span as error if tracing
            if self.enable_tracing and "span" in context:
                span = context["span"]
                span.set_tag("error", True)
                span.set_tag("error.type", type(error).__name__)
                span.set_tag("error.message", str(error))
        else:
            # Track successful improvement
            self.statsd.increment("improvements", tags=tags + ["status:success"])

            # Track duration
            self.statsd.histogram("improvement.duration", duration * 1000, tags=tags)

            # Track iterations
            self.statsd.histogram(
                "improvement.iterations", result.iterations, tags=tags
            )

            # Track tokens
            self.statsd.increment(
                "tokens.used",
                result.total_tokens,
                tags=[f"model:{context['model']}", "operation:improvement"],
            )

            # Track cost
            self.statsd.histogram(
                "improvement.cost",
                result.total_cost,
                tags=[f"model:{context['model']}"],
            )

            # Track output text length
            self.statsd.histogram(
                "text.length.output",
                len(result.final_text),
                tags=[f"critic:{context['critic']}"],
            )

            # Add result data to span if tracing
            if self.enable_tracing and "span" in context:
                span = context["span"]
                span.set_tags(
                    {
                        "iterations": result.iterations,
                        "total_tokens": result.total_tokens,
                        "total_cost": result.total_cost,
                        "output.length": len(result.final_text),
                        "improvement_ratio": len(result.final_text)
                        / len(result.history[0].text),
                    }
                )

        # Finish span if tracing
        if self.enable_tracing and "span" in context:
            context["span"].finish()

    async def on_critique(self, critique: str, iteration: int, **kwargs):
        """Track critique generation."""
        with tracer.start_span(
            f"sifaka.critique.iteration_{iteration}", service=self.prefix
        ) as span:
            span.set_tags({"iteration": iteration, "critique.length": len(critique)})

            # Track critique generation
            self.statsd.increment(
                "critiques.generated",
                tags=[
                    f"iteration:{iteration}",
                    f"critic:{kwargs.get('critic', 'unknown')}",
                ],
            )

    async def on_generation(self, text: str, iteration: int, **kwargs):
        """Track text generation."""
        with tracer.start_span(
            f"sifaka.generation.iteration_{iteration}", service=self.prefix
        ) as span:
            span.set_tags({"iteration": iteration, "generation.length": len(text)})

            # Track generation
            self.statsd.increment(
                "generations.created",
                tags=[
                    f"iteration:{iteration}",
                    f"critic:{kwargs.get('critic', 'unknown')}",
                ],
            )

    async def on_validation(self, validation_result: Any, validator: str, **kwargs):
        """Track validation results."""
        result = "passed" if validation_result.is_valid else "failed"
        self.statsd.increment(
            "validations", tags=[f"validator:{validator}", f"result:{result}"]
        )

        # Send validation event if failed
        if not validation_result.is_valid:
            api.Event.create(
                title=f"Validation Failed: {validator}",
                text=f"Validation failed with errors: {validation_result.errors}",
                alert_type="warning",
                tags=[f"validator:{validator}", "service:sifaka"],
            )


def setup_datadog(
    api_key: Optional[str] = None,
    app_key: Optional[str] = None,
    host: str = "localhost",
    port: int = 8125,
    service_name: str = "sifaka",
    environment: str = "development",
    enable_profiling: bool = False,
) -> DataDogMiddleware:
    """Set up DataDog integration.

    Args:
        api_key: DataDog API key
        app_key: DataDog app key
        host: DogStatsD host
        port: DogStatsD port
        service_name: Service name for tagging
        environment: Environment name
        enable_profiling: Enable continuous profiling

    Returns:
        Configured DataDogMiddleware instance
    """
    # Enable automatic instrumentation
    patch_all()

    # Enable profiling if requested
    if enable_profiling:
        from ddtrace.profiling import Profiler

        prof = Profiler(
            env=environment, service=service_name, tags={"version": "0.0.7"}
        )
        prof.start()

    # Create middleware
    return DataDogMiddleware(
        api_key=api_key,
        app_key=app_key,
        host=host,
        port=port,
        prefix=service_name,
        tags={"env": environment, "service": service_name, "version": "0.0.7"},
    )


# Custom metrics for business logic
class SifakaBusinessMetrics:
    """Track business-level metrics."""

    def __init__(self, statsd_client):
        self.statsd = statsd_client

    def track_improvement_quality(self, original: str, improved: str, critic: str):
        """Track quality metrics for improvements."""
        # Calculate improvement ratio
        ratio = len(improved) / len(original) if original else 1.0
        self.statsd.histogram(
            "improvement.quality.ratio", ratio, tags=[f"critic:{critic}"]
        )

        # Track if text was shortened or lengthened
        if ratio > 1.2:
            self.statsd.increment(
                "improvement.quality.expanded", tags=[f"critic:{critic}"]
            )
        elif ratio < 0.8:
            self.statsd.increment(
                "improvement.quality.condensed", tags=[f"critic:{critic}"]
            )
        else:
            self.statsd.increment(
                "improvement.quality.maintained", tags=[f"critic:{critic}"]
            )

    def track_user_satisfaction(self, user_id: str, satisfied: bool, critic: str):
        """Track user satisfaction metrics."""
        self.statsd.increment(
            "user.satisfaction",
            tags=[
                f"satisfied:{'yes' if satisfied else 'no'}",
                f"critic:{critic}",
                f"user_id:{user_id}",
            ],
        )

    def track_api_usage(self, user_id: str, endpoint: str, response_time: float):
        """Track API usage patterns."""
        self.statsd.histogram(
            "api.response_time",
            response_time * 1000,
            tags=[f"endpoint:{endpoint}", f"user_id:{user_id}"],
        )

        self.statsd.increment(
            "api.requests", tags=[f"endpoint:{endpoint}", f"user_id:{user_id}"]
        )


# Example usage
if __name__ == "__main__":
    # Note: This example requires DataDog agent running locally
    # Install: https://docs.datadoghq.com/agent/

    # Set up DataDog middleware
    dd_middleware = setup_datadog(
        service_name="sifaka-example", environment="development", enable_profiling=True
    )

    print("DataDog integration configured")
    print("Make sure DataDog agent is running locally")
    print("View metrics at https://app.datadoghq.com")

    # Example 1: Simple improvement with DataDog metrics
    text = "Machine learning is a type of AI that helps computers learn from data."

    try:
        # Use tracer for custom span
        with tracer.trace("user.request", service="sifaka-example") as span:
            span.set_tag("user.id", "user123")
            span.set_tag("request.type", "improve_text")

            result = improve_sync(
                text,
                critics=["reflexion"],
                max_iterations=2,
                middleware=[dd_middleware],
            )

            print(f"\nOriginal: {text}")
            print(f"Improved: {result.final_text}")
            print(f"Iterations: {result.iterations}")
            print(f"Cost: ${result.total_cost:.4f}")

            # Track business metrics
            business_metrics = SifakaBusinessMetrics(dd_middleware.statsd)
            business_metrics.track_improvement_quality(
                text, result.final_text, "reflexion"
            )

    except Exception as e:
        print(f"Error: {e}")
        # Send error event to DataDog
        api.Event.create(
            title="Sifaka Improvement Failed",
            text=f"Error improving text: {str(e)}",
            alert_type="error",
            tags=["service:sifaka-example", "error:improvement"],
        )

    # Example 2: Batch processing with metrics
    async def process_batch():
        """Process a batch of texts with detailed metrics."""
        texts = [
            ("doc1", "AI is changing the world."),
            ("doc2", "Climate change is a serious problem."),
            ("doc3", "Python is a programming language."),
        ]

        with tracer.trace("batch.processing", service="sifaka-example") as batch_span:
            batch_span.set_tag("batch.size", len(texts))

            tasks = []
            for doc_id, text in texts:
                # Create task with custom span
                async def process_one(doc_id, text):
                    with tracer.trace("document.processing") as doc_span:
                        doc_span.set_tag("document.id", doc_id)
                        return await improve_async(
                            text, critics=["self_refine"], middleware=[dd_middleware]
                        )

                tasks.append(process_one(doc_id, text))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Track batch results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            dd_middleware.statsd.gauge(
                "batch.success_rate", success_count / len(texts) * 100
            )

            print(
                f"\nBatch processing completed: {success_count}/{len(texts)} successful"
            )

    # Run batch processing
    # asyncio.run(process_batch())

    # Example 3: Custom dashboard metrics
    print("\n--- DataDog Dashboard Widgets ---")
    print("1. Improvement Success Rate:")
    print(
        "   sum:sifaka.improvements{status:success}.as_count() / sum:sifaka.improvements{*}.as_count() * 100"
    )

    print("\n2. Average Cost by Model:")
    print("   avg:sifaka.improvement.cost{*} by {model}")

    print("\n3. Token Usage Rate:")
    print("   sum:sifaka.tokens.used{*}.as_rate()")

    print("\n4. Error Rate by Critic:")
    print("   sum:sifaka.errors{*}.as_rate() by {critic}")

    print("\n5. P95 Improvement Duration:")
    print("   percentile:sifaka.improvement.duration{*}:95")

    # Send a test event
    api.Event.create(
        title="Sifaka Monitoring Started",
        text="DataDog integration is active and collecting metrics",
        alert_type="info",
        tags=["service:sifaka-example", "monitoring:started"],
    )

    print("\nDataDog integration is active. Check your DataDog dashboard for metrics.")
