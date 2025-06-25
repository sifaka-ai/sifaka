"""Prometheus metrics integration example for Sifaka.

This example shows how to expose Sifaka metrics in Prometheus format
for collection by a Prometheus server.
"""

import asyncio
import time
from typing import Optional, Dict, Any

# Prometheus client imports
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    start_http_server,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

from sifaka import improve_sync, improve_async
from sifaka.core.middleware import BaseMiddleware
from sifaka.core.models import SifakaResult


class PrometheusMiddleware(BaseMiddleware):
    """Middleware to expose Sifaka metrics in Prometheus format."""

    def __init__(
        self, registry: Optional[CollectorRegistry] = None, prefix: str = "sifaka"
    ):
        """Initialize Prometheus middleware.

        Args:
            registry: Prometheus registry (creates new if None)
            prefix: Metric name prefix
        """
        self.registry = registry or CollectorRegistry()
        self.prefix = prefix

        # Create metrics
        self._create_metrics()

    def _create_metrics(self):
        """Create Prometheus metrics."""
        # Counter for total improvements
        self.improvement_counter = Counter(
            f"{self.prefix}_improvements_total",
            "Total number of text improvements",
            ["critic", "model", "status"],
            registry=self.registry,
        )

        # Histogram for improvement duration
        self.duration_histogram = Histogram(
            f"{self.prefix}_improvement_duration_seconds",
            "Duration of text improvement operations",
            ["critic"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        # Histogram for iterations
        self.iterations_histogram = Histogram(
            f"{self.prefix}_improvement_iterations",
            "Number of iterations per improvement",
            ["critic"],
            buckets=(1, 2, 3, 4, 5, 10),
            registry=self.registry,
        )

        # Counter for tokens used
        self.tokens_counter = Counter(
            f"{self.prefix}_tokens_total",
            "Total tokens consumed",
            ["model", "operation"],
            registry=self.registry,
        )

        # Summary for text length
        self.text_length_summary = Summary(
            f"{self.prefix}_text_length_characters",
            "Length of texts being improved",
            ["type"],  # 'input' or 'output'
            registry=self.registry,
        )

        # Counter for errors
        self.error_counter = Counter(
            f"{self.prefix}_errors_total",
            "Total number of errors",
            ["error_type", "critic"],
            registry=self.registry,
        )

        # Gauge for active improvements
        self.active_improvements = Gauge(
            f"{self.prefix}_active_improvements",
            "Number of improvements currently in progress",
            registry=self.registry,
        )

        # Histogram for cost
        self.cost_histogram = Histogram(
            f"{self.prefix}_improvement_cost_dollars",
            "Cost of improvements in USD",
            ["model"],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0),
            registry=self.registry,
        )

        # Counter for validations
        self.validation_counter = Counter(
            f"{self.prefix}_validations_total",
            "Total number of validations",
            ["validator", "result"],  # result: 'passed' or 'failed'
            registry=self.registry,
        )

    async def pre_improve(self, text: str, **kwargs) -> Dict[str, Any]:
        """Start tracking metrics for improvement."""
        # Increment active improvements
        self.active_improvements.inc()

        # Record input text length
        self.text_length_summary.labels(type="input").observe(len(text))

        # Store start time
        return {
            "start_time": time.time(),
            "critic": kwargs.get("critics", ["reflexion"])[0],
            "model": kwargs.get("model", "default"),
        }

    async def post_improve(
        self,
        result: SifakaResult,
        context: Dict[str, Any],
        error: Optional[Exception] = None,
    ):
        """Record metrics after improvement."""
        # Decrement active improvements
        self.active_improvements.dec()

        # Calculate duration
        duration = time.time() - context["start_time"]
        critic = context["critic"]
        model = context["model"]

        if error:
            # Record error
            self.error_counter.labels(
                error_type=type(error).__name__, critic=critic
            ).inc()

            # Record as failed improvement
            self.improvement_counter.labels(
                critic=critic, model=model, status="failed"
            ).inc()
        else:
            # Record successful improvement
            self.improvement_counter.labels(
                critic=critic, model=model, status="success"
            ).inc()

            # Record duration
            self.duration_histogram.labels(critic=critic).observe(duration)

            # Record iterations
            self.iterations_histogram.labels(critic=critic).observe(result.iterations)

            # Record tokens
            self.tokens_counter.labels(model=model, operation="improvement").add(
                result.total_tokens
            )

            # Record cost
            self.cost_histogram.labels(model=model).observe(result.total_cost)

            # Record output text length
            self.text_length_summary.labels(type="output").observe(
                len(result.final_text)
            )

    async def on_validation(self, validation_result: Any, validator: str, **kwargs):
        """Record validation metrics."""
        result = "passed" if validation_result.is_valid else "failed"
        self.validation_counter.labels(validator=validator, result=result).inc()


class SifakaMetricsCollector:
    """Custom collector for runtime Sifaka metrics."""

    def __init__(self, sifaka_app):
        """Initialize collector with Sifaka app reference."""
        self.app = sifaka_app

    def collect(self):
        """Collect current metrics."""
        # Example: Collect cache metrics if available
        if hasattr(self.app, "cache"):
            yield GaugeMetricFamily(
                "sifaka_cache_size_bytes",
                "Current cache size in bytes",
                value=self.app.cache.size_bytes,
            )

            yield CounterMetricFamily(
                "sifaka_cache_hits_total", "Total cache hits", value=self.app.cache.hits
            )

            yield CounterMetricFamily(
                "sifaka_cache_misses_total",
                "Total cache misses",
                value=self.app.cache.misses,
            )

        # Example: Collect model pool metrics
        if hasattr(self.app, "model_pool"):
            for model_name, pool in self.app.model_pool.items():
                yield GaugeMetricFamily(
                    f"sifaka_model_pool_active_{model_name}",
                    f"Active connections for {model_name}",
                    value=pool.active_connections,
                )


def setup_prometheus(
    port: int = 8000, registry: Optional[CollectorRegistry] = None
) -> PrometheusMiddleware:
    """Set up Prometheus metrics endpoint.

    Args:
        port: Port to expose metrics on
        registry: Custom registry (creates new if None)

    Returns:
        Configured PrometheusMiddleware instance
    """
    # Create middleware
    middleware = PrometheusMiddleware(registry=registry)

    # Start HTTP server for metrics
    start_http_server(port, registry=middleware.registry)

    return middleware


# Example Flask integration
def create_flask_app(prometheus_middleware: PrometheusMiddleware):
    """Create Flask app with Prometheus metrics endpoint."""
    from flask import Flask, Response

    app = Flask(__name__)

    @app.route("/metrics")
    def metrics():
        """Expose metrics endpoint."""
        return Response(
            generate_latest(prometheus_middleware.registry),
            mimetype=CONTENT_TYPE_LATEST,
        )

    return app


# Example usage
if __name__ == "__main__":
    # Set up Prometheus middleware
    prom_middleware = setup_prometheus(port=8000)

    print("Prometheus metrics server started on http://localhost:8000")
    print("Metrics available at http://localhost:8000/metrics")

    # Example 1: Simple improvement with metrics
    text = "Machine learning is a type of AI that helps computers learn from data."

    try:
        result = improve_sync(
            text, critics=["reflexion"], max_iterations=2, middleware=[prom_middleware]
        )

        print(f"\nOriginal: {text}")
        print(f"Improved: {result.final_text}")
        print(f"Iterations: {result.iterations}")
        print(f"Cost: ${result.total_cost:.4f}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Multiple improvements to generate metrics
    async def run_multiple_improvements():
        """Run multiple improvements to generate interesting metrics."""
        texts = [
            "AI is changing the world.",
            "Climate change is a serious problem.",
            "Python is a programming language.",
            "Space exploration is important for humanity.",
            "Renewable energy is the future.",
        ]

        critics = ["reflexion", "self_refine", "chain_of_thought"]

        tasks = []
        for i, text in enumerate(texts):
            critic = critics[i % len(critics)]
            task = improve_async(
                text, critics=[critic], max_iterations=3, middleware=[prom_middleware]
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        print(f"\nCompleted {success_count}/{len(texts)} improvements")
        print("Check metrics at http://localhost:8000/metrics")

    # Run multiple improvements
    # asyncio.run(run_multiple_improvements())

    # Example 3: Prometheus queries
    print("\n--- Example Prometheus Queries ---")
    print("# Request rate (last 5 minutes):")
    print("rate(sifaka_improvements_total[5m])")
    print("\n# Average improvement duration by critic:")
    print(
        "sifaka_improvement_duration_seconds_sum / sifaka_improvement_duration_seconds_count"
    )
    print("\n# Error rate:")
    print("rate(sifaka_errors_total[5m]) / rate(sifaka_improvements_total[5m])")
    print("\n# Token usage by model:")
    print("sum by (model) (rate(sifaka_tokens_total[5m]))")
    print("\n# 95th percentile improvement duration:")
    print(
        "histogram_quantile(0.95, rate(sifaka_improvement_duration_seconds_bucket[5m]))"
    )

    # Keep the metrics server running
    print("\nMetrics server is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
