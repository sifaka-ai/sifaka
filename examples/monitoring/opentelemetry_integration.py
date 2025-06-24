"""OpenTelemetry integration example for Sifaka.

This example shows how to instrument Sifaka with OpenTelemetry
for distributed tracing and metrics collection.
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import contextmanager

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

from sifaka import improve_sync, improve_async
from sifaka.core.middleware import BaseMiddleware
from sifaka.core.models import SifakaResult


class OpenTelemetryMiddleware(BaseMiddleware):
    """Middleware to add OpenTelemetry instrumentation to Sifaka."""
    
    def __init__(
        self,
        service_name: str = "sifaka",
        otlp_endpoint: str = "localhost:4317",
        enable_traces: bool = True,
        enable_metrics: bool = True
    ):
        """Initialize OpenTelemetry middleware.
        
        Args:
            service_name: Name of the service for telemetry
            otlp_endpoint: OTLP collector endpoint
            enable_traces: Whether to enable distributed tracing
            enable_metrics: Whether to enable metrics collection
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enable_traces = enable_traces
        self.enable_metrics = enable_metrics
        
        # Set up OpenTelemetry
        self._setup_telemetry()
    
    def _setup_telemetry(self):
        """Set up OpenTelemetry providers."""
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: "0.0.7",
        })
        
        # Set up tracing
        if self.enable_traces:
            trace_provider = TracerProvider(resource=resource)
            
            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True  # Use insecure for local development
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace_provider.add_span_processor(span_processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(trace_provider)
            self.tracer = trace.get_tracer(__name__)
        
        # Set up metrics
        if self.enable_metrics:
            # Configure metric reader and exporter
            metric_exporter = OTLPMetricExporter(
                endpoint=self.otlp_endpoint,
                insecure=True
            )
            
            metric_reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=10000  # Export every 10 seconds
            )
            
            # Create meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
            
            # Set global meter provider
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
            # Create metrics
            self._create_metrics()
    
    def _create_metrics(self):
        """Create OpenTelemetry metrics."""
        # Counter for total improvements
        self.improvement_counter = self.meter.create_counter(
            name="sifaka.improvements.total",
            description="Total number of text improvements",
            unit="1"
        )
        
        # Histogram for improvement duration
        self.duration_histogram = self.meter.create_histogram(
            name="sifaka.improvement.duration",
            description="Duration of text improvement operations",
            unit="s"
        )
        
        # Histogram for iterations
        self.iterations_histogram = self.meter.create_histogram(
            name="sifaka.improvement.iterations",
            description="Number of iterations per improvement",
            unit="1"
        )
        
        # Counter for tokens used
        self.tokens_counter = self.meter.create_counter(
            name="sifaka.tokens.total",
            description="Total tokens consumed",
            unit="1"
        )
        
        # Gauge for text length
        self.text_length_histogram = self.meter.create_histogram(
            name="sifaka.text.length",
            description="Length of texts being improved",
            unit="characters"
        )
        
        # Counter for errors
        self.error_counter = self.meter.create_counter(
            name="sifaka.errors.total",
            description="Total number of errors",
            unit="1"
        )
    
    async def pre_improve(self, text: str, **kwargs) -> Dict[str, Any]:
        """Start a span for the improvement operation."""
        context = {}
        
        if self.enable_traces:
            # Create a new span
            span = self.tracer.start_span(
                "sifaka.improve",
                attributes={
                    "text.length": len(text),
                    "critics": str(kwargs.get("critics", ["reflexion"])),
                    "max_iterations": kwargs.get("max_iterations", 3),
                    "model": kwargs.get("model", "default")
                }
            )
            context["span"] = span
            context["start_time"] = asyncio.get_event_loop().time()
        
        if self.enable_metrics:
            # Record text length
            self.text_length_histogram.record(
                len(text),
                attributes={"critic": kwargs.get("critics", ["reflexion"])[0]}
            )
        
        return context
    
    async def post_improve(
        self,
        result: SifakaResult,
        context: Dict[str, Any],
        error: Optional[Exception] = None
    ):
        """End the span and record metrics."""
        if self.enable_traces and "span" in context:
            span = context["span"]
            
            if error:
                # Record error
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
                span.record_exception(error)
                
                if self.enable_metrics:
                    self.error_counter.add(
                        1,
                        attributes={"error_type": type(error).__name__}
                    )
            else:
                # Add result attributes
                span.set_attributes({
                    "result.iterations": result.iterations,
                    "result.total_tokens": result.total_tokens,
                    "result.total_cost": result.total_cost,
                    "result.final_text_length": len(result.final_text)
                })
                
                if self.enable_metrics:
                    # Record successful improvement
                    self.improvement_counter.add(
                        1,
                        attributes={
                            "critic": result.metadata.get("critic", "unknown"),
                            "model": result.metadata.get("model", "unknown")
                        }
                    )
                    
                    # Record iterations
                    self.iterations_histogram.record(
                        result.iterations,
                        attributes={"critic": result.metadata.get("critic", "unknown")}
                    )
                    
                    # Record tokens
                    self.tokens_counter.add(
                        result.total_tokens,
                        attributes={"model": result.metadata.get("model", "unknown")}
                    )
                    
                    # Record duration
                    if "start_time" in context:
                        duration = asyncio.get_event_loop().time() - context["start_time"]
                        self.duration_histogram.record(
                            duration,
                            attributes={"critic": result.metadata.get("critic", "unknown")}
                        )
            
            # End the span
            span.end()
    
    async def on_critique(self, critique: str, iteration: int, **kwargs):
        """Create a span for each critique."""
        if self.enable_traces:
            with self.tracer.start_as_current_span(
                f"sifaka.critique.iteration_{iteration}",
                attributes={
                    "iteration": iteration,
                    "critique_length": len(critique)
                }
            ):
                pass  # Span will auto-close when exiting context
    
    async def on_generation(self, text: str, iteration: int, **kwargs):
        """Create a span for each generation."""
        if self.enable_traces:
            with self.tracer.start_as_current_span(
                f"sifaka.generation.iteration_{iteration}",
                attributes={
                    "iteration": iteration,
                    "generated_length": len(text)
                }
            ):
                pass


def setup_opentelemetry(
    service_name: str = "sifaka-app",
    otlp_endpoint: str = "localhost:4317"
) -> OpenTelemetryMiddleware:
    """Set up OpenTelemetry for Sifaka.
    
    Args:
        service_name: Name of your service
        otlp_endpoint: OTLP collector endpoint
        
    Returns:
        Configured OpenTelemetryMiddleware instance
    """
    return OpenTelemetryMiddleware(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        enable_traces=True,
        enable_metrics=True
    )


# Example usage
if __name__ == "__main__":
    # Note: This example requires an OpenTelemetry collector running
    # You can start one with Docker:
    # docker run -p 4317:4317 -p 55679:55679 otel/opentelemetry-collector:latest
    
    # Set up OpenTelemetry middleware
    otel_middleware = setup_opentelemetry(
        service_name="sifaka-example",
        otlp_endpoint="localhost:4317"
    )
    
    # Example 1: Simple improvement with telemetry
    text = "Machine learning is a type of AI that helps computers learn from data."
    
    try:
        result = improve_sync(
            text,
            critics=["reflexion"],
            max_iterations=2,
            middleware=[otel_middleware]
        )
        
        print(f"Original: {text}")
        print(f"Improved: {result.final_text}")
        print(f"Iterations: {result.iterations}")
        print(f"Tokens used: {result.total_tokens}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Async improvement with custom span
    async def improve_with_context():
        """Example showing custom span context."""
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span(
            "user_request",
            attributes={
                "user_id": "user123",
                "request_type": "blog_improvement"
            }
        ):
            result = await improve_async(
                "Write about the benefits of solar energy.",
                critics=["self_refine"],
                middleware=[otel_middleware]
            )
            return result
    
    # Run async example
    # asyncio.run(improve_with_context())
    
    print("\nTelemetry data is being sent to the OTLP collector.")
    print("View traces in Jaeger UI: http://localhost:16686")
    print("View metrics in Prometheus: http://localhost:9090")