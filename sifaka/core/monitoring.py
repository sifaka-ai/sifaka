"""Performance monitoring and observability for Sifaka operations.

This module provides comprehensive monitoring capabilities for tracking
the performance and behavior of text improvement operations. It collects
detailed metrics about LLM calls, critic evaluations, validation checks,
and overall processing time.

## Key Features:

- **Detailed Metrics**: Track timing, token usage, and call counts
- **Performance Analysis**: Identify bottlenecks and optimization opportunities
- **Error Tracking**: Capture and categorize failures for debugging
- **Integration**: Optional Logfire integration for production monitoring
- **Context Managers**: Easy instrumentation of operations

## Usage:

    >>> # Basic monitoring
    >>> monitor = PerformanceMonitor()
    >>> async with monitor.track_operation("improve") as metrics:
    ...     result = await improve("text")
    >>> print(metrics.to_dict())

    >>> # Global monitoring
    >>> from sifaka import get_global_monitor
    >>> monitor = get_global_monitor()
    >>>
    >>> # Context manager for specific operations
    >>> async with monitor_context("critical_operation"):
    ...     await process_important_text()

## Metrics Collected:

- **Timing**: Total duration, LLM time, critic time, validator time
- **LLM Usage**: Call count, tokens used, tokens per second
- **Critics**: Which critics ran, how long they took
- **Validators**: Which validators ran, pass/fail rates
- **Iterations**: How many improvement rounds occurred
- **Confidence**: Tracking confidence scores over time
- **Errors**: Detailed error information for debugging

## Integration with Logfire:

Set the LOGFIRE_TOKEN environment variable to enable production monitoring:

    export LOGFIRE_TOKEN=your_token_here

This will send detailed traces and metrics to Logfire for analysis.
"""

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .models import SifakaResult

if TYPE_CHECKING:
    import logfire as logfire_module
else:
    try:
        import logfire as logfire_module
    except ImportError:
        logfire_module = None  # type: ignore[assignment]

logfire = logfire_module

# Configure logfire if token is available
if logfire and os.getenv("LOGFIRE_TOKEN"):
    # Set service name in environment for OpenTelemetry
    os.environ["OTEL_SERVICE_NAME"] = "sifaka"
    os.environ["OTEL_SERVICE_VERSION"] = "0.1.6"

    logfire.configure(
        token=os.getenv("LOGFIRE_TOKEN"), service_name="sifaka", service_version="0.1.6"
    )


@dataclass
class PerformanceMetrics:
    """Comprehensive metrics for a single text improvement operation.

    This dataclass collects all performance-related data during the
    execution of an improve() call. It tracks timing, resource usage,
    and quality metrics to provide full observability.

    The metrics are organized into categories:
    - Timing: How long different operations took
    - LLM: Language model usage and performance
    - Critics: Critic evaluation metrics
    - Validators: Validation check metrics
    - Iterations: Progress through improvement cycles
    - Results: Quality and confidence measurements
    - Memory: Resource usage tracking
    - Errors: Failure information

    Example:
        >>> metrics = PerformanceMetrics(start_time=time.time())
        >>> # ... perform operations ...
        >>> metrics.llm_calls += 1
        >>> metrics.tokens_used += 150
        >>> metrics.end_time = time.time()
        >>> metrics.finalize()
        >>> print(f"Total time: {metrics.total_duration:.2f}s")
        >>> print(f"Tokens/sec: {metrics.tokens_per_second:.1f}")

    Attributes:
        start_time: Unix timestamp when operation began
        end_time: Unix timestamp when operation completed
        total_duration: Total seconds elapsed (calculated)
        llm_calls: Number of LLM API calls made
        llm_time: Total seconds spent in LLM calls
        tokens_used: Total tokens consumed across all calls
        tokens_per_second: Token generation rate (calculated)
        critic_calls: Number of critic evaluations
        critic_time: Total seconds spent in critic evaluation
        critics_used: List of critic names that were called
        validator_calls: Number of validation checks
        validator_time: Total seconds spent in validation
        validators_used: List of validator names that were called
        iterations_completed: Number of improvement iterations finished
        max_iterations: Maximum iterations allowed
        confidence_scores: Confidence score from each iteration
        final_confidence: Final confidence score achieved
        improvement_achieved: Whether text was successfully improved
        generations_count: Number of text generations created
        critiques_count: Number of critiques generated
        validations_count: Number of validation results
        errors: List of error details for debugging
    """

    # Timing metrics
    start_time: float
    end_time: float = 0.0
    total_duration: float = 0.0

    # LLM metrics
    llm_calls: int = 0
    llm_time: float = 0.0
    tokens_used: int = 0
    tokens_per_second: float = 0.0

    # Critic metrics
    critic_calls: int = 0
    critic_time: float = 0.0
    critics_used: List[str] = field(default_factory=list)

    # Validator metrics
    validator_calls: int = 0
    validator_time: float = 0.0
    validators_used: List[str] = field(default_factory=list)

    # Iteration metrics
    iterations_completed: int = 0
    max_iterations: int = 0

    # Result metrics
    confidence_scores: List[float] = field(default_factory=list)
    final_confidence: float = 0.0
    improvement_achieved: bool = False

    # Memory metrics
    generations_count: int = 0
    critiques_count: int = 0
    validations_count: int = 0

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def finalize(self) -> None:
        """Calculate derived metrics after operation completes.

        This method should be called after all operations are finished
        to calculate final values like total duration and rates.

        Calculates:
        - total_duration from start and end times
        - tokens_per_second from tokens and time
        - final_confidence from confidence history
        """
        self.total_duration = self.end_time - self.start_time

        if self.tokens_used > 0 and self.llm_time > 0:
            self.tokens_per_second = self.tokens_used / self.llm_time

        if self.confidence_scores:
            self.final_confidence = self.confidence_scores[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a structured dictionary for reporting.

        Organizes metrics into logical groups for easy analysis and
        serialization. All timing values are rounded to 3 decimal places
        for readability.

        Returns:
            Dictionary with metrics organized by category:
            - timing: Duration measurements
            - llm: Language model metrics
            - critics: Critic evaluation metrics
            - validators: Validation metrics
            - iterations: Progress metrics
            - quality: Confidence and improvement metrics
            - errors: Error count and details

        Example:
            >>> metrics_dict = metrics.to_dict()
            >>> print(f"Total time: {metrics_dict['timing']['total_duration']}")
            >>> print(f"LLM calls: {metrics_dict['llm']['calls']}")
        """
        return {
            "timing": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration": round(self.total_duration, 3),
                "llm_time": round(self.llm_time, 3),
                "critic_time": round(self.critic_time, 3),
                "validator_time": round(self.validator_time, 3),
            },
            "llm": {
                "calls": self.llm_calls,
                "tokens_used": self.tokens_used,
                "tokens_per_second": round(self.tokens_per_second, 1),
            },
            "critics": {
                "calls": self.critic_calls,
                "critics_used": list(set(self.critics_used)),
                "avg_time_per_call": (
                    round(self.critic_time / self.critic_calls, 3)
                    if self.critic_calls > 0
                    else 0
                ),
            },
            "validators": {
                "calls": self.validator_calls,
                "validators_used": list(set(self.validators_used)),
                "avg_time_per_call": (
                    round(self.validator_time / self.validator_calls, 3)
                    if self.validator_calls > 0
                    else 0
                ),
            },
            "iterations": {
                "completed": self.iterations_completed,
                "max": self.max_iterations,
                "completion_rate": (
                    round(self.iterations_completed / self.max_iterations, 2)
                    if self.max_iterations > 0
                    else 0
                ),
            },
            "results": {
                "confidence_progression": [round(c, 3) for c in self.confidence_scores],
                "final_confidence": round(self.final_confidence, 3),
                "improvement_achieved": self.improvement_achieved,
            },
            "memory": {
                "generations": self.generations_count,
                "critiques": self.critiques_count,
                "validations": self.validations_count,
            },
            "errors": self.errors,
        }

    def __str__(self) -> str:
        """Generate a human-readable performance summary.

        Provides a concise overview of key metrics suitable for logging
        or console output. Formats percentages and rates for readability.

        Returns:
            Multi-line string with performance highlights
        """
        return (
            f"Performance Summary:\n"
            f"  Duration: {self.total_duration:.2f}s\n"
            f"  LLM Calls: {self.llm_calls} ({self.tokens_used} tokens)\n"
            f"  Critic Calls: {self.critic_calls}\n"
            f"  Iterations: {self.iterations_completed}/{self.max_iterations}\n"
            f"  Final Confidence: {self.final_confidence:.2%}\n"
            f"  Tokens/sec: {self.tokens_per_second:.1f}"
        )


class PerformanceMonitor:
    """Central performance monitoring system for Sifaka operations.

    This class provides comprehensive tracking of all performance-related
    metrics during text improvement operations. It integrates with Logfire
    for production monitoring and maintains a history of operations for
    analysis.

    The monitor tracks:
    - LLM API calls and token usage
    - Critic evaluation performance
    - Validator execution time
    - Overall operation timing
    - Error rates and types

    Example:
        >>> monitor = PerformanceMonitor()
        >>> metrics = monitor.start_monitoring(max_iterations=3)
        >>>
        >>> # Track operations
        >>> await monitor.track_llm_call(async_llm_function)
        >>> await monitor.track_critic_call("reflexion", critic_func)
        >>>
        >>> # Finalize and get results
        >>> final_metrics = monitor.end_monitoring()
        >>> print(final_metrics)

    The monitor is thread-safe for the current operation but should not
    be shared across concurrent improve() calls.
    """

    def __init__(self) -> None:
        """Initialize a new performance monitor.

        Creates an empty monitor ready to track operations. Each monitor
        instance should be used for a single improve() call or a series
        of related operations.
        """
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.history: List[PerformanceMetrics] = []

    def start_monitoring(self, max_iterations: int = 0) -> PerformanceMetrics:
        """Begin monitoring a new text improvement operation.

        Initializes metrics collection and starts timing. Only one operation
        can be monitored at a time per monitor instance.

        Args:
            max_iterations: Maximum iterations configured for this operation.
                Used to calculate completion percentage.

        Returns:
            New PerformanceMetrics instance that will be populated during
            the operation. The same instance is available via current_metrics.

        Raises:
            RuntimeError: If monitoring is already active

        Example:
            >>> metrics = monitor.start_monitoring(max_iterations=5)
            >>> # ... perform operations ...
            >>> metrics.iterations_completed = 3
        """
        self.current_metrics = PerformanceMetrics(
            start_time=time.time(), max_iterations=max_iterations
        )
        return self.current_metrics

    def end_monitoring(self) -> PerformanceMetrics:
        """Complete monitoring and finalize all metrics.

        Stops timing, calculates derived metrics, and adds the completed
        metrics to history. The monitor is then ready for a new operation.

        Returns:
            Finalized PerformanceMetrics with all calculations complete

        Raises:
            RuntimeError: If no monitoring session is active

        Example:
            >>> metrics = monitor.end_monitoring()
            >>> print(f"Operation took {metrics.total_duration:.2f} seconds")
            >>> print(f"Used {metrics.tokens_used} tokens")
        """
        if not self.current_metrics:
            raise RuntimeError("No active monitoring session")

        self.current_metrics.end_time = time.time()
        self.current_metrics.finalize()

        # Add to history
        self.history.append(self.current_metrics)

        # Return and clear current
        metrics = self.current_metrics
        self.current_metrics = None
        return metrics

    async def track_llm_call(self, func: Callable[[], Any]) -> Any:
        """Track execution of an LLM API call.

        Wraps an async function that makes an LLM call, tracking its
        duration and handling errors. Integrates with Logfire for
        distributed tracing.

        Args:
            func: Async function that makes an LLM API call

        Returns:
            Whatever the wrapped function returns

        Raises:
            Any exception from the wrapped function (after logging)

        Example:
            >>> async def generate_text():
            ...     return await llm_client.complete(prompt)
            >>>
            >>> result = await monitor.track_llm_call(generate_text)
        """
        if not self.current_metrics:
            return await func()

        start = time.time()
        try:
            # Use logfire span if available with rich attributes
            if logfire:
                with logfire.span(
                    "llm_call",
                    llm_call_number=self.current_metrics.llm_calls + 1,
                    llm_tokens_before=self.current_metrics.tokens_used,
                    llm_type=(
                        "generation"
                        if self.current_metrics.critic_calls > 0
                        else "critic"
                    ),
                ) as span:
                    result = await func()
                    duration = time.time() - start

                    # Update metrics
                    self.current_metrics.llm_calls += 1
                    self.current_metrics.llm_time += duration

                    # Extract token usage if available
                    tokens_added = 0
                    if isinstance(result, tuple) and len(result) >= 3:
                        # Result is (text, prompt, tokens, time) from generator
                        tokens_added = result[2] if result[2] else 0
                        self.current_metrics.tokens_used += tokens_added
                    elif hasattr(result, "usage") and result.usage:
                        tokens_added = result.usage.get("total_tokens", 0)
                        self.current_metrics.tokens_used += tokens_added

                    # Log rich metrics
                    span.set_attribute("llm.duration_seconds", round(duration, 3))
                    span.set_attribute("llm.tokens_used", tokens_added)
                    span.set_attribute(
                        "llm.total_tokens_so_far", self.current_metrics.tokens_used
                    )
                    span.set_attribute(
                        "llm.tokens_per_second",
                        round(tokens_added / duration, 1) if duration > 0 else 0,
                    )
            else:
                result = await func()
                self.current_metrics.llm_calls += 1
                self.current_metrics.llm_time += time.time() - start

            return result
        except Exception as e:
            self.current_metrics.errors.append(
                {"type": "llm_error", "error": str(e), "timestamp": time.time()}
            )
            if logfire:
                logfire.error(
                    "LLM call failed", error=str(e), error_type=type(e).__name__
                )
            raise

    async def track_critic_call(self, critic_name: str, func: Callable[[], Any]) -> Any:
        """Track execution of a critic evaluation.

        Wraps an async function that runs a critic, tracking its duration
        and which critic was used. Helps identify slow critics.

        Args:
            critic_name: Name of the critic being called (e.g., "reflexion")
            func: Async function that runs the critic

        Returns:
            Whatever the wrapped function returns

        Raises:
            Any exception from the wrapped function (after logging)

        Example:
            >>> async def run_critic():
            ...     return await critic.critique(text, result)
            >>>
            >>> critique = await monitor.track_critic_call("reflexion", run_critic)
        """
        if not self.current_metrics:
            return await func()

        start = time.time()
        try:
            # Use logfire span if available with detailed attributes
            if logfire:
                with logfire.span(
                    "critic_call",
                    critic_name=critic_name,
                    critic_call_number=self.current_metrics.critic_calls + 1,
                    iteration=self.current_metrics.iterations_completed,
                ) as span:
                    result = await func()
                    duration = time.time() - start

                    # Update metrics
                    self.current_metrics.critic_calls += 1
                    self.current_metrics.critic_time += duration
                    self.current_metrics.critics_used.append(critic_name)

                    # Extract critique details if available
                    if hasattr(result, "confidence"):
                        span.set_attribute(
                            "critic.confidence", round(result.confidence, 3)
                        )
                    if hasattr(result, "needs_improvement"):
                        span.set_attribute(
                            "critic.needs_improvement", result.needs_improvement
                        )
                    if hasattr(result, "suggestions") and result.suggestions:
                        span.set_attribute(
                            "critic.suggestion_count", len(result.suggestions)
                        )
                        # Log first few suggestions for debugging
                        span.set_attribute(
                            "critic.suggestions_preview", result.suggestions[:2]
                        )
                    if hasattr(result, "model_used"):
                        span.set_attribute("critic.model_used", result.model_used)
                    if hasattr(result, "tokens_used"):
                        span.set_attribute("critic.tokens_used", result.tokens_used)

                    span.set_attribute("critic.duration_seconds", round(duration, 3))
            else:
                result = await func()
                self.current_metrics.critic_calls += 1
                self.current_metrics.critic_time += time.time() - start
                self.current_metrics.critics_used.append(critic_name)

            return result
        except Exception as e:
            self.current_metrics.errors.append(
                {
                    "type": "critic_error",
                    "critic": critic_name,
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )
            if logfire:
                logfire.error(
                    "Critic evaluation failed",
                    critic_name=critic_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
            raise

    async def track_validator_call(
        self, validator_name: str, func: Callable[[], Any]
    ) -> Any:
        """Track a validator call."""
        if not self.current_metrics:
            return await func()

        start = time.time()
        try:
            result = await func()
            self.current_metrics.validator_calls += 1
            self.current_metrics.validator_time += time.time() - start
            self.current_metrics.validators_used.append(validator_name)
            return result
        except Exception as e:
            self.current_metrics.errors.append(
                {
                    "type": "validator_error",
                    "validator": validator_name,
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )
            raise

    def update_from_result(self, result: SifakaResult) -> None:
        """Update metrics from a result object."""
        if not self.current_metrics:
            return

        # Update iteration count
        self.current_metrics.iterations_completed = result.iteration

        # Update confidence scores from critiques
        for critique in result.critiques:
            if critique.confidence is not None and critique.confidence > 0:
                self.current_metrics.confidence_scores.append(critique.confidence)

        # Update improvement status
        self.current_metrics.improvement_achieved = not result.needs_improvement

        # Update memory metrics
        self.current_metrics.generations_count = len(result.generations)
        self.current_metrics.critiques_count = len(result.critiques)
        self.current_metrics.validations_count = len(result.validations)

        # Update token count
        for gen in result.generations:
            self.current_metrics.tokens_used += gen.tokens_used

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from history."""
        if not self.history:
            return {}

        durations = [m.total_duration for m in self.history]
        tokens = [m.tokens_used for m in self.history]
        iterations = [m.iterations_completed for m in self.history]
        confidences = [m.final_confidence for m in self.history]

        return {
            "total_operations": len(self.history),
            "average_duration": sum(durations) / len(durations),
            "total_tokens": sum(tokens),
            "average_iterations": sum(iterations) / len(iterations),
            "average_confidence": sum(confidences) / len(confidences),
            "success_rate": sum(1 for m in self.history if m.improvement_achieved)
            / len(self.history),
        }


# Global monitor instance
_global_monitor = PerformanceMonitor()


@asynccontextmanager
async def monitor(
    track_llm: bool = True,
    track_critics: bool = True,
    track_validators: bool = True,
    print_summary: bool = True,
) -> AsyncIterator[PerformanceMonitor]:
    """Context manager for performance monitoring.

    Args:
        track_llm: Whether to track LLM calls
        track_critics: Whether to track critic calls
        track_validators: Whether to track validator calls
        print_summary: Whether to print summary on exit

    Yields:
        PerformanceMonitor instance
    """
    # Use the global monitor instead of creating a new one
    monitor = get_global_monitor()

    # Start logfire span for the entire operation if available
    try:
        if logfire:
            # Use logfire context with rich initial attributes
            with logfire.span(
                "sifaka_improve",
                track_llm=track_llm,
                track_critics=track_critics,
                track_validators=track_validators,
                sifaka_version="0.1.6",
            ) as span:
                # Don't start monitoring here - let the caller do it
                yield monitor

                # Finalize and log comprehensive metrics
                if monitor.current_metrics:
                    final_metrics = monitor.end_monitoring()

                    # Log detailed performance data
                    span.set_attribute(
                        "performance.total_duration_seconds",
                        round(final_metrics.total_duration, 3),
                    )
                    span.set_attribute("performance.llm_calls", final_metrics.llm_calls)
                    span.set_attribute(
                        "performance.llm_time_seconds", round(final_metrics.llm_time, 3)
                    )
                    span.set_attribute(
                        "performance.tokens_used", final_metrics.tokens_used
                    )
                    span.set_attribute(
                        "performance.tokens_per_second",
                        round(final_metrics.tokens_per_second, 1),
                    )

                    span.set_attribute(
                        "critics.total_calls", final_metrics.critic_calls
                    )
                    span.set_attribute(
                        "critics.total_time_seconds",
                        round(final_metrics.critic_time, 3),
                    )
                    span.set_attribute(
                        "critics.unique_critics", len(set(final_metrics.critics_used))
                    )
                    span.set_attribute(
                        "critics.names", list(set(final_metrics.critics_used))
                    )

                    span.set_attribute(
                        "result.iterations_completed",
                        final_metrics.iterations_completed,
                    )
                    span.set_attribute(
                        "result.max_iterations", final_metrics.max_iterations
                    )
                    span.set_attribute(
                        "result.final_confidence",
                        round(final_metrics.final_confidence, 3),
                    )
                    span.set_attribute(
                        "result.improvement_achieved",
                        final_metrics.improvement_achieved,
                    )
                    span.set_attribute("result.error_count", len(final_metrics.errors))

                    # Log confidence progression
                    if final_metrics.confidence_scores:
                        span.set_attribute(
                            "confidence.progression",
                            [round(c, 3) for c in final_metrics.confidence_scores],
                        )
                        span.set_attribute(
                            "confidence.initial",
                            round(final_metrics.confidence_scores[0], 3),
                        )
                        span.set_attribute(
                            "confidence.improvement",
                            round(
                                final_metrics.final_confidence
                                - final_metrics.confidence_scores[0],
                                3,
                            ),
                        )

                    if print_summary:
                        print(final_metrics)
        else:
            # Just yield the monitor without logfire
            yield monitor

    finally:
        if print_summary and monitor.current_metrics and not logfire:
            metrics = monitor.end_monitoring()
            print(metrics)


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _global_monitor
