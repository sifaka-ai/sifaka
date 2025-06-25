"""Performance monitoring for Sifaka operations."""

import time
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

from .models import SifakaResult


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for an improvement operation."""

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
        """Calculate final metrics."""
        self.total_duration = self.end_time - self.start_time

        if self.tokens_used > 0 and self.llm_time > 0:
            self.tokens_per_second = self.tokens_used / self.llm_time

        if self.confidence_scores:
            self.final_confidence = self.confidence_scores[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
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
        """Human-readable summary."""
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
    """Monitors performance of Sifaka operations."""

    def __init__(self) -> None:
        """Initialize monitor."""
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.history: List[PerformanceMetrics] = []

    def start_monitoring(self, max_iterations: int = 0) -> PerformanceMetrics:
        """Start monitoring a new operation."""
        self.current_metrics = PerformanceMetrics(
            start_time=time.time(), max_iterations=max_iterations
        )
        return self.current_metrics

    def end_monitoring(self) -> PerformanceMetrics:
        """End monitoring and finalize metrics."""
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
        """Track an LLM call."""
        if not self.current_metrics:
            return await func()

        start = time.time()
        try:
            result = await func()
            self.current_metrics.llm_calls += 1
            self.current_metrics.llm_time += time.time() - start
            return result
        except Exception as e:
            self.current_metrics.errors.append(
                {"type": "llm_error", "error": str(e), "timestamp": time.time()}
            )
            raise

    async def track_critic_call(self, critic_name: str, func: Callable[[], Any]) -> Any:
        """Track a critic call."""
        if not self.current_metrics:
            return await func()

        start = time.time()
        try:
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
    monitor = PerformanceMonitor()

    # Could patch engine methods here for automatic tracking
    # For now, manual tracking is required

    yield monitor

    if print_summary and monitor.current_metrics:
        metrics = monitor.end_monitoring()
        print(metrics)


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _global_monitor
