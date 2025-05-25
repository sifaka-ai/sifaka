"""Metrics storage implementation using unified 3-tier architecture.

This module provides CachedMetricsStorage for storing and analyzing
performance metrics across all Sifaka components with vector search
capabilities for pattern analysis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from sifaka.utils.logging import get_logger

from .base import CachedStorage, StorageError

logger = get_logger(__name__)


class PerformanceMetric(BaseModel):
    """Performance metric data point.

    Represents a single performance measurement that can be stored
    and analyzed for bottleneck identification and optimization.

    Attributes:
        metric_id: Unique identifier for this metric.
        timestamp: When this metric was recorded.
        operation: Name of the operation being measured.
        component: Component that performed the operation.
        duration_ms: Duration in milliseconds.
        metadata: Additional metric metadata.
        tags: Tags for categorization and filtering.
    """

    # Identity
    metric_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    # Operation details
    operation: str
    component: str
    duration_ms: float

    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    def get_search_text(self) -> str:
        """Get text representation for vector similarity search."""
        return f"""
        Operation: {self.operation}
        Component: {self.component}
        Duration: {self.duration_ms}ms
        Tags: {', '.join(self.tags)}
        Metadata: {str(self.metadata)}
        """.strip()


class CachedMetricsStorage:
    """Storage for performance metrics.

    Provides metrics storage with vector similarity search for finding
    performance patterns and bottleneck analysis.

    Attributes:
        storage: Underlying CachedStorage instance.
    """

    def __init__(self, storage: CachedStorage):
        """Initialize cached metrics storage.

        Args:
            storage: CachedStorage instance for 3-tier storage.
        """
        self.storage = storage
        logger.debug("Initialized CachedMetricsStorage")

    def record_metric(
        self,
        operation: str,
        component: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Record a performance metric.

        Args:
            operation: Name of the operation.
            component: Component that performed the operation.
            duration_ms: Duration in milliseconds.
            metadata: Optional additional metadata.
            tags: Optional tags for categorization.

        Returns:
            The metric ID.

        Raises:
            StorageError: If the record operation fails.
        """
        try:
            metric = PerformanceMetric(
                operation=operation,
                component=component,
                duration_ms=duration_ms,
                metadata=metadata or {},
                tags=tags or [],
            )

            key = f"metric:{metric.metric_id}"

            # Prepare metadata for vector storage
            storage_metadata = {
                "operation": [operation],
                "component": [component],
                "duration_ms": [duration_ms],
                "timestamp": [metric.timestamp.isoformat()],
                "tags": tags or [],
            }

            # Save to all tiers
            self.storage.set(key, metric, storage_metadata)

            logger.debug(
                f"Recorded metric {metric.metric_id}: {component}.{operation} "
                f"took {duration_ms:.2f}ms"
            )

            return metric.metric_id

        except Exception as e:
            raise StorageError(
                f"Failed to record metric for {component}.{operation}",
                operation="record_metric",
                storage_type="CachedMetricsStorage",
                metadata={
                    "operation": operation,
                    "component": component,
                    "duration_ms": duration_ms,
                },
            ) from e

    def get_metric(self, metric_id: str) -> Optional[PerformanceMetric]:
        """Retrieve a metric by ID.

        Args:
            metric_id: The ID of the metric to retrieve.

        Returns:
            The metric if found, None otherwise.

        Raises:
            StorageError: If the retrieval operation fails.
        """
        try:
            key = f"metric:{metric_id}"
            metric = self.storage.get(key)

            if metric:
                logger.debug(f"Retrieved metric {metric_id}")
            else:
                logger.debug(f"Metric {metric_id} not found")

            return metric

        except Exception as e:
            raise StorageError(
                f"Failed to retrieve metric {metric_id}",
                operation="get_metric",
                storage_type="CachedMetricsStorage",
                metadata={"metric_id": metric_id},
            ) from e

    def get_metrics_by_operation(
        self, operation: str, component: Optional[str] = None, limit: int = 100
    ) -> List[PerformanceMetric]:
        """Get metrics for a specific operation.

        Args:
            operation: Operation name to filter by.
            component: Optional component name to filter by.
            limit: Maximum number of metrics to return.

        Returns:
            List of metrics for the operation.
        """
        try:
            # Create search query
            query_parts = [f"Operation: {operation}"]
            if component:
                query_parts.append(f"Component: {component}")

            query_text = " ".join(query_parts)
            similar_items = self.storage.search_similar(query_text, limit)

            # Filter to only return PerformanceMetric objects
            metrics = [
                item
                for item in similar_items
                if isinstance(item, PerformanceMetric) and item.operation == operation
            ]

            # Additional component filter if specified
            if component:
                metrics = [m for m in metrics if m.component == component]

            # Sort by timestamp (most recent first)
            metrics.sort(key=lambda m: m.timestamp, reverse=True)

            logger.debug(f"Found {len(metrics)} metrics for {operation}")
            return metrics[:limit]

        except Exception as e:
            logger.warning(f"Failed to get metrics for operation {operation}: {e}")
            return []

    def get_recent_metrics(self, hours: int = 24, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics within the specified time window.

        Args:
            hours: Number of hours to look back.
            limit: Maximum number of metrics to return.

        Returns:
            List of recent metrics.
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Get metrics from memory first
            recent_metrics = []
            for key, value in self.storage.memory.data.items():
                if (
                    key.startswith("metric:")
                    and isinstance(value, PerformanceMetric)
                    and value.timestamp >= cutoff_time
                ):
                    recent_metrics.append(value)

            # Sort by timestamp (most recent first)
            recent_metrics.sort(key=lambda m: m.timestamp, reverse=True)

            logger.debug(f"Found {len(recent_metrics)} recent metrics (last {hours}h)")
            return recent_metrics[:limit]

        except Exception as e:
            logger.warning(f"Failed to get recent metrics: {e}")
            return []

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time window.

        Args:
            hours: Number of hours to analyze.

        Returns:
            Dictionary containing performance summary statistics.
        """
        try:
            recent_metrics = self.get_recent_metrics(hours, limit=1000)

            if not recent_metrics:
                return {
                    "time_window_hours": hours,
                    "total_operations": 0,
                    "components": {},
                    "operations": {},
                    "bottlenecks": [],
                }

            # Analyze by component
            components = {}
            operations = {}

            for metric in recent_metrics:
                # Component stats
                if metric.component not in components:
                    components[metric.component] = {
                        "operation_count": 0,
                        "total_duration_ms": 0.0,
                        "avg_duration_ms": 0.0,
                        "max_duration_ms": 0.0,
                        "operations": set(),
                    }

                comp_stats = components[metric.component]
                comp_stats["operation_count"] = int(comp_stats["operation_count"]) + 1  # type: ignore
                comp_stats["total_duration_ms"] = (
                    float(comp_stats["total_duration_ms"]) + metric.duration_ms  # type: ignore
                )
                comp_stats["max_duration_ms"] = max(
                    float(comp_stats["max_duration_ms"]), metric.duration_ms  # type: ignore
                )
                if isinstance(comp_stats["operations"], set):
                    comp_stats["operations"].add(metric.operation)

                # Operation stats
                op_key = f"{metric.component}.{metric.operation}"
                if op_key not in operations:
                    operations[op_key] = {
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "avg_duration_ms": 0.0,
                        "max_duration_ms": 0.0,
                        "min_duration_ms": float("inf"),
                    }

                op_stats = operations[op_key]
                op_stats["count"] += 1
                op_stats["total_duration_ms"] += metric.duration_ms
                op_stats["max_duration_ms"] = max(op_stats["max_duration_ms"], metric.duration_ms)
                op_stats["min_duration_ms"] = min(op_stats["min_duration_ms"], metric.duration_ms)

            # Calculate averages
            for comp_stats in components.values():
                op_count = int(comp_stats["operation_count"])  # type: ignore
                if op_count > 0:
                    comp_stats["avg_duration_ms"] = (
                        float(comp_stats["total_duration_ms"]) / op_count  # type: ignore
                    )
                if isinstance(comp_stats["operations"], set):
                    comp_stats["operations"] = list(comp_stats["operations"])

            for op_stats in operations.values():
                if op_stats["count"] > 0:
                    op_stats["avg_duration_ms"] = op_stats["total_duration_ms"] / op_stats["count"]
                if op_stats["min_duration_ms"] == float("inf"):
                    op_stats["min_duration_ms"] = 0.0

            # Identify bottlenecks (operations with high average duration)
            bottlenecks = [
                {
                    "operation": op_key,
                    "avg_duration_ms": stats["avg_duration_ms"],
                    "count": stats["count"],
                }
                for op_key, stats in operations.items()
                if stats["avg_duration_ms"] > 100  # Operations taking > 100ms on average
            ]
            bottlenecks.sort(key=lambda b: b["avg_duration_ms"], reverse=True)  # type: ignore

            summary = {
                "time_window_hours": hours,
                "total_operations": len(recent_metrics),
                "components": components,
                "operations": operations,
                "bottlenecks": bottlenecks[:10],  # Top 10 bottlenecks
            }

            logger.debug(f"Generated performance summary for {hours}h window")
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate performance summary: {e}")
            return {"error": str(e)}

    def find_similar_performance_patterns(
        self, metric: PerformanceMetric, limit: int = 5
    ) -> List[PerformanceMetric]:
        """Find metrics with similar performance patterns.

        Args:
            metric: The metric to find similar patterns for.
            limit: Maximum number of similar metrics to return.

        Returns:
            List of metrics with similar performance patterns.
        """
        try:
            query_text = metric.get_search_text()
            similar_items = self.storage.search_similar(query_text, limit)

            # Filter to only return PerformanceMetric objects
            metrics = [
                item
                for item in similar_items
                if isinstance(item, PerformanceMetric) and item.metric_id != metric.metric_id
            ]

            logger.debug(f"Found {len(metrics)} similar performance patterns")
            return metrics

        except Exception as e:
            logger.warning(f"Similar performance patterns search failed: {e}")
            return []

    def cleanup_old_metrics(self, max_age_days: int = 30) -> int:
        """Clean up old metrics to save storage space.

        Args:
            max_age_days: Maximum age of metrics to keep.

        Returns:
            Number of metrics cleaned up.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0

            # Clean up from memory storage
            keys_to_remove = []
            for key, value in self.storage.memory.data.items():
                if (
                    key.startswith("metric:")
                    and isinstance(value, PerformanceMetric)
                    and value.timestamp < cutoff_date
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.storage.memory.data[key]
                cleaned_count += 1

            logger.debug(f"Cleaned up {cleaned_count} old metrics (older than {max_age_days} days)")
            return cleaned_count

        except Exception as e:
            logger.warning(f"Metrics cleanup failed: {e}")
            return 0

    def clear(self) -> None:
        """Clear all metrics storage."""
        self.storage.clear()
        logger.debug("Cleared all metrics storage")

    def get_stats(self) -> Dict[str, Any]:
        """Get metrics storage statistics."""
        base_stats = self.storage.get_stats()

        # Add metrics-specific stats
        metric_count = sum(
            1 for key in self.storage.memory.data.keys() if key.startswith("metric:")
        )

        return {
            **base_stats,
            "metric_count_in_memory": metric_count,
            "storage_type": "CachedMetricsStorage",
        }
