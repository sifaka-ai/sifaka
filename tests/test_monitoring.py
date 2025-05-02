"""Test module for sifaka.monitoring."""

import time
from unittest.mock import patch

import pytest

from sifaka.monitoring import PerformanceMonitor, TimingStats


class TestTimingStats:
    """Tests for the TimingStats class."""

    def test_empty_timing_stats(self):
        """Test an empty TimingStats object."""
        stats = TimingStats()
        assert stats.count == 0
        assert stats.total_time == 0.0
        assert stats.min_time is None
        assert stats.max_time is None
        assert stats.avg_time == 0.0
        assert stats.recent_avg_time == 0.0

    def test_add_measurement(self):
        """Test adding measurements to TimingStats."""
        stats = TimingStats()
        stats.add_measurement(0.1)

        assert stats.count == 1
        assert stats.total_time == 0.1
        assert stats.min_time == 0.1
        assert stats.max_time == 0.1
        assert stats.avg_time == 0.1
        assert stats.recent_avg_time == 0.1

        # Add another measurement
        stats.add_measurement(0.3)
        assert stats.count == 2
        assert stats.total_time == 0.4
        assert stats.min_time == 0.1
        assert stats.max_time == 0.3
        assert stats.avg_time == 0.2
        assert stats.recent_avg_time == 0.2

    def test_min_max_updates(self):
        """Test that min and max values are properly updated."""
        stats = TimingStats()
        stats.add_measurement(0.5)
        assert stats.min_time == 0.5
        assert stats.max_time == 0.5

        # Add a lower value
        stats.add_measurement(0.2)
        assert stats.min_time == 0.2
        assert stats.max_time == 0.5

        # Add a higher value
        stats.add_measurement(0.8)
        assert stats.min_time == 0.2
        assert stats.max_time == 0.8

    def test_recent_avg_time(self):
        """Test recent average time calculations."""
        stats = TimingStats()

        # Fill with more than maxlen measurements to test deque behavior
        for i in range(101):
            stats.add_measurement(i)

        # The deque maxlen is 100, so only last 100 should be used
        # Average of 1..100 = 50.5
        assert len(stats.times) == 100
        assert stats.recent_avg_time == 50.5


class TestPerformanceMonitor:
    """Tests for the PerformanceMonitor class."""

    def test_init(self):
        """Test initialization of PerformanceMonitor."""
        monitor = PerformanceMonitor()
        assert monitor.total_attempts == 0
        assert monitor.successful_attempts == 0
        assert monitor.failed_attempts == 0
        assert monitor.cache_hits == 0
        assert monitor.cache_misses == 0
        assert isinstance(monitor.rule_times, dict)
        assert monitor.rule_times == {}
        assert isinstance(monitor.generation_times, TimingStats)
        assert isinstance(monitor.critique_times, TimingStats)
        assert isinstance(monitor.validation_times, TimingStats)

    def test_record_rule_time(self):
        """Test recording rule validation times."""
        monitor = PerformanceMonitor()

        # First time for a rule
        monitor.record_rule_time("test_rule", 0.1)
        assert "test_rule" in monitor.rule_times
        assert monitor.rule_times["test_rule"].count == 1
        assert monitor.rule_times["test_rule"].total_time == 0.1

        # Second time for the same rule
        monitor.record_rule_time("test_rule", 0.2)
        assert monitor.rule_times["test_rule"].count == 2
        assert monitor.rule_times["test_rule"].total_time == 0.3

        # First time for another rule
        monitor.record_rule_time("another_rule", 0.5)
        assert "another_rule" in monitor.rule_times
        assert monitor.rule_times["another_rule"].count == 1
        assert monitor.rule_times["another_rule"].total_time == 0.5

    def test_record_generation_time(self):
        """Test recording model generation times."""
        monitor = PerformanceMonitor()
        monitor.record_generation_time(0.1)
        assert monitor.generation_times.count == 1
        assert monitor.generation_times.total_time == 0.1

        monitor.record_generation_time(0.2)
        assert monitor.generation_times.count == 2
        assert monitor.generation_times.total_time == 0.3

    def test_record_critique_time(self):
        """Test recording critique times."""
        monitor = PerformanceMonitor()
        monitor.record_critique_time(0.1)
        assert monitor.critique_times.count == 1
        assert monitor.critique_times.total_time == 0.1

        monitor.record_critique_time(0.2)
        assert monitor.critique_times.count == 2
        assert monitor.critique_times.total_time == 0.3

    def test_record_validation_time(self):
        """Test recording validation times."""
        monitor = PerformanceMonitor()
        monitor.record_validation_time(0.1)
        assert monitor.validation_times.count == 1
        assert monitor.validation_times.total_time == 0.1

        monitor.record_validation_time(0.2)
        assert monitor.validation_times.count == 2
        assert monitor.validation_times.total_time == 0.3

    def test_record_attempt(self):
        """Test recording validation attempts."""
        monitor = PerformanceMonitor()

        # Successful attempt
        monitor.record_attempt(True)
        assert monitor.total_attempts == 1
        assert monitor.successful_attempts == 1
        assert monitor.failed_attempts == 0

        # Failed attempt
        monitor.record_attempt(False)
        assert monitor.total_attempts == 2
        assert monitor.successful_attempts == 1
        assert monitor.failed_attempts == 1

    def test_record_cache_access(self):
        """Test recording cache accesses."""
        monitor = PerformanceMonitor()

        # Cache hit
        monitor.record_cache_access(True)
        assert monitor.cache_hits == 1
        assert monitor.cache_misses == 0

        # Cache miss
        monitor.record_cache_access(False)
        assert monitor.cache_hits == 1
        assert monitor.cache_misses == 1

    def test_get_metrics(self):
        """Test getting performance metrics."""
        monitor = PerformanceMonitor()

        # Add some test data
        monitor.record_rule_time("test_rule", 0.1)
        monitor.record_generation_time(0.2)
        monitor.record_critique_time(0.3)
        monitor.record_validation_time(0.4)
        monitor.record_attempt(True)
        monitor.record_attempt(False)
        monitor.record_cache_access(True)
        monitor.record_cache_access(False)

        # Get metrics
        metrics = monitor.get_metrics()

        # Check all components are present
        assert "uptime_seconds" in metrics
        assert "rule_validation" in metrics
        assert "model_generation" in metrics
        assert "critique" in metrics
        assert "validation" in metrics
        assert "attempts" in metrics
        assert "cache" in metrics

        # Check rule validation details
        assert "test_rule" in metrics["rule_validation"]
        assert metrics["rule_validation"]["test_rule"]["count"] == 1
        assert metrics["rule_validation"]["test_rule"]["avg_time_ms"] == 100.0  # 0.1 seconds = 100ms

        # Check model generation
        assert metrics["model_generation"]["count"] == 1
        assert metrics["model_generation"]["avg_time_ms"] == 200.0  # 0.2 seconds = 200ms

        # Check critique
        assert metrics["critique"]["count"] == 1
        assert metrics["critique"]["avg_time_ms"] == 300.0  # 0.3 seconds = 300ms

        # Check validation
        assert metrics["validation"]["count"] == 1
        assert metrics["validation"]["avg_time_ms"] == 400.0  # 0.4 seconds = 400ms

        # Check attempts
        assert metrics["attempts"]["total"] == 2
        assert metrics["attempts"]["successful"] == 1
        assert metrics["attempts"]["failed"] == 1
        assert metrics["attempts"]["success_rate"] == 0.5

        # Check cache
        assert metrics["cache"]["hits"] == 1
        assert metrics["cache"]["misses"] == 1
        assert metrics["cache"]["hit_rate"] == 0.5

    @patch("sifaka.monitoring.time.time")
    def test_uptime(self, mock_time):
        """Test uptime calculation."""
        # Mock time.time() to return fixed values
        mock_time.side_effect = [100.0, 150.0]  # First call returns 100, second call returns 150

        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()

        # Uptime should be the difference between the two time.time() calls
        assert metrics["uptime_seconds"] == 50.0

    @patch("sifaka.monitoring.logger")
    def test_log_metrics(self, mock_logger):
        """Test logging metrics."""
        monitor = PerformanceMonitor()

        # Add some test data
        monitor.record_rule_time("test_rule", 0.1)
        monitor.record_generation_time(0.2)

        # Call log_metrics
        monitor.log_metrics()

        # Verify that logger.info was called
        assert mock_logger.info.called

        # First call should be the header
        assert mock_logger.info.call_args_list[0][0][0] == "Performance metrics:"