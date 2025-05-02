"""Tests for the monitoring module."""

import time
import unittest
from unittest.mock import patch, MagicMock
from sifaka.monitoring import TimingStats, PerformanceMonitor
from tests.base.test_base import BaseTestCase


class TestTimingStats(BaseTestCase):
    """Tests for TimingStats class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.stats = TimingStats()

    def test_initial_state(self):
        """Test initial state of TimingStats."""
        self.assertEqual(self.stats.count, 0)
        self.assertEqual(self.stats.total_time, 0.0)
        self.assertIsNone(self.stats.min_time)
        self.assertIsNone(self.stats.max_time)
        self.assertEqual(len(self.stats.times), 0)

    def test_add_measurement(self):
        """Test adding measurements."""
        # Add first measurement
        self.stats.add_measurement(1.0)
        self.assertEqual(self.stats.count, 1)
        self.assertEqual(self.stats.total_time, 1.0)
        self.assertEqual(self.stats.min_time, 1.0)
        self.assertEqual(self.stats.max_time, 1.0)
        self.assertEqual(len(self.stats.times), 1)

        # Add second measurement
        self.stats.add_measurement(2.0)
        self.assertEqual(self.stats.count, 2)
        self.assertEqual(self.stats.total_time, 3.0)
        self.assertEqual(self.stats.min_time, 1.0)
        self.assertEqual(self.stats.max_time, 2.0)
        self.assertEqual(len(self.stats.times), 2)

    def test_avg_time(self):
        """Test average time calculation."""
        self.stats.add_measurement(1.0)
        self.stats.add_measurement(2.0)
        self.stats.add_measurement(3.0)
        self.assertEqual(self.stats.avg_time, 2.0)

    def test_recent_avg_time(self):
        """Test recent average time calculation."""
        # Add measurements up to maxlen
        for i in range(1, 101):
            self.stats.add_measurement(float(i))

        # Recent average should be average of last 100 measurements
        self.assertAlmostEqual(self.stats.recent_avg_time, 50.5)

        # Add one more measurement
        self.stats.add_measurement(101.0)
        # Oldest measurement (1.0) should be dropped
        self.assertAlmostEqual(self.stats.recent_avg_time, 51.5)


class TestPerformanceMonitor(BaseTestCase):
    """Tests for PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.monitor = PerformanceMonitor()

    def test_initial_state(self):
        """Test initial state of PerformanceMonitor."""
        self.assertEqual(len(self.monitor.rule_times), 0)
        self.assertEqual(self.monitor.generation_times.count, 0)
        self.assertEqual(self.monitor.critique_times.count, 0)
        self.assertEqual(self.monitor.validation_times.count, 0)
        self.assertEqual(self.monitor.total_attempts, 0)
        self.assertEqual(self.monitor.successful_attempts, 0)
        self.assertEqual(self.monitor.failed_attempts, 0)
        self.assertEqual(self.monitor.cache_hits, 0)
        self.assertEqual(self.monitor.cache_misses, 0)

    def test_record_rule_time(self):
        """Test recording rule validation times."""
        self.monitor.record_rule_time("test_rule", 1.0)
        self.assertIn("test_rule", self.monitor.rule_times)
        self.assertEqual(self.monitor.rule_times["test_rule"].count, 1)
        self.assertEqual(self.monitor.rule_times["test_rule"].total_time, 1.0)

    def test_record_generation_time(self):
        """Test recording model generation times."""
        self.monitor.record_generation_time(1.0)
        self.assertEqual(self.monitor.generation_times.count, 1)
        self.assertEqual(self.monitor.generation_times.total_time, 1.0)

    def test_record_critique_time(self):
        """Test recording critique times."""
        self.monitor.record_critique_time(1.0)
        self.assertEqual(self.monitor.critique_times.count, 1)
        self.assertEqual(self.monitor.critique_times.total_time, 1.0)

    def test_record_validation_time(self):
        """Test recording validation times."""
        self.monitor.record_validation_time(1.0)
        self.assertEqual(self.monitor.validation_times.count, 1)
        self.assertEqual(self.monitor.validation_times.total_time, 1.0)

    def test_record_attempt(self):
        """Test recording validation attempts."""
        # Test successful attempt
        self.monitor.record_attempt(True)
        self.assertEqual(self.monitor.total_attempts, 1)
        self.assertEqual(self.monitor.successful_attempts, 1)
        self.assertEqual(self.monitor.failed_attempts, 0)

        # Test failed attempt
        self.monitor.record_attempt(False)
        self.assertEqual(self.monitor.total_attempts, 2)
        self.assertEqual(self.monitor.successful_attempts, 1)
        self.assertEqual(self.monitor.failed_attempts, 1)

    def test_record_cache_access(self):
        """Test recording cache accesses."""
        # Test cache hit
        self.monitor.record_cache_access(True)
        self.assertEqual(self.monitor.cache_hits, 1)
        self.assertEqual(self.monitor.cache_misses, 0)

        # Test cache miss
        self.monitor.record_cache_access(False)
        self.assertEqual(self.monitor.cache_hits, 1)
        self.assertEqual(self.monitor.cache_misses, 1)

    def test_get_metrics(self):
        """Test getting performance metrics."""
        # Record some test data
        self.monitor.record_rule_time("test_rule", 1.0)
        self.monitor.record_generation_time(2.0)
        self.monitor.record_critique_time(3.0)
        self.monitor.record_validation_time(4.0)
        self.monitor.record_attempt(True)
        self.monitor.record_attempt(False)
        self.monitor.record_cache_access(True)
        self.monitor.record_cache_access(False)

        # Get metrics
        metrics = self.monitor.get_metrics()

        # Verify metrics structure and values
        self.assertIn("uptime_seconds", metrics)
        self.assertIn("rule_validation", metrics)
        self.assertIn("model_generation", metrics)
        self.assertIn("critique", metrics)
        self.assertIn("validation", metrics)
        self.assertIn("attempts", metrics)
        self.assertIn("cache", metrics)

        # Verify specific values
        self.assertEqual(metrics["attempts"]["total"], 2)
        self.assertEqual(metrics["attempts"]["successful"], 1)
        self.assertEqual(metrics["attempts"]["failed"], 1)
        self.assertEqual(metrics["attempts"]["success_rate"], 0.5)

        self.assertEqual(metrics["cache"]["hits"], 1)
        self.assertEqual(metrics["cache"]["misses"], 1)
        self.assertEqual(metrics["cache"]["hit_rate"], 0.5)

    @patch('sifaka.monitoring.logger')
    def test_log_metrics(self, mock_logger):
        """Test logging performance metrics."""
        # Record some test data
        self.monitor.record_rule_time("test_rule", 1.0)
        self.monitor.record_generation_time(2.0)

        # Log metrics
        self.monitor.log_metrics()

        # Verify logger was called
        self.assertTrue(mock_logger.info.called)
        self.assertGreater(mock_logger.info.call_count, 0)


if __name__ == "__main__":
    unittest.main()