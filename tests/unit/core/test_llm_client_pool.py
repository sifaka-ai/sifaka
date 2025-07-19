"""Tests for LLM client connection pooling."""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.core.llm_client import Provider
from sifaka.core.llm_client_pool import (
    ConnectionMetrics,
    LLMClientPool,
    PoolConfig,
    PooledConnection,
    close_global_pool,
    get_global_pool,
)


class TestPoolConfig:
    """Test pool configuration."""

    def test_default_config(self):
        """Test default pool configuration."""
        config = PoolConfig()

        assert config.max_pool_size == 10
        assert config.min_pool_size == 1
        assert config.max_idle_time == 300.0
        assert config.max_connection_age == 1800.0
        assert config.health_check_interval == 60.0
        assert config.connection_timeout == 10.0
        assert config.enable_health_checks is True
        assert config.enable_metrics is True

    def test_custom_config(self):
        """Test custom pool configuration."""
        config = PoolConfig(
            max_pool_size=5,
            min_pool_size=2,
            max_idle_time=180.0,
            enable_health_checks=False,
        )

        assert config.max_pool_size == 5
        assert config.min_pool_size == 2
        assert config.max_idle_time == 180.0
        assert config.enable_health_checks is False


class TestConnectionMetrics:
    """Test connection metrics."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = ConnectionMetrics()

        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.idle_connections == 0
        assert metrics.created_connections == 0
        assert metrics.destroyed_connections == 0
        assert metrics.borrowed_connections == 0
        assert metrics.returned_connections == 0
        assert metrics.failed_connections == 0
        assert metrics.pool_hits == 0
        assert metrics.pool_misses == 0
        assert metrics.last_reset > 0

    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        metrics = ConnectionMetrics()

        # Set some values
        metrics.total_connections = 5
        metrics.active_connections = 3
        metrics.pool_hits = 10

        # Reset
        metrics.reset()

        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.pool_hits == 0


class TestPooledConnection:
    """Test pooled connection wrapper."""

    def test_connection_creation(self):
        """Test pooled connection creation."""
        mock_client = MagicMock()
        connection = PooledConnection(
            client=mock_client,
            created_at=1000.0,
            last_used=1000.0,
        )

        assert connection.client is mock_client
        assert connection.created_at == 1000.0
        assert connection.last_used == 1000.0
        assert connection.use_count == 0
        assert connection.is_healthy is True

    def test_mark_used(self):
        """Test marking connection as used."""
        mock_client = MagicMock()
        connection = PooledConnection(
            client=mock_client,
            created_at=1000.0,
            last_used=1000.0,
        )

        connection.mark_used()

        assert connection.use_count == 1
        assert connection.last_used > 1000.0

    def test_is_expired(self):
        """Test connection expiration check."""
        mock_client = MagicMock()
        connection = PooledConnection(
            client=mock_client,
            created_at=1000.0,
            last_used=1000.0,
        )

        with patch("time.time", return_value=1500.0):
            assert connection.is_expired(400.0) is True
            assert connection.is_expired(600.0) is False

    def test_is_idle(self):
        """Test connection idle check."""
        mock_client = MagicMock()
        connection = PooledConnection(
            client=mock_client,
            created_at=1000.0,
            last_used=1000.0,
        )

        with patch("time.time", return_value=1400.0):
            assert connection.is_idle(300.0) is True
            assert connection.is_idle(500.0) is False


class TestLLMClientPool:
    """Test LLM client connection pool."""

    @pytest.fixture
    def pool_config(self):
        """Test pool configuration."""
        return PoolConfig(
            max_pool_size=3,
            min_pool_size=1,
            max_idle_time=60.0,
            enable_health_checks=False,  # Disable for testing
        )

    @pytest.fixture
    def pool(self, pool_config):
        """Test pool instance."""
        return LLMClientPool(pool_config)

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool):
        """Test pool initialization."""
        try:
            assert pool.config.max_pool_size == 3
            assert pool.config.min_pool_size == 1
            assert len(pool._pools) == 0
            assert len(pool._active_connections) == 0
        finally:
            await pool.close()

    @pytest.mark.asyncio
    async def test_get_client_creates_new_connection(self, pool):
        """Test getting client creates new connection when pool is empty."""
        with patch("sifaka.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

            assert client is mock_client
            mock_client_class.assert_called_once_with(
                Provider.OPENAI, "gpt-4o-mini", 0.7, None
            )

            # Check metrics
            metrics = pool.get_metrics()
            assert metrics.created_connections == 1
            assert metrics.active_connections == 1
            assert metrics.pool_misses == 1

    @pytest.mark.asyncio
    async def test_return_client_to_pool(self, pool):
        """Test returning client to pool."""
        with patch("sifaka.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Get client
            client = await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

            # Return client
            await pool.return_client(client)

            # Check metrics
            metrics = pool.get_metrics()
            assert metrics.returned_connections == 1
            assert metrics.active_connections == 0
            assert metrics.idle_connections == 1

            # Check pool state
            pool_key = "openai:gpt-4o-mini:0.7"
            assert pool_key in pool._pools
            assert len(pool._pools[pool_key]) == 1

    @pytest.mark.asyncio
    async def test_reuse_pooled_connection(self, pool):
        """Test reusing connection from pool."""
        with patch("sifaka.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Get and return client
            client1 = await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)
            await pool.return_client(client1)

            # Get client again (should reuse)
            client2 = await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

            assert client2 is mock_client
            assert mock_client_class.call_count == 1  # Called only once

            # Check metrics
            metrics = pool.get_metrics()
            assert metrics.created_connections == 1
            assert metrics.pool_hits == 1
            assert metrics.pool_misses == 1

    @pytest.mark.asyncio
    async def test_warm_up_connections(self, pool):
        """Test warming up connections."""
        with patch("sifaka.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Warm up 2 connections
            await pool.warm_up(Provider.OPENAI, "gpt-4o-mini", 0.7, 2)

            # Check pool state
            pool_key = "openai:gpt-4o-mini:0.7"
            assert pool_key in pool._pools
            assert len(pool._pools[pool_key]) == 2

            # Check metrics
            metrics = pool.get_metrics()
            assert metrics.created_connections == 2
            assert metrics.idle_connections == 2

    @pytest.mark.asyncio
    async def test_pool_status(self, pool):
        """Test getting pool status."""
        with patch("sifaka.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Create some connections
            await pool.warm_up(Provider.OPENAI, "gpt-4o-mini", 0.7, 2)
            await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

            # Get status
            status = await pool.get_pool_status()

            pool_key = "openai:gpt-4o-mini:0.7"
            assert pool_key in status
            assert status[pool_key]["idle_connections"] == 1
            assert status[pool_key]["active_connections"] == 1
            assert status[pool_key]["total_connections"] == 2
            assert status[pool_key]["max_pool_size"] == 3

    @pytest.mark.asyncio
    async def test_pool_cleanup(self, pool):
        """Test pool cleanup on close."""
        with patch("sifaka.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Create some connections
            await pool.warm_up(Provider.OPENAI, "gpt-4o-mini", 0.7, 2)

            # Verify connections exist
            assert len(pool._pools) > 0

            # Close pool
            await pool.close()

            # Check cleanup
            assert len(pool._pools) == 0
            assert len(pool._active_connections) == 0


class TestGlobalPool:
    """Test global pool management."""

    @pytest.mark.asyncio
    async def test_get_global_pool(self):
        """Test getting global pool."""
        pool = get_global_pool()

        assert pool is not None
        assert isinstance(pool, LLMClientPool)

        # Getting again should return same instance
        pool2 = get_global_pool()
        assert pool is pool2

        # Clean up
        await close_global_pool()

    @pytest.mark.asyncio
    async def test_close_global_pool(self):
        """Test closing global pool."""
        pool = get_global_pool()

        # Close global pool
        await close_global_pool()

        # Getting again should create new instance
        pool2 = get_global_pool()
        assert pool is not pool2

        # Clean up
        await close_global_pool()
