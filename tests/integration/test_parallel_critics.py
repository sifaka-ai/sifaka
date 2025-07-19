"""Tests for parallel critic evaluation."""

import asyncio
from datetime import datetime

import pytest

from sifaka.core.config import Config
from sifaka.core.engine.orchestration import CriticOrchestrator
from sifaka.core.interfaces import Critic
from sifaka.core.models import CritiqueResult, SifakaResult


class MockCritic(Critic):
    """Mock critic for testing."""

    def __init__(self, name: str, delay: float = 0.1, should_timeout: bool = False):
        self._name = name
        self.delay = delay
        self.should_timeout = should_timeout
        self.call_count = 0
        self.call_times = []

    @property
    def name(self) -> str:
        return self._name

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Mock critique method."""
        self.call_count += 1
        self.call_times.append(asyncio.get_event_loop().time())

        if self.should_timeout:
            # Simulate a timeout by sleeping longer than the timeout
            await asyncio.sleep(100)

        # Simulate work
        await asyncio.sleep(self.delay)

        return CritiqueResult(
            critic=self._name,
            feedback=f"Feedback from {self._name}",
            suggestions=[f"Suggestion from {self._name}"],
            needs_improvement=True,
            confidence=0.8,
        )


class TestParallelCritics:
    """Test parallel critic evaluation."""

    def create_mock_result(self) -> SifakaResult:
        """Create a mock SifakaResult for testing."""
        return SifakaResult(
            id="test_id",
            original_text="test text",
            final_text="test text",
            iteration=1,
            processing_time=0.1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            generations=[],
            critiques=[],
            validations=[],
        )

    @pytest.mark.asyncio
    async def test_parallel_critics_enabled(self):
        """Test that parallel critics are enabled by default."""
        config = Config()

        # Create orchestrator with parallel critics enabled
        CriticOrchestrator(
            critic_names=["test_critic_1", "test_critic_2"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        assert config.engine.parallel_critics is True
        assert config.engine.max_parallel_critics == 3

    @pytest.mark.asyncio
    async def test_parallel_execution_timing(self):
        """Test that parallel execution is faster than sequential."""
        config = Config()

        # Create mock critics with delays
        mock_critics = [
            MockCritic("critic_1", delay=0.1),
            MockCritic("critic_2", delay=0.1),
            MockCritic("critic_3", delay=0.1),
        ]

        orchestrator = CriticOrchestrator(
            critic_names=["critic_1", "critic_2", "critic_3"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test parallel execution
        result = self.create_mock_result()
        start_time = asyncio.get_event_loop().time()

        critiques = await orchestrator.run_critics("test text", result)

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Should take around 0.1 seconds (parallel) rather than 0.3 (sequential)
        assert execution_time < 0.25  # Allow some overhead
        assert len(critiques) == 3
        assert all(isinstance(c, CritiqueResult) for c in critiques)

        # Check that all critics were called
        assert all(critic.call_count == 1 for critic in mock_critics)

    @pytest.mark.asyncio
    async def test_sequential_execution_timing(self):
        """Test sequential execution timing."""
        from sifaka.core.config.engine import EngineConfig

        config = Config(engine=EngineConfig(parallel_critics=False))  # Disable parallel

        # Create mock critics with delays
        mock_critics = [
            MockCritic("critic_1", delay=0.1),
            MockCritic("critic_2", delay=0.1),
            MockCritic("critic_3", delay=0.1),
        ]

        orchestrator = CriticOrchestrator(
            critic_names=["critic_1", "critic_2", "critic_3"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test sequential execution
        result = self.create_mock_result()
        start_time = asyncio.get_event_loop().time()

        critiques = await orchestrator.run_critics("test text", result)

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Should take around 0.3 seconds (sequential)
        assert execution_time >= 0.25  # Should be close to 0.3
        assert len(critiques) == 3
        assert all(isinstance(c, CritiqueResult) for c in critiques)

        # Check that all critics were called
        assert all(critic.call_count == 1 for critic in mock_critics)

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency is limited by max_parallel_critics."""
        from sifaka.core.config.engine import EngineConfig

        config = Config(engine=EngineConfig(max_parallel_critics=2))

        # Create 5 mock critics
        mock_critics = [MockCritic(f"critic_{i}", delay=0.1) for i in range(5)]

        orchestrator = CriticOrchestrator(
            critic_names=[f"critic_{i}" for i in range(5)],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test parallel execution with limit
        result = self.create_mock_result()
        start_time = asyncio.get_event_loop().time()

        critiques = await orchestrator.run_critics("test text", result)

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # With 5 critics and max_parallel=2, should take around 0.3 seconds
        # (2 batches: first 2 in parallel, then 2 more, then 1)
        assert execution_time >= 0.25
        assert len(critiques) == 5
        assert all(isinstance(c, CritiqueResult) for c in critiques)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that critic timeouts are handled properly."""
        from sifaka.core.config.engine import EngineConfig

        config = Config(
            engine=EngineConfig(critic_timeout_seconds=0.2)
        )  # Short timeout

        # Create critics with one that will timeout
        mock_critics = [
            MockCritic("critic_1", delay=0.1),
            MockCritic("critic_2", delay=0.1, should_timeout=True),
            MockCritic("critic_3", delay=0.1),
        ]

        orchestrator = CriticOrchestrator(
            critic_names=["critic_1", "critic_2", "critic_3"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test execution with timeout
        result = self.create_mock_result()
        critiques = await orchestrator.run_critics("test text", result)

        # Should get 3 critiques (including timeout error)
        assert len(critiques) == 3

        # Check that one critique is a timeout error
        timeout_critiques = [c for c in critiques if "timed out" in c.feedback]
        assert len(timeout_critiques) == 1
        assert timeout_critiques[0].critic == "critic_2"
        assert timeout_critiques[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_batched_execution(self):
        """Test batched execution for many critics."""
        from sifaka.core.config.engine import EngineConfig

        config = Config(engine=EngineConfig(max_parallel_critics=2))

        # Create 7 mock critics (will trigger batching)
        mock_critics = [MockCritic(f"critic_{i}", delay=0.05) for i in range(7)]

        orchestrator = CriticOrchestrator(
            critic_names=[f"critic_{i}" for i in range(7)],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test batched execution
        result = self.create_mock_result()
        critiques = await orchestrator.run_critics("test text", result)

        # Should get all 7 critiques
        assert len(critiques) == 7
        assert all(isinstance(c, CritiqueResult) for c in critiques)

        # Check that all critics were called
        assert all(critic.call_count == 1 for critic in mock_critics)

    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test that performance metrics are tracked."""
        config = Config()

        # Create mock critics with different delays
        mock_critics = [
            MockCritic("fast_critic", delay=0.05),
            MockCritic("slow_critic", delay=0.15),
            MockCritic("medium_critic", delay=0.10),
        ]

        orchestrator = CriticOrchestrator(
            critic_names=["fast_critic", "slow_critic", "medium_critic"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test execution
        result = self.create_mock_result()
        await orchestrator.run_critics("test text", result)

        # Check performance metrics
        metrics = orchestrator.get_performance_metrics()
        assert len(metrics) == 3
        assert "fast_critic" in metrics
        assert "slow_critic" in metrics
        assert "medium_critic" in metrics

        # Check that the fastest critic is identified correctly
        fastest = orchestrator.get_fastest_critic()
        assert fastest == "fast_critic"

        # Check that the slowest critic is identified correctly
        slowest = orchestrator.get_slowest_critic()
        assert slowest == "slow_critic"

        # Check average execution time
        avg_time = orchestrator.get_average_execution_time()
        assert avg_time > 0.05  # Should be around 0.10
        assert avg_time < 0.20

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that critic errors are handled properly."""
        config = Config()

        # Create a critic that will raise an exception
        class ErrorCritic(Critic):
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
                raise ValueError("Test error")

        mock_critics = [
            MockCritic("good_critic", delay=0.1),
            ErrorCritic("error_critic"),
        ]

        orchestrator = CriticOrchestrator(
            critic_names=["good_critic", "error_critic"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test execution with error
        result = self.create_mock_result()
        critiques = await orchestrator.run_critics("test text", result)

        # Should get 2 critiques (including error)
        assert len(critiques) == 2

        # Check that one critique is an error
        error_critiques = [
            c for c in critiques if "Error during critique" in c.feedback
        ]
        assert len(error_critiques) == 1
        assert error_critiques[0].critic == "error_critic"
        assert error_critiques[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_consensus_analysis(self):
        """Test consensus analysis across multiple critics."""
        config = Config()

        # Create mock critics with different opinions
        class OpinionCritic(Critic):
            def __init__(self, name: str, needs_improvement: bool):
                self._name = name
                self.needs_improvement = needs_improvement

            @property
            def name(self) -> str:
                return self._name

            async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
                return CritiqueResult(
                    critic=self._name,
                    feedback=f"Feedback from {self._name}",
                    suggestions=[f"Suggestion from {self._name}"],
                    needs_improvement=self.needs_improvement,
                    confidence=0.8,
                )

        mock_critics = [
            OpinionCritic("critic_1", needs_improvement=True),
            OpinionCritic("critic_2", needs_improvement=True),
            OpinionCritic("critic_3", needs_improvement=False),
        ]

        orchestrator = CriticOrchestrator(
            critic_names=["critic_1", "critic_2", "critic_3"],
            model="gpt-4o-mini",
            temperature=0.7,
            config=config,
        )

        # Replace critics with mocks
        orchestrator._critics = mock_critics

        # Test consensus analysis
        result = self.create_mock_result()
        critiques = await orchestrator.run_critics("test text", result)

        # Analyze consensus (2 out of 3 think improvement is needed)
        consensus = orchestrator.analyze_consensus(critiques)
        assert consensus is True  # Majority vote

        # Test aggregated confidence
        confidence = orchestrator.get_aggregated_confidence(critiques)
        assert confidence == 0.8  # All critics have 0.8 confidence
