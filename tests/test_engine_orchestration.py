"""Tests for the critic orchestration engine module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from sifaka.core.engine.orchestration import CriticOrchestrator
from sifaka.core.models import SifakaResult, CritiqueResult


class TestCriticOrchestrator:
    """Test the CriticOrchestrator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Current version")

    def test_initialization_with_defaults(self):
        """Test orchestrator initialization with default values."""
        orchestrator = CriticOrchestrator(
            critic_names=["clarity", "accuracy"],
            model="gpt-4o-mini",
            temperature=0.7,
        )

        assert orchestrator.critic_names == ["clarity", "accuracy"]
        assert orchestrator.model == "gpt-4o-mini"
        assert orchestrator.temperature == 0.7
        assert orchestrator._critics is None

    def test_initialization_with_overrides(self):
        """Test orchestrator initialization with critic overrides."""
        orchestrator = CriticOrchestrator(
            critic_names=["clarity", "accuracy"],
            model="gpt-4o-mini",
            temperature=0.7,
            critic_model="gpt-4",
            critic_temperature=0.5,
        )

        assert orchestrator.critic_names == ["clarity", "accuracy"]
        assert orchestrator.model == "gpt-4"  # Override
        assert orchestrator.temperature == 0.5  # Override
        assert orchestrator._critics is None

    def test_critics_property_lazy_loading(self):
        """Test lazy loading of critics."""
        orchestrator = CriticOrchestrator(
            critic_names=["clarity"], model="gpt-4o-mini", temperature=0.7
        )

        # Initially None
        assert orchestrator._critics is None

        # Mock create_critics
        with patch("sifaka.core.engine.orchestration.create_critics") as mock_create:
            mock_critic = Mock()
            mock_critic.name = "clarity"
            mock_create.return_value = [mock_critic]

            # First access creates critics
            critics = orchestrator.critics
            assert critics == [mock_critic]
            assert orchestrator._critics == [mock_critic]
            mock_create.assert_called_once_with(
                ["clarity"], model="gpt-4o-mini", temperature=0.7
            )

            # Second access reuses
            critics2 = orchestrator.critics
            assert critics2 == [mock_critic]
            assert mock_create.call_count == 1

    def test_critics_property_empty_list(self):
        """Test critics property with empty critic names."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        with patch("sifaka.core.engine.orchestration.create_critics") as mock_create:
            mock_create.return_value = []

            critics = orchestrator.critics
            assert critics == []

    @pytest.mark.asyncio
    async def test_run_critics_empty_list(self, sample_result):
        """Test running with no critics."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        # Mock empty critics
        orchestrator._critics = []
        critiques = await orchestrator.run_critics("Test text", sample_result)
        assert critiques == []

    @pytest.mark.asyncio
    async def test_run_critics_single_success(self, sample_result):
        """Test running a single critic successfully."""
        orchestrator = CriticOrchestrator(
            critic_names=["clarity"], model="gpt-4o-mini", temperature=0.7
        )

        # Mock critic
        mock_critic = Mock()
        mock_critic.name = "clarity"
        mock_critique = CritiqueResult(
            critic="clarity",
            feedback="Text needs improvement",
            suggestions=["Add examples"],
            needs_improvement=True,
            confidence=0.8,
        )
        mock_critic.critique = AsyncMock(return_value=mock_critique)

        orchestrator._critics = [mock_critic]
        critiques = await orchestrator.run_critics("Test text", sample_result)

        assert len(critiques) == 1
        assert critiques[0] == mock_critique
        mock_critic.critique.assert_called_once_with("Test text", sample_result)

    @pytest.mark.asyncio
    async def test_run_critics_multiple_success(self, sample_result):
        """Test running multiple critics successfully."""
        orchestrator = CriticOrchestrator(
            critic_names=["clarity", "accuracy", "style"],
            model="gpt-4o-mini",
            temperature=0.7,
        )

        # Mock critics
        mock_critiques = []
        mock_critics = []

        for i, name in enumerate(["clarity", "accuracy", "style"]):
            mock_critic = Mock()
            mock_critic.name = name
            mock_critique = CritiqueResult(
                critic=name,
                feedback=f"{name} feedback",
                suggestions=[f"{name} suggestion"],
                needs_improvement=i < 2,  # First two need improvement
                confidence=0.7 + i * 0.1,
            )
            mock_critic.critique = AsyncMock(return_value=mock_critique)
            mock_critics.append(mock_critic)
            mock_critiques.append(mock_critique)

        orchestrator._critics = mock_critics
        critiques = await orchestrator.run_critics("Test text", sample_result)

        assert len(critiques) == 3
        for i, critique in enumerate(critiques):
            assert critique == mock_critiques[i]
            mock_critics[i].critique.assert_called_once_with("Test text", sample_result)

    @pytest.mark.asyncio
    async def test_run_critics_with_exception(self, sample_result):
        """Test handling critic exceptions."""
        orchestrator = CriticOrchestrator(
            critic_names=["good", "bad", "also_good"],
            model="gpt-4o-mini",
            temperature=0.7,
        )

        # Mock critics
        mock_critic1 = Mock()
        mock_critic1.name = "good"
        mock_critique1 = CritiqueResult(
            critic="good",
            feedback="Good feedback",
            suggestions=["Suggestion"],
            needs_improvement=True,
            confidence=0.8,
        )
        mock_critic1.critique = AsyncMock(return_value=mock_critique1)

        mock_critic2 = Mock()
        mock_critic2.name = "bad"
        mock_critic2.critique = AsyncMock(side_effect=Exception("Critic failed"))

        mock_critic3 = Mock()
        mock_critic3.name = "also_good"
        mock_critique3 = CritiqueResult(
            critic="also_good",
            feedback="Also good feedback",
            suggestions=["Another suggestion"],
            needs_improvement=False,
            confidence=0.9,
        )
        mock_critic3.critique = AsyncMock(return_value=mock_critique3)

        orchestrator._critics = [mock_critic1, mock_critic2, mock_critic3]
        critiques = await orchestrator.run_critics("Test text", sample_result)

        assert len(critiques) == 3

        # First critic succeeded
        assert critiques[0] == mock_critique1

        # Second critic failed - error critique created
        assert critiques[1].critic == "bad"
        assert "Error during critique: Critic failed" in critiques[1].feedback
        assert critiques[1].suggestions == ["Review the text manually"]
        assert critiques[1].needs_improvement is True
        assert critiques[1].confidence == 0.0

        # Third critic succeeded
        assert critiques[2] == mock_critique3

    @pytest.mark.asyncio
    async def test_run_critics_all_exceptions(self, sample_result):
        """Test when all critics raise exceptions."""
        orchestrator = CriticOrchestrator(
            critic_names=["bad1", "bad2"], model="gpt-4o-mini", temperature=0.7
        )

        # Mock critics that all fail
        mock_critics = []
        for name in ["bad1", "bad2"]:
            mock_critic = Mock()
            mock_critic.name = name
            mock_critic.critique = AsyncMock(side_effect=Exception(f"{name} failed"))
            mock_critics.append(mock_critic)

        orchestrator._critics = mock_critics
        critiques = await orchestrator.run_critics("Test text", sample_result)

        assert len(critiques) == 2
        for i, critique in enumerate(critiques):
            assert critique.critic == mock_critics[i].name
            assert f"{mock_critics[i].name} failed" in critique.feedback
            assert critique.needs_improvement is True
            assert critique.confidence == 0.0

    def test_analyze_consensus_empty_list(self):
        """Test consensus analysis with no critiques."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        assert orchestrator.analyze_consensus([]) is False

    def test_analyze_consensus_unanimous_yes(self):
        """Test consensus when all critics say improvement needed."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Needs work",
                suggestions=[],
                needs_improvement=True,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Also needs work",
                suggestions=[],
                needs_improvement=True,
            ),
            CritiqueResult(
                critic="critic3",
                feedback="Definitely needs work",
                suggestions=[],
                needs_improvement=True,
            ),
        ]

        assert orchestrator.analyze_consensus(critiques) is True

    def test_analyze_consensus_unanimous_no(self):
        """Test consensus when no critics say improvement needed."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Looks good",
                suggestions=[],
                needs_improvement=False,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Also good",
                suggestions=[],
                needs_improvement=False,
            ),
        ]

        assert orchestrator.analyze_consensus(critiques) is False

    def test_analyze_consensus_majority_yes(self):
        """Test consensus with majority saying improvement needed."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Needs work",
                suggestions=[],
                needs_improvement=True,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Also needs work",
                suggestions=[],
                needs_improvement=True,
            ),
            CritiqueResult(
                critic="critic3",
                feedback="Good",
                suggestions=[],
                needs_improvement=False,
            ),
        ]

        assert orchestrator.analyze_consensus(critiques) is True

    def test_analyze_consensus_majority_no(self):
        """Test consensus with majority saying no improvement needed."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Needs work",
                suggestions=[],
                needs_improvement=True,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Good",
                suggestions=[],
                needs_improvement=False,
            ),
            CritiqueResult(
                critic="critic3",
                feedback="Also good",
                suggestions=[],
                needs_improvement=False,
            ),
        ]

        assert orchestrator.analyze_consensus(critiques) is False

    def test_analyze_consensus_tie(self):
        """Test consensus with tie (50/50)."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Needs work",
                suggestions=[],
                needs_improvement=True,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Good",
                suggestions=[],
                needs_improvement=False,
            ),
        ]

        # Tie should return False (not > 50%)
        assert orchestrator.analyze_consensus(critiques) is False

    def test_get_aggregated_confidence_empty_list(self):
        """Test aggregated confidence with no critiques."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        assert orchestrator.get_aggregated_confidence([]) == 0.0

    def test_get_aggregated_confidence_single_critique(self):
        """Test aggregated confidence with single critique."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Feedback",
                suggestions=[],
                needs_improvement=True,
                confidence=0.8,
            )
        ]

        assert orchestrator.get_aggregated_confidence(critiques) == 0.8

    def test_get_aggregated_confidence_multiple_critiques(self):
        """Test aggregated confidence with multiple critiques."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Feedback1",
                suggestions=[],
                needs_improvement=True,
                confidence=0.8,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Feedback2",
                suggestions=[],
                needs_improvement=True,
                confidence=0.6,
            ),
            CritiqueResult(
                critic="critic3",
                feedback="Feedback3",
                suggestions=[],
                needs_improvement=False,
                confidence=0.9,
            ),
        ]

        # Average of 0.8, 0.6, 0.9
        expected = (0.8 + 0.6 + 0.9) / 3
        assert abs(orchestrator.get_aggregated_confidence(critiques) - expected) < 0.001

    def test_get_aggregated_confidence_with_none_values(self):
        """Test aggregated confidence with None confidence values."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Feedback1",
                suggestions=[],
                needs_improvement=True,
                confidence=0.8,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Feedback2",
                suggestions=[],
                needs_improvement=True,
                confidence=None,  # None value
            ),
            CritiqueResult(
                critic="critic3",
                feedback="Feedback3",
                suggestions=[],
                needs_improvement=False,
                confidence=0.6,
            ),
        ]

        # Should only average 0.8 and 0.6
        expected = (0.8 + 0.6) / 2
        assert orchestrator.get_aggregated_confidence(critiques) == expected

    def test_get_aggregated_confidence_with_zero_values(self):
        """Test aggregated confidence with zero confidence values."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Feedback1",
                suggestions=[],
                needs_improvement=True,
                confidence=0.8,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Error",
                suggestions=[],
                needs_improvement=True,
                confidence=0.0,  # Zero value (should be excluded)
            ),
            CritiqueResult(
                critic="critic3",
                feedback="Feedback3",
                suggestions=[],
                needs_improvement=False,
                confidence=0.6,
            ),
        ]

        # Should only average 0.8 and 0.6 (excluding 0.0)
        expected = (0.8 + 0.6) / 2
        assert orchestrator.get_aggregated_confidence(critiques) == expected

    def test_get_aggregated_confidence_all_invalid(self):
        """Test aggregated confidence when all values are invalid."""
        orchestrator = CriticOrchestrator(
            critic_names=[], model="gpt-4o-mini", temperature=0.7
        )

        critiques = [
            CritiqueResult(
                critic="critic1",
                feedback="Error",
                suggestions=[],
                needs_improvement=True,
                confidence=0.0,
            ),
            CritiqueResult(
                critic="critic2",
                feedback="Error",
                suggestions=[],
                needs_improvement=True,
                confidence=None,
            ),
        ]

        assert orchestrator.get_aggregated_confidence(critiques) == 0.0
