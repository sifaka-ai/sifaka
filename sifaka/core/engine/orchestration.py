"""Critic orchestration component of the Sifaka engine."""

import asyncio
from typing import List, Optional

from ..models import SifakaResult, CritiqueResult
from ..interfaces import Critic
from ...critics import create_critics

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


class CriticOrchestrator:
    """Orchestrates critic execution and feedback collection."""

    def __init__(
        self,
        critic_names: List[str],
        model: str,
        temperature: float,
        critic_model: Optional[str] = None,
        critic_temperature: Optional[float] = None,
        config: Optional["Config"] = None,
    ):
        """Initialize orchestrator.

        Args:
            critic_names: Names of critics to use
            model: Default model
            temperature: Default temperature
            critic_model: Override model for critics
            critic_temperature: Override temperature for critics
        """
        self.critic_names = critic_names
        self.model = critic_model or model
        self.temperature = critic_temperature or temperature
        self.config = config
        self._critics: Optional[List[Critic]] = None

    @property
    def critics(self) -> List[Critic]:
        """Get or create critics."""
        if self._critics is None:
            self._critics = create_critics(
                self.critic_names,
                model=self.model,
                temperature=self.temperature,
                config=self.config,
            )
        return self._critics

    async def run_critics(
        self, text: str, result: SifakaResult
    ) -> List[CritiqueResult]:
        """Run all critics on the text.

        Args:
            text: Text to critique
            result: Result object with history

        Returns:
            List of critique results
        """
        if not self.critics:
            return []

        # Run critics in parallel
        critique_tasks = [critic.critique(text, result) for critic in self.critics]

        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)

        # Process results
        valid_critiques: List[CritiqueResult] = []
        for i, critique in enumerate(critiques):
            if isinstance(critique, Exception):
                # Create error critique
                error_critique = CritiqueResult(
                    critic=self.critics[i].name,
                    feedback=f"Error during critique: {str(critique)}",
                    suggestions=["Review the text manually"],
                    needs_improvement=True,
                    confidence=0.0,
                )
                valid_critiques.append(error_critique)
            else:
                # critique is CritiqueResult here
                assert isinstance(critique, CritiqueResult)
                valid_critiques.append(critique)

        return valid_critiques

    def analyze_consensus(self, critiques: List[CritiqueResult]) -> bool:
        """Analyze if critics agree improvement is needed.

        Args:
            critiques: List of critique results

        Returns:
            True if majority think improvement is needed
        """
        if not critiques:
            return False

        needs_improvement_count = sum(1 for c in critiques if c.needs_improvement)

        # Majority vote
        return needs_improvement_count > len(critiques) / 2

    def get_aggregated_confidence(self, critiques: List[CritiqueResult]) -> float:
        """Get aggregated confidence from all critics.

        Args:
            critiques: List of critique results

        Returns:
            Average confidence score
        """
        if not critiques:
            return 0.0

        confidences = [
            c.confidence
            for c in critiques
            if c.confidence is not None and c.confidence > 0
        ]

        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)
