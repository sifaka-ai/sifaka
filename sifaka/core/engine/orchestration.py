"""Critic orchestration component of the Sifaka engine."""

import asyncio
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from ...critics import create_critics
from ..interfaces import Critic
from ..models import CritiqueResult, SifakaResult
from ..monitoring import get_global_monitor

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
        self._performance_metrics: Dict[str, float] = {}

    @property
    def critics(self) -> List[Critic]:
        """Get or create critics."""
        if self._critics is None:
            # Determine provider from config
            provider = None
            if self.config and self.config.llm.provider:
                provider = self.config.llm.provider

            self._critics = create_critics(
                self.critic_names,
                model=self.model,
                temperature=self.temperature,
                config=self.config,
                provider=provider,
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

        # Check if parallel execution is enabled
        if self.config and self.config.engine.parallel_critics:
            return await self._run_critics_parallel(text, result)
        else:
            return await self._run_critics_sequential(text, result)

    async def _run_critics_parallel(
        self, text: str, result: SifakaResult
    ) -> List[CritiqueResult]:
        """Run critics in parallel with concurrency limits."""
        max_parallel = self.config.engine.max_parallel_critics if self.config else 3

        # If we have many critics, process them in batches
        if len(self.critics) > max_parallel * 2:
            return await self._run_critics_batched(text, result, max_parallel)

        # Use semaphore to limit concurrent critics
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_single_critic(critic: Critic) -> CritiqueResult:
            async with semaphore:
                timeout = (
                    self.config.engine.critic_timeout_seconds if self.config else 60.0
                )
                start_time = time.time()
                try:
                    monitor = get_global_monitor()
                    result_critique = await asyncio.wait_for(
                        monitor.track_critic_call(
                            critic.name, lambda: critic.critique(text, result)
                        ),
                        timeout=timeout,
                    )
                    # Track performance
                    execution_time = time.time() - start_time
                    self._performance_metrics[critic.name] = execution_time
                    return result_critique  # type: ignore[no-any-return]
                except asyncio.TimeoutError:
                    # Track timeout
                    self._performance_metrics[critic.name] = timeout
                    # Create timeout error critique
                    return CritiqueResult(
                        critic=critic.name,
                        feedback=f"Critic timed out after {timeout} seconds",
                        suggestions=["Review the text manually"],
                        needs_improvement=True,
                        confidence=0.0,
                    )

        # Create tasks for all critics
        critique_tasks = [run_single_critic(critic) for critic in self.critics]

        # Run with gather to handle exceptions
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)

        return self._process_critique_results(critiques)

    async def _run_critics_batched(
        self, text: str, result: SifakaResult, batch_size: int
    ) -> List[CritiqueResult]:
        """Run critics in batches to manage resource usage."""
        all_critiques = []
        timeout = self.config.engine.critic_timeout_seconds if self.config else 60.0

        # Process critics in batches
        for i in range(0, len(self.critics), batch_size):
            batch = self.critics[i : i + batch_size]

            async def run_critic_with_timeout(critic: Critic) -> CritiqueResult:
                try:
                    monitor = get_global_monitor()
                    return await asyncio.wait_for(
                        monitor.track_critic_call(
                            critic.name, lambda: critic.critique(text, result)
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return CritiqueResult(
                        critic=critic.name,
                        feedback=f"Critic timed out after {timeout} seconds",
                        suggestions=["Review the text manually"],
                        needs_improvement=True,
                        confidence=0.0,
                    )

            batch_tasks = [run_critic_with_timeout(critic) for critic in batch]

            # Run batch in parallel
            batch_critiques = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_critiques.extend(batch_critiques)

        return self._process_critique_results(all_critiques)

    async def _run_critics_sequential(
        self, text: str, result: SifakaResult
    ) -> List[CritiqueResult]:
        """Run critics sequentially."""
        valid_critiques: List[CritiqueResult] = []
        timeout = self.config.engine.critic_timeout_seconds if self.config else 60.0

        monitor = get_global_monitor()
        for critic in self.critics:
            try:
                critique = await asyncio.wait_for(
                    monitor.track_critic_call(
                        critic.name, lambda: critic.critique(text, result)
                    ),
                    timeout=timeout,
                )
                valid_critiques.append(critique)
            except asyncio.TimeoutError:
                # Create timeout error critique
                timeout_critique = CritiqueResult(
                    critic=critic.name,
                    feedback=f"Critic timed out after {timeout} seconds",
                    suggestions=["Review the text manually"],
                    needs_improvement=True,
                    confidence=0.0,
                )
                valid_critiques.append(timeout_critique)
            except Exception as e:
                # Create error critique
                error_critique = CritiqueResult(
                    critic=critic.name,
                    feedback=f"Error during critique: {e!s}",
                    suggestions=["Review the text manually"],
                    needs_improvement=True,
                    confidence=0.0,
                )
                valid_critiques.append(error_critique)

        return valid_critiques

    def _process_critique_results(self, critiques: List) -> List[CritiqueResult]:
        """Process critic results and handle exceptions."""
        valid_critiques: List[CritiqueResult] = []

        for i, critique in enumerate(critiques):
            if isinstance(critique, Exception):
                # Create error critique
                error_critique = CritiqueResult(
                    critic=self.critics[i].name,
                    feedback=f"Error during critique: {critique!s}",
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

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the last critic execution.

        Returns:
            Dictionary mapping critic names to execution times in seconds
        """
        return self._performance_metrics.copy()

    def get_fastest_critic(self) -> Optional[str]:
        """Get the name of the fastest critic from the last execution.

        Returns:
            Name of the fastest critic, or None if no metrics available
        """
        if not self._performance_metrics:
            return None

        return min(
            self._performance_metrics, key=lambda x: self._performance_metrics[x]
        )

    def get_slowest_critic(self) -> Optional[str]:
        """Get the name of the slowest critic from the last execution.

        Returns:
            Name of the slowest critic, or None if no metrics available
        """
        if not self._performance_metrics:
            return None

        return max(
            self._performance_metrics, key=lambda x: self._performance_metrics[x]
        )

    def get_average_execution_time(self) -> float:
        """Get the average execution time across all critics.

        Returns:
            Average execution time in seconds, or 0.0 if no metrics available
        """
        if not self._performance_metrics:
            return 0.0

        return sum(self._performance_metrics.values()) / len(self._performance_metrics)
