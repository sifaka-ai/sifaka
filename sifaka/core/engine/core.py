"""Core engine that orchestrates the text improvement process.

This module contains the SifakaEngine class which coordinates all components
of the text improvement pipeline:
- Critics provide feedback on text quality
- Validators ensure text meets requirements
- Generators create improved versions based on feedback
- Storage persists results for analysis

The engine manages the iterative improvement loop, handling timeouts,
errors, and convergence detection."""

import time
from datetime import datetime
from typing import List, Optional

from ...storage import FileStorage, StorageBackend
from ...validators import LengthValidator
from ..config import Config
from ..exceptions import ModelProviderError, TimeoutError
from ..interfaces import Validator
from ..models import SifakaResult
from ..monitoring import get_global_monitor
from .generation import TextGenerator
from .orchestration import CriticOrchestrator
from .validation import ValidationRunner


class SifakaEngine:
    """Central orchestrator for the text improvement process.

    The SifakaEngine coordinates the interaction between critics, validators,
    and text generators to iteratively improve text. It implements the core
    improvement loop:

    1. Validate current text against requirements
    2. Run critics to get improvement feedback
    3. Generate improved text based on feedback
    4. Repeat until quality targets are met or max iterations reached

    The engine handles:
    - Component initialization and configuration
    - Timeout management to prevent runaway processes
    - Error handling and recovery
    - Result persistence through storage backends
    - Convergence detection based on critic consensus

    Example:
        >>> engine = SifakaEngine(config=Config(model="gpt-4"))
        >>> result = await engine.improve(
        ...     "Initial text",
        ...     validators=[LengthValidator(min_length=100)]
        ... )
    """

    def __init__(
        self, config: Optional[Config] = None, storage: Optional[StorageBackend] = None
    ):
        """Initialize the engine with configuration and storage.

        Args:
            config: Configuration object controlling all aspects of the
                improvement process. If not provided, uses defaults.
            storage: Storage backend for persisting results. If not provided,
                uses FileStorage with default settings.
        """
        self.config = config or Config()
        self.storage = storage or FileStorage()

        # Initialize components
        self.generator = TextGenerator(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            provider=self.config.llm.provider,
        )

        # Convert critics to strings if they are enums
        critic_names = [
            c.value if hasattr(c, "value") else c for c in self.config.critic.critics
        ]
        self.orchestrator = CriticOrchestrator(
            critic_names=critic_names,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            critic_model=self.config.llm.critic_model,
            critic_temperature=self.config.llm.effective_critic_temperature,
            config=self.config,
        )

        self.validator = ValidationRunner()

    async def improve(
        self, text: str, validators: Optional[List[Validator]] = None
    ) -> SifakaResult:
        """Improve text through iterative critique and refinement.

        This is the main entry point that runs the complete improvement
        pipeline. It coordinates critics, validators, and generators to
        progressively enhance the input text.

        Args:
            text: The initial text to improve. Can be any length or format.
            validators: Optional quality validators

        Returns:
            SifakaResult with improved text and audit trail
        """
        start_time = time.time()

        # Initialize result with config for traceability (excluding sensitive/verbose/irrelevant fields)
        config_dict = self.config.model_dump(
            exclude={
                "logfire_token",
                "critic_tool_settings",
                "tool_cache_ttl",
                "tool_timeout",
                "self_consistency_num_samples",  # Only relevant for self_consistency critic
                "constitutional_principles",  # Tracked in critic metadata instead
                "enable_tools",  # Not used
            }
        )
        result = SifakaResult(
            original_text=text, final_text=text, config_used=config_dict
        )

        # Use default validators if none provided
        if validators is None:
            validators = [LengthValidator(min_length=50)]

        try:
            # Improvement loop
            for iteration in range(self.config.engine.max_iterations):
                result.increment_iteration()
                current_text = result.current_text

                # Check timeout
                self._check_timeout(start_time)

                # Run validation
                monitor = get_global_monitor()
                validation_passed = await self.validator.run_validators(
                    current_text, result, validators
                )

                # Run critics
                try:
                    # Don't wrap orchestrator calls - let it track individual critics
                    critiques = await self.orchestrator.run_critics(
                        current_text, result
                    )
                except ModelProviderError:
                    # Re-raise authentication and provider errors
                    raise
                except Exception as e:
                    # Log other critic errors and continue
                    result.add_critique(
                        critic="system",
                        feedback=f"Critics failed: {e!s}",
                        suggestions=["Continue without critic feedback"],
                        needs_improvement=False,
                        confidence=0.0,
                    )
                    critiques = []

                # Add critiques to result (preserve all traceability data)
                for critique in critiques:
                    result.critiques.append(critique)
                    # Add any tools used by this critique
                    for tool_usage in critique.tools_used:
                        result.tools_used.append(tool_usage)
                    result.updated_at = datetime.now()

                # Check if improvement needed
                needs_improvement = self.orchestrator.analyze_consensus(critiques)

                # Decide whether to continue
                if not self._should_continue(validation_passed, needs_improvement):
                    break

                # Generate improvement
                try:
                    (
                        improved_text,
                        prompt,
                        tokens,
                        processing_time,
                    ) = await monitor.track_llm_call(
                        lambda: self.generator.generate_improvement(
                            current_text,
                            result,
                            self.config.engine.show_improvement_prompt,
                        )
                    )

                    if improved_text:
                        # Post-process to ensure validator constraints
                        improved_text = await self._post_process_text(
                            improved_text, validators
                        )

                        # Add generation to result
                        result.add_generation(
                            text=improved_text,
                            model=self.config.llm.model,
                            prompt=prompt,
                            tokens=tokens,
                            processing_time=processing_time,
                        )
                        result.final_text = improved_text

                except ModelProviderError:
                    raise
                except Exception as e:
                    # Log generation error
                    result.add_critique(
                        critic="system",
                        feedback=f"Generation failed: {e!s}",
                        suggestions=["Try with different parameters"],
                        needs_improvement=False,
                        confidence=0.0,
                    )
                    break

                # Check memory bounds
                self.validator.check_memory_bounds(result)

        except TimeoutError:
            raise
        except ModelProviderError:
            raise
        except Exception as e:
            # Log unexpected errors as system critique
            result.add_critique(
                critic="system",
                feedback=f"Unexpected error: {type(e).__name__}: {e!s}",
                suggestions=["Check logs for details"],
                needs_improvement=False,
                confidence=0.0,
            )

        finally:
            # Finalize result
            await self._finalize_result(result, start_time)

        return result

    def _check_timeout(self, start_time: float) -> None:
        """Check if operation has timed out."""
        elapsed = time.time() - start_time
        if elapsed > self.config.engine.total_timeout_seconds:
            raise TimeoutError(
                elapsed_time=elapsed, limit=self.config.engine.total_timeout_seconds
            )

    def _should_continue(
        self, validation_passed: bool, needs_improvement: bool
    ) -> bool:
        """Determine if improvement should continue."""
        if self.config.engine.force_improvements:
            return True

        return not validation_passed or needs_improvement

    async def _post_process_text(self, text: str, validators: List[Validator]) -> str:
        """Post-process text to ensure hard constraints are met."""
        processed_text = text

        # Check each validator for hard constraints
        for validator in validators:
            # Handle LengthValidator max_length constraint
            if hasattr(validator, "name") and validator.name == "length":
                if (
                    hasattr(validator, "max_length")
                    and validator.max_length is not None
                ):
                    if len(processed_text) > validator.max_length:
                        # Truncate to max length, trying to end at a sentence
                        processed_text = self._truncate_text(
                            processed_text, validator.max_length
                        )

        return processed_text

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length, preferring sentence boundaries."""
        if len(text) <= max_length:
            return text

        # Try to find a sentence ending before max_length
        truncated = text[:max_length]

        # Look for sentence endings
        last_period = truncated.rfind(".")
        last_exclaim = truncated.rfind("!")
        last_question = truncated.rfind("?")

        # Find the last sentence ending
        last_sentence_end = max(last_period, last_exclaim, last_question)

        if (
            last_sentence_end > max_length * 0.8
        ):  # If we have a sentence ending in last 20%
            return text[: last_sentence_end + 1]

        # Otherwise, try to break at a word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.9:  # If we have a space in last 10%
            return text[:last_space] + "..."

        # Last resort: hard truncate
        return text[: max_length - 3] + "..."

    async def _finalize_result(self, result: SifakaResult, start_time: float) -> None:
        """Finalize the result object."""
        result.processing_time = time.time() - start_time

        # Store result if storage is configured
        try:
            await self.storage.save(result)
        except Exception:
            # Storage errors are non-fatal
            pass
