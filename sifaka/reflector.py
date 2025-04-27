"""
Reflector class for Sifaka.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
import concurrent.futures
import logging

from pydantic import BaseModel, ConfigDict, computed_field, Field

from sifaka.critics.base import Critic
from sifaka.rules.base import Rule, RuleResult
from sifaka.models.base import ModelProvider
from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer
from sifaka.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


class Reflector(BaseModel):
    """
    A reflector that validates and critiques prompts.

    Attributes:
        name: Name of the reflector
        description: Description of what this reflector does
        rules: List of rules to apply
        critic: Optional critic to use
        model: Optional model provider to use
        config: Additional configuration parameters
        tracer: Optional tracer for debugging
    """

    name: str = Field(default="reflector", description="Name of the reflector")
    description: str = Field(
        default="Validates and critiques prompts",
        description="Description of what this reflector does",
    )
    rules: List[Rule] = Field(default_factory=list, description="List of rules to apply")
    critic: Optional[Critic] = Field(default=None, description="Optional critic to use")
    model: Optional[ModelProvider] = Field(
        default=None, description="Optional model provider to use"
    )
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    tracer: Optional[Tracer] = Field(default=None, description="Optional tracer for debugging")
    parallel_validation: bool = Field(default=False, description="Whether to run rules in parallel")
    max_workers: Optional[int] = Field(
        default=None, description="Maximum number of worker threads for parallel validation"
    )
    performance_monitor: Optional[PerformanceMonitor] = Field(
        default_factory=PerformanceMonitor, description="Optional performance monitor"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    def critique(self) -> bool:
        """Whether critique functionality is enabled (True if critic is provided)."""
        return self.critic is not None

    def __init__(
        self,
        name: str,
        model: ModelProvider,
        rules: Optional[List[Rule]] = None,
        tracer: Optional[Tracer] = None,
        critic: Optional[Critic] = None,
        parallel_validation: bool = False,
        max_workers: Optional[int] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a reflector.

        Args:
            name: The name of the reflector
            model: The model provider to use
            rules: List of rules to apply
            tracer: Optional tracer for debugging
            critic: Optional critique system for improving outputs
            parallel_validation: Whether to run rules in parallel
            max_workers: Maximum number of worker threads for parallel validation
            performance_monitor: Optional performance monitor
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            model=model,
            rules=rules or [],
            tracer=tracer,
            critic=critic,
            parallel_validation=parallel_validation,
            max_workers=max_workers,
            performance_monitor=performance_monitor or PerformanceMonitor(),
            **kwargs,
        )

    @property
    def has_rules(self) -> bool:
        """Return whether the reflector has any rules."""
        return bool(self.rules)

    @property
    def has_tracer(self) -> bool:
        """Return whether the reflector has a tracer."""
        return self.tracer is not None

    @property
    def has_critic(self) -> bool:
        """Return whether the reflector has a critic."""
        return self.critic is not None

    def _sort_rules(self) -> List[Rule]:
        """
        Sort rules by priority (highest first) and cost (lowest first).

        Returns:
            Sorted list of rules
        """
        return sorted(self.rules, key=lambda r: (-r.priority, r.cost))

    def _validate_rule(self, rule: Rule, output: str) -> Optional[Dict[str, Any]]:
        """
        Validate a single rule.

        Args:
            rule: The rule to validate
            output: The output to validate

        Returns:
            Dict with violation info if rule failed, None if passed
        """
        start_time = time.time()
        result = rule.validate(output)
        elapsed = time.time() - start_time

        # Record performance metrics
        self.performance_monitor.record_rule_time(rule.name, elapsed)

        if not result.passed:
            return {"rule": rule.name, "message": result.message, "metadata": result.metadata}
        return None

    def _validate_output(self, output: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the output using the reflector's rules.

        Args:
            output: The output to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the output passed all rules
                - violations: List of rule violations
        """
        violations = []

        if not self.parallel_validation:
            # Sequential validation with early exit
            for rule in self.rules:
                violation = self._validate_rule(rule, output)
                if violation:
                    violations.append(violation)
                    # Early exit if critique is enabled (since we'll need to improve it anyway)
                    if self.critique:
                        break
        else:
            # Parallel validation using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_rule = {
                    executor.submit(self._validate_rule, rule, output): rule for rule in self.rules
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_rule):
                    violation = future.result()
                    if violation:
                        violations.append(violation)
                        # Early exit if critique is enabled
                        if self.critique:
                            # Cancel remaining futures
                            for f in future_to_rule:
                                if not f.done():
                                    f.cancel()
                            break

        passed = not violations
        self.performance_monitor.record_attempt(passed)
        return passed, violations

    def _improve_output(self, output: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve the output using the critique system.

        Args:
            output: The output to improve
            violations: List of violations to fix

        Returns:
            The improved output

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if not self.has_critic:
            raise ValueError(
                "Critique is enabled but no critic is configured. "
                "Please provide a critic when initializing the reflector."
            )

        # Get the improved output from the critic
        start_time = time.time()
        improved_output = self.critic.improve(output, violations)
        elapsed = time.time() - start_time
        self.performance_monitor.record_critique_time(elapsed)

        logger.debug("Improved output: %s", improved_output)

        # Validate the improved output
        passed, new_violations = self._validate_output(improved_output)
        if not passed:
            logger.warning("Improved output still has violations: %s", new_violations)

        return improved_output

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Trace an event if a tracer is configured.

        Args:
            event_type: The type of event
            data: The event data
        """
        if self.has_tracer:
            self.tracer.add_event("reflector", event_type, data)

    def reflect(self, prompt: str, max_attempts: int = 3, **kwargs) -> str:
        """
        Reflect on a prompt and generate an improved output.

        Args:
            prompt: The prompt to reflect on
            max_attempts: Maximum number of attempts to fix violations
            **kwargs: Additional arguments for the model

        Returns:
            The improved output

        Raises:
            ValueError: If the output fails validation and critique is disabled
            RuntimeError: If max attempts reached and output still has violations
        """
        # Start tracing if configured
        if self.has_tracer:
            self.tracer.start_trace("reflector")

        # Generate the output
        self._trace_event("start", {"prompt": prompt})
        start_time = time.time()
        output = self.model.generate(prompt, **kwargs)
        elapsed = time.time() - start_time
        self.performance_monitor.record_generation_time(elapsed)
        self._trace_event("end", {"output": output})

        # Validate and improve the output
        attempt = 1
        passed, violations = self._validate_output(output)

        while not passed and attempt <= max_attempts:
            if not self.critique:
                raise ValueError(f"Output failed validation: {violations}")

            # Log attempt number and violations
            logger.info(f"Attempt {attempt}/{max_attempts} to fix violations:")
            for v in violations:
                logger.info(f"- {v['rule']}: {v['message']}")

            # Improve the output using the critique system
            self._trace_event("critique_start", {"attempt": attempt, "violations": violations})
            output = self._improve_output(output, violations)
            self._trace_event("critique_end", {"attempt": attempt, "output": output})

            # Check if violations are fixed
            passed, violations = self._validate_output(output)
            attempt += 1

        if not passed:
            logger.warning(
                f"Failed to fix all violations after {max_attempts} attempts. Remaining violations:"
            )
            for v in violations:
                logger.warning(f"- {v['rule']}: {v['message']}")
            raise RuntimeError(f"Failed to fix all violations after {max_attempts} attempts")

        return output

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Reflect on a prompt and generate an improved output.

        This allows the reflector to be called like a function.

        Args:
            prompt: The prompt to reflect on
            **kwargs: Additional arguments for the model

        Returns:
            The improved output
        """
        return self.reflect(prompt, **kwargs)

    def reflect_batch(
        self, prompts: List[str], batch_size: int = 10, max_attempts: int = 3, **kwargs
    ) -> List[str]:
        """
        Reflect on multiple prompts in batches.

        Args:
            prompts: List of prompts to reflect on
            batch_size: Size of batches for processing
            max_attempts: Maximum number of attempts to fix violations
            **kwargs: Additional arguments for the model

        Returns:
            List of improved outputs

        Raises:
            ValueError: If outputs fail validation and critique is disabled
            RuntimeError: If max attempts reached and outputs still have violations
        """
        results = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            # Generate outputs for the batch
            if self.has_tracer:
                self.tracer.start_trace("reflector_batch")

            self._trace_event("batch_start", {"batch_size": len(batch)})
            outputs = self.model.generate_batch(batch, **kwargs)
            self._trace_event("batch_end", {"outputs": outputs})

            # Validate and improve each output in the batch
            for j, output in enumerate(outputs):
                attempt = 1
                passed, violations = self._validate_output(output)

                while not passed and attempt <= max_attempts:
                    if not self.critique:
                        raise ValueError(f"Output {i+j} failed validation: {violations}")

                    # Log attempt number and violations
                    logger.info(
                        f"Output {i+j}, Attempt {attempt}/{max_attempts} to fix violations:"
                    )
                    for v in violations:
                        logger.info(f"- {v['rule']}: {v['message']}")

                    # Improve the output using the critique system
                    self._trace_event(
                        "critique_start",
                        {"batch_index": i + j, "attempt": attempt, "violations": violations},
                    )
                    output = self._improve_output(output, violations)
                    self._trace_event(
                        "critique_end", {"batch_index": i + j, "attempt": attempt, "output": output}
                    )

                    # Check if violations are fixed
                    passed, violations = self._validate_output(output)
                    attempt += 1

                if not passed:
                    logger.warning(
                        f"Output {i+j}: Failed to fix violations after {max_attempts} attempts. "
                        f"Remaining violations:"
                    )
                    for v in violations:
                        logger.warning(f"- {v['rule']}: {v['message']}")
                    raise RuntimeError(
                        f"Output {i+j}: Failed to fix violations after {max_attempts} attempts"
                    )

                results.append(output)

        return results
