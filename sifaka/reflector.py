"""
Reflector class for Sifaka.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from sifaka.critique.base import Critique
from sifaka.models.base import ModelProvider
from sifaka.rules.base import Rule, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)


class Reflector(BaseModel):
    """
    A reflector that validates and improves LLM outputs.

    Attributes:
        name: The name of the reflector
        model: The model provider to use
        rules: List of rules to apply
        critique: Whether to enable critique
        tracer: Optional tracer for debugging
        critic: Optional critique system for improving outputs
    """

    name: str
    model: ModelProvider
    rules: List[Rule] = []
    critique: bool = True
    tracer: Optional[Tracer] = None
    critic: Optional[Critique] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str,
        model: ModelProvider,
        rules: Optional[List[Rule]] = None,
        critique: bool = True,
        tracer: Optional[Tracer] = None,
        critic: Optional[Critique] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a reflector.

        Args:
            name: The name of the reflector
            model: The model provider to use
            rules: List of rules to apply
            critique: Whether to enable critique
            tracer: Optional tracer for debugging
            critic: Optional critique system for improving outputs
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            model=model,
            rules=rules or [],
            critique=critique,
            tracer=tracer,
            critic=critic,
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
        for rule in self.rules:
            result = rule.validate(output)
            if not result.passed:
                violations.append(
                    {"rule": rule.name, "message": result.message, "metadata": result.metadata}
                )

        return not violations, violations

    def _improve_output(self, output: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve the output using the critique system.

        Args:
            output: The output to improve
            violations: List of rule violations

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

        # Create a critique prompt
        critique_prompt = f"""
        The following output failed validation with the following violations:
        {violations}

        Original output:
        {output}

        Please provide an improved version that addresses these issues.
        """

        # Get the improved output from the critic
        improved_output = self.critic.critique(critique_prompt)
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

    def reflect(self, prompt: str, **kwargs) -> str:
        """
        Reflect on a prompt and generate an improved output.

        Args:
            prompt: The prompt to reflect on
            **kwargs: Additional arguments for the model

        Returns:
            The improved output

        Raises:
            ValueError: If the output fails validation and critique is disabled
        """
        # Start tracing if configured
        if self.has_tracer:
            self.tracer.start_trace("reflector")

        # Generate the output
        self._trace_event("start", {"prompt": prompt})
        output = self.model.generate(prompt, **kwargs)
        self._trace_event("end", {"output": output})

        # Validate the output
        passed, violations = self._validate_output(output)
        if not passed:
            if self.critique:
                # Improve the output using the critique system
                self._trace_event("critique_start", {"violations": violations})
                output = self._improve_output(output, violations)
                self._trace_event("critique_end", {"output": output})
            else:
                raise ValueError(f"Output failed validation: {violations}")

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
