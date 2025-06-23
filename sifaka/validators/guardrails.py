"""GuardrailsAI integration for Sifaka validators."""

import asyncio
from typing import List, Optional, Tuple

try:
    import guardrails as gr  # type: ignore[import-not-found]
    from guardrails.hub import install  # type: ignore[import-not-found]

    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False

from ..core.models import SifakaResult
from .base import BaseValidator, ValidatorConfig


class GuardrailsValidator(BaseValidator):
    """Validator that uses GuardrailsAI for content validation.

    This validator provides integration with GuardrailsAI's validator hub,
    allowing you to use pre-built validators for various content checks.

    Common validators:
    - toxic-language: Detects toxic or offensive language
    - detect-pii: Identifies personally identifiable information
    - profanity-free: Ensures text is free from profanity
    - gibberish: Detects nonsensical text
    - valid-url: Validates URLs in text
    - competitors-check: Checks for competitor mentions
    - one-line: Ensures text is a single line
    - length: Validates text length

    Example:
        ```python
        validator = GuardrailsValidator([
            "toxic-language",
            "detect-pii",
            "profanity-free"
        ])
        ```
    """

    def __init__(self, validators: List[str], on_fail: str = "fix", config: Optional[ValidatorConfig] = None):
        """Initialize GuardrailsValidator.

        Args:
            validators: List of validator names from GuardrailsAI hub
            on_fail: Action to take on validation failure ("fix", "filter", "refrain", "exception")
            config: Optional validator configuration
        """
        super().__init__(config)
        if not HAS_GUARDRAILS:
            raise ImportError(
                "GuardrailsAI is not installed. Install with: pip install sifaka[guardrails]"
            )

        self.validators = validators
        self.on_fail = on_fail
        self._installed_validators: set[str] = set()
        self._guard = None
        self._lock = asyncio.Lock()  # Thread-safe access to shared state

    def _ensure_validators_installed(self) -> None:
        """Ensure all required validators are installed."""
        for validator_name in self.validators:
            if validator_name not in self._installed_validators:
                try:
                    # Install validator from hub
                    install(validator_name)
                    self._installed_validators.add(validator_name)
                except Exception as e:
                    raise ValueError(
                        f"Failed to install GuardrailsAI validator '{validator_name}': {e}"
                    )

    def _build_guard(self) -> None:
        """Build the Guard object with validators."""
        if self._guard is not None:
            return

        self._ensure_validators_installed()

        # Create Guard with validators
        rail_spec = """
<rail version="0.1">
<output>
    <string name="text" description="The validated text">
        """

        # Add each validator
        for validator_name in self.validators:
            rail_spec += f'<validator name="{validator_name}" on-fail="{self.on_fail}"/>\n        '

        rail_spec += """
    </string>
</output>
</rail>
"""

        self._guard = gr.Guard.from_rail_string(rail_spec)

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> Tuple[bool, float, str]:
        """Validate text using GuardrailsAI validators.

        Args:
            text: Text to validate
            result: Current Sifaka result (unused)

        Returns:
            Tuple of (passed, score, details)
        """
        # Thread-safe guard initialization
        async with self._lock:
            self._build_guard()

        try:
            # Run validation
            if self._guard is None:
                raise RuntimeError("Guard not initialized")
            validated_output = self._guard.validate({"text": text})

            # Check if validation passed
            if validated_output.validation_passed:
                return True, 1.0, "All GuardrailsAI validators passed"
            else:
                # Collect failure details
                failures = []
                for fail in validated_output.validator_logs:
                    if not fail.passed:
                        failures.append(f"{fail.validator_name}: {fail.failure_reason}")

                details = "; ".join(failures) if failures else "Validation failed"
                
                # Calculate score based on how many validators passed
                total_validators = len(self.validators)
                failed_count = len(failures)
                score = max(0.0, (total_validators - failed_count) / total_validators) if total_validators > 0 else 0.0
                
                return False, score, details

        except Exception:
            # Let BaseValidator handle the error
            raise

    @property
    def name(self) -> str:
        """Return validator name."""
        validator_list = ", ".join(self.validators)
        return f"guardrails[{validator_list}]"
