"""GuardrailsAI integration for advanced content validation and safety.

TODO: GuardrailsAI support is temporarily disabled due to a dependency conflict
with griffe. The issue has been fixed in guardrails main branch but not yet
released. Once guardrails releases a new version with the updated griffe
dependency, uncomment this module to re-enable guardrails support.

See: https://github.com/guardrails-ai/guardrails (main branch has the fix)

This module will provide seamless integration with GuardrailsAI's comprehensive
validator hub, enabling advanced content validation capabilities including
toxicity detection, PII identification, profanity filtering, and custom
business rule validation.
"""

# Temporary placeholder to maintain module structure
from typing import List, Optional

from ..core.models import SifakaResult
from .base import BaseValidator, ValidatorConfig

# TODO: Uncomment the entire implementation below once guardrails releases
# a new version with the updated griffe dependency

"""
## Key Features (when re-enabled):

- **Pre-built Validators**: Access to 50+ validators from GuardrailsAI hub
- **Safety & Compliance**: Built-in toxicity, PII, and profanity detection
- **Business Rules**: Custom validation for domain-specific requirements
- **Automatic Installation**: Validators are installed on-demand from hub
- **Flexible Actions**: Configure how to handle validation failures

## Popular Validators:

- **toxic-language**: Detects toxic or offensive language
- **detect-pii**: Identifies personally identifiable information
- **profanity-free**: Ensures text is free from profanity
- **gibberish**: Detects nonsensical or meaningless text
- **valid-url**: Validates URL formats and accessibility
- **competitors-check**: Checks for competitor mentions
- **one-line**: Ensures text is a single line
- **length**: Validates text length constraints

## Usage Examples (when re-enabled):

    >>> # Content safety validator
    >>> safety_validator = GuardrailsValidator([
    ...     "toxic-language",
    ...     "detect-pii",
    ...     "profanity-free"
    ... ])
    >>>
    >>> # Business compliance validator
    >>> compliance_validator = GuardrailsValidator([
    ...     "competitors-check",
    ...     "valid-url",
    ...     "gibberish"
    ... ], on_fail="exception")

## Installation (when re-enabled):

Requires GuardrailsAI package:
```bash
pip install sifaka[guardrails]
# or
pip install guardrails-ai
```

## Error Handling:

The validator automatically handles validator installation, Guard creation,
and provides detailed feedback on validation failures. Failed validations
include specific validator names and failure reasons for debugging.

## Performance Considerations:

- Validators are installed once and cached for reuse
- Guard objects are built lazily and reused across validations
- Thread-safe operation for concurrent validation scenarios

import asyncio
import warnings
from typing import List, Optional, Tuple

# GuardrailsAI optional dependency handling
# Allows graceful degradation when GuardrailsAI is not installed
try:
    # Suppress pkg_resources deprecation warning from guardrails
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="guardrails.hub.install"
        )
        import guardrails as gr
        from guardrails.hub import install

    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False
    # Module can still be imported, but GuardrailsValidator will raise
    # ImportError with helpful installation instructions when used

from ..core.models import SifakaResult
from .base import BaseValidator, ValidatorConfig


class GuardrailsValidator(BaseValidator):
    ""\"Validator that integrates GuardrailsAI for advanced content validation.

    Provides access to GuardrailsAI's extensive validator hub for sophisticated
    content validation including safety checks, compliance validation, and
    business rule enforcement. Automatically handles validator installation
    and Guard creation for seamless integration.

    Key capabilities:
    - 50+ pre-built validators from GuardrailsAI hub
    - Automatic validator installation on first use
    - Thread-safe Guard creation and reuse
    - Configurable failure handling actions
    - Detailed validation reporting with specific failure reasons

    Popular validator categories:

    **Safety & Moderation:**
    - toxic-language: Detects toxic or offensive language
    - profanity-free: Ensures text is free from profanity
    - gibberish: Detects nonsensical or meaningless text

    **Privacy & Compliance:**
    - detect-pii: Identifies personally identifiable information
    - competitors-check: Checks for competitor mentions
    - valid-url: Validates URLs in text

    **Format & Structure:**
    - one-line: Ensures text is a single line
    - length: Validates text length constraints
    - json: Validates JSON format

    Example:
        >>> # Comprehensive safety validator
        >>> safety_validator = GuardrailsValidator([
        ...     "toxic-language",
        ...     "detect-pii",
        ...     "profanity-free",
        ...     "gibberish"
        ... ])
        >>>
        >>> # Business compliance validator
        >>> business_validator = GuardrailsValidator([
        ...     "competitors-check",
        ...     "valid-url"
        ... ], on_fail="exception")
        >>>
        >>> # Use in validation
        >>> result = await safety_validator.validate(text, sifaka_result)
        >>> if not result.passed:
        ...     print(f"Safety violations: {result.details}")

    Performance notes:
        - Validators are installed once and cached for subsequent use
        - Guard objects are reused across multiple validations
        - Thread-safe design supports concurrent validation scenarios
    ""\"

    def __init__(
        self,
        validators: List[str],
        on_fail: str = "fix",
        config: Optional[ValidatorConfig] = None,
    ):
        ""\"Initialize GuardrailsValidator with specified validators.

        Creates a validator that uses GuardrailsAI validators for content
        validation. Validators are installed automatically on first use.

        Args:
            validators: List of validator names from GuardrailsAI hub.
                See https://hub.guardrailsai.com for available validators.
                Common examples: ["toxic-language", "detect-pii", "profanity-free"]
            on_fail: Action to take when validation fails:
                - "fix": Attempt to fix the content (default)
                - "filter": Filter out problematic content
                - "refrain": Stop processing and return failure
                - "exception": Raise an exception on failure
            config: Optional validator configuration for advanced settings

        Raises:
            ImportError: If GuardrailsAI is not installed
            ValueError: If a validator fails to install from the hub

        Example:
            >>> # Content moderation validator
            >>> validator = GuardrailsValidator([
            ...     "toxic-language",
            ...     "profanity-free",
            ...     "detect-pii"
            ... ])
            >>>
            >>> # Strict compliance validator that raises exceptions
            >>> strict_validator = GuardrailsValidator([
            ...     "competitors-check",
            ...     "valid-url"
            ... ], on_fail="exception")

        Installation note:
            Requires GuardrailsAI package. Install with:
            pip install sifaka[guardrails] or pip install guardrails-ai
        ""\"
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
        ""\"Ensure all required validators are installed from GuardrailsAI hub.

        Downloads and installs validators on-demand from the GuardrailsAI hub.
        Validators are cached after installation to avoid repeated downloads.

        Raises:
            ValueError: If any validator fails to install from the hub

        Note:
            This method is called automatically during Guard creation and
            uses caching to avoid redundant installation attempts.
        ""\"
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
        ""\"Build the GuardrailsAI Guard object with configured validators.

        Creates a Guard object using RAIL (Reliable AI Language) specification
        that includes all configured validators. The Guard is cached for reuse
        across multiple validations.

        Note:
            This method is called automatically during first validation and
            uses lazy initialization for optimal performance. The Guard object
            is thread-safe and can be reused across concurrent validations.
        ""\"
        if self._guard is not None:
            return  # type: ignore[unreachable]  # Already initialized

        self._ensure_validators_installed()

        # Create Guard with validators
        rail_spec = ""\"
<rail version="0.1">
<output>
    <string name="text" description="The validated text">
        ""\"

        # Add each validator
        for validator_name in self.validators:
            rail_spec += f'<validator name="{validator_name}" on-fail="{self.on_fail}"/>\n        '

        rail_spec += ""\"
    </string>
</output>
</rail>
""\"

        self._guard = gr.Guard.from_rail_string(rail_spec)

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> Tuple[bool, float, str]:
        ""\"Validate text using GuardrailsAI validators.

        Executes all configured GuardrailsAI validators against the text and
        returns comprehensive validation results with detailed failure information.

        Args:
            text: Text content to validate against all configured validators
            result: SifakaResult for context (not currently used but available
                for future enhancements)

        Returns:
            Tuple containing:
            - bool: True if all validators pass, False if any validator fails
            - float: Score from 0.0-1.0 based on proportion of validators that passed
            - str: Detailed feedback including specific validator failures

        Process:
        1. Initialize Guard object with validators (thread-safe, cached)
        2. Execute validation using GuardrailsAI's validation engine
        3. Collect detailed failure information from validator logs
        4. Calculate proportional score based on validator success rate
        5. Return comprehensive results with actionable feedback

        Error handling:
            Validator installation errors and Guard creation errors are
            propagated as exceptions. Runtime validation errors are handled
            by the base validator's error handling mechanism.

        Example output:
            >>> # All validators pass
            >>> (True, 1.0, "All GuardrailsAI validators passed")
            >>>
            >>> # Some validators fail
            >>> (False, 0.67, "toxic-language: Detected offensive content; detect-pii: Found email address")
        ""\"
        # Thread-safe guard initialization
        async with self._lock:
            self._build_guard()

        try:
            # Run validation
            guard = self._guard
            if guard is None:
                raise RuntimeError("Guard not initialized after _build_guard")
            validated_output = guard.validate({"text": text})  # type: ignore[unreachable]

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
                score = (
                    max(0.0, (total_validators - failed_count) / total_validators)
                    if total_validators > 0
                    else 0.0
                )

                return False, score, details

        except Exception:
            # Let BaseValidator handle the error
            raise

    @property
    def name(self) -> str:
        ""\"Return descriptive validator name including configured validators.

        Returns:
            String identifier that includes all configured validator names
            for easy identification in logs and validation results

        Example:
            >>> validator = GuardrailsValidator(["toxic-language", "detect-pii"])
            >>> print(validator.name)
            "guardrails[toxic-language, detect-pii]"
        ""\"
        validator_list = ", ".join(self.validators)
        return f"guardrails[{validator_list}]"
"""


class GuardrailsValidator(BaseValidator):
    """Placeholder for GuardrailsValidator - temporarily disabled.

    GuardrailsAI integration is temporarily disabled due to a dependency
    conflict. This will be re-enabled once guardrails releases a new version
    with the updated griffe dependency.
    """

    def __init__(
        self,
        validators: List[str],
        on_fail: str = "fix",
        config: Optional[ValidatorConfig] = None,
    ):
        """Initialize placeholder GuardrailsValidator."""
        raise NotImplementedError(
            "GuardrailsValidator is temporarily disabled due to a dependency conflict. "
            "It will be re-enabled once guardrails releases a new version with the "
            "updated griffe dependency. See the module docstring for more information."
        )

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> tuple[bool, float, str]:
        """Placeholder validation method."""
        raise NotImplementedError("GuardrailsValidator is temporarily disabled")

    @property
    def name(self) -> str:
        """Return validator name."""
        return "guardrails[disabled]"
