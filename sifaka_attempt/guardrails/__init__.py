"""
Guardrails for the Sifaka library.

This package provides guardrail components for adding safety and compliance
features to text processing pipelines.
"""

from typing import Dict, Any, List, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Protocol for guardrails that process text for safety and compliance.

    Guardrails implement this protocol to provide safety and compliance features
    such as filtering inappropriate content, detecting PII, or preventing prompt
    injection attacks.
    """

    def process(self, text: str) -> str:
        """
        Process text to ensure safety and compliance.

        Args:
            text: The text to process

        Returns:
            The processed text (may be modified or redacted)
        """
        ...

    def validate(self, text: str) -> bool:
        """
        Validate if text meets safety and compliance requirements.

        Args:
            text: The text to validate

        Returns:
            True if the text is safe and compliant, False otherwise
        """
        ...


class GuardrailResult:
    """
    Result of a guardrail operation.

    This class represents the result of processing text through a guardrail,
    including the processed text, whether it passed validation, and any issues found.
    """

    def __init__(
        self,
        processed_text: str,
        passed: bool = True,
        message: str = "",
        issues: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a guardrail result.

        Args:
            processed_text: The text after processing (may be modified)
            passed: Whether the text passed validation
            message: Message describing the result
            issues: List of issues found
            metadata: Additional information about the result
        """
        self.processed_text = processed_text
        self.passed = passed
        self.message = message
        self.issues = issues or []
        self.metadata = metadata or {}


# Import guardrail implementations
try:
    from .guardrails_ai import GuardrailsValidator, GuardrailsRail, GuardrailsAIAdapter

    GUARDRAILS_AI_AVAILABLE = True
except ImportError:
    GUARDRAILS_AI_AVAILABLE = False

# Export guardrails
__all__ = [
    "GuardrailProtocol",
    "GuardrailResult",
]

# Add guardrails-ai exports if available
if GUARDRAILS_AI_AVAILABLE:
    __all__.extend(["GuardrailsValidator", "GuardrailsRail", "GuardrailsAIAdapter"])
