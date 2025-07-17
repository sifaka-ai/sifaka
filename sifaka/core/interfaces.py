"""Core interfaces defining the contracts for Sifaka components.

This module contains the abstract base classes that define the interfaces
for the two main extension points in Sifaka:

1. Validators - Ensure text meets quality criteria
2. Critics - Analyze text and suggest improvements

These interfaces enable a plugin architecture where new validators and
critics can be added without modifying core Sifaka code."""

from abc import ABC, abstractmethod

from .models import CritiqueResult, SifakaResult, ValidationResult


class Validator(ABC):
    """Abstract base class for text quality validators.

    Validators check if text meets specific criteria and can block
    further processing if critical requirements aren't met. They're
    used to enforce constraints like minimum length, content safety,
    or domain-specific rules.

    Validators run after each text generation to ensure quality standards
    are maintained throughout the improvement process.

    Example:
        >>> class MinLengthValidator(Validator):
        ...     def __init__(self, min_length: int):
        ...         self.min_length = min_length
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return f"min_length_{self.min_length}"
        ...
        ...     async def validate(self, text, result):
        ...         passed = len(text) >= self.min_length
        ...         return ValidationResult(
        ...             validator=self.name,
        ...             passed=passed,
        ...             details=f"Length: {len(text)}/{self.min_length}"
        ...         )
    """

    @abstractmethod
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate the given text against this validator's criteria.

        Validators should be deterministic - the same text should always
        produce the same validation result. They should also be fast since
        they run on every iteration.

        Args:
            text: The current text to validate. This could be the original
                text or an improved version from a previous iteration.
            result: The complete SifakaResult object containing all history.
                Validators can use this for context-aware validation (e.g.,
                ensuring improvements don't make text shorter).

        Returns:
            ValidationResult containing:
            - validator: Name of this validator
            - passed: Boolean indicating if validation passed
            - score: Optional float score (0.0-1.0) for scoring validators
            - details: Human-readable explanation of the result
            - timestamp: When validation occurred (auto-set)

        Note:
            Validators should NOT modify the text. They only evaluate it.
            Text transformation should be done by critics and generators.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this validator.

        The name is used for tracking validation results and should be
        descriptive of what the validator checks. It should be stable
        across runs to enable proper result tracking.

        Returns:
            String identifier like 'min_length_50' or 'no_profanity'
        """


class Critic(ABC):
    """Abstract base class for text improvement critics.

    Critics are the core of Sifaka's improvement system. They analyze
    text from different perspectives and provide structured feedback
    about quality issues and improvement opportunities.

    Each critic represents a different improvement strategy or evaluation
    perspective (e.g., clarity, coherence, factual accuracy). Multiple
    critics can be combined to improve text from multiple angles.

    The critic system is inspired by research showing that iterative
    refinement with specific feedback produces better results than
    single-pass generation.

    Example:
        >>> class ClarityCritic(Critic):
        ...     @property
        ...     def name(self) -> str:
        ...         return "clarity"
        ...
        ...     async def critique(self, text, result):
        ...         # Analyze text for clarity issues
        ...         return CritiqueResult(
        ...             critic="clarity",
        ...             feedback="Some sentences are too complex.",
        ...             suggestions=["Break up the third sentence."],
        ...             needs_improvement=True,
        ...             confidence=0.8
        ...         )
    """

    @abstractmethod
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Analyze text and provide structured improvement feedback.

        This is the main method critics implement to provide their unique
        perspective on text quality and improvement opportunities.

        Args:
            text: The current text to critique. This is typically either
                the original input or the most recent improvement.
            result: The complete SifakaResult containing all history from
                previous iterations. Critics can use this to:
                - See previous feedback to avoid repetition
                - Track if their suggestions were implemented
                - Understand the evolution of the text
                - Access the original text for comparison

        Returns:
            CritiqueResult containing:
            - critic: Name of this critic (should match self.name)
            - feedback: Qualitative assessment explaining strengths/weaknesses
            - suggestions: List of specific, actionable improvements
            - needs_improvement: Whether this critic thinks more work is needed
            - confidence: How certain the critic is (0.0-1.0)
            - metadata: Optional dict with critic-specific data
            - Additional fields for traceability (model, tokens, timing)

        Note:
            Critics should focus on analysis and feedback. They don't
            generate improved text - that's done by the generator using
            the critics' feedback.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this critic.

        Used for registration, configuration, and tracking which critic
        provided which feedback. Should be lowercase with underscores.

        Returns:
            String identifier like 'reflexion', 'self_refine', or 'clarity'
        """
