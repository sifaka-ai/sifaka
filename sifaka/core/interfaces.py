"""Core interfaces for Sifaka components."""

from abc import ABC, abstractmethod

from .models import SifakaResult, ValidationResult, CritiqueResult


class Validator(ABC):
    """Base interface for text validators."""

    @abstractmethod
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text and return result.

        Args:
            text: Text to validate
            result: Current SifakaResult for context

        Returns:
            ValidationResult with pass/fail and details
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this validator."""
        pass


class Critic(ABC):
    """Base interface for text critics."""

    @abstractmethod
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Critique text and provide improvement suggestions.

        Args:
            text: Text to critique
            result: Current SifakaResult for context

        Returns:
            CritiqueResult with feedback and suggestions
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this critic."""
        pass
