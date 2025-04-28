"""
LLM-based classifier implementation.
"""

import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Final,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from typing_extensions import TypeGuard

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7
    ) -> str: ...

@dataclass(frozen=True)
class LLMPromptConfig:
    """Configuration for LLM prompts."""

    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    temperature: float = 0.1
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

@dataclass(frozen=True)
class LLMResponse:
    """Container for parsed LLM responses."""

    label: str
    confidence: float
    explanation: str = ""
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))

class LLMClassifier(BaseClassifier):
    """
    A classifier that uses an LLM for predictions.

    This allows for flexible classification tasks using LLM capabilities.
    """

    # Class-level constants
    DEFAULT_COST: Final[int] = 5  # Higher cost for LLM API calls

    def __init__(
        self,
        name: str,
        description: str,
        model: LLMProvider,
        labels: List[str],
        prompt_config: Optional[LLMPromptConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LLM classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            model: The LLM provider to use
            labels: List of possible labels/classes
            prompt_config: Configuration for LLM prompts
            **kwargs: Additional configuration parameters
        """
        config = ClassifierConfig(labels=labels, cost=self.DEFAULT_COST, **kwargs)
        super().__init__(name=name, description=description, config=config)

        self._validate_model(model)
        self._model = model
        self._prompt_config = prompt_config or self._create_default_prompt_config(labels)

    def _validate_model(self, model: Any) -> TypeGuard[LLMProvider]:
        """Validate that a model implements the required protocol."""
        if not isinstance(model, LLMProvider):
            raise ValueError(f"Model must implement LLMProvider protocol, got {type(model)}")
        return True

    def _create_default_prompt_config(self, labels: List[str]) -> LLMPromptConfig:
        """Create default prompt configuration."""
        return LLMPromptConfig(
            system_prompt=(
                f"You are a classifier that assigns one of the following labels: {', '.join(labels)}. "
                "Respond with a JSON object containing 'label' and 'confidence' (0-1) fields."
            ),
            user_prompt_template=(
                "Classify the following text:\n\n{text}\n\n"
                "Respond with a JSON object containing:\n"
                "- label: one of {labels}\n"
                "- confidence: number between 0 and 1\n"
                "- explanation: brief explanation of the classification"
            ),
        )

    def _parse_llm_response(self, response: str) -> LLMResponse:
        """
        Parse the LLM response into structured data.

        Args:
            response: Raw LLM response

        Returns:
            LLMResponse with parsed data
        """
        try:
            # Try to find JSON in the response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return LLMResponse(
                    label=str(data["label"]).lower(),
                    confidence=float(data["confidence"]),
                    explanation=str(data.get("explanation", "")),
                    raw_response=response,
                )

            raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            # Fallback: try to extract label and confidence using simple heuristics
            lines = response.lower().split("\n")
            extracted = {}

            for line in lines:
                if "label" in line and ":" in line:
                    extracted["label"] = line.split(":")[1].strip().strip("\"'")
                elif "confidence" in line and ":" in line:
                    try:
                        conf = float(line.split(":")[1].strip().strip("\"'"))
                        extracted["confidence"] = min(max(conf, 0), 1)  # Clamp to [0,1]
                    except ValueError:
                        pass
                elif "explanation" in line and ":" in line:
                    extracted["explanation"] = line.split(":")[1].strip().strip("\"'")

            if "label" in extracted and "confidence" in extracted:
                return LLMResponse(
                    label=extracted["label"],
                    confidence=extracted["confidence"],
                    explanation=extracted.get("explanation", "Extracted from text response"),
                    raw_response=response,
                )

            # If all else fails, make a best effort guess
            return LLMResponse(
                label=self.config.labels[0],
                confidence=0.5,
                explanation="Failed to parse LLM response",
                raw_response=response,
            )

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement LLM classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with LLM's prediction
        """
        prompt = self._prompt_config.user_prompt_template.format(
            text=text, labels=self.config.labels
        )

        try:
            response = self._model.generate(
                prompt,
                system_prompt=self._prompt_config.system_prompt,
                temperature=self._prompt_config.temperature,
            )

            result = self._parse_llm_response(response)

            return ClassificationResult(
                label=result.label,
                confidence=result.confidence,
                metadata={
                    "explanation": result.explanation,
                    "raw_response": result.raw_response,
                },
            )

        except Exception as e:
            logger.error("Failed to classify text with LLM: %s", e)
            return ClassificationResult(
                label=self.config.labels[0],
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "reason": "llm_classification_error",
                },
            )

    @classmethod
    def create_with_custom_model(
        cls,
        model: LLMProvider,
        name: str = "custom_llm_classifier",
        description: str = "Custom LLM classifier",
        labels: List[str] = None,
        prompt_config: Optional[LLMPromptConfig] = None,
        **kwargs,
    ) -> "LLMClassifier":
        """
        Factory method to create a classifier with a custom model.

        Args:
            model: Custom LLM provider implementation
            name: Name of the classifier
            description: Description of the classifier
            labels: List of possible labels/classes
            prompt_config: Custom prompt configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured LLMClassifier instance
        """
        if not labels:
            raise ValueError("labels must be provided")

        instance = cls(
            name=name,
            description=description,
            model=model,
            labels=labels,
            prompt_config=prompt_config,
            **kwargs,
        )
        instance._validate_model(model)
        return instance
