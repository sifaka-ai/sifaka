"""
Toxicity classifier using the Detoxify model.
"""

import importlib
from typing import (
    Any,
    Dict,
    List,
    Optional,
    ClassVar,
    TypeVar,
    Type,
    Union,
    Callable,
    cast,
    overload,
)

import numpy as np
from typing_extensions import TypeGuard
from pydantic import PrivateAttr

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
    TextClassifier,
)
from sifaka.utils.logging import get_logger
from sifaka.classifiers.toxicity_model import ToxicityModel

logger = get_logger(__name__)


class ToxicityClassifier(TextClassifier):
    """
    A lightweight toxicity classifier using the Detoxify model.

    This provides a fast, local alternative to API-based toxicity detection.
    Requires the 'toxicity' extra to be installed:
    pip install sifaka[toxicity]
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    DEFAULT_COST: ClassVar[int] = 2  # Moderate cost for local ML model

    # Default thresholds
    DEFAULT_SEVERE_TOXIC_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_THREAT_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_GENERAL_THRESHOLD: ClassVar[float] = 0.5

    # Private attributes using PrivateAttr for state management
    _model: Optional[ToxicityModel] = PrivateAttr(default=None)
    _labels: List[str] = PrivateAttr(default_factory=list)
    _thresholds: Dict[str, float] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        name: str = "toxicity_classifier",
        description: str = "Detects toxic content using Detoxify",
        config: Optional[ClassifierConfig[str]] = None,
        labels: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize the toxicity classifier with optional config and thresholds."""
        if config is None:
            config = ClassifierConfig[str](
                labels=self.DEFAULT_LABELS,
                cost=self.DEFAULT_COST,
                params={},
            )

        super().__init__(name=name, description=description, config=config)

        # Set labels
        self._labels = labels or self.DEFAULT_LABELS

        # Set thresholds with defaults
        params = thresholds or {}
        params["general_threshold"] = params.get("general_threshold", self.DEFAULT_GENERAL_THRESHOLD)
        params["severe_toxic_threshold"] = params.get(
            "severe_toxic_threshold", self.DEFAULT_SEVERE_TOXIC_THRESHOLD
        )
        params["threat_threshold"] = params.get("threat_threshold", self.DEFAULT_THREAT_THRESHOLD)
        self._thresholds = params

        # Initialize model
        self._model = None

    def _validate_model(self, model: Any) -> TypeGuard[ToxicityModel]:
        """Validate that a model implements the required protocol."""
        if not isinstance(model, ToxicityModel):
            raise ValueError(f"Model must implement ToxicityModel protocol, got {type(model)}")
        return True

    def _load_detoxify(self) -> ToxicityModel:
        """Load the Detoxify package and model."""
        try:
            # Get model_name from config.params
            model_name = self.config.params.get("model_name", "original")

            detoxify_module = importlib.import_module("detoxify")
            model = detoxify_module.Detoxify(model_type=model_name)
            self._validate_model(model)
            return model
        except ImportError:
            raise ImportError(
                "Detoxify package is required for ToxicityClassifier. "
                "Install it with: pip install sifaka[toxicity]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Detoxify: {e}")

    def warm_up(self) -> None:
        """Initialize the model if needed."""
        if not self._model:
            self._model = self._model or self._load_detoxify()

    def _get_toxicity_label(self, scores: Dict[str, float]) -> tuple[str, float]:
        """Get toxicity label and confidence based on scores."""
        # Get thresholds from config.params
        severe_toxic_threshold = self.config.params.get(
            "severe_toxic_threshold", self.DEFAULT_SEVERE_TOXIC_THRESHOLD
        )
        threat_threshold = self.config.params.get("threat_threshold", self.DEFAULT_THREAT_THRESHOLD)
        general_threshold = self.config.params.get(
            "general_threshold", self.DEFAULT_GENERAL_THRESHOLD
        )

        # Check for severe toxicity or threats first
        if scores.get("severe_toxic", 0) >= severe_toxic_threshold:
            return "severe_toxic", scores["severe_toxic"]
        elif scores.get("threat", 0) >= threat_threshold:
            return "threat", scores["threat"]

        # Get highest toxicity score and category
        max_category = max(scores.items(), key=lambda x: x[1])
        label, confidence = max_category

        # Only return toxic label if it meets the general threshold
        if confidence >= general_threshold:
            return label, confidence

        # Special case: If all toxicity scores are extremely low (e.g., < 0.01),
        # this is clearly non-toxic content, so return high confidence for non_toxic
        max_toxicity_score = max(scores.values()) if scores else 0.0
        if max_toxicity_score < 0.01:
            # If content is extremely non-toxic, return high confidence (0.95)
            return "non_toxic", 0.95

        return "non_toxic", confidence

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Implement toxicity classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity scores
        """
        self.warm_up()

        try:
            scores = self._model.predict(text)
            scores = {k: float(v) for k, v in scores.items()}
            label, confidence = self._get_toxicity_label(scores)

            return ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={"all_scores": scores},
            )
        except Exception as e:
            logger.error("Failed to classify text: %s", e)
            return ClassificationResult[str](
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e), "reason": "classification_error"},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[str]]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)
        self.warm_up()

        try:
            batch_scores = self._model.predict(texts)
            results = []

            for i in range(len(texts)):
                scores = {k: float(v[i]) for k, v in batch_scores.items()}
                label, confidence = self._get_toxicity_label(scores)

                results.append(
                    ClassificationResult[str](
                        label=label,
                        confidence=confidence,
                        metadata={"all_scores": scores},
                    )
                )

            return results
        except Exception as e:
            logger.error("Failed to batch classify texts: %s", e)
            return [
                ClassificationResult[str](
                    label="unknown",
                    confidence=0.0,
                    metadata={"error": str(e), "reason": "batch_classification_error"},
                )
                for _ in texts
            ]

    @classmethod
    def create_with_custom_model(
        cls,
        model: ToxicityModel,
        name: str = "custom_toxicity_classifier",
        description: str = "Custom toxicity model",
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs: Any,
    ) -> "ToxicityClassifier":
        """
        Factory method to create a classifier with a custom model.

        Args:
            model: Custom toxicity model implementation
            name: Name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured ToxicityClassifier instance
        """
        # Validate model first
        if not isinstance(model, ToxicityModel):
            raise ValueError(f"Model must implement ToxicityModel protocol, got {type(model)}")

        # Create default config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Ensure we have model_name and thresholds
            if "model_name" not in params:
                params["model_name"] = "original"
            if "general_threshold" not in params:
                params["general_threshold"] = cls.DEFAULT_GENERAL_THRESHOLD
            if "severe_toxic_threshold" not in params:
                params["severe_toxic_threshold"] = cls.DEFAULT_SEVERE_TOXIC_THRESHOLD
            if "threat_threshold" not in params:
                params["threat_threshold"] = cls.DEFAULT_THREAT_THRESHOLD

            # Create config with params
            config = ClassifierConfig[str](
                labels=cls.DEFAULT_LABELS,
                cost=cls.DEFAULT_COST,
                params=params,
            )

        # Create instance with validated model
        instance = cls(
            name=name,
            description=description,
            config=config,
            **kwargs,
        )

        # Set the model directly
        instance._model = model

        return instance


@overload
def create_toxicity_classifier(
    model_name: str = "original",
    name: str = "toxicity_classifier",
    description: str = "Detects toxic content using Detoxify",
) -> ToxicityClassifier: ...


@overload
def create_toxicity_classifier(
    model_name: str = "original",
    name: str = "toxicity_classifier",
    description: str = "Detects toxic content using Detoxify",
    general_threshold: float = 0.5,
    severe_toxic_threshold: float = 0.7,
    threat_threshold: float = 0.7,
    cache_size: int = 0,
    cost: int = 2,
) -> ToxicityClassifier: ...


def create_toxicity_classifier(
    model_name: str = "original",
    name: str = "toxicity_classifier",
    description: str = "Detects toxic content using Detoxify",
    general_threshold: float = 0.5,
    severe_toxic_threshold: float = 0.7,
    threat_threshold: float = 0.7,
    cache_size: int = 0,
    cost: int = 2,
    **kwargs: Any,
) -> ToxicityClassifier:
    """
    Factory function to create a toxicity classifier.

    Args:
        model_name: Name of the Detoxify model to use ('original', 'unbiased', etc.)
        name: Name of the classifier
        description: Description of the classifier
        general_threshold: General toxicity threshold (0-1)
        severe_toxic_threshold: Severe toxicity threshold (0-1)
        threat_threshold: Threat threshold (0-1)
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured ToxicityClassifier instance
    """
    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})
    params.update({
        "model_name": model_name,
        "general_threshold": general_threshold,
        "severe_toxic_threshold": severe_toxic_threshold,
        "threat_threshold": threat_threshold,
    })

    # Create config
    config = ClassifierConfig[str](
        labels=ToxicityClassifier.DEFAULT_LABELS,
        cache_size=cache_size,
        cost=cost,
        params=params,
    )

    # Create and return classifier
    return ToxicityClassifier(
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
