"""
Refactored ToxicityClassifierImplementation using standardized state management.

This file demonstrates how to refactor the ToxicityClassifierImplementation
to use the standardized state management pattern with _state_manager.
"""

import importlib
import logging
from typing import Any, ClassVar, Dict, List, Tuple, TypeGuard, TypeVar, cast

from pydantic import BaseModel, PrivateAttr

from sifaka.classifiers.models import ClassificationResult, ClassifierConfig
from sifaka.utils.state import StateManager, create_classifier_state, ClassifierState

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar("T")
R = TypeVar("R")


# Protocol for toxicity models
class ToxicityModel(Protocol):
    """Protocol for toxicity models."""

    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]: ...


class ToxicityClassifierImplementation:
    """
    Implementation of toxicity classification logic using the Detoxify model.

    This implementation uses the Detoxify model to detect toxic content in text.
    It provides a fast, local alternative to API-based toxicity detection and
    can identify various forms of toxic content including severe toxicity,
    obscenity, threats, insults, and identity-based hate.

    ## Architecture

    ToxicityClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements toxicity detection
    2. **State Management**: Uses StateManager for internal state
    3. **Resource Management**: Loads and manages Detoxify model

    ## Lifecycle

    1. **Initialization**: Set up configuration and parameters
       - Initialize with config
       - Extract thresholds from config.params
       - Set up default values

    2. **Warm-up**: Load Detoxify resources
       - Load Detoxify model when needed
       - Initialize only once
       - Handle initialization errors gracefully

    3. **Classification**: Process input text
       - Apply Detoxify toxicity detection
       - Convert scores to standardized format
       - Handle empty text and edge cases

    4. **Result Creation**: Return standardized results
       - Map toxicity scores to labels
       - Convert scores to confidence values
       - Include detailed scores in metadata
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "non_toxic",
    ]
    DEFAULT_COST: ClassVar[int] = 2  # Moderate cost for local ML model

    # Default thresholds
    DEFAULT_SEVERE_TOXIC_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_THREAT_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_GENERAL_THRESHOLD: ClassVar[float] = 0.5

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def __init__(
        self,
        config: ClassifierConfig,
    ) -> None:
        """
        Initialize the toxicity classifier implementation.

        Args:
            config: Configuration for the classifier
        """
        self.config = config
        # State is managed by StateManager, no need to initialize here

    def _validate_model(self, model: Any) -> TypeGuard[ToxicityModel]:
        """
        Validate that a model implements the required protocol.

        Args:
            model: The model to validate

        Returns:
            True if the model is valid

        Raises:
            ValueError: If the model doesn't implement the ToxicityModel protocol
        """
        if not isinstance(model, ToxicityModel):
            raise ValueError(f"Model must implement ToxicityModel protocol, got {type(model)}")
        return True

    def _load_detoxify(self) -> ToxicityModel:
        """
        Load the Detoxify package and model.

        Returns:
            Initialized Detoxify model

        Raises:
            ImportError: If the Detoxify package is not installed
            RuntimeError: If Detoxify initialization fails
        """
        try:
            detoxify_module = importlib.import_module("detoxify")
            # Get model name from config params
            model_name = self.config.params.get("model_name", "original")
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

    def _get_thresholds(self) -> Dict[str, float]:
        """
        Get toxicity thresholds from state cache.

        Returns:
            Dictionary of thresholds
        """
        # Get state
        state = self._state_manager.get_state()
        
        # Return thresholds from cache
        return state.cache.get("thresholds", {
            "general_threshold": self.DEFAULT_GENERAL_THRESHOLD,
            "severe_toxic_threshold": self.DEFAULT_SEVERE_TOXIC_THRESHOLD,
            "threat_threshold": self.DEFAULT_THREAT_THRESHOLD,
        })

    def warm_up_impl(self) -> None:
        """
        Initialize the model if needed.

        This method loads the Detoxify model if it hasn't been loaded yet.
        It is called automatically when needed but can also be called
        explicitly to pre-initialize resources.

        Raises:
            RuntimeError: If model initialization fails
        """
        # Get state
        state = self._state_manager.get_state()
        
        # Check if already initialized
        if not state.initialized:
            try:
                # Load model
                state.model = self._load_detoxify()

                # Store thresholds in state cache
                state.cache["thresholds"] = {
                    "general_threshold": self.config.params.get(
                        "general_threshold", self.DEFAULT_GENERAL_THRESHOLD
                    ),
                    "severe_toxic_threshold": self.config.params.get(
                        "severe_toxic_threshold", self.DEFAULT_SEVERE_TOXIC_THRESHOLD
                    ),
                    "threat_threshold": self.config.params.get(
                        "threat_threshold", self.DEFAULT_THREAT_THRESHOLD
                    ),
                }

                # Mark as initialized
                state.initialized = True
            except Exception as e:
                state.error = f"Failed to initialize toxicity classifier: {e}"
                raise RuntimeError(f"Failed to initialize toxicity classifier: {e}")

    def _get_toxicity_label(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Get toxicity label and confidence based on scores.

        This method analyzes the toxicity scores from the model and determines
        the most appropriate label and confidence level based on configured thresholds.

        Args:
            scores: Dictionary of toxicity scores from the model

        Returns:
            Tuple of (toxicity_label, confidence)
        """
        # Get thresholds
        thresholds = self._get_thresholds()

        # Check for severe toxicity or threats first (high priority categories)
        if scores.get("severe_toxic", 0) >= thresholds["severe_toxic_threshold"]:
            return "severe_toxic", scores["severe_toxic"]
        elif scores.get("threat", 0) >= thresholds["threat_threshold"]:
            return "threat", scores["threat"]

        # Get highest toxicity score and category
        max_category = max(scores.items(), key=lambda x: x[1])
        label, confidence = max_category

        # Only return toxic label if it meets the general threshold
        if confidence >= thresholds["general_threshold"]:
            return label, confidence

        # Special case: If all toxicity scores are extremely low (e.g., < 0.01),
        # this is clearly non-toxic content, so return high confidence for non_toxic
        max_toxicity_score = max(scores.values()) if scores else 0.0
        if max_toxicity_score < 0.01:
            # If content is extremely non-toxic, return high confidence (0.95)
            return "non_toxic", 0.95

        # Default case: content is non-toxic but with lower confidence
        return "non_toxic", confidence

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """
        Implement toxicity classification logic.

        This method contains the core toxicity detection logic using the Detoxify model.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity scores
        """
        # Ensure resources are initialized
        if not self._state_manager.get_state().initialized:
            self.warm_up_impl()

        # Get state
        state = self._state_manager.get_state()

        # Handle empty or whitespace-only text
        if not text.strip():
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={
                    "reason": "empty_input",
                    "all_scores": {
                        "toxic": 0.0,
                        "severe_toxic": 0.0,
                        "obscene": 0.0,
                        "threat": 0.0,
                        "insult": 0.0,
                        "identity_hate": 0.0,
                    },
                },
            )

        try:
            # Get toxicity scores from Detoxify
            scores = state.model.predict(text)
            scores = {k: float(v) for k, v in scores.items()}

            # Determine toxicity label and confidence
            label, confidence = self._get_toxicity_label(scores)

            # Return result with detailed metadata
            return ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={"all_scores": scores},
            )
        except Exception as e:
            # Log the error and return a fallback result
            logger.error("Failed to classify text: %s", e)
            state.error = f"Failed to classify text: {e}"
            return ClassificationResult[str](
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "reason": "classification_error",
                },
            )
