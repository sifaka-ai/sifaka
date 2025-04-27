"""
Toxicity classifier using the Detoxify model.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import importlib
import logging

from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Only import type hints during type checking
if TYPE_CHECKING:
    from detoxify import Detoxify


class ToxicityClassifier(Classifier):
    """
    A lightweight toxicity classifier using the Detoxify model.

    This provides a fast, local alternative to API-based toxicity detection.
    Requires the 'toxicity' extra to be installed:
    pip install sifaka[toxicity]

    Attributes:
        model_name: The Detoxify model to use
        labels: The toxicity categories to check
    """

    model_name: str = "original"

    def __init__(
        self,
        name: str = "toxicity_classifier",
        description: str = "Detects toxic content using Detoxify",
        model_name: str = "original",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the toxicity classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            model_name: The Detoxify model to use ('original', 'unbiased', or 'multilingual')
            config: Additional configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
            cost=2,  # Moderate cost for local ML model
            **kwargs,
        )
        self.model_name = model_name
        self._model = None
        self._detoxify = None

    def _load_detoxify(self) -> None:
        """Load the Detoxify package."""
        try:
            detoxify_module = importlib.import_module("detoxify")
            self._detoxify = detoxify_module.Detoxify
        except ImportError:
            raise ImportError(
                "Detoxify package is required for ToxicityClassifier. "
                "Install it with: pip install sifaka[toxicity]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Detoxify: {e}")

    def warm_up(self) -> None:
        """Load the model if not already loaded."""
        if self._model is None:
            if self._detoxify is None:
                self._load_detoxify()
            try:
                logger.info("Loading Detoxify model %s...", self.model_name)
                self._model = self._detoxify(model_type=self.model_name)
                logger.info("Detoxify model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Detoxify model: {e}")

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text toxicity.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity scores
        """
        self.warm_up()
        try:
            scores = self._model.predict(text)

            # Get highest toxicity score and category
            max_category = max(scores.items(), key=lambda x: x[1])
            label, confidence = max_category

            return ClassificationResult(
                label=label,
                confidence=float(confidence),
                metadata={"all_scores": {k: float(v) for k, v in scores.items()}},
            )
        except Exception as e:
            logger.error("Failed to classify text: %s", e)
            # Return a safe default
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.warm_up()
        try:
            batch_scores = self._model.predict(texts)

            results = []
            for i in range(len(texts)):
                scores = {k: float(v[i]) for k, v in batch_scores.items()}
                max_category = max(scores.items(), key=lambda x: x[1])
                label, confidence = max_category

                results.append(
                    ClassificationResult(
                        label=label,
                        confidence=confidence,
                        metadata={"all_scores": scores},
                    )
                )

            return results
        except Exception as e:
            logger.error("Failed to batch classify texts: %s", e)
            # Return safe defaults
            return [
                ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={"error": str(e)},
                )
                for _ in texts
            ]
