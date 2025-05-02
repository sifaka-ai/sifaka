"""
Template for standardized Sifaka classifier implementation.

This template provides the standard structure that all classifier implementations
should follow to ensure consistency across the Sifaka framework.

Usage Example:
    from sifaka.classifiers.my_domain import MyDomainClassifier

    # Create classifier
    classifier = MyDomainClassifier()

    # Classify text
    result = classifier.classify("This is a test.")
    print(f"Label: {result.label}, Confidence: {result.confidence}")
"""

from typing import Any, Dict, List, Optional, Union, ClassVar

from pydantic import BaseModel, Field, ConfigDict, field_validator

from sifaka.classifiers.base import ClassificationResult


class ClassifierNameConfig(BaseModel):
    """
    Configuration for classifier_name.

    All configuration parameters should be defined here
    with proper typing, validation, and documentation.

    Attributes:
        param1: Description of parameter 1
        param2: Description of parameter 2
        labels: Valid classification labels (required)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param1: type = Field(
        default=default_value,
        description="Description of parameter 1",
        # Add validation constraints as needed
    )

    param2: type = Field(
        default=default_value,
        description="Description of parameter 2",
        # Add validation constraints as needed
    )

    labels: List[str] = Field(
        ...,  # Required field
        description="Valid classification labels",
        min_length=1,
    )

    # Add field validators if needed
    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: List[str]) -> List[str]:
        """Validate labels are non-empty and unique."""
        if not v:
            raise ValueError("At least one label must be provided")
        if len(v) != len(set(v)):
            raise ValueError("Labels must be unique")
        return v


class ClassifierName:
    """
    Classifier for classifier_name.

    This classifier implements the specific classification logic
    for this classifier type.

    Lifecycle:
    1. Initialization: Set up with configuration
    2. Classification: Process input and apply classification logic
    3. Result: Return standardized classification results

    Examples:
        ```python
        from sifaka.classifiers.domain import ClassifierName

        classifier = ClassifierName(
            param1=value1,
            param2=value2
        )

        result = classifier.classify("Text to classify")
        print(f"Label: {result.label}, Confidence: {result.confidence}")
        ```
    """

    # Class-level defaults
    DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {
        "param1": default_value,
        "param2": default_value,
        "labels": ["label1", "label2", "label3"],
    }

    def __init__(
        self,
        param1: type = None,
        param2: type = None,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize classifier with configuration.

        Args:
            param1: Parameter 1
            param2: Parameter 2
            labels: Valid classification labels
            **kwargs: Additional configuration parameters
        """
        # Apply defaults for None values
        config = self.DEFAULT_CONFIG.copy()
        if param1 is not None:
            config["param1"] = param1
        if param2 is not None:
            config["param2"] = param2
        if labels is not None:
            config["labels"] = labels

        # Update with any additional kwargs
        for key, value in kwargs.items():
            if key in config:
                config[key] = value

        # Store configuration
        self._config = ClassifierNameConfig(**config)

        # Initialize any resources needed
        self._initialize_resources()

    def _initialize_resources(self) -> None:
        """
        Initialize any resources needed by the classifier.

        This might include loading models, setting up connections, etc.
        """
        # Implement resource initialization here
        pass

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return "classifier_name"

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return "Classifies text based on classifier_name criteria"

    @property
    def config(self) -> ClassifierNameConfig:
        """Get the classifier configuration."""
        return self._config

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before classification.

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        # Implement text preprocessing here
        return text

    def _classify_internal(self, text: str) -> Dict[str, Any]:
        """
        Internal method to implement classification logic.

        Args:
            text: Preprocessed text to classify

        Returns:
            Classification details including label and confidence

        Raises:
            Exception: Any classification-specific exceptions
        """
        # Implement classification logic here
        # For example:
        label = "label1"  # Determine label based on text
        confidence = 0.9  # Calculate confidence score

        return {
            "label": label,
            "confidence": confidence,
            "raw_scores": {
                "label1": 0.9,
                "label2": 0.05,
                "label3": 0.05,
            },
        }

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text.

        Args:
            text: Text to classify

        Returns:
            Classification result with label, confidence, and metadata

        Raises:
            ValueError: If text is empty or None
        """
        # Validate input
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            # Preprocess text
            preprocessed_text = self._preprocess_text(text)

            # Classify text
            result = self._classify_internal(preprocessed_text)

            # Create and return standardized result
            return ClassificationResult(
                label=result["label"],
                confidence=result["confidence"],
                metadata={
                    "raw_scores": result.get("raw_scores", {}),
                    "text_length": len(text),
                    "preprocessed_length": len(preprocessed_text),
                },
            )
        except Exception as e:
            # Handle exceptions by returning a low-confidence result
            return ClassificationResult(
                label=self._config.labels[0],  # Default to first label
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "text_length": len(text),
                },
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]


# Export public components
__all__ = [
    # Config classes
    "ClassifierNameConfig",
    # Classifier classes
    "ClassifierName",
]