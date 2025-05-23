"""Base classes and protocols for Sifaka classifiers.

This module provides the foundation for all text classifiers in the Sifaka framework.
It defines the common interfaces, result types, and base implementations that all
classifiers should follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sifaka.utils.error_handling import SifakaError
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ClassifierError(SifakaError):
    """Error raised by classifier components."""
    pass


class ClassificationResult(BaseModel):
    """Result of a text classification operation.
    
    This class represents the result of classifying a piece of text,
    including the predicted label, confidence score, and additional metadata.
    
    Attributes:
        label: The predicted class label
        confidence: Confidence score between 0.0 and 1.0
        metadata: Additional information about the classification
    """
    
    label: str = Field(..., description="The predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional classification metadata")
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class TextClassifier(ABC):
    """Abstract base class for text classifiers.
    
    This class defines the interface that all text classifiers must implement.
    It provides both a high-level classify() method and scikit-learn compatible
    predict() and predict_proba() methods for integration with validators.
    
    Attributes:
        name: The name of the classifier
        description: A description of what the classifier does
    """
    
    def __init__(self, name: str = "TextClassifier", description: str = ""):
        """Initialize the classifier.
        
        Args:
            name: The name of the classifier
            description: A description of what the classifier does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        """Classify a single text.
        
        Args:
            text: The text to classify
            
        Returns:
            A ClassificationResult with the prediction
            
        Raises:
            ClassifierError: If classification fails
        """
        pass
    
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts.
        
        This default implementation calls classify() for each text.
        Subclasses can override this for more efficient batch processing.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of ClassificationResults
            
        Raises:
            ClassifierError: If classification fails
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify text: {e}")
                # Create a failed result
                results.append(ClassificationResult(
                    label="error",
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))
        return results
    
    # Scikit-learn compatible interface for validator integration
    
    def predict(self, X: List[str]) -> List[str]:
        """Predict class labels for samples (scikit-learn interface).
        
        Args:
            X: List of text samples to classify
            
        Returns:
            List of predicted class labels
        """
        results = self.batch_classify(X)
        return [result.label for result in results]
    
    def predict_proba(self, X: List[str]) -> List[List[float]]:
        """Predict class probabilities for samples (scikit-learn interface).
        
        Args:
            X: List of text samples to classify
            
        Returns:
            List of probability arrays for each sample
            
        Note:
            This default implementation returns confidence as probability for the
            predicted class and (1-confidence) for the other class. Subclasses
            should override this for multi-class scenarios.
        """
        results = self.batch_classify(X)
        probabilities = []
        
        for result in results:
            # Simple binary probability: [other_class_prob, predicted_class_prob]
            # This works for most binary classifiers
            other_prob = 1.0 - result.confidence
            predicted_prob = result.confidence
            probabilities.append([other_prob, predicted_prob])
            
        return probabilities
    
    def get_classes(self) -> List[str]:
        """Get the list of possible class labels.
        
        Returns:
            List of possible class labels
            
        Note:
            Subclasses should override this to return their specific classes.
        """
        return ["negative", "positive"]  # Default binary classes
    
    def __str__(self) -> str:
        """String representation of the classifier."""
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the classifier."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"
