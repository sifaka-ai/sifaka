"""Spam classifier for detecting spam content in text.

This module provides a classifier for detecting spam content using
machine learning with fallback to rule-based detection.
"""

from typing import List, Optional

from sifaka.classifiers.base import ClassificationResult, ClassifierError, TextClassifier
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Configure logger
logger = get_logger(__name__)

# Sample spam text for training
SPAM_SAMPLES = [
    "URGENT! You have won $1,000,000! Click here now!",
    "FREE MONEY! No strings attached! Act now!",
    "Congratulations! You are our lucky winner!",
    "CLICK HERE FOR AMAZING DEALS! LIMITED TIME ONLY!",
    "Make money fast! Work from home! No experience needed!",
    "LOSE WEIGHT FAST! Miracle pill! Doctors hate this trick!",
    "Hot singles in your area! Meet them tonight!",
    "Your account will be suspended! Verify now!",
    "FINAL NOTICE: Your warranty is about to expire!",
    "Get rich quick! This one simple trick!",
]

HAM_SAMPLES = [
    "Hi, how are you doing today?",
    "The meeting is scheduled for 3 PM tomorrow.",
    "Thanks for your help with the project.",
    "Can you please review this document?",
    "I'll be working from home tomorrow.",
    "The weather is nice today, isn't it?",
    "Let's grab lunch sometime this week.",
    "The report is due by Friday.",
    "Happy birthday! Hope you have a great day.",
    "The presentation went well yesterday.",
]

# Spam indicators for rule-based detection
SPAM_INDICATORS = {
    "urgent", "free", "money", "winner", "congratulations", "click here",
    "act now", "limited time", "amazing deals", "work from home",
    "lose weight", "miracle", "hot singles", "verify now", "final notice",
    "get rich", "no experience", "doctors hate", "one simple trick",
    "suspended", "warranty", "expires", "claim", "prize", "lottery"
}

SPAM_PATTERNS = [
    "!!!", "URGENT", "FREE", "CLICK HERE", "ACT NOW", "LIMITED TIME",
    "$$$", "100% FREE", "NO COST", "RISK FREE", "GUARANTEED"
]


class SpamClassifier(TextClassifier):
    """Classifier for detecting spam content in text.
    
    This classifier uses machine learning when scikit-learn is available,
    with fallback to rule-based spam detection. It identifies common
    spam patterns and indicators.
    
    Attributes:
        threshold: Confidence threshold for spam detection
        model: The trained classification model
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        name: str = "SpamClassifier",
        description: str = "Detects spam content in text",
    ):
        """Initialize the spam classifier.
        
        Args:
            threshold: Confidence threshold for spam detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.threshold = threshold
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize and train the spam detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            
            # Prepare training data
            X = SPAM_SAMPLES + HAM_SAMPLES
            y = [1] * len(SPAM_SAMPLES) + [0] * len(HAM_SAMPLES)
            
            # Create and train the model
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            # Train the model
            self.model.fit(X, y)
            
            logger.debug(f"Initialized spam classifier with {len(X)} training samples")
            
        except ImportError:
            logger.warning(
                "scikit-learn not available. SpamClassifier will use rule-based detection. "
                "Install scikit-learn for better accuracy: pip install scikit-learn"
            )
            self.model = None
    
    def classify(self, text: str) -> ClassificationResult:
        """Classify text for spam.
        
        Args:
            text: The text to classify
            
        Returns:
            ClassificationResult with spam prediction
            
        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label="ham",
                confidence=0.5,
                metadata={"reason": "empty_text", "input_length": 0}
            )
        
        try:
            if self.model is not None:
                return self._classify_with_ml(text)
            else:
                return self._classify_with_rules(text)
                
        except Exception as e:
            logger.error(f"Spam classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for spam: {str(e)}",
                component="SpamClassifier",
                operation="classification"
            )
    
    def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using machine learning model."""
        # Get prediction probabilities
        probabilities = self.model.predict_proba([text])[0]
        
        # Get the predicted class (0 = ham, 1 = spam)
        predicted_class = self.model.predict([text])[0]
        confidence = float(probabilities[predicted_class])
        
        # Map class to label
        label = "spam" if predicted_class == 1 else "ham"
        
        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "machine_learning",
                "spam_probability": float(probabilities[1]),
                "ham_probability": float(probabilities[0]),
                "input_length": len(text),
            }
        )
    
    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using rule-based approach."""
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Count spam indicators
        indicator_count = sum(1 for indicator in SPAM_INDICATORS if indicator in text_lower)
        
        # Count spam patterns
        pattern_count = sum(1 for pattern in SPAM_PATTERNS if pattern in text_upper)
        
        # Check for excessive capitalization
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        else:
            caps_ratio = 0
        
        # Check for excessive punctuation
        exclamation_count = text.count('!')
        dollar_count = text.count('$')
        
        # Calculate spam score
        spam_score = 0
        spam_score += indicator_count * 0.2
        spam_score += pattern_count * 0.3
        spam_score += caps_ratio * 0.5 if caps_ratio > 0.3 else 0
        spam_score += min(exclamation_count * 0.1, 0.3)
        spam_score += min(dollar_count * 0.15, 0.3)
        
        # Determine label and confidence
        if spam_score > 0.5:
            label = "spam"
            confidence = min(0.6 + spam_score * 0.3, 0.9)
        else:
            label = "ham"
            confidence = max(0.6, 0.9 - spam_score)
        
        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "spam_score": spam_score,
                "indicator_count": indicator_count,
                "pattern_count": pattern_count,
                "caps_ratio": caps_ratio,
                "exclamation_count": exclamation_count,
                "dollar_count": dollar_count,
                "input_length": len(text),
            }
        )
    
    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["ham", "spam"]


def create_spam_validator(
    threshold: float = 0.7,
    name: str = "SpamValidator"
) -> ClassifierValidator:
    """Create a validator that detects spam in text.
    
    Args:
        threshold: Confidence threshold for spam detection
        name: Name of the validator
        
    Returns:
        A ClassifierValidator configured for spam detection
    """
    classifier = SpamClassifier(threshold=threshold)
    
    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        invalid_labels=["spam"],  # Spam text is invalid
        name=name
    )
