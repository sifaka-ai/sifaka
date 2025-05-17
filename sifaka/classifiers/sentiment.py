"""
Sentiment classifier for Sifaka.

This module provides a simple sentiment classifier that categorizes text as
positive, negative, or neutral based on keyword matching.
"""

import re
from typing import List, Dict, Any, Optional

from sifaka.classifiers import ClassificationResult


class SentimentClassifier:
    """
    A simple sentiment classifier that categorizes text as positive, negative, or neutral.

    This classifier uses a simple keyword-based approach to classify text sentiment.
    It's intended as a basic example and for testing purposes, not for production use.
    For production, consider using a more sophisticated sentiment analysis model.

    Attributes:
        positive_words: List of words that indicate positive sentiment.
        negative_words: List of words that indicate negative sentiment.
        name: The name of the classifier.
        description: The description of the classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
        name: str = "sentiment_classifier",
        description: str = "Classifies text sentiment as positive, negative, or neutral",
    ):
        """
        Initialize the sentiment classifier.

        Args:
            positive_words: Optional list of words that indicate positive sentiment.
            negative_words: Optional list of words that indicate negative sentiment.
            name: The name of the classifier.
            description: The description of the classifier.
        """
        self._name = name
        self._description = description
        
        # Default positive and negative word lists
        self._positive_words = positive_words or [
            "good", "great", "excellent", "wonderful", "amazing", "fantastic",
            "terrific", "outstanding", "superb", "brilliant", "awesome",
            "happy", "joy", "love", "like", "enjoy", "pleased", "satisfied",
            "positive", "beautiful", "perfect", "best", "better", "success",
            "recommend", "impressive", "delighted", "excited", "thrilled"
        ]
        
        self._negative_words = negative_words or [
            "bad", "terrible", "horrible", "awful", "poor", "disappointing",
            "frustrating", "annoying", "irritating", "unpleasant", "negative",
            "hate", "dislike", "angry", "sad", "unhappy", "dissatisfied",
            "worst", "worse", "failure", "problem", "issue", "trouble",
            "difficult", "hard", "impossible", "ugly", "disgusting", "broken"
        ]

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text sentiment as positive, negative, or neutral.

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the sentiment label and confidence score.
        """
        if not text:
            return ClassificationResult(
                label="neutral",
                confidence=1.0,
                metadata={"input_length": 0, "reason": "empty_text"}
            )

        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self._positive_words if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
        negative_count = sum(1 for word in self._negative_words if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
        
        # Calculate total sentiment words found
        total_count = positive_count + negative_count
        
        # Determine sentiment based on counts
        if total_count == 0:
            return ClassificationResult(
                label="neutral",
                confidence=0.7,  # Moderate confidence for neutral
                metadata={
                    "input_length": len(text),
                    "positive_count": 0,
                    "negative_count": 0,
                    "reason": "no_sentiment_words"
                }
            )
        
        # Calculate sentiment score (-1.0 to 1.0)
        sentiment_score = (positive_count - negative_count) / total_count
        
        # Determine label based on sentiment score
        if sentiment_score > 0.2:
            label = "positive"
            # Scale confidence from 0.6 to 1.0 based on sentiment score
            confidence = 0.6 + (sentiment_score * 0.4)
        elif sentiment_score < -0.2:
            label = "negative"
            # Scale confidence from 0.6 to 1.0 based on absolute sentiment score
            confidence = 0.6 + (abs(sentiment_score) * 0.4)
        else:
            label = "neutral"
            # Scale confidence from 0.5 to 0.7 based on how close to zero
            confidence = 0.5 + (0.2 * (1 - abs(sentiment_score) / 0.2))
        
        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "input_length": len(text),
                "sentiment_score": sentiment_score,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "total_sentiment_words": total_count
            }
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: The list of texts to classify.

        Returns:
            A list of ClassificationResults.
        """
        return [self.classify(text) for text in texts]
