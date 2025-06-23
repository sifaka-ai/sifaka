"""Confidence calculation for critic assessments."""

from typing import List, Optional


class ConfidenceCalculator:
    """Calculates confidence scores for critic assessments."""
    
    def __init__(self, base_confidence: float = 0.7):
        """Initialize calculator with base confidence."""
        self.base_confidence = base_confidence
    
    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[dict] = None
    ) -> float:
        """Calculate confidence score based on multiple factors.
        
        Args:
            feedback: The feedback text
            suggestions: List of suggestions made
            response_length: Length of full response
            metadata: Additional metadata
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = self.base_confidence
        
        # Adjust based on response length
        if response_length > 500:
            confidence += 0.1  # Thorough response
        elif response_length < 100:
            confidence -= 0.1  # Too brief
        
        # Adjust based on specificity
        confidence += self._score_specificity(feedback) * 0.15
        
        # Adjust based on uncertainty
        confidence -= self._score_uncertainty(feedback) * 0.15
        
        # Adjust based on number of suggestions
        if len(suggestions) > 3:
            confidence += 0.05  # Many specific suggestions
        elif len(suggestions) == 0:
            confidence -= 0.1   # No actionable suggestions
        
        # Ensure valid range
        return max(0.0, min(1.0, confidence))
    
    def _score_specificity(self, text: str) -> float:
        """Score how specific the feedback is (0.0-1.0)."""
        text_lower = text.lower()
        
        specificity_indicators = [
            "specifically", "particularly", "exactly", "precisely",
            "clearly", "definitely", "certainly", "for example",
            "such as", "including", "namely", "in particular"
        ]
        
        count = sum(1 for ind in specificity_indicators if ind in text_lower)
        return min(count / 3.0, 1.0)  # Cap at 1.0
    
    def _score_uncertainty(self, text: str) -> float:
        """Score how uncertain the feedback is (0.0-1.0)."""
        text_lower = text.lower()
        
        uncertainty_indicators = [
            "might", "maybe", "perhaps", "possibly", "could be",
            "seems", "appears", "somewhat", "relatively", "fairly",
            "probably", "potentially", "unclear if", "not sure"
        ]
        
        count = sum(1 for ind in uncertainty_indicators if ind in text_lower)
        return min(count / 3.0, 1.0)  # Cap at 1.0