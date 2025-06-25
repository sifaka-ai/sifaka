"""Advanced confidence calculation with critic-specific strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class CriticConfidenceStrategy(ABC):
    """Base class for critic-specific confidence calculation strategies."""

    @abstractmethod
    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence score for this specific critic type."""
        pass


class ReflexionConfidenceStrategy(CriticConfidenceStrategy):
    """Confidence calculation for Reflexion critic."""

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate based on reflection depth and history usage."""
        base = 0.7

        # Check if previous attempts were referenced
        history_references = ["previous", "earlier", "last attempt", "before", "tried"]
        history_score = sum(
            1 for ref in history_references if ref in feedback.lower()
        ) / len(history_references)
        base += history_score * 0.15

        # Check for learning indicators
        learning_indicators = [
            "learned",
            "realized",
            "understood",
            "noticed",
            "discovered",
        ]
        learning_score = sum(
            1 for ind in learning_indicators if ind in feedback.lower()
        ) / len(learning_indicators)
        base += learning_score * 0.1

        # Penalize if no improvements suggested despite history
        if metadata and metadata.get("iteration", 0) > 2 and len(suggestions) == 0:
            base -= 0.15

        return max(0.0, min(1.0, base))


class ConstitutionalConfidenceStrategy(CriticConfidenceStrategy):
    """Confidence calculation for Constitutional critic."""

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate based on principle coverage and clarity."""
        base = 0.75  # Higher base for principle-based evaluation

        # Check principle mentions
        principles = [
            "safety",
            "accuracy",
            "clarity",
            "respect",
            "helpfulness",
            "balance",
        ]
        principle_coverage = sum(1 for p in principles if p in feedback.lower()) / len(
            principles
        )
        base += principle_coverage * 0.15

        # Check for balanced evaluation (not just criticism)
        positive_indicators = ["good", "well", "effective", "strong", "clear"]
        has_positives = any(ind in feedback.lower() for ind in positive_indicators)
        if has_positives and len(suggestions) > 0:
            base += 0.05  # Balanced feedback

        # High confidence if clear principle violations found
        violation_indicators = [
            "violates",
            "contradicts",
            "unsafe",
            "harmful",
            "misleading",
        ]
        if any(ind in feedback.lower() for ind in violation_indicators):
            base = max(base, 0.9)

        return max(0.0, min(1.0, base))


class SelfConsistencyConfidenceStrategy(CriticConfidenceStrategy):
    """Confidence calculation for Self-Consistency critic."""

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate based on consensus strength and variance."""
        if not metadata:
            return 0.7

        # Use variance from metadata if available
        variance = metadata.get("variance", 0.5)
        consensus_strength = metadata.get("consensus_strength", 0.5)

        # Low variance = high confidence
        base = 0.5 + (1 - variance) * 0.3

        # Strong consensus = high confidence
        base += consensus_strength * 0.2

        # Adjust for number of evaluations
        num_evaluations = metadata.get("num_evaluations", 3)
        if num_evaluations > 3:
            base += 0.05  # More evaluations = more confidence

        return max(0.0, min(1.0, base))


class MetaRewardingConfidenceStrategy(CriticConfidenceStrategy):
    """Confidence calculation for Meta-Rewarding critic."""

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate based on meta-evaluation quality."""
        if not metadata:
            return 0.75

        # Use meta-evaluation score directly if available
        meta_score = metadata.get("meta_evaluation_score", 0.75)

        # Adjust based on self-critique depth
        self_critique_indicators = [
            "my analysis",
            "my evaluation",
            "i should",
            "could have",
        ]
        self_critique_depth = sum(
            1 for ind in self_critique_indicators if ind in feedback.lower()
        )

        # Higher self-awareness = higher confidence
        base = meta_score + (self_critique_depth * 0.05)

        return max(0.0, min(1.0, base))


class NCriticsConfidenceStrategy(CriticConfidenceStrategy):
    """Confidence calculation for N-Critics ensemble."""

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate based on perspective coverage and synthesis quality."""
        base = 0.8  # Higher base for ensemble approach

        # Check perspective mentions
        perspective_keywords = [
            "perspective",
            "viewpoint",
            "considers",
            "from the standpoint",
        ]
        perspective_score = sum(
            1 for kw in perspective_keywords if kw in feedback.lower()
        )
        base += min(perspective_score * 0.05, 0.15)

        # Check for synthesis/integration
        synthesis_keywords = [
            "overall",
            "synthesizing",
            "across all",
            "considering all",
            "balanced",
        ]
        synthesis_score = sum(1 for kw in synthesis_keywords if kw in feedback.lower())
        if synthesis_score > 0:
            base += 0.05

        return max(0.0, min(1.0, base))


class DefaultConfidenceStrategy(CriticConfidenceStrategy):
    """Default confidence calculation (original implementation)."""

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Default confidence calculation."""
        confidence = 0.7

        # Adjust based on response length
        if response_length > 500:
            confidence += 0.1
        elif response_length < 100:
            confidence -= 0.1

        # Adjust based on specificity
        confidence += self._score_specificity(feedback) * 0.15

        # Adjust based on uncertainty
        confidence -= self._score_uncertainty(feedback) * 0.15

        # Adjust based on suggestions
        if len(suggestions) > 3:
            confidence += 0.05
        elif len(suggestions) == 0:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _score_specificity(self, text: str) -> float:
        """Score how specific the feedback is."""
        text_lower = text.lower()
        specificity_indicators = [
            "specifically",
            "particularly",
            "exactly",
            "precisely",
            "clearly",
            "definitely",
            "for example",
            "such as",
        ]
        count = sum(1 for ind in specificity_indicators if ind in text_lower)
        return min(count / 3.0, 1.0)

    def _score_uncertainty(self, text: str) -> float:
        """Score how uncertain the feedback is."""
        text_lower = text.lower()
        uncertainty_indicators = [
            "might",
            "maybe",
            "perhaps",
            "possibly",
            "could be",
            "seems",
            "appears",
            "somewhat",
            "unclear if",
        ]
        count = sum(1 for ind in uncertainty_indicators if ind in text_lower)
        return min(count / 3.0, 1.0)


class AdvancedConfidenceCalculator:
    """Advanced confidence calculator with critic-specific strategies."""

    def __init__(self):
        """Initialize with strategy mappings."""
        self.strategies: Dict[str, CriticConfidenceStrategy] = {
            "reflexion": ReflexionConfidenceStrategy(),
            "constitutional": ConstitutionalConfidenceStrategy(),
            "self_consistency": SelfConsistencyConfidenceStrategy(),
            "meta_rewarding": MetaRewardingConfidenceStrategy(),
            "n_critics": NCriticsConfidenceStrategy(),
        }
        self.default_strategy = DefaultConfidenceStrategy()

    def calculate(
        self,
        critic_name: str,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence using critic-specific strategy."""
        strategy = self.strategies.get(critic_name, self.default_strategy)
        return strategy.calculate(feedback, suggestions, response_length, metadata)

    def register_strategy(
        self, critic_name: str, strategy: CriticConfidenceStrategy
    ) -> None:
        """Register a custom strategy for a critic."""
        self.strategies[critic_name] = strategy
