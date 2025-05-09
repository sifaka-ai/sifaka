"""
Simplified Adapter Pattern Example

This example demonstrates the adapter pattern by showing how to:
1. Convert classifiers to rules using an adapter
2. Compose multiple rules with logical operations
"""

from typing import Dict, List, Optional, Protocol


# --- Core Interfaces ---


class RuleResult:
    """Result of a rule validation."""

    def __init__(self, passed: bool, message: str, metadata=None):
        self.passed = passed
        self.message = message
        self.metadata = metadata or {}


class Rule(Protocol):
    """Protocol for rule components."""

    @property
    def name(self) -> str: ...

    def validate(self, text: str) -> RuleResult: ...


class Classifier(Protocol):
    """Protocol for classifier components."""

    @property
    def name(self) -> str: ...

    def classify(self, text: str) -> Dict[str, float]: ...


# --- Simple Implementations ---


class ToxicityClassifier:
    """Simple classifier that detects toxic content."""

    @property
    def name(self) -> str:
        return "toxicity_classifier"

    def classify(self, text: str) -> Dict[str, float]:
        """Classify text as toxic or non-toxic."""
        lower_text = text.lower()
        toxic_words = ["stupid", "hate", "worst", "terrible"]

        # Count toxic words and calculate score
        toxic_count = sum(1 for word in toxic_words if word in lower_text)
        toxicity = min(1.0, toxic_count / 2)

        return {"toxic": toxicity, "non_toxic": 1.0 - toxicity}


class LengthRule:
    """Rule that validates text length."""

    def __init__(self, min_chars: int = 0, max_chars: Optional[int] = None):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._name = "length_rule"

    @property
    def name(self) -> str:
        return self._name

    def validate(self, text: str) -> RuleResult:
        """Validate text length."""
        length = len(text)

        # Check minimum length
        if length < self.min_chars:
            return RuleResult(
                passed=False,
                message=f"Text is too short ({length} chars, minimum {self.min_chars})",
            )

        # Check maximum length if specified
        if self.max_chars is not None and length > self.max_chars:
            return RuleResult(
                passed=False, message=f"Text is too long ({length} chars, maximum {self.max_chars})"
            )

        return RuleResult(passed=True, message=f"Text length is valid ({length} chars)")


# --- Adapters ---


class ClassifierAdapter:
    """Adapter that converts classifiers to rules."""

    def __init__(
        self, classifier: Classifier, threshold: float = 0.5, valid_labels: List[str] = None
    ):
        self.classifier = classifier
        self.threshold = threshold
        self.valid_labels = valid_labels or []
        self._name = f"{classifier.name}_rule"

    @property
    def name(self) -> str:
        return self._name

    def validate(self, text: str) -> RuleResult:
        """Validate text using the classifier."""
        # Handle empty text
        if not text or not text.strip():
            return RuleResult(passed=False, message="Text is empty")

        # Classify the text
        classification = self.classifier.classify(text)

        # Find the highest scoring valid label
        valid_scores = {
            label: score for label, score in classification.items() if label in self.valid_labels
        }

        if not valid_scores:
            return RuleResult(
                passed=False, message=f"No valid labels found in classification result"
            )

        best_label = max(valid_scores.items(), key=lambda x: x[1])
        label, confidence = best_label

        # Check if confidence exceeds threshold
        passed = confidence >= self.threshold

        if passed:
            message = f"Text is classified as {label} with confidence {confidence:.2f}"
        else:
            message = f"Text failed to meet threshold for {label} (got {confidence:.2f}, needed {self.threshold})"

        return RuleResult(passed=passed, message=message)


class CompositeRule:
    """Composes multiple rules with logical operations."""

    def __init__(self, rules: List[Rule], operator: str = "AND", name: str = "composite_rule"):
        self.rules = rules
        self.operator = operator.upper()
        self._name = name

        if self.operator not in ["AND", "OR"]:
            raise ValueError(f"Operator must be 'AND' or 'OR', got {operator}")

    @property
    def name(self) -> str:
        return self._name

    def validate(self, text: str) -> RuleResult:
        """Run all rules and combine results according to the operator."""
        results = [rule.validate(text) for rule in self.rules]

        if not results:
            return RuleResult(passed=True, message="No rules to validate with")

        if self.operator == "AND":
            passed = all(r.passed for r in results)
        else:  # OR
            passed = any(r.passed for r in results)

        # Collect messages from failed rules
        messages = [r.message for r in results if not r.passed]
        if not messages and passed:
            messages = ["All validations passed"]

        return RuleResult(
            passed=passed, message="; ".join(messages) if messages else "Validation passed"
        )


# --- Example Usage ---


def main():
    """Demonstrate the adapter pattern."""
    print("=== Adapter Pattern Example ===\n")

    # Create components
    toxicity_classifier = ToxicityClassifier()
    length_rule = LengthRule(min_chars=10, max_chars=100)

    # Create adapter to convert classifier to rule
    toxicity_rule = ClassifierAdapter(
        classifier=toxicity_classifier, threshold=0.5, valid_labels=["non_toxic"]
    )

    # Create composite rule
    content_rule = CompositeRule(
        rules=[length_rule, toxicity_rule], operator="AND", name="content_rule"
    )

    # Test texts
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst thing I've ever seen.",
        "You are stupid and I hate you.",
        "OK",
    ]

    # Validate texts
    for text in test_texts:
        print(f"\nAnalyzing: '{text}'")

        # Validate with individual rules
        length_result = length_rule.validate(text)
        toxicity_result = toxicity_rule.validate(text)

        print(f"Length rule: {'✓' if length_result.passed else '✗'} - {length_result.message}")
        print(
            f"Toxicity rule: {'✓' if toxicity_result.passed else '✗'} - {toxicity_result.message}"
        )

        # Validate with composite rule
        composite_result = content_rule.validate(text)
        print(
            f"Composite rule: {'✓' if composite_result.passed else '✗'} - {composite_result.message}"
        )


if __name__ == "__main__":
    main()
