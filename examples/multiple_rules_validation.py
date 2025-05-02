from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.rules import (
    RuleConfig,
    create_classifier_rule,
    create_length_rule,
)
from sifaka.rules.base import RulePriority
from sifaka.rules.content.prohibited import create_prohibited_content_rule


# Create a simplified mock profanity classifier
class SimpleProfanityClassifier(BaseClassifier):
    """A simple mock profanity classifier for demonstration purposes."""

    def __init__(
        self,
        name: str = "simple_profanity_classifier",
        description: str = "Simple profanity detection for examples",
        config: ClassifierConfig = None,
    ):
        if config is None:
            config = ClassifierConfig(
                labels=["clean", "profane"],
                params={"profane_words": ["f***", "stupid", "garbage", "scam"]},
            )
        super().__init__(name=name, description=description, config=config)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Simple implementation that checks for bad words."""
        profane_words = self.config.params.get("profane_words", [])

        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Check if any profane words are in the text
        found_words = [word for word in profane_words if word.lower() in text_lower]
        contains_profanity = len(found_words) > 0

        # Calculate confidence based on count of profane words found
        confidence = min(1.0, len(found_words) / 10) if contains_profanity else 0.0

        return ClassificationResult(
            label="profane" if contains_profanity else "clean",
            confidence=confidence if contains_profanity else 1.0 - confidence,
            metadata={
                "contains_profanity": contains_profanity,
                "found_words": found_words,
                "word_count": len(text.split()),
            },
        )


# Create a simplified mock toxicity classifier
class SimpleToxicityClassifier(BaseClassifier):
    """A simple mock toxicity classifier for demonstration purposes."""

    def __init__(
        self,
        name: str = "simple_toxicity_classifier",
        description: str = "Simple toxicity detection for examples",
        config: ClassifierConfig = None,
    ):
        if config is None:
            config = ClassifierConfig(
                labels=["non-toxic", "toxic"],
                params={"toxic_patterns": ["terrible", "useless", "garbage", "stupid", "hate"]},
            )
        super().__init__(name=name, description=description, config=config)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Simple implementation that checks for toxic patterns."""
        toxic_patterns = self.config.params.get("toxic_patterns", [])

        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Check if any toxic patterns are in the text
        found_patterns = [pattern for pattern in toxic_patterns if pattern.lower() in text_lower]
        is_toxic = len(found_patterns) > 0

        # Calculate confidence based on count of toxic patterns found
        confidence = min(1.0, len(found_patterns) / 5) if is_toxic else 0.0

        return ClassificationResult(
            label="toxic" if is_toxic else "non-toxic",
            confidence=confidence if is_toxic else 1.0 - confidence,
            metadata={
                "is_toxic": is_toxic,
                "found_patterns": found_patterns,
                "pattern_count": len(found_patterns),
            },
        )


# Sample text for validation
sample_texts = [
    "I love this product! It's absolutely fantastic and works perfectly.",
    "This is a terrible product. It's completely useless and a waste of money.",
    "F*** this stupid product. It's complete garbage and the company is a scam!",
    "Product works as expected.",  # Very short text
    "This review contains inappropriate content and references to illegal activities like drug trafficking and other prohibited topics that should not be allowed in reviews.",
]

# Set up classifiers
sentiment_config = ClassifierConfig(
    labels=["positive", "neutral", "negative", "unknown"],
    params={"positive_threshold": 0.05, "negative_threshold": -0.05},
)
sentiment_classifier = SentimentClassifier(config=sentiment_config)
profanity_classifier = SimpleProfanityClassifier()
toxicity_classifier = SimpleToxicityClassifier()

# Create rules
length_rule = create_length_rule(
    min_chars=20,
    max_chars=200,
    rule_id="length_validation",
    description="Validates text length",
    config=RuleConfig(params={"priority": RulePriority.MEDIUM}),
)

prohibited_rule = create_prohibited_content_rule(
    name="prohibited_content",
    terms=["illegal", "trafficking", "scam"],
    rule_config={"priority": RulePriority.HIGH},
)

# Create classifier rules
sentiment_rule = create_classifier_rule(
    classifier=sentiment_classifier,
    name="sentiment_validation",
    threshold=0.3,  # Minimum confidence threshold
    valid_labels=["positive", "neutral"],  # Allow positive or neutral sentiment
    rule_config={
        "threshold": 0.3,  # Minimum positive sentiment score required
        "negative_threshold": 0.7,  # Maximum negative sentiment allowed
    },
)

profanity_rule = create_classifier_rule(
    classifier=profanity_classifier,
    name="profanity_validation",
    threshold=0.5,
    valid_labels=["clean"],
    rule_config={"priority": RulePriority.HIGH},
)

toxicity_rule = create_classifier_rule(
    classifier=toxicity_classifier,
    name="toxicity_validation",
    threshold=0.5,
    valid_labels=["non-toxic"],
    rule_config={"priority": RulePriority.CRITICAL},
)

# Create a list of all rules to apply
rules = [length_rule, prohibited_rule, sentiment_rule, profanity_rule, toxicity_rule]


# Function to validate text against all rules
def validate_text(text, rules):
    results = {}
    all_passed = True

    for rule in rules:
        try:
            result = rule.validate(text)
            results[rule.name] = result
            if not result.passed:
                all_passed = False
        except Exception as e:
            # Handle validation errors gracefully
            print(f"Error validating with rule {rule.name}: {str(e)}")
            results[rule.name] = None
            all_passed = False

    return all_passed, results


# Validate each sample text
if __name__ == "__main__":
    print("Testing multiple rules validation example")
    print("----------------------------------------")

    for i, text in enumerate(sample_texts):
        print(f"\nValidating text #{i+1}:")
        print(f"Text: {text}")

        passed, results = validate_text(text, rules)

        print(f"Overall validation passed: {passed}")
        print("Rule results:")
        for rule_name, result in results.items():
            if result is None:
                status = "❌ ERROR"
                print(f"  - {rule_name}: {status}")
                continue

            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"  - {rule_name}: {status}")
            if not result.passed:
                print(f"    Message: {result.message}")
                if result.score is not None:
                    print(f"    Score: {result.score}")
                if result.metadata:
                    print(f"    Details: {result.metadata}")

    print("\nMultiple rules validation example completed.")
