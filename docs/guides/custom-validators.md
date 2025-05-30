# Custom Validators Guide

Learn how to create custom validators for Sifaka to implement domain-specific validation logic and quality assurance rules.

## Overview

Validators in Sifaka check if generated text meets specific criteria. They return detailed `ValidationResult` objects with pass/fail status, scores, issues, and improvement suggestions.

## Validator Protocol

All validators must implement the `Validator` protocol:

```python
from typing import Protocol
from sifaka.core.thought import Thought, ValidationResult

class Validator(Protocol):
    def validate(self, thought: Thought) -> ValidationResult:
        """Validate text against specific criteria.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with validation results.
        """
        ...
```

## ValidationResult Structure

Validators return `ValidationResult` objects with comprehensive information:

```python
from sifaka.core.thought import ValidationResult

result = ValidationResult(
    passed=True,           # Whether validation passed
    message="Text is valid",  # Human-readable message
    score=0.95,           # Numeric score (0.0-1.0)
    issues=[],            # List of problems found
    suggestions=[],       # List of improvement suggestions
    metadata={}           # Additional validator-specific data
)
```

## Quick Start: Simple Custom Validator

Here's a minimal custom validator:

```python
from sifaka.core.thought import Thought, ValidationResult

class WordCountValidator:
    """Validates text has specific word count."""

    def __init__(self, min_words: int = 0, max_words: int = 1000):
        self.min_words = min_words
        self.max_words = max_words
        self.name = f"WordCount({min_words}-{max_words})"

    def validate(self, thought: Thought) -> ValidationResult:
        """Validate word count."""
        if not thought.text:
            return ValidationResult(
                passed=False,
                message="No text to validate",
                score=0.0,
                issues=["Text is empty"],
                suggestions=["Generate text first"]
            )

        word_count = len(thought.text.split())

        # Check constraints
        issues = []
        suggestions = []

        if word_count < self.min_words:
            issues.append(f"Text has {word_count} words, minimum is {self.min_words}")
            suggestions.append(f"Add {self.min_words - word_count} more words")

        if word_count > self.max_words:
            issues.append(f"Text has {word_count} words, maximum is {self.max_words}")
            suggestions.append(f"Remove {word_count - self.max_words} words")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.5)

        return ValidationResult(
            passed=passed,
            message=f"Word count: {word_count}" if passed else "Word count validation failed",
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={"word_count": word_count, "validator": self.name}
        )

# Usage
validator = WordCountValidator(min_words=50, max_words=200)
```

## Using BaseValidatorImplementation

For production validators, extend the base class for error handling and utilities:

```python
from sifaka.validators.shared import BaseValidatorImplementation
from sifaka.core.thought import Thought, ValidationResult

class EmailValidator(BaseValidatorImplementation):
    """Validates text contains valid email addresses."""

    def __init__(self, require_email: bool = True, max_emails: int = 5):
        super().__init__(name="EmailValidator")
        self.require_email = require_email
        self.max_emails = max_emails

    def _validate_impl(self, thought: Thought) -> ValidationResult:
        """Implement email validation logic."""
        import re

        text = thought.text or ""

        # Email regex pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        issues = []
        suggestions = []

        # Check if email is required
        if self.require_email and not emails:
            issues.append("No valid email addresses found")
            suggestions.append("Include at least one valid email address")

        # Check maximum emails
        if len(emails) > self.max_emails:
            issues.append(f"Found {len(emails)} emails, maximum is {self.max_emails}")
            suggestions.append(f"Remove {len(emails) - self.max_emails} email addresses")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.3)

        return self.create_validation_result(
            passed=passed,
            message=f"Found {len(emails)} valid email(s)" if passed else "Email validation failed",
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "email_count": len(emails),
                "emails_found": emails,
                "validator": self.name
            }
        )

# Factory function
def create_email_validator(require_email: bool = True, max_emails: int = 5) -> EmailValidator:
    """Create an email validator."""
    return EmailValidator(require_email=require_email, max_emails=max_emails)
```

## Advanced Example: Content Quality Validator

Here's a comprehensive validator for content quality:

```python
import re
from typing import List, Dict, Any
from sifaka.validators.shared import BaseValidatorImplementation
from sifaka.core.thought import Thought, ValidationResult

class ContentQualityValidator(BaseValidatorImplementation):
    """Validates content quality using multiple criteria."""

    def __init__(
        self,
        min_sentences: int = 3,
        max_sentences: int = 20,
        min_avg_words_per_sentence: float = 8.0,
        max_avg_words_per_sentence: float = 25.0,
        require_punctuation: bool = True,
        forbidden_words: List[str] = None,
        required_keywords: List[str] = None
    ):
        super().__init__(name="ContentQualityValidator")
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_avg_words = min_avg_words_per_sentence
        self.max_avg_words = max_avg_words_per_sentence
        self.require_punctuation = require_punctuation
        self.forbidden_words = [word.lower() for word in (forbidden_words or [])]
        self.required_keywords = [word.lower() for word in (required_keywords or [])]

    def _validate_impl(self, thought: Thought) -> ValidationResult:
        """Validate content quality across multiple dimensions."""
        text = thought.text or ""

        # Analyze text structure
        analysis = self._analyze_text(text)

        # Run all quality checks
        issues = []
        suggestions = []

        self._check_sentence_count(analysis, issues, suggestions)
        self._check_sentence_length(analysis, issues, suggestions)
        self._check_punctuation(analysis, issues, suggestions)
        self._check_forbidden_words(analysis, issues, suggestions)
        self._check_required_keywords(analysis, issues, suggestions)

        # Calculate overall score
        total_checks = 5
        failed_checks = len([issue for issue in issues if "failed" in issue.lower()])
        score = max(0.0, (total_checks - failed_checks) / total_checks)

        passed = len(issues) == 0

        return self.create_validation_result(
            passed=passed,
            message="Content quality is good" if passed else "Content quality issues found",
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "analysis": analysis,
                "validator": self.name,
                "quality_score": score
            }
        )

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and properties."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Calculate metrics
        total_words = len(text.split())
        avg_words_per_sentence = total_words / len(sentences) if sentences else 0

        # Check punctuation
        has_punctuation = bool(re.search(r'[.!?]', text))

        # Word analysis
        words = [word.lower().strip('.,!?;:"()[]') for word in text.split()]

        return {
            "sentence_count": len(sentences),
            "word_count": total_words,
            "avg_words_per_sentence": avg_words_per_sentence,
            "has_punctuation": has_punctuation,
            "words": words,
            "sentences": sentences
        }

    def _check_sentence_count(self, analysis: Dict, issues: List, suggestions: List):
        """Check sentence count requirements."""
        count = analysis["sentence_count"]

        if count < self.min_sentences:
            issues.append(f"Only {count} sentences, minimum is {self.min_sentences}")
            suggestions.append(f"Add {self.min_sentences - count} more sentences")

        if count > self.max_sentences:
            issues.append(f"Has {count} sentences, maximum is {self.max_sentences}")
            suggestions.append(f"Reduce to {self.max_sentences} sentences or fewer")

    def _check_sentence_length(self, analysis: Dict, issues: List, suggestions: List):
        """Check average sentence length."""
        avg_length = analysis["avg_words_per_sentence"]

        if avg_length < self.min_avg_words:
            issues.append(f"Average sentence length is {avg_length:.1f} words, minimum is {self.min_avg_words}")
            suggestions.append("Use longer, more detailed sentences")

        if avg_length > self.max_avg_words:
            issues.append(f"Average sentence length is {avg_length:.1f} words, maximum is {self.max_avg_words}")
            suggestions.append("Break long sentences into shorter ones")

    def _check_punctuation(self, analysis: Dict, issues: List, suggestions: List):
        """Check punctuation requirements."""
        if self.require_punctuation and not analysis["has_punctuation"]:
            issues.append("Text lacks proper punctuation")
            suggestions.append("Add periods, exclamation marks, or question marks")

    def _check_forbidden_words(self, analysis: Dict, issues: List, suggestions: List):
        """Check for forbidden words."""
        words = analysis["words"]
        found_forbidden = [word for word in words if word in self.forbidden_words]

        if found_forbidden:
            issues.append(f"Contains forbidden words: {', '.join(set(found_forbidden))}")
            suggestions.append("Remove or replace forbidden words")

    def _check_required_keywords(self, analysis: Dict, issues: List, suggestions: List):
        """Check for required keywords."""
        if not self.required_keywords:
            return

        words = analysis["words"]
        missing_keywords = [kw for kw in self.required_keywords if kw not in words]

        if missing_keywords:
            issues.append(f"Missing required keywords: {', '.join(missing_keywords)}")
            suggestions.append(f"Include these keywords: {', '.join(missing_keywords)}")

# Factory function
def create_content_quality_validator(**kwargs) -> ContentQualityValidator:
    """Create a content quality validator with custom parameters."""
    return ContentQualityValidator(**kwargs)
```

## ML-Based Validators

Create validators using machine learning models:

```python
from sifaka.validators.shared import BaseValidatorImplementation
from sifaka.core.thought import Thought, ValidationResult

class SentimentValidator(BaseValidatorImplementation):
    """Validates text sentiment using ML."""

    def __init__(self, target_sentiment: str = "positive", confidence_threshold: float = 0.7):
        super().__init__(name="SentimentValidator")
        self.target_sentiment = target_sentiment.lower()
        self.confidence_threshold = confidence_threshold
        self._load_model()

    def _load_model(self):
        """Load sentiment analysis model."""
        try:
            from textblob import TextBlob
            self.analyzer = TextBlob
        except ImportError:
            raise ImportError("TextBlob required for sentiment validation: pip install textblob")

    def _validate_impl(self, thought: Thought) -> ValidationResult:
        """Validate sentiment using TextBlob."""
        text = thought.text or ""

        # Analyze sentiment
        blob = self.analyzer(text)
        polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)

        # Determine sentiment
        if polarity > 0.1:
            detected_sentiment = "positive"
        elif polarity < -0.1:
            detected_sentiment = "negative"
        else:
            detected_sentiment = "neutral"

        # Calculate confidence (distance from neutral)
        confidence = abs(polarity)

        # Check if sentiment matches target
        sentiment_matches = detected_sentiment == self.target_sentiment
        confidence_sufficient = confidence >= self.confidence_threshold

        issues = []
        suggestions = []

        if not sentiment_matches:
            issues.append(f"Sentiment is {detected_sentiment}, expected {self.target_sentiment}")
            suggestions.append(f"Adjust tone to be more {self.target_sentiment}")

        if not confidence_sufficient:
            issues.append(f"Sentiment confidence is {confidence:.2f}, minimum is {self.confidence_threshold}")
            suggestions.append("Use stronger emotional language")

        passed = sentiment_matches and confidence_sufficient
        score = confidence if sentiment_matches else 0.0

        return self.create_validation_result(
            passed=passed,
            message=f"Sentiment: {detected_sentiment} (confidence: {confidence:.2f})",
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "detected_sentiment": detected_sentiment,
                "polarity": polarity,
                "confidence": confidence,
                "target_sentiment": self.target_sentiment
            }
        )
```

## Integration with Chains

Use your custom validators in Sifaka chains:

```python
from sifaka.agents import create_pydantic_chain
from sifaka.models import create_model
from pydantic_ai import Agent

# Create model and validators
model = create_model("openai:gpt-4")
word_validator = WordCountValidator(min_words=50, max_words=200)
email_validator = create_email_validator(require_email=True)
quality_validator = create_content_quality_validator(
    min_sentences=3,
    required_keywords=["innovation", "technology"]
)

# Create PydanticAI agent
agent = Agent("openai:gpt-4", system_prompt="Write a professional email about our new technology innovation.")

# Build modern PydanticAI chain with multiple validators
chain = create_pydantic_chain(
    agent=agent,
    validators=[word_validator, email_validator, quality_validator],
    critics=[]
)

# Run chain (validators are automatically applied)
result = chain.run("Write a professional email about our new technology innovation.")

# Check validation results
for validator_name, validation_result in result.validation_results.items():
    print(f"{validator_name}: {'✓' if validation_result.passed else '✗'}")
    if not validation_result.passed:
        print(f"  Issues: {validation_result.issues}")
        print(f"  Suggestions: {validation_result.suggestions}")
```

## Best Practices

### 1. Comprehensive Error Handling

```python
def _validate_impl(self, thought: Thought) -> ValidationResult:
    try:
        # Your validation logic
        return self.create_validation_result(...)
    except Exception as e:
        return self.create_validation_result(
            passed=False,
            message=f"Validation error: {str(e)}",
            score=0.0,
            issues=[f"Internal error: {str(e)}"],
            suggestions=["Check validator configuration"]
        )
```

### 2. Meaningful Scores

```python
def _calculate_score(self, issues_count: int, total_checks: int) -> float:
    """Calculate meaningful validation score."""
    if total_checks == 0:
        return 1.0

    # Exponential decay for multiple issues
    base_score = (total_checks - issues_count) / total_checks
    return max(0.0, base_score ** (1 + issues_count * 0.1))
```

### 3. Helpful Suggestions

```python
def _generate_suggestions(self, issues: List[str]) -> List[str]:
    """Generate actionable improvement suggestions."""
    suggestions = []

    for issue in issues:
        if "too short" in issue:
            suggestions.append("Add more detail and examples")
        elif "too long" in issue:
            suggestions.append("Remove unnecessary content and be more concise")
        elif "missing" in issue:
            suggestions.append("Include the required information")

    return suggestions
```

## Testing Custom Validators

Always test your validators thoroughly:

```python
def test_word_count_validator():
    """Test word count validator."""
    from sifaka.core.thought import Thought

    validator = WordCountValidator(min_words=5, max_words=10)

    # Test valid text
    thought = Thought(prompt="test", text="This is a valid text with seven words.")
    result = validator.validate(thought)
    assert result.passed
    assert result.score > 0.8

    # Test too short
    thought = Thought(prompt="test", text="Too short.")
    result = validator.validate(thought)
    assert not result.passed
    assert "minimum" in result.issues[0]

    # Test too long
    thought = Thought(prompt="test", text="This text is way too long and exceeds the maximum word count limit.")
    result = validator.validate(thought)
    assert not result.passed
    assert "maximum" in result.issues[0]

    print("✅ All validator tests passed!")

test_word_count_validator()
```

Your custom validators are now ready to ensure high-quality text generation in your Sifaka chains!
