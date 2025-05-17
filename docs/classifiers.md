# Classifiers in Sifaka

Classifiers are components that categorize text into specific classes or labels with confidence scores. They can be used directly or adapted into validators using the classifier validator.

## Overview

The classifier system in Sifaka consists of:

1. **Classifier Protocol**: Defines the interface that all classifiers must implement
2. **ClassificationResult**: Represents the result of a classification operation
3. **Classifier Implementations**: Concrete classifier classes that implement the protocol
4. **ClassifierValidator**: Adapts classifiers to the validator interface

## Classifier Protocol

The `Classifier` protocol defines the interface that all classifiers must implement:

```python
@runtime_checkable
class Classifier(Protocol):
    def classify(self, text: str) -> ClassificationResult:
        """Classify text into a specific category."""
        ...

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts efficiently."""
        ...

    @property
    def name(self) -> str:
        """Get the classifier name."""
        ...

    @property
    def description(self) -> str:
        """Get the classifier description."""
        ...
```

## Classification Result

The `ClassificationResult` class represents the result of a classification operation:

```python
@dataclass
class ClassificationResult:
    label: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
```

- `label`: The class label assigned to the text
- `confidence`: The confidence score for the classification (0.0 to 1.0)
- `metadata`: Optional additional information about the classification

## Classifier Implementations

Sifaka provides several classifier implementations:

### Sentiment Classifier

The `SentimentClassifier` categorizes text as positive, negative, or neutral based on keyword matching:

```python
from sifaka.classifiers import SentimentClassifier

# Create a sentiment classifier
classifier = SentimentClassifier()

# Classify text
result = classifier.classify("I love this product!")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

### Toxicity Classifier

The `ToxicityClassifier` categorizes text as toxic or non-toxic using the Detoxify library:

```python
from sifaka.classifiers import ToxicityClassifier

# Create a toxicity classifier
classifier = ToxicityClassifier()

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

**Note:** Requires the `detoxify` package: `pip install detoxify`

### Profanity Classifier

The `ProfanityClassifier` detects profanity and inappropriate language using the better_profanity library:

```python
from sifaka.classifiers import ProfanityClassifier

# Create a profanity classifier
classifier = ProfanityClassifier()

# Classify text
result = classifier.classify("This is a clean message.")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

**Note:** Requires the `better_profanity` package: `pip install better_profanity`

### Spam Classifier

The `SpamClassifier` categorizes text as spam or ham (non-spam) using scikit-learn:

```python
from sifaka.classifiers import SpamClassifier

# Create a spam classifier
classifier = SpamClassifier()

# Classify text
result = classifier.classify("Please review the attached document.")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

**Note:** Requires the `scikit-learn` and `numpy` packages: `pip install scikit-learn numpy`

### Language Classifier

The `LanguageClassifier` detects the language of text using the langdetect library:

```python
from sifaka.classifiers import LanguageClassifier

# Create a language classifier
classifier = LanguageClassifier()

# Classify text
result = classifier.classify("Hello world!")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

**Note:** Requires the `langdetect` package: `pip install langdetect`

## Classifier Validator

The `ClassifierValidator` adapts classifiers to the validator interface, allowing them to be used in validation chains:

```python
from sifaka.classifiers import SentimentClassifier
from sifaka.validators import classifier_validator

# Create a sentiment classifier
classifier = SentimentClassifier()

# Create a classifier validator
validator = classifier_validator(
    classifier=classifier,
    threshold=0.7,
    valid_labels=["positive"],
)

# Validate text
result = validator.validate("I love this product!")
print(f"Passed: {result.passed}, Message: {result.message}")
```

### Configuration Options

The `ClassifierValidator` supports the following configuration options:

- `threshold`: Confidence threshold for accepting a classification (0.0 to 1.0)
- `valid_labels`: List of labels considered valid
- `invalid_labels`: Optional list of labels considered invalid
- `extraction_function`: Optional function to extract text for classification

## Using Classifiers with Chains

Classifiers can be used with the chain system through the classifier validator:

```python
from sifaka.classifiers import SentimentClassifier
from sifaka.validators import classifier_validator
from sifaka.chain import Chain

# Create a sentiment classifier
classifier = SentimentClassifier()

# Create a classifier validator
validator = classifier_validator(
    classifier=classifier,
    threshold=0.7,
    valid_labels=["positive"],
)

# Use the validator in a chain
result = (
    Chain()
    .with_model("openai:gpt-3.5-turbo")
    .with_prompt("Write a short paragraph about something you love.")
    .validate_with(validator)
    .run()
)

print(f"Passed validation: {result.passed}")
print(result.text)
```

## Creating Custom Classifiers

You can create custom classifiers by implementing the `Classifier` protocol:

```python
from sifaka.classifiers import Classifier, ClassificationResult

class CustomClassifier:
    @property
    def name(self) -> str:
        return "custom_classifier"

    @property
    def description(self) -> str:
        return "A custom classifier"

    def classify(self, text: str) -> ClassificationResult:
        # Implement your classification logic here
        return ClassificationResult(
            label="custom_label",
            confidence=0.9,
            metadata={"custom_key": "custom_value"},
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        return [self.classify(text) for text in texts]
```

## Best Practices

1. **Confidence Scores**: Ensure confidence scores are between 0.0 and 1.0
2. **Metadata**: Include useful information in the metadata field
3. **Batch Classification**: Implement efficient batch classification when possible
4. **Error Handling**: Handle edge cases gracefully (e.g., empty text)
5. **Validation**: Use appropriate thresholds and label lists for validation
