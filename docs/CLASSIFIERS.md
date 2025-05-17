# Classifiers in Sifaka

Classifiers are components in the Sifaka framework that categorize text into specific classes or labels. They can be used directly or adapted into validators using the classifier validator adapter.

## Overview

Classifiers implement a common interface defined by the `Classifier` protocol, which requires implementing methods for classifying text and providing metadata about the classifier.

The Sifaka framework provides several classifier implementations:

1. **SentimentClassifier** - Classifies text sentiment as positive, negative, or neutral
2. **ToxicityClassifier** - Detects toxic content in text
3. **SpamClassifier** - Identifies spam or unwanted content
4. **ProfanityClassifier** - Detects profane language in text
5. **LanguageClassifier** - Identifies the language of text

## Classifier Interface

All classifiers in Sifaka implement the following interface:

```python
class Classifier(Protocol):
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text into a specific category.

        Args:
            text: The text to classify

        Returns:
            A ClassificationResult with the class label and confidence score
        """
        ...

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: The list of texts to classify

        Returns:
            A list of ClassificationResults
        """
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

The result of a classification operation is represented by the `ClassificationResult` class:

```python
@dataclass
class ClassificationResult:
    """
    Result of a classification operation.

    Attributes:
        label: The class label assigned to the text.
        confidence: The confidence score for the classification (0.0 to 1.0).
        metadata: Optional additional information about the classification.
    """

    label: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
```

## Available Classifiers

### SentimentClassifier

The `SentimentClassifier` analyzes the sentiment of text, categorizing it as positive, negative, or neutral.

```python
from sifaka.classifiers import SentimentClassifier

# Create a sentiment classifier
classifier = SentimentClassifier()

# Classify text sentiment
result = classifier.classify("I love using the Sifaka framework!")
print(f"Label: {result.label}, Confidence: {result.confidence}")
# Output: Label: positive, Confidence: 0.95
```

### ToxicityClassifier

The `ToxicityClassifier` detects toxic content in text, such as hate speech, threats, or insults.

```python
from sifaka.classifiers import ToxicityClassifier

# Create a toxicity classifier
classifier = ToxicityClassifier()

# Check if text contains toxic content
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}, Confidence: {result.confidence}")
# Output: Label: non-toxic, Confidence: 0.98
```

### SpamClassifier

The `SpamClassifier` identifies spam or unwanted content in text.

```python
from sifaka.classifiers import SpamClassifier

# Create a spam classifier
classifier = SpamClassifier()

# Check if text is spam
result = classifier.classify("Check out this amazing offer! Click here to claim your prize!")
print(f"Label: {result.label}, Confidence: {result.confidence}")
# Output: Label: spam, Confidence: 0.87
```

### ProfanityClassifier

The `ProfanityClassifier` detects profane language in text.

```python
from sifaka.classifiers import ProfanityClassifier

# Create a profanity classifier
classifier = ProfanityClassifier()

# Check if text contains profanity
result = classifier.classify("This is a clean and professional message.")
print(f"Label: {result.label}, Confidence: {result.confidence}")
# Output: Label: clean, Confidence: 0.99
```

### LanguageClassifier

The `LanguageClassifier` identifies the language of text.

```python
from sifaka.classifiers import LanguageClassifier

# Create a language classifier
classifier = LanguageClassifier()

# Identify the language of text
result = classifier.classify("Hello, how are you today?")
print(f"Label: {result.label}, Confidence: {result.confidence}")
# Output: Label: en, Confidence: 0.98

result = classifier.classify("Bonjour, comment allez-vous aujourd'hui?")
print(f"Label: {result.label}, Confidence: {result.confidence}")
# Output: Label: fr, Confidence: 0.97
```

## Using Classifiers with Validators

Classifiers can be adapted into validators using the `classifier_validator` adapter:

```python
from sifaka import Chain
from sifaka.models import create_model
from sifaka.validators.classifier import classifier_validator
from sifaka.classifiers import ToxicityClassifier

# Create a model
model = create_model("openai:gpt-4")

# Create a toxicity classifier validator
toxicity_validator = classifier_validator(
    classifier=ToxicityClassifier(),
    valid_labels=["non-toxic"],
    min_confidence=0.8
)

# Use the validator in a chain
result = (Chain()
    .with_model(model)
    .with_prompt("Write a friendly message about teamwork.")
    .validate_with(toxicity_validator)
    .run())

if result.passed:
    print("Generated text passed toxicity validation!")
    print(result.text)
else:
    print("Generated text failed toxicity validation.")
    print(result.validation_results[0].message)
```

## Batch Classification

All classifiers support batch classification for efficiently processing multiple texts:

```python
from sifaka.classifiers import SentimentClassifier

# Create a sentiment classifier
classifier = SentimentClassifier()

# Classify multiple texts at once
texts = [
    "I love this product!",
    "This is terrible service.",
    "I'm neutral about this."
]

results = classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result.label}, Confidence: {result.confidence}")
    print()
```

## Creating Custom Classifiers

You can create custom classifiers by implementing the `Classifier` protocol:

```python
from sifaka.classifiers import Classifier, ClassificationResult
from typing import List

class CustomClassifier:
    def classify(self, text: str) -> ClassificationResult:
        # Implement your custom classification logic here
        # This is a simple example that classifies text length
        if len(text) < 10:
            return ClassificationResult(label="short", confidence=1.0)
        elif len(text) < 50:
            return ClassificationResult(label="medium", confidence=1.0)
        else:
            return ClassificationResult(label="long", confidence=1.0)
    
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        # Implement batch classification
        return [self.classify(text) for text in texts]
    
    @property
    def name(self) -> str:
        return "TextLengthClassifier"
    
    @property
    def description(self) -> str:
        return "Classifies text based on its length."
```

## Next Steps

- Learn more about [Validators](VALIDATORS.md) that use classifiers
- Explore the [Classifier Validator](VALIDATORS.md#classifier-validator) adapter
- Check out the [Chain](CHAIN.md) documentation for using validators in text generation
