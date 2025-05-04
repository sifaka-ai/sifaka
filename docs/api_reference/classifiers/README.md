# Classifiers API Reference

Classifiers are components in Sifaka that analyze and categorize text. Unlike rules, which provide binary pass/fail validation, classifiers assign labels with confidence scores, providing more nuanced analysis.

## Core Classes and Protocols

### BaseClassifier

`BaseClassifier` is the abstract base class for all classifiers in Sifaka.

```python
from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig

class MyClassifier(BaseClassifier[str, str]):
    """Custom classifier implementation."""
    
    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """Classify the text (core implementation)."""
        if len(text) > 100:
            return ClassificationResult(
                label="long",
                confidence=0.9,
                metadata={"length": len(text)}
            )
        return ClassificationResult(
            label="short",
            confidence=0.9,
            metadata={"length": len(text)}
        )
```

### ClassifierProtocol

`ClassifierProtocol` defines the interface for all classifier-like objects.

```python
from sifaka.classifiers.base import ClassifierProtocol, ClassificationResult

# Any class implementing ClassifierProtocol can be used where a Classifier is expected
def process_classifier(classifier: ClassifierProtocol[str, str]):
    """Process a classifier."""
    result = classifier.classify("This is a test")
    print(f"Classifier {classifier.name} assigned label: {result.label}")
```

### TextProcessor

`TextProcessor` is a protocol for text processing components.

```python
from sifaka.classifiers.base import TextProcessor

class MyTextProcessor(TextProcessor):
    """Custom text processor implementation."""
    
    def process(self, text: str) -> str:
        """Process the text."""
        return text.lower()
```

## Configuration

### ClassifierConfig

`ClassifierConfig` is the configuration class for classifiers.

```python
from sifaka.classifiers.base import ClassifierConfig

# Create a classifier configuration
config = ClassifierConfig(
    name="my_classifier",
    description="A custom classifier",
    labels=["short", "long"],
    cache_size=100,
    min_confidence=0.7,
    params={
        "threshold": 100,
    }
)

# Access configuration values
print(f"Name: {config.name}")
print(f"Labels: {config.labels}")
print(f"Min confidence: {config.min_confidence}")
print(f"Threshold: {config.params['threshold']}")

# Create a new configuration with updated options
updated_config = config.with_options(
    min_confidence=0.8,
    params={"threshold": 200}
)
```

## Results

### ClassificationResult

`ClassificationResult` represents the result of a classification.

```python
from sifaka.classifiers.base import ClassificationResult

# Create a classification result
result = ClassificationResult(
    label="long",
    confidence=0.9,
    metadata={"length": 150}
)

# Access result values
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence}")
print(f"Length: {result.metadata['length']}")
```

## Classifier Types

Sifaka provides several types of classifiers:

### Content Analysis Classifiers

Content analysis classifiers analyze the semantic content of text.

```python
from sifaka.classifiers.toxicity import create_toxicity_classifier
from sifaka.classifiers.profanity import create_profanity_classifier
from sifaka.classifiers.spam import create_spam_classifier
from sifaka.classifiers.bias import create_bias_detector

# Create content analysis classifiers
toxicity_classifier = create_toxicity_classifier()
profanity_classifier = create_profanity_classifier()
spam_classifier = create_spam_classifier()
bias_detector = create_bias_detector()
```

### Text Properties Classifiers

Text properties classifiers analyze the structural properties of text.

```python
from sifaka.classifiers.readability import create_readability_classifier
from sifaka.classifiers.language import create_language_classifier
from sifaka.classifiers.topic import create_topic_classifier
from sifaka.classifiers.genre import create_genre_classifier

# Create text properties classifiers
readability_classifier = create_readability_classifier()
language_classifier = create_language_classifier()
topic_classifier = create_topic_classifier()
genre_classifier = create_genre_classifier()
```

### Entity Analysis Classifiers

Entity analysis classifiers identify and categorize entities in text.

```python
from sifaka.classifiers.ner import create_ner_classifier

# Create an entity analysis classifier
ner_classifier = create_ner_classifier()
```

## Usage Examples

### Basic Classifier Usage

```python
from sifaka.classifiers.toxicity import create_toxicity_classifier

# Create a classifier
classifier = create_toxicity_classifier()

# Classify text
result = classifier.classify("This is a test")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

### Custom Classifier Implementation

```python
from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig

class SentimentClassifier(BaseClassifier[str, str]):
    """Classifier for sentiment analysis."""
    
    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """Classify the sentiment of the text."""
        # Simple implementation for demonstration
        positive_words = ["good", "great", "excellent", "happy", "positive"]
        negative_words = ["bad", "terrible", "awful", "sad", "negative"]
        
        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)
        
        if positive_count > negative_count:
            return ClassificationResult(
                label="positive",
                confidence=0.7 + (0.1 * (positive_count - negative_count)),
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
        elif negative_count > positive_count:
            return ClassificationResult(
                label="negative",
                confidence=0.7 + (0.1 * (negative_count - positive_count)),
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
        else:
            return ClassificationResult(
                label="neutral",
                confidence=0.7,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )

# Create the classifier using the factory method
classifier = SentimentClassifier.create(
    name="sentiment_classifier",
    description="Classifies text sentiment as positive, negative, or neutral",
    labels=["positive", "negative", "neutral"],
    cache_size=100
)

# Classify text
result = classifier.classify("This is a great day!")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

### Batch Classification

```python
from sifaka.classifiers.toxicity import create_toxicity_classifier

# Create a classifier
classifier = create_toxicity_classifier()

# Classify multiple texts
texts = [
    "This is a test",
    "This is another test",
    "This is a third test"
]

results = classifier.batch_classify(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: Label: {result.label}, Confidence: {result.confidence}")
```

### Using Classifiers with Rules

Classifiers can be adapted to work as rules using the ClassifierAdapter:

```python
from sifaka.classifiers.toxicity import create_toxicity_classifier
from sifaka.adapters.rules.classifier import create_classifier_rule

# Create a classifier
classifier = create_toxicity_classifier()

# Create a rule from the classifier
rule = create_classifier_rule(
    classifier=classifier,
    valid_labels=["safe"],
    name="toxicity_rule",
    description="Ensures text is not toxic"
)

# Validate text using the rule
result = rule.validate("This is a test")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
