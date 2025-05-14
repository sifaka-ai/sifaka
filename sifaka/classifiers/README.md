# Sifaka Classifiers

This package provides a robust implementation of the classifiers system for categorizing and analyzing text.

## Architecture

The classifiers architecture follows a component-based design:

```
Classifier
├── Engine (core classification logic)
│   ├── CacheManager (manages result caching)
│   └── StateManager (centralized state management)
├── Components (pluggable components)
│   └── ClassifierImplementation (text classification)
├── Factory Functions
│   ├── create_classifier (generic classifier creation)
│   ├── create_toxicity_classifier (toxicity classification)
│   ├── create_sentiment_classifier (sentiment analysis)
│   └── create_profanity_classifier (profanity detection)
└── Implementations
    ├── Content (toxicity, sentiment, profanity, spam, bias)
    ├── Properties (language, readability, genre, topic)
    └── Entities (named entity recognition)
```

### Core Components

- **Classifier**: Main user-facing class for classification
- **Engine**: Core classification engine that coordinates the flow
- **StateManager**: Centralized state management
- **ClassifierImplementation**: Interface for classifier implementations
- **ClassificationResult**: Result of a classification operation
- **Adapters**: Adapts various implementations to the common interface

## Usage

### Basic Usage

```python
from sifaka.classifiers import Classifier
from sifaka.classifiers.implementations.content import ToxicityClassifier

# Create classifier implementation
implementation = ToxicityClassifier()

# Create classifier
classifier = Classifier(implementation=implementation)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Execution time: {result.execution_time:.2f}s")
```

### Using Factory Functions

```python
from sifaka.classifiers import create_toxicity_classifier, create_sentiment_classifier, create_profanity_classifier

# Create toxicity classifier
toxicity_classifier = create_toxicity_classifier(
    general_threshold=0.5,
    cache_enabled=True,
    cache_size=100
)

# Create sentiment classifier
sentiment_classifier = create_sentiment_classifier(
    positive_threshold=0.05,
    negative_threshold=-0.05,
    cache_enabled=True
)

# Create profanity classifier
profanity_classifier = create_profanity_classifier(
    threshold=0.5,
    cache_enabled=True
)

# Classify text
toxicity_result = toxicity_classifier.classify("This is a friendly message.")
sentiment_result = sentiment_classifier.classify("I love this product!")
profanity_result = profanity_classifier.classify("This is clean text.")

print(f"Toxicity: {toxicity_result.label} ({toxicity_result.confidence:.2f})")
print(f"Sentiment: {sentiment_result.label} ({sentiment_result.confidence:.2f})")
print(f"Profanity: {profanity_result.label} ({profanity_result.confidence:.2f})")
```

### Batch Classification

```python
from sifaka.classifiers import create_sentiment_classifier

# Create classifier
classifier = create_sentiment_classifier()

# Classify batch of texts
results = classifier.classify_batch([
    "I love this product!",
    "I'm not sure about this.",
    "This is terrible."
])

# Process results
for i, result in enumerate(results):
    print(f"Text {i+1}: {result.label} ({result.confidence:.2f})")
```

### Asynchronous Classification

```python
import asyncio
from sifaka.classifiers import Classifier
from sifaka.utils.config import ClassifierConfig
from sifaka.classifiers.implementations.content import ToxicityClassifier

# Create classifier with async enabled
config = ClassifierConfig(async_enabled=True)
classifier = Classifier(
    implementation=ToxicityClassifier(),
    config=config
)

# Classify text asynchronously
async def classify_async():
    result = await classifier.classify_async("This is a friendly message.")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.2f}")

    # Batch classification
    batch_results = await classifier.classify_batch_async([
        "This is a friendly message.",
        "This is another message."
    ])
    for i, res in enumerate(batch_results):
        print(f"Batch result {i+1}: {res.label} ({res.confidence:.2f})")

# Run async function
asyncio.run(classify_async())
```

### Statistics and Monitoring

```python
from sifaka.classifiers import create_toxicity_classifier

# Create classifier
classifier = create_toxicity_classifier()

# Classify some texts
classifier.classify("This is a friendly message.")
classifier.classify("This is another message.")

# Get statistics
stats = classifier.get_statistics()
print(f"Execution count: {stats['execution_count']}")
print(f"Average execution time: {stats['avg_execution_time']:.2f}s")
print(f"Success rate: {stats['success_count'] / stats['execution_count']:.2%}")

# Clear cache
classifier.clear_cache()
```

## Available Classifier Implementations

### Content Classifiers
- **ToxicityClassifier**: Detects toxic content using Detoxify
- **SentimentClassifier**: Analyzes sentiment using VADER
- **ProfanityClassifier**: Detects profanity using better-profanity
- **SpamClassifier**: Identifies spam content
- **BiasClassifier**: Detects biased language

### Property Classifiers
- **LanguageClassifier**: Identifies the language of text
- **ReadabilityClassifier**: Measures text readability metrics
- **GenreClassifier**: Classifies text by genre
- **TopicClassifier**: Identifies topics in text

### Entity Classifiers
- **NERClassifier**: Named Entity Recognition for identifying people, organizations, etc.

## Extending

### Creating a Custom Implementation

```python
from sifaka.classifiers.interfaces import ClassifierImplementation
from sifaka.core.results import ClassificationResult

class CustomClassifier(ClassifierImplementation):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.positive_words = ["good", "great", "excellent", "happy"]
        self.negative_words = ["bad", "terrible", "awful", "sad"]

    def classify(self, text: str) -> ClassificationResult:
        # Simple implementation that checks for positive/negative words
        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in self.positive_words)
        negative_count = sum(word in text_lower for word in self.negative_words)

        # Calculate confidence based on word count difference
        total_count = positive_count + negative_count
        if total_count == 0:
            confidence = 0.5
            label = "neutral"
        elif positive_count > negative_count:
            confidence = positive_count / total_count
            label = "positive"
        else:
            confidence = negative_count / total_count
            label = "negative"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "positive_words": positive_count,
                "negative_words": negative_count,
                "threshold": self.threshold
            }
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        # For simple implementations, async can just call the sync version
        return self.classify(text)
```

### Creating a Custom Factory

```python
from typing import Any
from sifaka.classifiers import Classifier, create_classifier
from sifaka.utils.config import ClassifierConfig

def create_custom_classifier(
    threshold: float = 0.5,
    name: str = "custom_classifier",
    description: str = "Custom text classifier",
    cache_enabled: bool = True,
    cache_size: int = 100,
    **kwargs: Any
) -> Classifier:
    """
    Create a custom classifier.

    Args:
        threshold: Classification threshold
        name: Classifier name
        description: Classifier description
        cache_enabled: Whether to enable result caching
        cache_size: Maximum number of cached results
        **kwargs: Additional parameters

    Returns:
        A custom classifier
    """
    from .custom_implementation import CustomClassifier

    implementation = CustomClassifier(threshold=threshold)

    return create_classifier(
        implementation=implementation,
        name=name,
        description=description,
        config=ClassifierConfig(
            cache_enabled=cache_enabled,
            cache_size=cache_size,
            **kwargs
        )
    )
```

### Error Handling

```python
from sifaka.classifiers import create_toxicity_classifier
from sifaka.utils.errors import ClassifierError

try:
    # Create classifier
    classifier = create_toxicity_classifier()

    # Classify text
    result = classifier.classify("This is a test.")

    # Check confidence threshold
    if result.confidence < 0.5:
        print(f"Low confidence result: {result.label} ({result.confidence:.2f})")

except ClassifierError as e:
    print(f"Classification error: {str(e)}")
    # Fallback strategy
    print("Using fallback classification: neutral")
```
