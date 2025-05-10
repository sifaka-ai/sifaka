# Sifaka Classifiers

This package provides a simplified and more maintainable implementation of the classifiers system for categorizing and analyzing text.

## Architecture

The classifiers architecture follows a simplified component-based design:

```
Classifier
├── Engine (core classification logic)
│   └── StateTracker (centralized state management)
├── Components (pluggable components)
│   └── ClassifierImplementation (text classification)
└── Plugins (extension mechanism)
    ├── PluginRegistry (plugin discovery and registration)
    └── PluginLoader (dynamic plugin loading)
```

### Core Components

- **Classifier**: Main user-facing class for classification
- **Engine**: Core classification engine that coordinates the flow
- **StateTracker**: Centralized state management
- **ClassifierImplementation**: Interface for classifier implementations
- **ClassificationResult**: Result of a classification operation
- **Plugin**: Interface for plugins

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
```

### Using Factory Functions

```python
from sifaka.classifiers import create_classifier
from sifaka.classifiers.implementations.content import ToxicityClassifier

# Create classifier implementation
implementation = ToxicityClassifier()

# Create classifier using factory
classifier = create_classifier(
    implementation=implementation,
    name="toxicity_classifier",
    description="Detects toxic content in text",
    cache_enabled=True,
    cache_size=100,
    min_confidence=0.7
)

# Classify text
result = classifier.classify("This is a friendly message.")
```

### Batch Classification

```python
from sifaka.classifiers import Classifier
from sifaka.classifiers.implementations.content import SentimentClassifier

# Create classifier
classifier = Classifier(implementation=SentimentClassifier())

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
from sifaka.classifiers.config import ClassifierConfig
from sifaka.classifiers.implementations.content import ToxicityClassifier

# Create classifier with async enabled
classifier = Classifier(
    implementation=ToxicityClassifier(),
    config=ClassifierConfig(async_enabled=True)
)

# Classify text asynchronously
async def classify_async():
    result = await classifier.classify_async("This is a friendly message.")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.2f}")

# Run async function
asyncio.run(classify_async())
```

## Extending

### Creating a Custom Implementation

```python
from sifaka.classifiers.interfaces import ClassifierImplementation
from sifaka.classifiers.result import ClassificationResult

class SentimentClassifier(ClassifierImplementation):
    def __init__(self):
        self.positive_words = ["good", "great", "excellent", "happy"]
        self.negative_words = ["bad", "terrible", "awful", "sad"]

    def classify(self, text: str) -> ClassificationResult:
        # Simple implementation that checks for positive/negative words
        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in self.positive_words)
        negative_count = sum(word in text_lower for word in self.negative_words)

        if positive_count > negative_count:
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
        elif negative_count > positive_count:
            return ClassificationResult(
                label="negative",
                confidence=0.8,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
        else:
            return ClassificationResult(
                label="neutral",
                confidence=0.6,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )

    async def classify_async(self, text: str) -> ClassificationResult:
        # For simple implementations, async can just call the sync version
        return self.classify(text)
```

### Creating a Plugin

```python
from typing import Any, Dict
from sifaka.classifiers.interfaces import Plugin, ClassifierImplementation
from sifaka.classifiers.result import ClassificationResult

class SentimentPlugin(Plugin):
    @property
    def name(self) -> str:
        return "sentiment_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def component_type(self) -> str:
        return "classifier_implementation"

    def create_component(self, config: Dict[str, Any]) -> ClassifierImplementation:
        positive_words = config.get("positive_words", ["good", "great", "excellent", "happy"])
        negative_words = config.get("negative_words", ["bad", "terrible", "awful", "sad"])

        class SentimentClassifier(ClassifierImplementation):
            def classify(self, text: str) -> ClassificationResult:
                text_lower = text.lower()
                positive_count = sum(word in text_lower for word in positive_words)
                negative_count = sum(word in text_lower for word in negative_words)

                if positive_count > negative_count:
                    return ClassificationResult(
                        label="positive",
                        confidence=0.8,
                        metadata={"positive_words": positive_count, "negative_words": negative_count}
                    )
                elif negative_count > positive_count:
                    return ClassificationResult(
                        label="negative",
                        confidence=0.8,
                        metadata={"positive_words": positive_count, "negative_words": negative_count}
                    )
                else:
                    return ClassificationResult(
                        label="neutral",
                        confidence=0.6,
                        metadata={"positive_words": positive_count, "negative_words": negative_count}
                    )

            async def classify_async(self, text: str) -> ClassificationResult:
                return self.classify(text)

        return SentimentClassifier()
```
