# Sifaka Classifiers

The Sifaka Classifiers module provides a flexible and extensible framework for text classification. It supports various classification tasks such as sentiment analysis, toxicity detection, and content moderation.

## Overview

The classifiers system is designed to be:
- **Modular**: Easy to add new classifier implementations
- **Configurable**: Flexible configuration options
- **Extensible**: Support for different classification types
- **Reliable**: Robust error handling and validation

## Architecture

The system follows a layered architecture:
1. **User Interface Layer**: Classifier class with public methods
2. **Engine Layer**: Core classification engine
3. **Implementation Layer**: Concrete classifier implementations
4. **State Management Layer**: State tracking and statistics

## Components

### Classifier

The main user-facing class that provides a simple interface for text classification:

```python
from sifaka.classifiers import Classifier
from sifaka.utils.config.classifiers import ClassifierConfig

# Create classifier with configuration
classifier = Classifier(
    name="sentiment_classifier",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"],
    config=ClassifierConfig(
        cache_enabled=True,
        cache_size=100,
        min_confidence=0.7
    )
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

### Engine

The core classification engine that coordinates the classification process:

```python
from sifaka.classifiers.engine import Engine
from sifaka.utils.state import StateManager
from sifaka.utils.config.classifiers import ClassifierConfig

# Create engine
engine = Engine(
    state_manager=StateManager(),
    config=ClassifierConfig(
        cache_enabled=True,
        cache_size=100,
        min_confidence=0.7
    )
)

# Classify text
result = engine.classify(
    text="This is a friendly message.",
    implementation=implementation
)
```

### Implementations

Concrete classifier implementations that provide the actual classification logic:

```python
from sifaka.classifiers.interfaces import ClassifierImplementation
from sifaka.core.results import ClassificationResult

class SentimentClassifier(ClassifierImplementation):
    def classify(self, text: str) -> ClassificationResult:
        # Implementation-specific classification logic
        return ClassificationResult(
            label="positive",
            confidence=0.8,
            metadata={"source": "sentiment_classifier"}
        )
```

### Adapters

Adapters for integrating existing classifiers with the system:

```python
from sifaka.classifiers.adapters import ImplementationAdapter
from sifaka.models import Model

# Create model
model = Model.create(
    name="sentiment_model",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"]
)

# Create adapter
implementation = ImplementationAdapter(model)

# Use adapter
result = implementation.classify("This is a friendly message.")
```

## Configuration

The system supports configuration through the ClassifierConfig class:

```python
from sifaka.utils.config.classifiers import ClassifierConfig

config = ClassifierConfig(
    cache_enabled=True,  # Enable result caching
    cache_size=100,      # Maximum number of cached results
    min_confidence=0.7   # Minimum confidence threshold
)
```

## Error Handling

The system provides robust error handling:
- ClassifierError: Base class for classifier errors
- ImplementationError: Raised when implementation fails
- ConfigurationError: Raised for invalid configuration

## State Management

The system tracks state and statistics:
- Execution counts
- Success/failure rates
- Timing information
- Error details
- Cache statistics

## Usage Examples

### Basic Classification

```python
from sifaka.classifiers import Classifier
from sifaka.utils.config.classifiers import ClassifierConfig

# Create classifier
classifier = Classifier(
    name="sentiment_classifier",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"],
    config=ClassifierConfig(
        cache_enabled=True,
        cache_size=100,
        min_confidence=0.7
    )
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

### Custom Implementation

```python
from sifaka.classifiers.interfaces import ClassifierImplementation
from sifaka.core.results import ClassificationResult

class CustomClassifier(ClassifierImplementation):
    def classify(self, text: str) -> ClassificationResult:
        # Custom classification logic
        return ClassificationResult(
            label="custom",
            confidence=0.9,
            metadata={"source": "custom_classifier"}
        )

# Create classifier with custom implementation
classifier = Classifier(
    name="custom_classifier",
    description="Custom text classifier",
    labels=["custom"],
    implementation=CustomClassifier()
)

# Classify text
result = classifier.classify("This is a test message.")
```

### Using Adapters

```python
from sifaka.classifiers.adapters import ImplementationAdapter
from sifaka.models import Model

# Create model
model = Model.create(
    name="sentiment_model",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"]
)

# Create adapter
implementation = ImplementationAdapter(model)

# Create classifier with adapted implementation
classifier = Classifier(
    name="sentiment_classifier",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"],
    implementation=implementation
)

# Classify text
result = classifier.classify("This is a friendly message.")
```

## Contributing

To contribute to the Sifaka Classifiers module:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
