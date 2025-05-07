# Composition Over Inheritance Implementation Plan for Classifiers

This document outlines the plan for implementing the Composition Over Inheritance pattern in the Sifaka classifier system.

## Current Architecture

Currently, the classifier system uses inheritance:
- `BaseClassifier` is an abstract base class that provides common functionality
- Specific classifiers like `ToxicityClassifier`, `BiasDetector`, etc. inherit from `BaseClassifier`
- The adapter system already uses composition to adapt classifiers to work as validators

## Target Architecture

We'll refactor the classifier system to use composition over inheritance:

1. Create a `ClassifierImplementation` protocol that defines the core classification logic
2. Create a `Classifier` class that delegates to a `ClassifierImplementation`
3. Create specific implementations like `ToxicityClassifierImplementation` that follow the protocol
4. Update factory functions to create classifiers with their implementations

## Implementation Status

| Classifier | Implementation Status | Factory Function Updated | Notes |
|------------|----------------------|-------------------------|-------|
| NERClassifier | ✅ Completed | ✅ Completed | Already using composition pattern |
| ToxicityClassifier | ✅ Completed | ✅ Completed | Implemented with batch_classify_impl support |
| BiasDetector | ✅ Completed | ✅ Completed | Implemented with batch_classify_impl and fit_impl support |
| SpamClassifier | ❌ Pending | ❌ Pending | |
| ProfanityClassifier | ❌ Pending | ❌ Pending | |
| SentimentClassifier | ❌ Pending | ❌ Pending | |
| ReadabilityClassifier | ❌ Pending | ❌ Pending | |
| LanguageClassifier | ❌ Pending | ❌ Pending | |
| TopicClassifier | ❌ Pending | ❌ Pending | |
| GenreClassifier | ❌ Pending | ❌ Pending | |

## Implementation Steps

### 1. ClassifierImplementation Protocol

The `ClassifierImplementation` protocol has already been added to `base.py`:

```python
@runtime_checkable
class ClassifierImplementation(Protocol[T, R]):
    """
    Protocol for classifier implementations.

    This protocol defines the core classification logic that can be composed with
    the Classifier class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.
    """

    def classify_impl(self, text: T) -> "ClassificationResult[R]": ...
    def warm_up_impl(self) -> None: ...
```

### 2. Classifier Class

The `Classifier` class has already been added to `base.py`:

```python
class Classifier(BaseModel, Generic[T, R]):
    """
    Classifier that uses composition over inheritance.

    This class delegates classification to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    name: str = Field(description="Name of the classifier", min_length=1)
    description: str = Field(description="Description of the classifier", min_length=1)
    config: ClassifierConfig[T]
    _implementation: ClassifierImplementation[T, R] = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        config: ClassifierConfig[T],
        implementation: ClassifierImplementation[T, R],
        **kwargs: Any,
    ):
        """Initialize the classifier."""
        super().__init__(name=name, description=description, config=config, **kwargs)
        self._implementation = implementation

        # Initialize cache if needed
        if self.config.cache_size > 0:
            self._result_cache: Dict[str, ClassificationResult[R]] = {}
```

### 3. Implementation Plan for Each Classifier

For each classifier, we need to:

1. Create an implementation class that follows the `ClassifierImplementation` protocol
2. Move the core logic from the classifier to the implementation class
3. Update the factory function to create a `Classifier` with the implementation

#### 3.1 ToxicityClassifier Implementation

```python
class ToxicityClassifierImplementation:
    """Implementation of toxicity classification logic."""

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_toxic",
    ]
    DEFAULT_COST: ClassVar[int] = 2
    DEFAULT_SEVERE_TOXIC_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_THREAT_THRESHOLD: ClassVar[float] = 0.7
    DEFAULT_GENERAL_THRESHOLD: ClassVar[float] = 0.5

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement toxicity classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.2 BiasDetector Implementation

```python
class BiasDetectorImplementation:
    """Implementation of bias detection logic."""

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.5

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement bias detection logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.3 SpamClassifier Implementation

```python
class SpamClassifierImplementation:
    """Implementation of spam classification logic."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement spam classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.4 ProfanityClassifier Implementation

```python
class ProfanityClassifierImplementation:
    """Implementation of profanity classification logic."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement profanity classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.5 SentimentClassifier Implementation

```python
class SentimentClassifierImplementation:
    """Implementation of sentiment classification logic."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement sentiment classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.6 ReadabilityClassifier Implementation

```python
class ReadabilityClassifierImplementation:
    """Implementation of readability classification logic."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement readability classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.7 LanguageClassifier Implementation

```python
class LanguageClassifierImplementation:
    """Implementation of language classification logic."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement language classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.8 TopicClassifier Implementation

```python
class TopicClassifierImplementation:
    """Implementation of topic classification logic."""

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.0

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement topic classification logic."""
        # Implementation from current _classify_impl
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

#### 3.9 GenreClassifier Implementation

```python
class GenreClassifierImplementation:
    """Implementation of genre classification logic."""

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.0

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False

    def classify_impl(self, text: str) -> ClassificationResult[str]:
        """Implement genre classification logic."""
        # Implementation from current _classify_impl_uncached
        # ...

    def warm_up_impl(self) -> None:
        """Prepare resources for classification."""
        # Implementation from current warm_up
        # ...
```

### 4. Update Factory Functions

For each classifier, update the factory function to create a `Classifier` with the implementation:

```python
def create_toxicity_classifier(
    model_name: str = "original",
    name: str = "toxicity_classifier",
    description: str = "Detects toxic content using Detoxify",
    general_threshold: float = 0.5,
    severe_toxic_threshold: float = 0.7,
    threat_threshold: float = 0.7,
    cache_size: int = 0,
    min_confidence: float = 0.0,
    cost: int = 2,
    **kwargs: Any,
) -> Classifier[str, str]:
    """Factory function to create a toxicity classifier."""
    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "model_name": model_name,
            "general_threshold": general_threshold,
            "severe_toxic_threshold": severe_toxic_threshold,
            "threat_threshold": threat_threshold,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=ToxicityClassifierImplementation.DEFAULT_LABELS,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = ToxicityClassifierImplementation(config)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
```

## Files to Modify

1. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/toxicity.py`
   - Add `ToxicityClassifierImplementation` class
   - Update `create_toxicity_classifier` function

2. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/bias.py`
   - Add `BiasDetectorImplementation` class
   - Update `create_bias_detector` function

3. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/spam.py`
   - Add `SpamClassifierImplementation` class
   - Update `create_spam_classifier` function

4. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/profanity.py`
   - Add `ProfanityClassifierImplementation` class
   - Update `create_profanity_classifier` function

5. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/sentiment.py`
   - Add `SentimentClassifierImplementation` class
   - Update `create_sentiment_classifier` function

6. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/readability.py`
   - Add `ReadabilityClassifierImplementation` class
   - Update `create_readability_classifier` function

7. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/language.py`
   - Add `LanguageClassifierImplementation` class
   - Update `create_language_classifier` function

8. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/topic.py`
   - Add `TopicClassifierImplementation` class
   - Update `create_topic_classifier` function

9. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/classifiers/genre.py`
   - Add `GenreClassifierImplementation` class
   - Update `create_genre_classifier` function

## Implementation Strategy

1. Start with the ToxicityClassifier as it's a commonly used classifier
2. Implement one classifier at a time, following the pattern:
   - Create the implementation class
   - Update the factory function
   - Test the changes
3. Update the implementation status in this document as each classifier is completed

## Benefits

- Reduced complexity by avoiding deep inheritance hierarchies
- Improved flexibility by allowing components to be combined in different ways
- Better testability by enabling testing of implementations in isolation
- Reduced coupling between components
- More consistent API across all classifiers

## Next Steps After Implementation

1. Update the adapter system to work with the new classifier pattern:
   - Ensure ClassifierAdapter works with the new Classifier class
   - Update any code that depends on the old BaseClassifier

2. Update tests to work with the new pattern:
   - Update test fixtures to create classifiers using the new pattern
   - Update assertions to work with the new classifier structure

3. Update documentation to reflect the new pattern:
   - Update docstrings to describe the new architecture
   - Update examples to use the new factory functions
   - Add implementation notes about the Composition Over Inheritance pattern
