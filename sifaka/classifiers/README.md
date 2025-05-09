# Sifaka Classifiers

This package provides a collection of text classifiers that analyze content for various characteristics. Each classifier follows a consistent interface and can be used independently or integrated with rules.

## Directory Structure

```
classifiers/
├── __init__.py           # Public API
├── base.py               # Base classifier classes
├── config.py             # Configuration classes
├── factories.py          # Factory functions
├── models.py             # Data models
├── interfaces/           # Protocol interfaces
│   ├── __init__.py
│   └── classifier.py     # Classifier protocols
├── managers/             # State management
│   ├── __init__.py
│   └── state.py          # State manager
├── strategies/           # Strategy implementations
│   ├── __init__.py
│   └── caching.py        # Caching strategy
└── implementations/      # Concrete classifier implementations
    ├── __init__.py
    ├── content/          # Content analysis classifiers
    │   ├── __init__.py
    │   ├── bias.py       # Bias detection
    │   ├── profanity.py  # Profanity detection
    │   ├── sentiment.py  # Sentiment analysis
    │   ├── spam.py       # Spam detection
    │   └── toxicity.py   # Toxicity detection
    ├── properties/       # Text properties classifiers
    │   ├── __init__.py
    │   ├── genre.py      # Genre classification
    │   ├── language.py   # Language detection
    │   ├── readability.py # Readability analysis
    │   └── topic.py      # Topic classification
    └── entities/         # Entity analysis classifiers
        ├── __init__.py
        └── ner.py        # Named entity recognition
```

## Architecture

The classifiers package follows a layered architecture:

1. **Public API**: The `__init__.py` file exports all public classes and functions
2. **Base Classes**: The `base.py` file provides abstract base classes
3. **Interfaces**: The `interfaces/` directory defines protocol interfaces
4. **Managers**: The `managers/` directory provides state management components
5. **Strategies**: The `strategies/` directory provides strategy implementations
6. **Implementations**: The `implementations/` directory contains concrete classifier implementations

## Usage

### Using Factory Functions (Recommended)

```python
from sifaka.classifiers import create_sentiment_classifier, create_toxicity_classifier

# Create classifiers using factory functions
sentiment = create_sentiment_classifier(
    positive_threshold=0.1,
    negative_threshold=-0.1,
    cache_size=100
)

# Analyze text
sentiment_result = sentiment.classify("This is fantastic!")
print(f"Sentiment: {sentiment_result.label}, Confidence: {sentiment_result.confidence:.2f}")
print(f"Compound score: {sentiment_result.metadata['compound_score']:.2f}")
```

### Using the Generic Factory Function

```python
from sifaka.classifiers import create_classifier_by_name

# Create classifiers using the generic factory function
sentiment = create_classifier_by_name(
    name="sentiment",
    positive_threshold=0.1,
    negative_threshold=-0.1,
    cache_size=100
)

toxicity = create_classifier_by_name(
    name="toxicity",
    general_threshold=0.5,
    cache_size=100
)

# Analyze text
result = toxicity.classify("This is a test.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
```

### Using with Rules

```python
from sifaka.classifiers import create_sentiment_classifier
from sifaka.adapters.classifier import create_classifier_rule

# Create a classifier
sentiment = create_sentiment_classifier(
    positive_threshold=0.1,
    negative_threshold=-0.1,
    cache_size=100
)

# Create a rule using the classifier
sentiment_rule = create_classifier_rule(
    classifier=sentiment,
    name="sentiment_rule",
    description="Ensures text has positive sentiment",
    threshold=0.6,
    valid_labels=["positive"]
)

# Apply the rule
result = sentiment_rule.apply("This is fantastic!")
print(f"Rule passed: {result.passed}")
print(f"Confidence: {result.confidence:.2f}")
```

## Available Classifiers

### Content Analysis

- **SentimentClassifier**: Analyzes text sentiment (positive/negative/neutral)
- **ProfanityClassifier**: Detects profane or inappropriate language
- **ToxicityClassifier**: Identifies toxic content
- **SpamClassifier**: Detects spam content in text
- **BiasDetector**: Identifies various forms of bias in text

### Text Properties

- **ReadabilityClassifier**: Evaluates reading difficulty level
- **LanguageClassifier**: Identifies the language of text
- **TopicClassifier**: Identifies topics in text using LDA
- **GenreClassifier**: Categorizes text into genres (news, fiction, academic, etc.)

### Entity Analysis

- **NERClassifier**: Identifies named entities (people, organizations, locations, etc.)
