# Classifiers Guide

This guide covers how to use Sifaka's built-in classifiers for text analysis and content validation.

## Overview

Sifaka provides several pre-built classifiers that can analyze text for various properties like toxicity, sentiment, bias, and more. These classifiers can be used standalone for analysis or integrated with validators for automated content filtering.

## Available Classifiers

### ToxicityClassifier

Detects toxic language including hate speech, threats, and abuse.

```python
from sifaka.classifiers import ToxicityClassifier, create_toxicity_validator

# Standalone usage
classifier = ToxicityClassifier()
result = classifier.classify("This is a sample text.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# As validator
validator = create_toxicity_validator(threshold=0.8)
```

### SentimentClassifier

Analyzes sentiment (positive, negative, neutral).

```python
from sifaka.classifiers import SentimentClassifier, create_sentiment_validator

classifier = SentimentClassifier()
result = classifier.classify("I love this framework!")
print(f"Sentiment: {result.label}")

# Validate for specific sentiment
validator = create_sentiment_validator(target_sentiment="positive", threshold=0.7)
```

### BiasClassifier

Detects potential bias in text using machine learning.

```python
from sifaka.classifiers import BiasClassifier, create_bias_validator

classifier = BiasClassifier()
result = classifier.classify("This text might contain bias.")

validator = create_bias_validator(threshold=0.7)
```

### LanguageClassifier

Detects the language of text.

```python
from sifaka.classifiers import LanguageClassifier, create_language_validator

classifier = LanguageClassifier()
result = classifier.classify("Hello world")  # Returns "en"

# Validate for specific language
validator = create_language_validator(expected_language="en")
```

### ProfanityClassifier

Detects profanity and inappropriate language.

```python
from sifaka.classifiers import ProfanityClassifier, create_profanity_validator

classifier = ProfanityClassifier()
validator = create_profanity_validator(threshold=0.8)
```

### SpamClassifier

Detects spam content.

```python
from sifaka.classifiers import SpamClassifier, create_spam_validator

classifier = SpamClassifier()
validator = create_spam_validator(threshold=0.8)
```

## Using Classifiers with Chains

### Basic Integration

```python
from sifaka import Chain
from sifaka.models import create_model
from sifaka.classifiers import create_toxicity_validator, create_sentiment_validator

model = create_model("openai:gpt-4")
chain = Chain(model=model, prompt="Write a positive product review")

# Add classifier-based validators
toxicity_validator = create_toxicity_validator(threshold=0.8)
sentiment_validator = create_sentiment_validator(target_sentiment="positive", threshold=0.7)

chain.validate_with(toxicity_validator).validate_with(sentiment_validator)

result = chain.run()
```

### Multiple Classifiers

```python
from sifaka.classifiers import (
    create_toxicity_validator,
    create_sentiment_validator,
    create_bias_validator,
    create_profanity_validator
)

# Create multiple validators
validators = [
    create_toxicity_validator(threshold=0.8),
    create_sentiment_validator(target_sentiment="positive", threshold=0.7),
    create_bias_validator(threshold=0.6),
    create_profanity_validator(threshold=0.9)
]

# Add all validators to chain
for validator in validators:
    chain.validate_with(validator)
```

## Custom Classifier Integration

You can integrate any classifier with Sifaka using the generic classifier validator:

```python
from sifaka.validators.classifier import create_classifier_validator
from sifaka.classifiers import BiasClassifier

# Use any classifier with the generic validator factory
bias_classifier = BiasClassifier()
custom_validator = create_classifier_validator(bias_classifier, threshold=0.6)

chain.validate_with(custom_validator)
```

## Caching for Performance

Some classifiers support caching to improve performance:

```python
from sifaka.classifiers import CachedToxicityClassifier, create_cached_toxicity_validator

# Use cached version for better performance
cached_classifier = CachedToxicityClassifier(cache_size=1000)
cached_validator = create_cached_toxicity_validator(threshold=0.8, cache_size=1000)
```

## Installation

To use classifiers, install Sifaka with the classifiers extra:

```bash
pip install sifaka[classifiers]
```

This installs optional dependencies like scikit-learn for machine learning-based classifiers.

## Configuration

### Thresholds

All classifiers support configurable thresholds:

- **Lower thresholds** (e.g., 0.3): More sensitive, may have false positives
- **Higher thresholds** (e.g., 0.8): Less sensitive, more confident predictions

### Fallback Behavior

Most classifiers have rule-based fallbacks when ML dependencies aren't available:

```python
# Will use ML if scikit-learn is available, otherwise rule-based
classifier = ToxicityClassifier()
```

## Best Practices

1. **Choose appropriate thresholds** based on your use case
2. **Use caching** for repeated text analysis
3. **Combine multiple classifiers** for comprehensive content filtering
4. **Test with your specific content** to tune thresholds
5. **Monitor performance** and adjust as needed

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the classifiers dependencies:

```bash
pip install sifaka[classifiers]
```

### Performance Issues

For better performance with repeated classifications:

1. Use cached classifiers
2. Increase cache size for frequently analyzed text
3. Consider batch processing for large volumes

### Accuracy Issues

To improve classification accuracy:

1. Tune thresholds based on your data
2. Use multiple classifiers for consensus
3. Consider domain-specific training data if available
