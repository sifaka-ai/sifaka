"""
Classifier Configuration Module

This module imports standardized configuration classes from utils/config.py and
extends them with classifier-specific functionality.

## Components
1. **ClassifierConfig**: Configuration for classifiers
2. **ImplementationConfig**: Configuration for classifier implementations

## Usage Examples
```python
from sifaka.utils.config import ClassifierConfig, standardize_classifier_config

# Create classifier configuration
config = ClassifierConfig(
    name="sentiment_classifier",
    description="Classifies text sentiment",
    cache_size=100,
    min_confidence=0.7,
    labels=["positive", "negative", "neutral"],
    params={
        "threshold": 0.8,
        "use_fallback": True,
    }
)

# Create classifier with configuration
classifier = Classifier(
    implementation=implementation,
    config=config
)

# Update configuration
updated_config = config.with_options(min_confidence=0.5)
classifier.update_config(updated_config)

# Standardize configuration
std_config = standardize_classifier_config(
    min_confidence=0.7,
    labels=["positive", "negative", "neutral"],
    params={"threshold": 0.8}
)
```
"""

from pydantic import Field

from sifaka.utils.config import BaseConfig


class ImplementationConfig(BaseConfig):
    """Configuration for classifier implementations."""

    timeout: float = Field(
        default=10.0, ge=0.0, description="Timeout for classification operations in seconds"
    )
    fallback_label: str = Field(
        default="unknown", description="Fallback label to use when classification fails"
    )
    fallback_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score for fallback label"
    )
