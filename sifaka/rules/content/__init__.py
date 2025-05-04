"""
Content validation rules for Sifaka.

This module provides a collection of content validation rules for text analysis.
Each submodule focuses on a specific aspect of content validation:

- :mod:`sifaka.rules.content.prohibited`: Prohibited content detection
- :mod:`sifaka.rules.content.tone`: Tone consistency validation
- :mod:`sifaka.rules.content.safety`: Safety-related content validation
- :mod:`sifaka.rules.content.sentiment`: Sentiment and emotional content analysis
- :mod:`sifaka.rules.content.language`: Language validation

The base module (:mod:`sifaka.rules.content.base`) provides common interfaces and
implementations for content analysis and validation.

Example:
    >>> from sifaka.rules.content.prohibited import create_prohibited_content_rule
    >>> from sifaka.rules.content.tone import create_tone_consistency_rule
    >>> from sifaka.rules.content.safety import create_toxicity_rule, create_bias_rule
    >>> from sifaka.rules.content.sentiment import create_sentiment_rule
    >>> from sifaka.rules.content.language import create_language_rule

    >>> # Create rules
    >>> prohibited_rule = create_prohibited_content_rule(
    ...     terms=["inappropriate", "offensive"],
    ...     threshold=0.5
    ... )
    >>> tone_rule = create_tone_consistency_rule(
    ...     expected_tone="formal",
    ...     threshold=0.8
    ... )
    >>> toxicity_rule = create_toxicity_rule(threshold=0.7)
    >>> bias_rule = create_bias_rule(threshold=0.6)
    >>> sentiment_rule = create_sentiment_rule(threshold=0.5)

    >>> # Create a language rule
    >>> language_rule = create_language_rule(allowed_languages=["en"])

    >>> # Validate text
    >>> text = "This is a formal and appropriate message."
    >>> prohibited_result = prohibited_rule.validate(text)
    >>> tone_result = tone_rule.validate(text)
    >>> toxicity_result = toxicity_rule.validate(text)
    >>> bias_result = bias_rule.validate(text)
    >>> sentiment_result = sentiment_rule.validate(text)
    >>> language_result = language_rule.validate(text)
"""

from sifaka.rules.content.base import (
    ContentAnalyzer,
    ContentValidator,
    DefaultContentAnalyzer,
    DefaultToneAnalyzer,
    ToneAnalyzer,
    IndicatorAnalyzer,
    CategoryAnalyzer,
    ContentAnalysis,
    ToneAnalysis,
)

from sifaka.rules.content.prohibited import (
    create_prohibited_content_rule,
)

from sifaka.rules.content.tone import (
    create_tone_consistency_rule,
    create_tone_consistency_validator,
)

from sifaka.rules.content.safety import (
    create_toxicity_rule,
    create_bias_rule,
    create_harmful_content_rule,
)

from sifaka.rules.content.sentiment import (
    create_sentiment_rule,
)

from sifaka.rules.content.language import (
    create_language_rule,
    create_language_validator,
)


__all__ = [
    # Base classes and protocols
    "ContentAnalyzer",
    "ContentValidator",
    "DefaultContentAnalyzer",
    "DefaultToneAnalyzer",
    "ToneAnalyzer",
    "IndicatorAnalyzer",
    "CategoryAnalyzer",
    "ContentAnalysis",
    "ToneAnalysis",
    # Prohibited content
    "create_prohibited_content_rule",
    # Tone consistency
    "create_tone_consistency_rule",
    "create_tone_consistency_validator",
    # Safety
    "create_toxicity_rule",
    "create_bias_rule",
    "create_harmful_content_rule",
    # Sentiment
    "create_sentiment_rule",
    # Language
    "create_language_rule",
    "create_language_validator",
]
