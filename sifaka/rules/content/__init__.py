"""
Content validation rules for Sifaka.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use the following modules instead:

   - :mod:`sifaka.rules.content.prohibited` for prohibited content validation
   - :mod:`sifaka.rules.content.tone` for tone consistency validation
   - :mod:`sifaka.rules.content.safety` for safety validation
   - :mod:`sifaka.rules.content.sentiment` for sentiment analysis

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.content import ProhibitedContentRule, ToneConsistencyRule
   - New: from sifaka.rules.content.prohibited import ProhibitedContentRule
         from sifaka.rules.content.tone import ToneConsistencyRule
         from sifaka.rules.content.safety import ToxicityRule, BiasRule, HarmfulContentRule  # Safety module
         from sifaka.rules.content.sentiment import SentimentRule, EmotionalContentRule  # Sentiment module

2. Update configuration:
   - Each module has its own set of parameters and validation logic
   - See the respective module documentation for details

Example:
    Old code:
    >>> from sifaka.rules.content import ProhibitedContentRule
    >>> rule = ProhibitedContentRule()

    New code:
    >>> from sifaka.rules.content.prohibited import ProhibitedContentRule
    >>> rule = ProhibitedContentRule()
"""

import warnings

# Re-export classes for backward compatibility
from sifaka.rules.content.base import (
    ContentAnalyzer,
    ContentValidator,
    DefaultContentAnalyzer,
    DefaultToneAnalyzer,
    ToneAnalyzer,
)
from sifaka.rules.content.prohibited import (
    DefaultProhibitedContentValidator,
    ProhibitedContentRule,
    ProhibitedContentValidator,
    ProhibitedTerms,
    create_prohibited_content_rule,
)
from sifaka.rules.content.tone import (
    DefaultToneValidator,
    ToneConsistencyRule,
    ToneConsistencyValidator,
    ToneIndicators,
    create_tone_consistency_rule,
)

# New imports from safety module
from sifaka.rules.content.safety import (
    BiasCategories,
    BiasRule,
    BiasValidator,
    DefaultBiasValidator,
    DefaultHarmfulContentValidator,
    DefaultToxicityValidator,
    HarmfulCategories,
    HarmfulContentRule,
    HarmfulContentValidator,
    ToxicityIndicators,
    ToxicityRule,
    ToxicityValidator,
    create_bias_rule,
    create_harmful_content_rule,
    create_toxicity_rule,
)

# New imports from sentiment module
from sifaka.rules.content.sentiment import (
    DEFAULT_EMOTION_CATEGORIES,
    DEFAULT_NEGATIVE_WORDS,
    DEFAULT_POSITIVE_WORDS,
    DefaultEmotionalContentValidator,
    DefaultSentimentValidator,
    EmotionalContentConfig,
    EmotionalContentRule,
    EmotionalContentValidator,
    EmotionCategories,
    SentimentConfig,
    SentimentRule,
    SentimentValidator,
    SentimentWords,
    create_emotional_content_rule,
    create_sentiment_rule,
)

warnings.warn(
    "The content module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.content.prohibited, sifaka.rules.content.tone, "
    "sifaka.rules.content.safety, and sifaka.rules.content.sentiment instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Export public classes and functions
__all__ = [
    # Original exports
    "ProhibitedContentRule",
    "DefaultProhibitedContentValidator",
    "ToneConsistencyRule",
    "DefaultToneValidator",
    "create_prohibited_content_rule",
    "create_tone_consistency_rule",
    "ContentAnalyzer",
    "ToneAnalyzer",
    "ContentValidator",
    "ProhibitedContentValidator",
    "ToneConsistencyValidator",
    "ProhibitedTerms",
    "ToneIndicators",
    "DefaultContentAnalyzer",
    "DefaultToneAnalyzer",
    # Safety module exports
    "ToxicityRule",
    "ToxicityIndicators",
    "ToxicityValidator",
    "DefaultToxicityValidator",
    "BiasRule",
    "BiasCategories",
    "BiasValidator",
    "DefaultBiasValidator",
    "HarmfulContentRule",
    "HarmfulCategories",
    "HarmfulContentValidator",
    "DefaultHarmfulContentValidator",
    "create_toxicity_rule",
    "create_bias_rule",
    "create_harmful_content_rule",
    # Sentiment module exports
    "SentimentRule",
    "SentimentConfig",
    "SentimentWords",
    "SentimentValidator",
    "DefaultSentimentValidator",
    "EmotionalContentRule",
    "EmotionalContentConfig",
    "EmotionCategories",
    "EmotionalContentValidator",
    "DefaultEmotionalContentValidator",
    "create_sentiment_rule",
    "create_emotional_content_rule",
    "DEFAULT_POSITIVE_WORDS",
    "DEFAULT_NEGATIVE_WORDS",
    "DEFAULT_EMOTION_CATEGORIES",
]
