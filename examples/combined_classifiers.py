"""
Example of using multiple classifiers with Sifaka.

This example demonstrates:
1. Creating custom classifiers for content analysis
2. Using critics for content reflection and improvement
3. Using pattern rules for structural analysis
4. Combining multiple validation strategies
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import Field

from sifaka.integrations.langchain import (
    ChainConfig,
    ChainValidator,
    wrap_chain,
)
from sifaka.models import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.classifiers.base import BaseClassifier, ClassificationResult
from sifaka.rules import Rule, RuleResult, SymmetryRule, RepetitionRule
from sifaka.rules.pattern_rules import SymmetryConfig, RepetitionConfig
from sifaka.critics import PromptCritic
from sifaka.critics.prompt import PromptCriticConfig
from sifaka.core import Chain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadabilityClassifier(BaseClassifier):
    """Classifier that analyzes text readability."""

    thresholds: Dict[str, int] = Field(
        default_factory=lambda: {"easy": 10, "moderate": 15, "complex": 20}
    )

    def __init__(
        self,
        name: str = "readability",
        description: str = "Analyzes text readability",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the readability classifier."""
        config = config or {}
        config["labels"] = ["easy", "moderate", "complex", "unknown"]
        super().__init__(name=name, description=description, config=config)

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate a simple readability score based on sentence length."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return 0.0

        words_per_sentence = [len(s.split()) for s in sentences]
        return sum(words_per_sentence) / len(words_per_sentence)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Classify text readability."""
        try:
            score = self._calculate_readability_score(text)

            if score <= self.thresholds["easy"]:
                label = "easy"
            elif score <= self.thresholds["moderate"]:
                label = "moderate"
            else:
                label = "complex"

            return ClassificationResult(
                label=label,
                confidence=0.8,
                metadata={"score": score, "thresholds": self.thresholds},
            )
        except Exception as e:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e)},
            )


class ToneClassifier(BaseClassifier):
    """Classifier that analyzes text tone."""

    tone_indicators: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "formal": ["therefore", "furthermore", "consequently", "however"],
            "casual": ["yeah", "cool", "awesome", "basically"],
            "technical": ["algorithm", "implementation", "function", "parameter"],
        }
    )

    def __init__(
        self,
        name: str = "tone",
        description: str = "Analyzes text tone",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the tone classifier."""
        config = config or {}
        config["labels"] = ["formal", "casual", "technical", "neutral", "unknown"]
        super().__init__(name=name, description=description, config=config)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Classify text tone."""
        try:
            text = text.lower()
            scores = {
                tone: sum(1 for word in indicators if word in text)
                for tone, indicators in self.tone_indicators.items()
            }

            max_score = max(scores.values())
            if max_score == 0:
                return ClassificationResult(
                    label="neutral",
                    confidence=0.5,
                    metadata={"scores": scores},
                )

            dominant_tone = max(scores.items(), key=lambda x: x[1])[0]
            confidence = max_score / (sum(scores.values()) + 1)

            return ClassificationResult(
                label=dominant_tone,
                confidence=confidence,
                metadata={"scores": scores},
            )
        except Exception as e:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"error": str(e)},
            )


@dataclass
class CombinedValidator(ChainValidator[str]):
    """Validator that combines multiple classifiers."""

    readability_classifier: ReadabilityClassifier
    tone_classifier: ToneClassifier
    required_tone: str = "formal"
    max_complexity: str = "moderate"

    def validate(self, output: str) -> RuleResult:
        """Validate output using both classifiers."""
        # Check readability
        readability = self.readability_classifier.classify(output)
        if readability.label == "unknown":
            return RuleResult(
                passed=False,
                message="Failed to analyze readability",
                metadata=readability.metadata,
            )

        if readability.label == "complex":
            return RuleResult(
                passed=False,
                message=f"Text is too complex (score: {readability.metadata['score']})",
                metadata=readability.metadata,
            )

        # Check tone
        tone = self.tone_classifier.classify(output)
        if tone.label == "unknown":
            return RuleResult(
                passed=False,
                message="Failed to analyze tone",
                metadata=tone.metadata,
            )

        if tone.label != self.required_tone:
            return RuleResult(
                passed=False,
                message=f"Tone is {tone.label}, expected {self.required_tone}",
                metadata=tone.metadata,
            )

        return RuleResult(
            passed=True,
            message="Content meets readability and tone requirements",
            metadata={
                "readability": readability.metadata,
                "tone": tone.metadata,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if the validator can handle the output."""
        return isinstance(output, str)


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the model provider with configuration
    config = ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=2000,
    )

    model = AnthropicProvider(
        model_name="claude-3-haiku-20240307",
        config=config,
    )

    # Create classifiers
    readability_classifier = ReadabilityClassifier()
    tone_classifier = ToneClassifier()

    # Create pattern detection rules
    symmetry_rule = SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=SymmetryConfig(mirror_mode="both", symmetry_threshold=0.8),
    )

    repetition_rule = RepetitionRule(
        name="repetition_check",
        description="Detects repetitive patterns",
        config=RepetitionConfig(pattern_type="repeat", pattern_length=3),
    )

    # Create the chain with all components
    chain = Chain(
        model=model,
        rules=[readability_classifier, tone_classifier, symmetry_rule, repetition_rule],
        critic=PromptCritic(
            name="content_critic",
            description="Provides feedback on content quality",
            config=PromptCriticConfig(
                prompt_template="""
                Please provide a critique of the following content:
                {content}

                Requirements:
                1. Use formal language and academic tone
                2. Keep sentences concise and clear
                3. Maintain consistent structure
                4. Use appropriate technical terms
                5. Avoid colloquialisms and casual language
                """,
                input_variable="content",
            ),
        ),
        max_attempts=3,
    )

    # Example topics
    topics = [
        "The role of quantum computing in cryptography",
        "The impact of artificial intelligence on healthcare",
        "The principles of sustainable urban development",
    ]

    # Process each topic
    for topic in topics:
        logger.info("\nProcessing topic: %s", topic)
        try:
            # Generate and validate content
            result = chain.run(topic)
            logger.info("\nGenerated content:")
            logger.info(result)

            # Log validation results
            logger.info("\nValidation Results:")
            for rule_result in result.rule_results:
                if isinstance(rule_result, ClassificationResult):
                    # Handle classifier results
                    logger.info("\n%s Classification:", rule_result.name)
                    logger.info("- Label: %s", rule_result.label)
                    logger.info("- Confidence: %.2f", rule_result.confidence)
                    if rule_result.metadata:
                        logger.info("- Details: %s", rule_result.metadata)
                else:
                    # Handle pattern rule results
                    logger.info("\n%s Analysis:", rule_result.name)
                    logger.info("- Passed: %s", rule_result.passed)
                    logger.info("- Message: %s", rule_result.message)
                    if rule_result.metadata:
                        logger.info("- Details: %s", rule_result.metadata)

            # Log critic's feedback if available
            if hasattr(result, "critique_details"):
                logger.info("\nCritic's Analysis:")
                logger.info("- Score: %s", result.critique_details.get("score"))
                logger.info("- Feedback: %s", result.critique_details.get("feedback"))
                if result.critique_details.get("issues"):
                    logger.info("- Issues:")
                    for issue in result.critique_details["issues"]:
                        logger.info("  * %s", issue)
                if result.critique_details.get("suggestions"):
                    logger.info("- Suggestions:")
                    for suggestion in result.critique_details["suggestions"]:
                        logger.info("  * %s", suggestion)

        except ValueError as e:
            logger.error("Validation failed: %s", str(e))


if __name__ == "__main__":
    main()
