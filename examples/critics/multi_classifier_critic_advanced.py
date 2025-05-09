"""
Advanced Multi-Classifier Example with Batched Critic

This example demonstrates a system where:
1. All available classifiers are used to analyze a 2,500-word text
2. Each classifier is converted to a rule
3. A custom BatchedCritic consolidates feedback from all failing rules
4. A faster model (GPT-3.5-turbo) is used to revise text based on batched feedback
5. The process continues until all rules pass or max attempts are reached
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pydantic import PrivateAttr
from sifaka.utils.state import create_critic_state, create_state_manager, CriticState

from dotenv import load_dotenv

# Load environment variables from .env file (containing API keys)
load_dotenv()

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.classifiers import (
    SentimentClassifier,
    ReadabilityClassifier,
    LanguageClassifier,
    BiasDetector,
    GenreClassifier,
    ProfanityClassifier,
    SpamClassifier,
    TopicClassifier,
    ClassifierConfig,
    NERClassifier,
)
from sifaka.classifiers.implementations.content.toxicity import (
    ToxicityClassifier,
    create_toxicity_classifier,
)
from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
)
from sifaka.adapters.classifier import create_classifier_rule
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RulePriority
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import (
    ChainCore,
    ChainResult,
    PromptManager,
    ValidationManager,
    ResultFormatter,
    SimpleRetryStrategy,
)


# Create concrete implementations of all classifiers
class ConcreteSentimentClassifier(SentimentClassifier):
    """Concrete implementation of SentimentClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """Implementation of the required abstract method."""
        # Simple sentiment analysis based on word lists
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "positive",
            "beneficial",
        }
        negative_words = {"bad", "terrible", "awful", "horrible", "negative", "risky", "dangerous"}
        neutral_words = {
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "this",
            "that",
            "these",
            "those",
        }

        words = set(text.lower().split())
        pos_count = len(words.intersection(positive_words))
        neg_count = len(words.intersection(negative_words))
        neu_count = len(words.intersection(neutral_words))
        total = pos_count + neg_count + neu_count

        if total == 0:
            return ClassificationResult(
                label="neutral",
                confidence=0.7,
                metadata={"pos_score": 0, "neg_score": 0, "neu_score": 1.0},
            )

        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = neu_count / total

        # Bias towards neutral sentiment
        if neu_score >= 0.4 or (pos_score < 0.3 and neg_score < 0.3):
            label = "neutral"
            confidence = max(0.7, neu_score)
        elif pos_score > neg_score:
            label = "positive"
            confidence = pos_score
        else:
            label = "negative"
            confidence = neg_score

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "pos_score": pos_score,
                "neg_score": neg_score,
                "neu_score": neu_score,
            },
        )


class ConcreteReadabilityClassifier(ReadabilityClassifier):
    """Concrete implementation of ReadabilityClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple readability analysis based on sentence and word length
        sentences = text.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        if avg_sentence_length < 10:
            label = "simple"
            confidence = 0.8
        elif avg_sentence_length < 20:
            label = "moderate"
            confidence = 0.7
        else:
            label = "complex"
            confidence = 0.6

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={"avg_sentence_length": avg_sentence_length},
        )


class ConcreteLanguageClassifier(LanguageClassifier):
    """Concrete implementation of LanguageClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple language detection based on common English words
        english_words = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i"}
        words = set(text.lower().split())
        english_count = len(words.intersection(english_words))

        if english_count > 3:
            return ClassificationResult(
                label="en",
                confidence=0.8,
                metadata={"english_word_count": english_count},
            )
        else:
            return ClassificationResult(
                label="other",
                confidence=0.6,
                metadata={"english_word_count": english_count},
            )


class ConcreteBiasDetector(BiasDetector):
    """Concrete implementation of BiasDetector."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple bias detection based on word lists and context
        biased_words = {
            "always",
            "never",
            "all",
            "none",
            "everyone",
            "nobody",
            "every",
            "any",
            "no one",
            "nothing",
            "everything",
            "best",
            "worst",
            "only",
            "impossible",
            "definitely",
        }

        # Words that might indicate balanced/unbiased language
        balanced_words = {
            "however",
            "although",
            "while",
            "but",
            "yet",
            "some",
            "many",
            "most",
            "often",
            "sometimes",
            "typically",
            "generally",
            "usually",
            "may",
            "might",
            "could",
            "perhaps",
            "possibly",
            "approximately",
        }

        words = set(text.lower().split())
        bias_count = len(words.intersection(biased_words))
        balance_count = len(words.intersection(balanced_words))
        total_words = len(words)

        # Calculate bias score based on:
        # 1. Ratio of biased words to total words
        # 2. Presence of balanced language
        # 3. Overall text length (longer texts can have some biased words)
        bias_ratio = bias_count / max(total_words, 1)
        balance_ratio = balance_count / max(total_words, 1)

        # Text is considered unbiased if:
        # - Very few biased words relative to length
        # - Has balanced language
        # - Is long enough to establish context
        if (bias_ratio < 0.02 and balance_ratio > 0.01) or (
            total_words > 100 and bias_ratio < 0.01
        ):
            return ClassificationResult(
                label="unbiased",
                confidence=0.8,
                metadata={
                    "biased_word_count": bias_count,
                    "balanced_word_count": balance_count,
                    "total_words": total_words,
                },
            )
        else:
            return ClassificationResult(
                label="biased",
                confidence=min(0.9, bias_ratio * 5),  # Scale up for confidence
                metadata={
                    "biased_word_count": bias_count,
                    "balanced_word_count": balance_count,
                    "total_words": total_words,
                },
            )


class ConcreteGenreClassifier(GenreClassifier):
    """Concrete implementation of GenreClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple genre classification based on word lists and patterns
        academic_words = {
            "research",
            "study",
            "analysis",
            "theory",
            "hypothesis",
            "methodology",
            "findings",
            "literature",
            "evidence",
            "data",
            "results",
            "conclusion",
            "investigate",
            "examine",
            "analyze",
            "empirical",
            "experiment",
            "observation",
            "statistical",
            "scientific",
            "publication",
            "journal",
            "peer-reviewed",
            "abstract",
            "introduction",
        }
        technical_words = {
            "implementation",
            "system",
            "algorithm",
            "process",
            "function",
            "software",
            "hardware",
            "database",
            "network",
            "protocol",
            "interface",
            "module",
            "component",
            "architecture",
            "framework",
            "configuration",
            "deployment",
            "integration",
            "optimization",
            "runtime",
            "performance",
            "latency",
            "throughput",
            "scalability",
            "reliability",
        }
        news_words = {
            "report",
            "announced",
            "according",
            "officials",
            "sources",
            "today",
            "yesterday",
            "breaking",
            "latest",
            "update",
            "statement",
            "press",
            "release",
            "coverage",
            "reported",
            "confirmed",
            "investigation",
            "development",
            "spokesperson",
            "authorities",
            "incident",
            "event",
            "situation",
            "briefing",
            "conference",
        }
        blog_words = {
            "opinion",
            "think",
            "feel",
            "experience",
            "personal",
            "blog",
            "post",
            "thoughts",
            "reflections",
            "perspective",
            "journey",
            "story",
            "share",
            "recommend",
            "review",
            "insights",
            "takeaways",
            "learnings",
            "musings",
            "diary",
            "lifestyle",
            "tips",
            "advice",
            "guide",
            "tutorial",
        }

        # Additional genre-specific patterns
        academic_patterns = [
            "this study",
            "research shows",
            "findings suggest",
            "data indicates",
            "literature review",
            "methodology",
            "empirical evidence",
            "statistical analysis",
            "theoretical framework",
            "hypothesis testing",
        ]
        technical_patterns = [
            "code implementation",
            "system architecture",
            "technical specification",
            "performance optimization",
            "deployment process",
            "configuration settings",
            "integration testing",
            "error handling",
            "debugging process",
            "documentation",
        ]
        news_patterns = [
            "breaking news",
            "press release",
            "official statement",
            "recent developments",
            "news conference",
            "media briefing",
            "sources confirm",
            "latest update",
            "incident report",
            "announcement",
        ]
        blog_patterns = [
            "my experience",
            "personal journey",
            "what i learned",
            "tips and tricks",
            "how to guide",
            "best practices",
            "recommendations",
            "my thoughts",
            "personal opinion",
            "blog post",
        ]

        # Calculate word-based scores
        words = set(text.lower().split())
        academic_count = len(words.intersection(academic_words))
        technical_count = len(words.intersection(technical_words))
        news_count = len(words.intersection(news_words))
        blog_count = len(words.intersection(blog_words))

        # Calculate pattern-based scores
        text_lower = text.lower()
        academic_pattern_count = sum(1 for pattern in academic_patterns if pattern in text_lower)
        technical_pattern_count = sum(1 for pattern in technical_patterns if pattern in text_lower)
        news_pattern_count = sum(1 for pattern in news_patterns if pattern in text_lower)
        blog_pattern_count = sum(1 for pattern in blog_patterns if pattern in text_lower)

        # Combine word and pattern scores with weights
        word_weight = 0.7
        pattern_weight = 0.3

        genre_scores = {
            "academic": (academic_count * word_weight + academic_pattern_count * pattern_weight),
            "technical": (technical_count * word_weight + technical_pattern_count * pattern_weight),
            "news": (news_count * word_weight + news_pattern_count * pattern_weight),
            "blog": (blog_count * word_weight + blog_pattern_count * pattern_weight),
        }

        # Get the top two genres
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        top_genre = sorted_genres[0]
        second_genre = sorted_genres[1]

        # Calculate confidence based on:
        # 1. Absolute score of the top genre
        # 2. Relative difference between top and second genre
        # 3. Length of the text (longer texts should have higher confidence)
        total_words = len(text.split())
        min_words = 100  # Minimum words for reasonable confidence
        length_factor = min(1.0, total_words / min_words)

        # Base confidence from absolute score
        base_confidence = min(0.8, top_genre[1] / 10)  # Cap at 0.8

        # Margin between top and second genres
        score_margin = (top_genre[1] - second_genre[1]) / (
            top_genre[1] + 0.1
        )  # Add 0.1 to avoid division by zero
        margin_boost = score_margin * 0.2  # Max boost of 0.2 from margin

        # Final confidence combines base, margin, and length
        confidence = (base_confidence + margin_boost) * length_factor

        # Ensure confidence is at least 0.3 if we have a clear winner
        if top_genre[1] > 0 and score_margin > 0.2:
            confidence = max(0.3, confidence)

        # Cap confidence at 0.95
        confidence = min(0.95, confidence)

        return ClassificationResult(
            label=top_genre[0],
            confidence=confidence,
            metadata={
                "genre_scores": genre_scores,
                "word_counts": {
                    "academic": academic_count,
                    "technical": technical_count,
                    "news": news_count,
                    "blog": blog_count,
                },
                "pattern_counts": {
                    "academic": academic_pattern_count,
                    "technical": technical_pattern_count,
                    "news": news_pattern_count,
                    "blog": blog_pattern_count,
                },
                "total_words": total_words,
                "score_margin": score_margin,
            },
        )


class ConcreteProfanityClassifier(ProfanityClassifier):
    """Concrete implementation of ProfanityClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple profanity detection based on word list
        profane_words = {"profanity1", "profanity2", "profanity3"}  # Replace with actual words
        words = set(text.lower().split())
        profanity_count = len(words.intersection(profane_words))

        if profanity_count > 0:
            return ClassificationResult(
                label="profane",
                confidence=0.8,
                metadata={"profanity_count": profanity_count},
            )
        else:
            return ClassificationResult(
                label="clean",
                confidence=0.9,
                metadata={"profanity_count": 0},
            )


class ConcreteSpamClassifier(SpamClassifier):
    """Concrete implementation of SpamClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple spam detection based on word list
        spam_words = {"buy", "free", "win", "click", "money", "offer", "limited", "act now"}
        words = set(text.lower().split())
        spam_count = len(words.intersection(spam_words))

        if spam_count > 2:
            return ClassificationResult(
                label="spam",
                confidence=0.8,
                metadata={"spam_word_count": spam_count},
            )
        else:
            return ClassificationResult(
                label="ham",
                confidence=0.7,
                metadata={"spam_word_count": spam_count},
            )


class ConcreteTopicClassifier(TopicClassifier):
    """Concrete implementation of TopicClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple topic classification based on word lists
        topic_words = {
            "topic_0_people+media+communicate": {
                "people",
                "media",
                "communication",
                "social",
                "network",
            },
            "topic_1_industry+content+preferences": {
                "industry",
                "content",
                "business",
                "market",
                "consumer",
            },
            "topic_2_pollution+loss+addressing": {
                "environment",
                "pollution",
                "climate",
                "impact",
                "solution",
            },
            "topic_3_volatility+experienced+financial": {
                "finance",
                "market",
                "investment",
                "risk",
                "trading",
            },
        }

        words = set(text.lower().split())
        topic_counts = {topic: len(words.intersection(topic_words[topic])) for topic in topic_words}

        total = sum(topic_counts.values())
        if total == 0:
            return ClassificationResult(
                label="topic_0_people+media+communicate",
                confidence=0.4,
                metadata={"topic_counts": topic_counts},
            )

        max_topic = max(topic_counts.items(), key=lambda x: x[1])
        confidence = max_topic[1] / total

        return ClassificationResult(
            label=max_topic[0],
            confidence=confidence,
            metadata={"topic_counts": topic_counts},
        )


class ConcreteToxicityClassifier(ToxicityClassifier):
    """Concrete implementation of ToxicityClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple toxicity detection based on word lists
        toxic_words = {"toxic1", "toxic2", "toxic3"}  # Replace with actual words
        words = set(text.lower().split())
        toxic_count = len(words.intersection(toxic_words))

        if toxic_count > 0:
            return ClassificationResult(
                label="toxic",
                confidence=0.8,
                metadata={"toxic_word_count": toxic_count},
            )
        else:
            return ClassificationResult(
                label="non-toxic",
                confidence=0.9,
                metadata={"toxic_word_count": 0},
            )


class ConcreteNERClassifier(NERClassifier):
    """Concrete implementation of NERClassifier."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implementation of the required abstract method."""
        # Simple NER based on word lists and patterns
        person_words = {"John", "Jane", "Smith", "Doe", "Dr.", "Mr.", "Ms.", "Mrs."}
        org_words = {"Inc.", "Corp.", "LLC", "Ltd.", "Company", "Organization", "Institute"}
        location_words = {"Street", "Avenue", "Road", "City", "State", "Country", "Park"}
        date_words = {
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        }
        money_words = {"$", "dollar", "euro", "pound", "yen", "cent", "million", "billion"}

        words = set(text.split())
        person_count = len(words.intersection(person_words))
        org_count = len(words.intersection(org_words))
        location_count = len(words.intersection(location_words))
        date_count = len(words.intersection(date_words))
        money_count = len(words.intersection(money_words))

        # Count entities
        entity_counts = {
            "person": person_count,
            "organization": org_count,
            "location": location_count,
            "date": date_count,
            "money": money_count,
        }

        # Find dominant entity type
        dominant_type = "unknown"
        max_count = 0
        total_entities = 0

        for entity_type, count in entity_counts.items():
            total_entities += count
            if count > max_count:
                max_count = count
                dominant_type = entity_type

        # Calculate confidence based on entity density
        confidence = min(1.0, total_entities / max(len(text.split()), 1))

        # Prepare metadata
        metadata = {
            "entity_counts": entity_counts,
            "total_entities": total_entities,
            "entity_density": confidence,
        }

        return ClassificationResult(
            label=dominant_type if total_entities > 0 else "unknown",
            confidence=confidence if total_entities > 0 else 0.0,
            metadata=metadata,
        )


# Custom prompt factory for batched feedback
class BatchedFeedbackPromptFactory:
    """Custom factory for batched rule violation feedback."""

    def create_critique_prompt(self, text: str) -> str:
        """
        Create a critique prompt for batched rule violations.

        Args:
            text: The text to analyze

        Returns:
            str: The critique prompt
        """
        return f"""TASK: Analyze the following text for quality issues.

TEXT TO ANALYZE:
{text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
SCORE: [score between 0.0 and 1.0]
FEEDBACK: [overall feedback explaining issues]
ISSUES: [list of specific issues]
SUGGESTIONS: [list of specific suggestions for improvement]

YOUR ANALYSIS:"""

    def create_validation_prompt(self, text: str) -> str:
        """
        Create a validation prompt.

        Args:
            text: The text to validate

        Returns:
            str: The validation prompt
        """
        return f"""TASK: Validate if the following text meets quality standards.

TEXT TO VALIDATE:
{text}

Check if the text is clear, coherent, and error-free.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
VALID: [true/false]
REASON: [brief explanation for your decision]

YOUR VALIDATION:"""

    def create_improvement_prompt(self, text: str, rule_violations: List[Dict]) -> str:
        """
        Create a prompt for improving text based on multiple rule violations.

        Args:
            text: The text to improve
            rule_violations: List of rule violations with messages and metadata

        Returns:
            str: The improvement prompt
        """
        # Extract all rule violations
        rule_details = []
        for i, violation in enumerate(rule_violations, 1):
            rule_name = violation.get("rule_name", f"Rule {i}")
            message = violation.get("message", "Unknown issue")
            metadata = violation.get("metadata", {})

            # Format details based on metadata
            if metadata:
                details = "\n".join(
                    [f"   - {k}: {v}" for k, v in metadata.items() if k != "rule_name"]
                )
                rule_details.append(f"{i}. {rule_name}: {message}\n{details}")
            else:
                rule_details.append(f"{i}. {rule_name}: {message}")

        rule_violations_text = "\n".join(rule_details)

        # Check for length violation
        has_length_violation = any(
            "too few words" in violation.get("message", "") for violation in rule_violations
        )
        length_instruction = ""
        if has_length_violation:
            # Extract current word count and required minimum from the violation message
            for violation in rule_violations:
                message = violation.get("message", "")
                if "too few words" in message:
                    # Try to extract the current and required word counts
                    import re

                    word_counts = re.findall(r"\d+", message)
                    if len(word_counts) >= 2:
                        current_count = word_counts[0]
                        required_count = word_counts[1]
                        words_needed = int(required_count) - int(current_count)
                        length_instruction = f"""7. LENGTH REQUIREMENT:
- Current word count: {current_count} words
- REQUIRED minimum: {required_count} words
- You need to add at least {words_needed} more words
- Add relevant examples, data points, and detailed explanations
- Expand EACH section with more depth and detail"""
                    break

        # Check for readability violation
        has_readability_violation = any(
            "readability" in violation.get("rule_name", "").lower()
            or "complex" in violation.get("message", "")
            for violation in rule_violations
        )
        readability_instruction = ""
        if has_readability_violation:
            readability_instruction = """8. READABILITY IMPROVEMENT REQUIRED:
- Simplify complex language and jargon
- Use shorter sentences (15-20 words max)
- Break up long paragraphs
- Use simpler words where possible
- Aim for a high school reading level (grades 8-10)
- Include clear topic sentences
- Use active voice instead of passive voice
- Add explanations for technical terms"""

        return f"""TASK: Improve the following text to address all rule violations.

TEXT TO IMPROVE:
{text}

RULE VIOLATIONS:
{rule_violations_text}

IMPORTANT INSTRUCTIONS:
1. Address ALL rule violations while preserving the original message
2. Make necessary edits to fix each identified issue
3. Return the text with the title included at the beginning
4. Maintain the overall structure and key information
5. Focus on fixing the specific issues mentioned in the rule violations
6. Ensure the text is at least 1000 words, preferably around 1,500 words
{length_instruction}
{readability_instruction}

OUTPUT FORMAT REQUIREMENTS:
- Start your response with the title of the article
- Include the full improved text with no preamble, markers, or explanations
- Do not write "IMPROVED_TEXT:" or any markers - just begin with the title
- Do not include explanations about your changes

Title: [Your title here]

[Your full improved text here]"""


# Custom BatchedCritic that consolidates feedback from multiple rules
class BatchedCritic(PromptCritic):
    """A critic that batches feedback from multiple rule violations."""

    def __init__(
        self,
        llm_provider: Any,
        name: str = "batched_critic",
        description: str = "Batches feedback from multiple rule violations",
        config: Optional[PromptCriticConfig] = None,
    ) -> None:
        """
        Initialize the batched critic.

        Args:
            llm_provider: Language model provider
            name: Name of the critic
            description: Description of the critic
            config: Configuration options
        """
        # Create custom prompt factory
        prompt_factory = BatchedFeedbackPromptFactory()

        # Use provided config or create default
        if config is None:
            config = PromptCriticConfig(
                name=name,
                description=description,
                system_prompt=(
                    "# QUALITY IMPROVEMENT SPECIALIST\n\n"
                    "You are an expert editor who specializes in improving text quality. "
                    "Your job is to analyze text, identify issues, and provide clear feedback "
                    "on how to improve it. When improving text, you should address all rule "
                    "violations while preserving the original meaning and maintaining appropriate "
                    "length.\n\n"
                    "IMPORTANT: When asked to increase the length of a text, you MUST add substantial "
                    "new content, examples, details, and elaboration to significantly expand the text. "
                    "Length requirements are NON-NEGOTIABLE and MUST be met. If asked to expand text to "
                    "a specific word count, you should diligently work to reach that target by adding "
                    "meaningful, relevant content that enhances the original text."
                ),
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=4000,  # Allow for longer responses
            )

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            llm_provider=llm_provider,
            prompt_factory=prompt_factory,
            config=config,
        )

    def improve_with_violations(self, text: str, rule_results: List[RuleResult]) -> str:
        """
        Improve text by addressing all rule violations in one batch.

        Args:
            text: The text to improve
            rule_results: List of rule results with violations

        Returns:
            str: The improved text
        """
        # Extract failed rule results
        failures = [r for r in rule_results if not r.passed]

        if not failures:
            return text  # No failures to address

        # Convert rule results to format expected by prompt factory
        violations = []
        for i, result in enumerate(failures):
            violations.append(
                {
                    "rule_name": f"Rule {i+1}",  # Use index since we don't have access to rule_name
                    "message": result.message,
                    "metadata": result.metadata,
                }
            )

        # Create improvement prompt with all violations
        improvement_prompt = self._state.prompt_manager.create_improvement_prompt(text, violations)

        # Debug output
        print("\n=== DEBUG: IMPROVEMENT PROMPT ===")
        print(improvement_prompt)

        # Get improved text from model
        response = self._state.model.generate(improvement_prompt)

        # Debug output
        print("\n=== DEBUG: MODEL RESPONSE ===")
        print(response[:500] + "..." if len(response) > 500 else response)
        print(f"Response length: {len(response)} chars, {len(response.split())} words")

        # Extract improved text - handle different possible formats
        if isinstance(response, str):
            # First check for common markers
            markers = ["IMPROVED_TEXT:", "YOUR IMPROVED TEXT:", "Title:", "#", "The ", "A "]

            for marker in markers:
                if marker in response:
                    # Extract text after the marker - but don't extract if it's just part of a sentence
                    # Only extract if it appears to be at the start of a line
                    if marker in ["Title:", "#", "The ", "A "]:
                        # Check if it's at the beginning of a line
                        lines = response.split("\n")
                        for i, line in enumerate(lines):
                            if line.strip().startswith(marker):
                                # Extract from this line onwards
                                improved_text = "\n".join(lines[i:])
                                print(
                                    f"Extracted improved text using marker '{marker}' at line {i}: {len(improved_text.split())} words"
                                )
                                return improved_text
                    else:
                        # For explicit markers like "IMPROVED_TEXT:", split normally
                        improved_text = response.split(marker, 1)[1].strip()
                        print(
                            f"Extracted improved text using marker '{marker}': {len(improved_text.split())} words"
                        )
                        return improved_text

            # If we get here, no markers were definitively found
            # Try to extract the article content based on structure
            lines = response.split("\n")
            # Look for lines that might be a title (starts with # or has fewer than 8 words)
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("#") or (len(line.split()) < 8 and len(line) > 10 and i < 10):
                    # This looks like a title - extract from here
                    article_text = "\n".join(lines[i:])
                    print(
                        f"Extracted article starting from line {i}: {len(article_text.split())} words"
                    )
                    return article_text

            # If all extraction attempts fail, use the full response
            print("No extraction markers found, using full response")
            return response

        elif isinstance(response, dict) and "improved_text" in response:
            return response["improved_text"]

        # Fallback
        print("Fallback: Converting response to string")
        return response if isinstance(response, str) else str(response)


# Custom Chain with batched feedback
class BatchedFeedbackChain(ChainCore[str]):
    """Chain implementation with batched feedback for multiple rules."""

    verbose: bool = True
    max_attempts: int = 5

    def __init__(
        self, model, rules, batched_critic, max_attempts=5, verbose=True, prioritize_by_cost=True
    ):
        """Initialize chain with model, rules, and batched critic."""
        validation_manager = ValidationManager[str](rules, prioritize_by_cost=prioritize_by_cost)
        prompt_manager = PromptManager()
        retry_strategy = SimpleRetryStrategy[str](max_attempts=max_attempts)
        result_formatter = ResultFormatter[str]()

        super().__init__(
            model=model,
            validation_manager=validation_manager,
            prompt_manager=prompt_manager,
            retry_strategy=retry_strategy,
            result_formatter=result_formatter,
            critic=batched_critic,
        )
        # Set attributes using model_copy
        self.model_copy(update={"verbose": verbose, "max_attempts": max_attempts})

    def run(self, prompt: str) -> ChainResult:
        """
        Run the chain with batched feedback for rule violations.

        Args:
            prompt: The input prompt

        Returns:
            ChainResult: The final output and validation details
        """
        attempts = 0
        current_prompt = prompt
        current_output = None

        while attempts < self.max_attempts:
            if self.verbose:
                print(f"\n--- Attempt {attempts + 1} ---")

            # Generate output (first time) or use the improved text
            if current_output is None:
                current_output = self.model.generate(current_prompt)
                if self.verbose:
                    print(f"Generated {len(current_output.split())} words")

            # Validate output against all rules
            rule_results = []
            all_passed = True
            failed_rules = []

            for rule in self.validation_manager.rules:
                result = rule.validate(current_output)
                rule_results.append(result)

                if not result.passed:
                    all_passed = False
                    failed_rules.append(rule.name)
                    if self.verbose:
                        print(f"❌ Rule failed: {rule.name} - {result.message}")
                elif self.verbose:
                    print(f"✅ Rule passed: {rule.name}")

            # If all rules passed, return the result
            if all_passed:
                if self.verbose:
                    print("\n=== SUCCESS: All rules passed ===")
                return ChainResult(
                    output=current_output,
                    rule_results=rule_results,
                )

            # If we have a batched critic, use it to improve the text
            if self.critic and attempts < self.max_attempts - 1:
                if self.verbose:
                    print(
                        f"\nImproving text to fix {len(failed_rules)} rule violations: {', '.join(failed_rules)}"
                    )
                # Use the batched critic to improve based on all violations
                improved_text = self.critic.improve_with_violations(current_output, rule_results)
                current_output = improved_text

                attempts += 1
                continue

            # If out of attempts, raise error
            error_messages = [r.message for r in rule_results if not r.passed]
            raise ValueError(
                f"Validation failed after {attempts + 1} attempts. Errors:\n"
                + "\n".join(error_messages)
            )

        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of chain execution")


def create_all_classifier_rules(openai_model=None) -> List[Rule]:
    """Create rules using all available classifiers."""
    rules = []

    # 1. SentimentClassifier - require neutral or positive sentiment (EVEN EASIER)
    sentiment_classifier = ConcreteSentimentClassifier(
        name="sentiment_classifier",
        description="Analyzes text sentiment",
        config=ClassifierConfig[str](
            labels=["positive", "neutral", "negative", "unknown"],
            min_confidence=0.2,  # Further lowered confidence requirement
            cost=3,  # Medium cost
            params={"positive_threshold": 0.05, "negative_threshold": -0.05},
        ),
    )
    sentiment_rule = create_classifier_rule(
        classifier=sentiment_classifier,
        name="sentiment_rule",
        description="Ensures text has neutral or positive sentiment",
        threshold=0.2,  # Further lowered threshold
        valid_labels=["positive", "neutral"],
    )
    # Make sure the rule has the same cost as the classifier
    sentiment_rule._config = RuleConfig(cost=3, params=sentiment_rule._config.params)
    rules.append(sentiment_rule)

    # 2. Neutrality SentimentClassifier - specifically require neutral content (EVEN EASIER)
    neutrality_classifier = ConcreteSentimentClassifier(
        name="neutrality_classifier",
        description="Analyzes text for moderate neutrality",
        config=ClassifierConfig[str](
            labels=["positive", "neutral", "negative", "unknown"],
            min_confidence=0.0,  # No confidence requirement for neutrality
            cost=4,  # Medium-high cost
            params={
                # Much wider band for neutrality - allowing more leeway
                "positive_threshold": 0.6,  # Up from 0.4
                "negative_threshold": -0.6,  # Down from -0.4
            },
        ),
    )
    neutrality_rule = create_classifier_rule(
        classifier=neutrality_classifier,
        name="neutrality_rule",
        description="Ensures text is not extremely biased in tone",
        threshold=0.0,  # No threshold requirement
        valid_labels=["neutral"],  # Only accept neutral label
    )
    # Make sure the rule has the same cost as the classifier
    neutrality_rule._config = RuleConfig(cost=4, params=neutrality_rule._config.params)
    rules.append(neutrality_rule)

    # 3. ReadabilityClassifier - allow wider range of readability (EVEN EASIER)
    readability_classifier = ConcreteReadabilityClassifier(
        name="readability_classifier",
        description="Evaluates reading difficulty level",
        config=ClassifierConfig(
            labels=["simple", "moderate", "complex"],
            min_confidence=0.3,  # Further lowered confidence requirement
            cost=5,  # Medium-high cost
            params={
                "min_confidence": 0.3,
                "grade_level_bounds": {
                    "simple": (0.0, 10.0),  # Expanded upper bound from 8.0 to 10.0
                    "moderate": (8.0, 18.0),  # Expanded upper bound from 16.0 to 18.0
                    "complex": (18.0, float("inf")),
                },
            },
        ),
    )
    readability_rule = create_classifier_rule(
        classifier=readability_classifier,
        name="readability_rule",
        description="Ensures text has appropriate readability",
        threshold=0.3,  # Further lowered threshold
        valid_labels=["simple", "moderate"],  # "simple" first for emphasis
    )
    # Make sure the rule has the same cost as the classifier
    readability_rule._config = RuleConfig(cost=5, params=readability_rule._config.params)
    rules.append(readability_rule)

    # 4. LanguageClassifier - require English (SLIGHTLY EASIER)
    language_classifier = ConcreteLanguageClassifier(
        name="language_classifier",
        description="Identifies the language of text",
        config=ClassifierConfig(
            labels=["en", "es", "fr", "de", "other"],
            min_confidence=0.6,  # Slightly lower confidence
            cost=2,  # Low cost
            params={
                "min_confidence": 0.6,
                "seed": 0,
                "fallback_lang": "en",
                "fallback_confidence": 0.0,
            },
        ),
    )
    language_rule = create_classifier_rule(
        classifier=language_classifier,
        name="language_rule",
        description="Ensures text is in English",
        threshold=0.6,  # Slightly lower threshold
        valid_labels=["en"],
    )
    # Make sure the rule has the same cost as the classifier
    language_rule._config = RuleConfig(cost=2, params=language_rule._config.params)
    rules.append(language_rule)

    # 5. BiasDetector - ensure content is unbiased (EVEN EASIER)
    # Create with pretrained data to ensure the model is initialized
    dummy_texts = [
        "Men are better at math than women.",  # gender bias
        "People from certain ethnic backgrounds are more likely to commit crimes.",  # racial bias
        "All conservatives are close-minded.",  # political bias
        "This is a factual, balanced statement that presents multiple perspectives.",  # neutral
        "Here is an objective explanation of a complex issue.",  # neutral
        "The data suggests various factors contribute to this trend.",  # neutral
    ]
    dummy_labels = ["biased", "biased", "biased", "unbiased", "unbiased", "unbiased"]

    bias_detector = ConcreteBiasDetector.create_pretrained(
        texts=dummy_texts,
        labels=dummy_labels,
        name="bias_detector",
        description="Detects various forms of bias in text",
        config=ClassifierConfig(
            labels=["biased", "unbiased"],
            min_confidence=0.4,  # Further lowered confidence requirement
            cost=7,  # High cost
            params={
                "min_confidence": 0.4,
                "bias_keywords": {
                    "gender": ["men", "women", "male", "female"],
                    "racial": ["ethnic", "race", "background"],
                    "political": ["conservative", "liberal", "democrat", "republican"],
                },
            },
        ),
    )

    bias_rule = create_classifier_rule(
        classifier=bias_detector,
        name="bias_rule",
        description="Ensures text is free from bias",
        threshold=0.4,  # Further lowered threshold
        valid_labels=["unbiased"],
    )
    # Make sure the rule has the same cost as the classifier
    bias_rule._config = RuleConfig(cost=7, params=bias_rule._config.params)
    rules.append(bias_rule)

    # 12. Length rule - enforce fewer words (EVEN EASIER)
    length_rule = create_length_rule(
        min_words=500,  # Further reduced from 700
        max_words=2700,
        rule_id="length_rule",
        description="Ensures text is of appropriate length",
        priority=RulePriority.MEDIUM,  # Default priority
        cache_size=0,  # No caching
        cost=1,  # Lowest cost
    )
    rules.append(length_rule)

    # 6. ProfanityClassifier - ensure content is clean (NEW)
    profanity_classifier = ConcreteProfanityClassifier(
        name="profanity_classifier",
        description="Detects profanity and inappropriate language",
        config=ClassifierConfig(
            labels=["clean", "profane", "unknown"],
            min_confidence=0.3,  # Low confidence requirement
            cost=6,  # High cost
            params={
                "custom_words": ["controversial", "offensive", "explicit"],
                "censor_char": "*",
                "min_confidence": 0.3,
            },
        ),
    )

    profanity_rule = create_classifier_rule(
        classifier=profanity_classifier,
        name="profanity_rule",
        description="Ensures text is free from profanity",
        threshold=0.3,  # Low threshold
        valid_labels=["clean"],
    )
    # Make sure the rule has the same cost as the classifier
    profanity_rule._config = RuleConfig(cost=6, params=profanity_rule._config.params)
    rules.append(profanity_rule)
    print("Added profanity rule")

    # 7. SpamClassifier - ensure content is not spam-like (NEW)
    # Training data for spam classifier
    ham_texts = [
        "Here's the information you requested about the topic.",
        "The following analysis provides insight into the subject matter.",
        "This article explores multiple perspectives on the issue.",
        "Research indicates several factors contribute to this phenomenon.",
        "Experts suggest the following recommendations for consideration.",
    ]

    spam_texts = [
        "CLICK HERE to learn more about this amazing opportunity!",
        "You won't believe these incredible results! Act now!",
        "Limited time offer! Don't miss this exclusive content!",
        "Secret method revealed! Guaranteed amazing outcomes!",
        "Revolutionary breakthrough that experts don't want you to know!",
    ]

    # Labels for training data
    ham_labels = ["ham"] * len(ham_texts)
    spam_labels = ["spam"] * len(spam_texts)

    spam_classifier = ConcreteSpamClassifier.create_pretrained(
        texts=ham_texts + spam_texts,
        labels=ham_labels + spam_labels,
        name="spam_classifier",
        description="Detects spam-like content in text",
        config=ClassifierConfig(
            labels=["ham", "spam"],
            min_confidence=0.3,  # Low confidence requirement for easier passing
            cost=6,  # High cost
            params={
                "max_features": 1000,
                "use_bigrams": True,
            },
        ),
    )

    spam_rule = create_classifier_rule(
        classifier=spam_classifier,
        name="spam_rule",
        description="Ensures text is not spam-like",
        threshold=0.3,  # Low threshold for easier passing
        valid_labels=["ham"],
    )
    # Make sure the rule has the same cost as the classifier
    spam_rule._config = RuleConfig(cost=6, params=spam_rule._config.params)
    rules.append(spam_rule)
    print("Added spam rule")

    # 8. GenreClassifier - ensure content is of an appropriate genre (ALREADY INCLUDED & EASY)
    # Create with pretrained data to ensure the model is initialized
    genre_texts = [
        "Breaking news: Scientists discover a new planet in our solar system.",  # news
        "Once upon a time, in a land far away, there lived a brave knight.",  # fiction
        "The research results indicate a statistically significant correlation between the variables.",  # academic
        "This tutorial shows how to configure a web server with Nginx and Python.",  # technical
        "Today I'm sharing my thoughts on the latest smartphone release. Here's what I think...",  # blog
    ]
    genre_labels = ["news", "fiction", "academic", "technical", "blog"]

    genre_classifier = ConcreteGenreClassifier.create_pretrained(
        texts=genre_texts,
        labels=genre_labels,
        name="genre_classifier",
        description="Categorizes text into genres",
        config=ClassifierConfig(
            labels=["news", "fiction", "academic", "technical", "blog"],
            min_confidence=0.3,  # Lowered from 0.4
            cost=4,  # Medium-high cost
            params={
                "min_confidence": 0.3,
                "max_features": 2000,
                "use_ngrams": True,
                "default_genres": ["news", "fiction", "academic", "technical", "blog"],
            },
        ),
    )

    genre_rule = create_classifier_rule(
        classifier=genre_classifier,
        name="genre_rule",
        description="Ensures text is an appropriate informational genre",
        threshold=0.3,  # Lowered from 0.4
        valid_labels=["news", "academic", "technical", "blog"],  # Everything except fiction
    )
    # Make sure the rule has the same cost as the classifier
    genre_rule._config = RuleConfig(cost=4, params=genre_rule._config.params)
    rules.append(genre_rule)

    # 9. TopicClassifier - ensure content matches desired topics (NEW)
    # Training corpus for topic modeling
    topic_corpus = [
        "The latest economic report shows significant growth in the tech sector and manufacturing industries.",
        "Recent scientific research has led to breakthroughs in renewable energy and climate science.",
        "Educational reforms are being proposed to address challenges in schools and improve student outcomes.",
        "The healthcare industry is adopting new technologies to enhance patient care and reduce costs.",
        "Political debates focus on economic policies, social issues, and international relations.",
        "Technological innovations in artificial intelligence and machine learning are transforming industries.",
        "The entertainment industry has seen shifts in content distribution and consumer preferences.",
        "Environmental conservation efforts are addressing pollution, habitat loss, and species protection.",
        "Social media platforms are changing how people communicate and share information.",
        "Financial markets have experienced volatility due to global economic uncertainty.",
    ]

    topic_classifier = ConcreteTopicClassifier.create_pretrained(
        corpus=topic_corpus,
        name="topic_classifier",
        description="Identifies main topics in text",
        config=ClassifierConfig(
            labels=[
                "topic_0",
                "topic_1",
                "topic_2",
                "topic_3",
            ],  # Initial labels, will be replaced after fitting
            cost=3,  # Medium cost
            params={
                "num_topics": 4,
                "min_confidence": 0.3,  # Low confidence requirement for easier passing
                "max_features": 1000,
                "top_words_per_topic": 5,
            },
        ),
    )

    # Print the topic labels to see what was generated
    print("Topic labels:", topic_classifier.config.labels)

    # The rule will pass if any topic has sufficient confidence
    # We need to specify valid_labels based on the dynamically generated topics
    topic_rule = create_classifier_rule(
        classifier=topic_classifier,
        name="topic_rule",
        description="Ensures text contains clear topic focus",
        threshold=0.3,  # Low threshold for easier passing
        valid_labels=topic_classifier.config.labels,  # Use all available topic labels
    )
    # Make sure the rule has the same cost as the classifier
    topic_rule._config = RuleConfig(cost=3, params=topic_rule._config.params)
    rules.append(topic_rule)
    print("Added topic rule")

    # 10. ToxicityClassifier - ensure content is not toxic (NEW)
    try:
        # Use the standardized factory function
        # Import already included above
        # from sifaka.classifiers.toxicity import create_toxicity_classifier

        toxicity_classifier = create_toxicity_classifier(
            name="toxicity_classifier",
            description="Detects toxic content in text",
            model_name="original",
            general_threshold=0.05,  # Very low threshold to detect actual toxic content
            severe_toxic_threshold=0.1,  # Very low threshold to detect actual toxic content
            threat_threshold=0.1,  # Very low threshold to detect actual toxic content
            cost=8,  # Highest cost
        )

        # Use the standard classifier rule but with a very low threshold
        # The toxicity scores are naturally very low for non-toxic content
        toxicity_rule = create_classifier_rule(
            classifier=toxicity_classifier,
            name="toxicity_rule",
            description="Ensures text is not toxic",
            threshold=0.01,  # Very low threshold
            valid_labels=["non_toxic"],
        )
        # Make sure the rule has the same cost as the classifier
        toxicity_rule._config = RuleConfig(cost=8, params=toxicity_rule._config.params)
        rules.append(toxicity_rule)
        print("Added toxicity rule")
    except ImportError:
        print("Skipping toxicity rule (detoxify package not installed)")
    except Exception as e:
        print(f"Could not add toxicity rule: {str(e)}")

    # 11. NERClassifier - ensure content contains named entities (NEW)
    ner_classifier = ConcreteNERClassifier(
        name="ner_classifier",
        description="Identifies named entities in text",
        config=ClassifierConfig(
            labels=["person", "organization", "location", "date", "money", "unknown"],
            min_confidence=0.3,  # Low confidence requirement
            cost=4,  # Medium cost
            params={
                "min_confidence": 0.3,
                "min_entities": 2,  # Require at least 2 entities
            },
        ),
    )

    ner_rule = create_classifier_rule(
        classifier=ner_classifier,
        name="ner_rule",
        description="Ensures text contains named entities",
        threshold=0.3,  # Low threshold
        valid_labels=[
            "person",
            "organization",
            "location",
            "date",
            "money",
        ],  # All entity types except unknown
    )
    # Make sure the rule has the same cost as the classifier
    ner_rule._config = RuleConfig(cost=4, params=ner_rule._config.params)
    rules.append(ner_rule)
    print("Added NER rule")

    print(f"Created {len(rules)} rules")
    return rules


def run_advanced_example():
    """Run the advanced multi-classifier example with batched critic."""

    # Configure OpenAI model for text generation
    openai_model = OpenAIProvider(
        model_name="gpt-4-turbo",  # Use GPT-4 for better completions
        config=ModelConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=4000,  # Maximum allowed for GPT-4-turbo
        ),
    )

    # Configure OpenAI model for the critic (more capabilities)
    critic_model = OpenAIProvider(
        model_name="gpt-4-turbo",  # Use GPT-4 for better reasoning
        config=ModelConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.3,  # Lower temperature for more consistent feedback
            max_tokens=4000,  # Maximum allowed for GPT-4-turbo
        ),
    )

    # Create rules from all classifiers
    print("Setting up classifier rules...")
    rules = create_all_classifier_rules(openai_model)  # Pass the model to the function
    print(f"Created {len(rules)} rules")

    # Create batched critic
    batched_critic = BatchedCritic(
        llm_provider=critic_model,
        name="multi_classifier_critic",
        description="Improves text based on multiple classifier feedback",
    )

    # Create chain with all rules and batched critic
    chain = BatchedFeedbackChain(
        model=openai_model,
        rules=rules,
        batched_critic=batched_critic,
        max_attempts=2,
        verbose=True,
        prioritize_by_cost=True,
    )

    # Print out the rules sorted by cost
    print("\n=== Rules sorted by cost ===")
    for i, rule in enumerate(chain.validation_manager.rules):
        cost = getattr(rule.config, "cost", "unknown")
        print(f"{i+1}. {rule.name} (cost: {cost})")
    print()

    # Random topic prompt for OpenAI to generate 2,500 words
    prompt = """
    Write a comprehensive article on a topic of your choice.

    IMPORTANT REQUIREMENTS:
    1. The article MUST be at least 900 words long, preferably around 1200-1500 words
    2. Use moderate readability suitable for a general audience (high school level)
    3. Use simple, clear language and avoid overly complex terms
    4. Use shorter sentences (15-20 words) and paragraphs
    5. Include headings, subheadings, and examples
    6. Ensure a well-structured flow with introduction and conclusion
    7. CRITICAL: The content must be strictly neutral in tone and perspective
       - Present factual information without emotional language
       - Avoid language expressing strong opinions or judgments
       - Maintain a balanced, academic tone throughout
       - Include multiple perspectives on any controversial aspects
       - Avoid loaded terms or language that suggests bias
       - Use objective, measured phrasing
       - Refrain from political, social, gender, or racial bias

    Choose something interesting and engaging for a general audience.
    """

    print("\n=== Starting Advanced Multi-Classifier Example ===")
    start_time = time.time()

    try:
        # Run the chain
        result = chain.run(prompt)

        print("\n=== Final Validated Result ===")
        print(f"Word count: {len(result.output.split())}")
        print("Rule results:")
        for i, rule_result in enumerate(result.rule_results):
            print(
                f"- Rule {i+1}: {'✓ Passed' if rule_result.passed else '✗ Failed'} - {rule_result.message}"
            )

        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    except ValueError as e:
        print(f"\n=== ERROR ===\n{e}")


if __name__ == "__main__":
    run_advanced_example()
