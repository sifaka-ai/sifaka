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

from dotenv import load_dotenv

# Load environment variables from .env file (containing API keys)
load_dotenv()

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.classifiers import (
    SentimentClassifier,
    ReadabilityClassifier,
    LanguageClassifier,
    ClassifierConfig,
)
from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
)
from sifaka.rules.adapters import create_classifier_rule
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain, ChainResult
from sifaka.rules.base import Rule, RuleResult


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
        improvement_prompt = self.prompt_factory.create_improvement_prompt(text, violations)

        # Debug output
        print("\n=== DEBUG: IMPROVEMENT PROMPT ===")
        print(improvement_prompt)

        # Get improved text from model
        response = self._model.generate(improvement_prompt)

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
class BatchedFeedbackChain(Chain):
    """Chain implementation with batched feedback for multiple rules."""

    def __init__(self, model, rules, batched_critic, max_attempts=5, verbose=True):
        """Initialize chain with model, rules, and batched critic."""
        super().__init__(model, rules, critic=batched_critic, max_attempts=max_attempts)
        self.verbose = verbose

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

            for rule in self.rules:
                result = rule.validate(current_output)
                rule_results.append(result)

                if not result.passed:
                    all_passed = False
                    failed_rules.append(rule.name)
                    if self.verbose:
                        # Access rule.name directly since RuleResult doesn't have rule_name
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


def create_all_classifier_rules() -> List[Rule]:
    """Create rules using all available classifiers."""
    rules = []

    # 1. SentimentClassifier - require neutral or positive sentiment
    sentiment_classifier = SentimentClassifier(
        name="sentiment_classifier",
        description="Analyzes text sentiment",
    )
    sentiment_rule = create_classifier_rule(
        classifier=sentiment_classifier,
        name="sentiment_rule",
        description="Ensures text has neutral or positive sentiment",
        threshold=0.6,
        valid_labels=["positive", "neutral"],
    )
    rules.append(sentiment_rule)

    # 2. ReadabilityClassifier - require moderately readable content
    readability_classifier = ReadabilityClassifier(
        name="readability_classifier",
        description="Evaluates reading difficulty level",
        config=ClassifierConfig(
            labels=["simple", "moderate", "complex"],
            min_confidence=0.6,
            params={
                "min_confidence": 0.6,
                "grade_level_bounds": {
                    "simple": (0.0, 8.0),
                    "moderate": (8.0, 14.0),
                    "complex": (14.0, float("inf")),
                },
            },
        ),
    )
    readability_rule = create_classifier_rule(
        classifier=readability_classifier,
        name="readability_rule",
        description="Ensures text has appropriate readability",
        threshold=0.7,
        valid_labels=["moderate"],  # Not too simple, not too complex
    )
    rules.append(readability_rule)

    # 3. LanguageClassifier - require English
    language_classifier = LanguageClassifier(
        name="language_classifier",
        description="Identifies the language of text",
        config=ClassifierConfig(
            labels=["en", "es", "fr", "de", "other"],
            min_confidence=0.7,
            params={
                "min_confidence": 0.7,
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
        threshold=0.8,
        valid_labels=["en"],
    )
    rules.append(language_rule)

    # 4. Length rule - enforce approximately 1,500 words
    length_rule = create_length_rule(
        min_words=900,
        max_words=2700,
        rule_id="length_rule",
        description="Ensures text is approximately 1,200 words",
    )
    rules.append(length_rule)

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
    rules = create_all_classifier_rules()
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
    )

    # Random topic prompt for OpenAI to generate 2,500 words
    prompt = """
    Write a comprehensive article on a topic of your choice.

    IMPORTANT REQUIREMENTS:
    1. The article MUST be at least 1000 words long, preferably around 1200-1500 words
    2. Use moderate readability suitable for a general audience (high school level)
    3. Use simple, clear language and avoid overly complex terms
    4. Use shorter sentences (15-20 words) and paragraphs
    5. Include headings, subheadings, and examples
    6. Ensure a well-structured flow with introduction and conclusion

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

        # Write output to file
        with open("multi_classifier_output.txt", "w") as f:
            f.write(result.output)
        print("\nOutput saved to 'multi_classifier_output.txt'")

    except ValueError as e:
        print(f"\n=== ERROR ===\n{e}")


if __name__ == "__main__":
    run_advanced_example()
