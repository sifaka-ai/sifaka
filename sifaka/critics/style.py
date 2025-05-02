"""
Style critic that analyzes and improves text style.

This module provides a StyleCritic that evaluates text for stylistic elements
and suggests improvements.

Examples:
    Basic usage of the StyleCritic:

    ```python
    from sifaka.critics.style import StyleCritic

    # Create a style critic
    critic = StyleCritic()

    # Analyze text
    text = "this is poorly formatted text with no capitalization."
    metadata = critic.critique(text)

    # Check the critique results
    print(f"Style score: {metadata.score:.2f}")
    print(f"Issues identified: {metadata.issues}")

    # Get improvement suggestions
    print(f"Suggestions: {metadata.suggestions}")

    # Improve the text
    improved = critic.improve_with_feedback(
        text,
        "Improve capitalization and punctuation"
    )
    print(f"Improved text: {improved}")
    ```
"""

from typing import Dict, List, Any, Optional, Final, ClassVar

from sifaka.critics.base import (
    BaseCritic,
    CriticConfig,
    CriticMetadata,
    create_critic,
)
from sifaka.models.base import ModelProvider
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class StyleCritic(BaseCritic[str, str]):
    """
    Critic for analyzing and improving text style.

    The StyleCritic evaluates text for stylistic elements such as:
    - Capitalization
    - Punctuation
    - Sentence structure
    - Paragraph organization
    - Word choice and variety

    It can suggest improvements and provide edited versions of the text
    with better style.

    Lifecycle:
        - Initialize with configuration (optional)
        - Use critique() to analyze text
        - Use improve() or improve_with_feedback() to get improved text
        - No explicit cleanup needed

    Examples:
        Creating and using a StyleCritic:

        ```python
        from sifaka.critics.style import create_style_critic

        # Create with custom parameters
        critic = create_style_critic(
            name="formal_style_critic",
            min_confidence=0.6,
            params={
                "style_guide": "formal",
                "check_capitalization": True,
                "check_punctuation": True
            }
        )

        # Analyze text
        result = critic.critique("this is informal text without proper capitalization.")

        # Print issues
        if result.score < 0.7:
            print("Style issues detected:")
            for issue in result.issues:
                print(f"- {issue}")

        # Improve the text
        improved = critic.improve_with_feedback(
            "this needs better style.",
            "Make it more formal with proper capitalization"
        )
        print(f"Improved: {improved}")
        ```

    Args:
        name: Name of the critic
        description: Description of the critic
        config: Configuration for the critic
        model: Optional model provider for advanced improvements
    """

    # Default style elements to check
    DEFAULT_STYLE_ELEMENTS: ClassVar[List[str]] = [
        "capitalization",
        "punctuation",
        "sentence_structure",
        "paragraph_breaks",
        "word_variety"
    ]

    def __init__(
        self,
        name: str = "style_critic",
        description: str = "Analyzes and improves text style",
        config: Optional[CriticConfig[str]] = None,
        model: Optional[ModelProvider] = None,
    ):
        """
        Initialize the StyleCritic.

        Args:
            name: Name of the critic
            description: Description of the critic
            config: Configuration for the critic
            model: Optional model provider for advanced improvements
        """
        # Use default config if none provided
        if config is None:
            config = CriticConfig[str](
                name=name,
                description=description,
                params={
                    "style_elements": self.DEFAULT_STYLE_ELEMENTS,
                    "formality_level": "standard"
                }
            )

        super().__init__(config)
        self.model = model

        # Initialize style elements from config
        self.style_elements = config.params.get(
            "style_elements", self.DEFAULT_STYLE_ELEMENTS
        )
        self.formality_level = config.params.get("formality_level", "standard")

        logger.info(
            "Initialized StyleCritic with %d style elements and %s formality",
            len(self.style_elements),
            self.formality_level
        )

    def validate(self, text: str) -> bool:
        """
        Validate text against style standards.

        This method checks if the text meets basic style standards.

        Lifecycle:
            - Called to quickly check if text meets style requirements
            - Returns Boolean result rather than detailed analysis
            - Used by process() when determining if improvement is needed

        Examples:
            ```python
            critic = StyleCritic()

            # Check if text meets style standards
            is_valid = critic.validate("This is a properly formatted sentence.")
            print(f"Text meets style standards: {is_valid}")

            # Check problematic text
            is_valid = critic.validate("this is missing capitalization")
            print(f"Text meets style standards: {is_valid}")
            ```

        Args:
            text: The text to validate

        Returns:
            True if the text meets style standards, False otherwise
        """
        if not self.is_valid_text(text):
            return False

        # Basic style checks
        sentences = text.split('.')

        # Check capitalization of sentences
        if "capitalization" in self.style_elements:
            for sentence in sentences:
                if sentence.strip() and not sentence.strip()[0].isupper():
                    return False

        # Check for proper ending punctuation
        if "punctuation" in self.style_elements:
            if not text.strip()[-1] in ".!?":
                return False

        # Passed basic checks
        return True

    def critique(self, text: str) -> CriticMetadata[str]:
        """
        Critique text and provide style feedback.

        This method performs a detailed style analysis of the text and
        returns structured feedback with issues and suggestions.

        Lifecycle:
            - Central method for style analysis
            - Called by process() and can be called directly
            - Returns detailed CriticMetadata with score and feedback

        Examples:
            ```python
            from sifaka.critics.style import StyleCritic

            critic = StyleCritic()

            # Analyze style of a text
            result = critic.critique(
                "this is poorly formatted text. it lacks proper capitalization"
            )

            # Print the analysis
            print(f"Style score: {result.score:.2f}")
            print("Issues:")
            for issue in result.issues:
                print(f"- {issue}")

            print("Suggestions:")
            for suggestion in result.suggestions:
                print(f"- {suggestion}")
            ```

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        if not self.is_valid_text(text):
            return CriticMetadata[str](
                score=0.0,
                feedback="Invalid or empty text",
                issues=["Text must be a non-empty string"],
                suggestions=["Provide non-empty text input"]
            )

        # Initialize score and feedback lists
        issues = []
        suggestions = []
        style_scores = {}

        # Check capitalization
        if "capitalization" in self.style_elements:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            missing_caps = sum(1 for s in sentences if s and not s[0].isupper())

            capitalization_score = max(0.0, 1.0 - (missing_caps / max(1, len(sentences))))
            style_scores["capitalization"] = capitalization_score

            if missing_caps > 0:
                issues.append(f"Missing capitalization in {missing_caps} sentences")
                suggestions.append("Capitalize the first letter of each sentence")

        # Check punctuation
        if "punctuation" in self.style_elements:
            if not text.strip()[-1] in ".!?":
                issues.append("Missing ending punctuation")
                suggestions.append("Add appropriate ending punctuation (., !, or ?)")
                style_scores["punctuation"] = 0.0
            else:
                style_scores["punctuation"] = 1.0

        # Check sentence variety
        if "sentence_structure" in self.style_elements:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) >= 3:
                lengths = [len(s.split()) for s in sentences]
                avg_length = sum(lengths) / len(lengths)
                variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)

                # Low variance means monotonous sentence structure
                sentence_variety_score = min(1.0, variance / 10.0)
                style_scores["sentence_variety"] = sentence_variety_score

                if sentence_variety_score < 0.3:
                    issues.append("Limited sentence length variety")
                    suggestions.append("Vary sentence lengths for better rhythm")

        # Check paragraph breaks
        if "paragraph_breaks" in self.style_elements and len(text) > 200:
            paragraphs = text.split('\n\n')
            if len(paragraphs) == 1:
                issues.append("No paragraph breaks in long text")
                suggestions.append("Add paragraph breaks for readability")
                style_scores["paragraph_breaks"] = 0.0
            else:
                style_scores["paragraph_breaks"] = 1.0

        # Check word variety (avoid repetition)
        if "word_variety" in self.style_elements:
            words = text.lower().split()
            if len(words) > 20:
                unique_words = set(words)
                variety_ratio = len(unique_words) / len(words)

                word_variety_score = min(1.0, variety_ratio * 2.0)  # Scale for readability
                style_scores["word_variety"] = word_variety_score

                if word_variety_score < 0.4:
                    issues.append("Limited vocabulary variety")
                    suggestions.append("Use more diverse word choices to avoid repetition")

        # Calculate overall score (average of individual scores)
        if style_scores:
            overall_score = sum(style_scores.values()) / len(style_scores)
        else:
            overall_score = 0.5  # Default middle score if no checks performed

        # Create feedback message
        if overall_score > 0.8:
            feedback = "Text has good style overall"
        elif overall_score > 0.5:
            feedback = "Text has adequate style with some issues"
        else:
            feedback = "Text has significant style issues"

        # Return metadata with critique details
        return CriticMetadata[str](
            score=overall_score,
            feedback=feedback,
            issues=issues,
            suggestions=suggestions,
            extra={"style_scores": style_scores}
        )

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on style violations.

        This method attempts to fix style issues in the text based on
        the provided list of violations.

        Lifecycle:
            - Called by process() when improvement is needed
            - Can leverage external model if available
            - Falls back to rule-based improvements if no model is available

        Examples:
            ```python
            from sifaka.critics.style import StyleCritic

            critic = StyleCritic()

            # Define style violations
            violations = [
                {"issue": "Missing capitalization", "fix": "capitalize"},
                {"issue": "Missing punctuation", "fix": "add_period"}
            ]

            # Improve text based on violations
            improved = critic.improve(
                "this needs improvement",
                violations
            )

            print(f"Original: this needs improvement")
            print(f"Improved: {improved}")
            ```

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text
        """
        if not self.is_valid_text(text):
            return text

        if not violations:
            return text

        # If we have a model, use it for improvements
        if self.model:
            try:
                prompt = (
                    f"Improve the style of the following text by fixing these issues:\n"
                    f"Issues: {', '.join(v.get('issue', str(v)) for v in violations)}\n\n"
                    f"Text: {text}\n\n"
                    f"Improved text:"
                )

                response = self.model.generate(prompt)
                if isinstance(response, dict) and "text" in response:
                    return response["text"].strip()
                return response.strip()
            except Exception as e:
                logger.warning("Model-based improvement failed: %s", str(e))
                # Fall back to rule-based improvement

        # Rule-based improvement if no model or model failed
        improved = text

        # Apply basic fixes
        for violation in violations:
            issue = violation.get("issue", "")

            # Handle capitalization issues
            if "capitalization" in issue.lower():
                sentences = improved.split('.')
                improved_sentences = []

                for sentence in sentences:
                    if sentence.strip():
                        improved_sentences.append(
                            sentence[0].upper() + sentence[1:]
                            if sentence[0].islower() else sentence
                        )
                    else:
                        improved_sentences.append(sentence)

                improved = '.'.join(improved_sentences)

            # Handle punctuation issues
            elif "punctuation" in issue.lower():
                if improved and improved[-1] not in ".!?":
                    improved = improved + "."

        return improved

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        This method improves the text based on the provided feedback,
        which can be more specific than the general improvements.

        Lifecycle:
            - Can be called directly with specific feedback
            - Uses model if available for intelligent improvements
            - Falls back to rule-based improvements if no model is available

        Examples:
            ```python
            from sifaka.critics.style import StyleCritic

            critic = StyleCritic()

            # Improve with specific feedback
            improved = critic.improve_with_feedback(
                "this text has style issues and lacks proper format.",
                "Make the text more formal with proper capitalization and punctuation"
            )

            print(f"Original: this text has style issues and lacks proper format.")
            print(f"Improved: {improved}")
            ```

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text
        """
        if not self.is_valid_text(text):
            return text

        # If we have a model, use it for improvements
        if self.model:
            try:
                prompt = (
                    f"Improve the style of the following text based on this feedback:\n"
                    f"Feedback: {feedback}\n\n"
                    f"Text: {text}\n\n"
                    f"Improved text:"
                )

                response = self.model.generate(prompt)
                if isinstance(response, dict) and "text" in response:
                    return response["text"].strip()
                return response.strip()
            except Exception as e:
                logger.warning("Model-based improvement failed: %s", str(e))
                # Fall back to rule-based improvement

        # Rule-based improvement if no model or model failed
        improved = text

        # Apply basic improvements based on feedback keywords
        if "capital" in feedback.lower():
            sentences = improved.split('.')
            improved_sentences = []

            for sentence in sentences:
                if sentence.strip():
                    improved_sentences.append(
                        sentence[0].upper() + sentence[1:]
                        if sentence[0].islower() else sentence
                    )
                else:
                    improved_sentences.append(sentence)

            improved = '.'.join(improved_sentences)

        if "punctuation" in feedback.lower():
            if improved and improved[-1] not in ".!?":
                improved = improved + "."

        if "formal" in feedback.lower():
            # Simple formality improvements
            informal_to_formal = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "i'm": "I am",
                "you're": "you are",
                "they're": "they are",
                "gonna": "going to",
                "wanna": "want to",
            }

            for informal, formal in informal_to_formal.items():
                improved = improved.replace(informal, formal)

        return improved


def create_style_critic(
    name: str = "style_critic",
    description: str = "Analyzes and improves text style",
    min_confidence: float = 0.7,
    formality_level: str = "standard",
    style_elements: Optional[List[str]] = None,
    model: Optional[ModelProvider] = None,
    **kwargs: Any,
) -> StyleCritic:
    """
    Create a style critic with the specified configuration.

    This factory function creates a StyleCritic with the given parameters,
    making it easy to configure and instantiate style critics.

    Examples:
        ```python
        from sifaka.critics.style import create_style_critic
        from sifaka.models.anthropic import AnthropicProvider

        # Create a model for better improvements
        model = AnthropicProvider(model="claude-3-haiku")

        # Create a style critic with custom configuration
        critic = create_style_critic(
            name="academic_style_critic",
            description="Improves academic writing style",
            formality_level="formal",
            style_elements=["capitalization", "punctuation", "word_variety"],
            model=model,
            min_confidence=0.6
        )

        # Use the critic
        result = critic.critique("this text needs improvement.")
        print(f"Score: {result.score}")
        ```

    Args:
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        formality_level: Level of formality to enforce (casual, standard, formal)
        style_elements: List of style elements to check
        model: Optional model provider for intelligent improvements
        **kwargs: Additional configuration parameters

    Returns:
        A configured StyleCritic instance
    """
    # Extract params from kwargs
    params = kwargs.pop("params", {})

    # Add style-specific parameters
    params["formality_level"] = formality_level
    if style_elements is not None:
        params["style_elements"] = style_elements

    # Create config
    config = CriticConfig[str](
        name=name,
        description=description,
        min_confidence=min_confidence,
        params=params,
        **kwargs
    )

    # Create and return the critic
    return StyleCritic(
        name=name,
        description=description,
        config=config,
        model=model
    )