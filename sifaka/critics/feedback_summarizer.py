"""Feedback summarizer for Sifaka critics.

This module provides a customizable feedback summarizer that can use various local
or API-based models to summarize validation results and critic feedback before
passing them to model prompts for improvement.

The summarizer supports multiple models including T5, BART, Pegasus, and others,
with T5 as the default, and provides fallback mechanisms when summarization fails.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import CriticFeedback, Thought, ValidationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class FeedbackSummarizer:
    """Customizable feedback summarizer for critic and validation feedback.

    This class provides the ability to summarize validation results and critic feedback
    using various local or API-based models. It supports multiple summarization models
    including T5, BART, Pegasus, and others, with T5 as the default.

    Features:
    - Multiple model support (T5, BART, Pegasus, custom models)
    - Configurable summarization prompts and parameters
    - Fallback mechanisms when summarization fails
    - Caching for improved performance
    - Both local and API-based model support

    Example:
        ```python
        # Create summarizer with T5 (default)
        summarizer = FeedbackSummarizer()

        # Create summarizer with BART
        summarizer = FeedbackSummarizer(
            model_name="facebook/bart-base",
            model_type="bart"
        )

        # Create summarizer with API model
        summarizer = FeedbackSummarizer(
            model_type="api",
            api_model="openai:gpt-3.5-turbo"
        )

        # Summarize feedback from a thought
        summary = summarizer.summarize_thought_feedback(thought)

        # Use in improvement prompt
        prompt = f"Improve the text based on: {summary}"
        ```
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        model_type: str = "auto",  # "t5", "bart", "pegasus", "auto", "api"
        api_model: Optional[str] = None,  # For API-based summarization
        max_length: int = 150,
        min_length: int = 30,
        custom_prompt: Optional[str] = None,
        cache_summaries: bool = True,
        fallback_to_truncation: bool = True,
        **model_kwargs: Any,
    ):
        """Initialize the feedback summarizer.

        Args:
            model_name: Name of the HuggingFace model to use (e.g., "t5-small", "facebook/bart-base").
            model_type: Type of model ("t5", "bart", "pegasus", "auto", "api").
            api_model: Model name for API-based summarization (e.g., "openai:gpt-3.5-turbo").
            max_length: Maximum length of generated summaries.
            min_length: Minimum length of generated summaries.
            custom_prompt: Custom prompt template for summarization.
            cache_summaries: Whether to cache summaries for repeated inputs.
            fallback_to_truncation: Whether to fall back to truncation if summarization fails.
            **model_kwargs: Additional arguments for model initialization.
        """
        self.config = {
            "model_name": model_name,
            "model_type": model_type,
            "api_model": api_model,
            "max_length": max_length,
            "min_length": min_length,
            "custom_prompt": custom_prompt,
            "cache_summaries": cache_summaries,
            "fallback_to_truncation": fallback_to_truncation,
            "model_kwargs": model_kwargs,
        }

        # Initialize model lazily
        self._model = None
        self._tokenizer = None
        self._summary_cache = {} if cache_summaries else None

        logger.debug(
            f"Initialized FeedbackSummarizer with model: {model_name} (type: {model_type})"
        )

    def summarize_thought_feedback(
        self,
        thought: Thought,
        include_validation: bool = True,
        include_critic_feedback: bool = True,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Summarize validation results and critic feedback from a thought.

        Args:
            thought: The Thought container with feedback to summarize.
            include_validation: Whether to include validation results.
            include_critic_feedback: Whether to include critic feedback.
            custom_prompt: Optional custom prompt for this specific summarization.

        Returns:
            Summarized feedback text ready for use in improvement prompts.
        """
        feedback_text = self._format_feedback_for_summarization(
            thought, include_validation, include_critic_feedback
        )

        if (
            not feedback_text
            or feedback_text.strip() == "No specific feedback available for improvement."
        ):
            return "No feedback available for improvement."

        return self._summarize_feedback(feedback_text, custom_prompt)

    def summarize_validation_results(
        self,
        validation_results: Dict[str, ValidationResult],
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Summarize validation results only.

        Args:
            validation_results: Dictionary of validation results.
            custom_prompt: Optional custom prompt for summarization.

        Returns:
            Summarized validation results.
        """
        feedback_text = self._format_validation_results(validation_results)

        if not feedback_text:
            return "All validations passed."

        return self._summarize_feedback(feedback_text, custom_prompt)

    def summarize_critic_feedback(
        self,
        critic_feedback: List[CriticFeedback],
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Summarize critic feedback only.

        Args:
            critic_feedback: List of critic feedback.
            custom_prompt: Optional custom prompt for summarization.

        Returns:
            Summarized critic feedback.
        """
        feedback_text = self._format_critic_feedback(critic_feedback)

        if not feedback_text:
            return "No critic feedback requiring improvement."

        return self._summarize_feedback(feedback_text, custom_prompt)

    def _get_model(self):
        """Lazy initialization of the summarization model."""
        if self._model is not None:
            return self._model, self._tokenizer

        model_name = self.config["model_name"]

        try:
            # Import transformers here to avoid dependency issues
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.debug(f"Loading summarization model: {model_name}")

            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, **self.config["model_kwargs"]
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **self.config["model_kwargs"]
            )

            logger.debug(f"Successfully loaded summarization model: {model_name}")
            return self._model, self._tokenizer

        except ImportError:
            logger.warning("transformers library not available, falling back to truncation")
            return None, None
        except Exception as e:
            logger.warning(f"Failed to load summarization model {model_name}: {e}")
            return None, None

    def _summarize_feedback(
        self,
        feedback_text: str,
        custom_prompt: Optional[str] = None,
        max_input_length: int = 1024,
    ) -> str:
        """Summarize feedback text using the configured model.

        Args:
            feedback_text: The feedback text to summarize.
            custom_prompt: Optional custom prompt for this specific summarization.
            max_input_length: Maximum input length for the model.

        Returns:
            Summarized feedback text.
        """
        if not feedback_text or not feedback_text.strip():
            return "No feedback available."

        # Check cache first
        if self._summary_cache is not None:
            cache_key = hash(feedback_text + str(custom_prompt))
            if cache_key in self._summary_cache:
                logger.debug("Using cached summary")
                return self._summary_cache[cache_key]

        # Handle API-based summarization
        if self.config["model_type"] == "api" and self.config["api_model"]:
            summary = self._summarize_with_api(feedback_text, custom_prompt)
            if summary:
                if self._summary_cache is not None:
                    self._summary_cache[cache_key] = summary
                return summary

        # Handle local model summarization
        model, tokenizer = self._get_model()
        if model is None or tokenizer is None:
            return self._fallback_summarization(feedback_text)

        try:
            # Prepare input text
            prompt = custom_prompt or self.config["custom_prompt"]
            if prompt:
                input_text = f"{prompt}\n\n{feedback_text}"
            else:
                # Use model-specific default prompts
                input_text = self._get_default_prompt(feedback_text)

            # Truncate input if too long
            if len(input_text) > max_input_length:
                input_text = input_text[: max_input_length - 3] + "..."

            # Tokenize and generate
            inputs = tokenizer.encode(
                input_text, return_tensors="pt", truncation=True, max_length=max_input_length
            )

            with tokenizer.as_target_tokenizer():
                outputs = model.generate(
                    inputs,
                    max_length=self.config["max_length"],
                    min_length=self.config["min_length"],
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Cache the result
            if self._summary_cache is not None:
                cache_key = hash(feedback_text + str(custom_prompt))
                self._summary_cache[cache_key] = summary

            logger.debug(f"Generated summary: {len(summary)} chars from {len(feedback_text)} chars")
            return summary

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return self._fallback_summarization(feedback_text)

    def _summarize_with_api(
        self, feedback_text: str, custom_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Summarize feedback using an API-based model.

        Args:
            feedback_text: The feedback text to summarize.
            custom_prompt: Optional custom prompt for summarization.

        Returns:
            Summarized text or None if API summarization fails.
        """
        try:
            from sifaka.models.base import create_model

            api_model = create_model(self.config["api_model"])

            prompt = (
                custom_prompt
                or self.config["custom_prompt"]
                or (
                    "Please provide a concise summary of the following feedback, "
                    "highlighting the key issues and suggestions:"
                )
            )

            full_prompt = f"{prompt}\n\n{feedback_text}\n\nSummary:"

            summary = api_model.generate(
                full_prompt,
                max_tokens=self.config["max_length"],
                temperature=0.3,  # Lower temperature for more focused summaries
            )

            return summary.strip()

        except Exception as e:
            logger.warning(f"API-based summarization failed: {e}")
            return None

    def _get_default_prompt(self, feedback_text: str) -> str:
        """Get default prompt based on model type.

        Args:
            feedback_text: The feedback text to summarize.

        Returns:
            Formatted input text with appropriate prompt.
        """
        model_type = self.config["model_type"]

        if model_type == "t5" or model_type == "auto":
            # T5 expects task prefix
            return f"summarize: {feedback_text}"
        elif model_type == "bart":
            # BART can work with direct input
            return feedback_text
        elif model_type == "pegasus":
            # Pegasus is designed for summarization
            return feedback_text
        else:
            # Default approach
            return f"Summarize the following feedback: {feedback_text}"

    def _fallback_summarization(self, feedback_text: str) -> str:
        """Fallback summarization using simple truncation and extraction.

        Args:
            feedback_text: The feedback text to summarize.

        Returns:
            A simplified summary using truncation and key phrase extraction.
        """
        if not self.config["fallback_to_truncation"]:
            return "Summarization unavailable."

        max_length = self.config["max_length"]

        # Simple extractive summarization
        sentences = feedback_text.split(". ")

        # Prioritize sentences with key feedback words
        key_words = [
            "issue",
            "problem",
            "suggest",
            "improve",
            "error",
            "fix",
            "better",
            "should",
            "must",
        ]
        scored_sentences = []

        for sentence in sentences:
            score = sum(1 for word in key_words if word.lower() in sentence.lower())
            scored_sentences.append((score, sentence))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        summary_parts = []
        current_length = 0

        for _, sentence in scored_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) + 2 <= max_length:  # +2 for '. '
                summary_parts.append(sentence)
                current_length += len(sentence) + 2
            else:
                break

        if not summary_parts:
            # If no sentences fit, just truncate
            return (
                feedback_text[: max_length - 3] + "..."
                if len(feedback_text) > max_length
                else feedback_text
            )

        summary = ". ".join(summary_parts)
        if not summary.endswith("."):
            summary += "."

        logger.debug(
            f"Fallback summarization: {len(summary)} chars from {len(feedback_text)} chars"
        )
        return summary

    def _format_feedback_for_summarization(
        self,
        thought: Thought,
        include_validation: bool = True,
        include_critic_feedback: bool = True,
    ) -> str:
        """Format validation results and critic feedback for summarization.

        Args:
            thought: The Thought container with feedback to format.
            include_validation: Whether to include validation results.
            include_critic_feedback: Whether to include critic feedback.

        Returns:
            Formatted feedback text ready for summarization.
        """
        feedback_parts = []

        # Add validation results
        if include_validation and thought.validation_results:
            validation_text = self._format_validation_results(thought.validation_results)
            if validation_text:
                feedback_parts.append(f"VALIDATION RESULTS:\n{validation_text}")

        # Add critic feedback
        if include_critic_feedback and thought.critic_feedback:
            critic_text = self._format_critic_feedback(thought.critic_feedback)
            if critic_text:
                feedback_parts.append(f"CRITIC FEEDBACK:\n{critic_text}")

        if not feedback_parts:
            return "No specific feedback available for improvement."

        return "\n\n".join(feedback_parts)

    def _format_validation_results(self, validation_results: Dict[str, ValidationResult]) -> str:
        """Format validation results for summarization."""
        validation_parts = []

        for validator_name, result in validation_results.items():
            if not result.passed:
                status = "FAILED"
                details = []
                if result.issues:
                    details.extend([f"Issue: {issue}" for issue in result.issues])
                if result.suggestions:
                    details.extend(
                        [f"Suggestion: {suggestion}" for suggestion in result.suggestions]
                    )

                validation_text = f"{validator_name}: {status}"
                if details:
                    validation_text += f" - {'; '.join(details)}"
                validation_parts.append(validation_text)

        return "\n".join(validation_parts)

    def _format_critic_feedback(self, critic_feedback: List[CriticFeedback]) -> str:
        """Format critic feedback for summarization."""
        critic_parts = []

        for feedback in critic_feedback:
            if feedback.needs_improvement:
                critic_text = f"{feedback.critic_name} (confidence: {feedback.confidence:.2f})"

                details = []
                if feedback.violations:
                    details.extend([f"Violation: {violation}" for violation in feedback.violations])
                if feedback.suggestions:
                    details.extend(
                        [f"Suggestion: {suggestion}" for suggestion in feedback.suggestions]
                    )
                if feedback.feedback and not feedback.violations and not feedback.suggestions:
                    # Include main feedback if no specific violations/suggestions
                    details.append(f"Feedback: {feedback.feedback}")

                if details:
                    critic_text += f" - {'; '.join(details)}"
                critic_parts.append(critic_text)

        return "\n".join(critic_parts)

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        if self._summary_cache is not None:
            self._summary_cache.clear()
            logger.debug("Summary cache cleared")
