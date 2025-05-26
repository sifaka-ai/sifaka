"""Mixins for Sifaka critics and models.

This module provides mixin classes that add common functionality to critics
and models without requiring inheritance from a common base class. Mixins are
a clean way to share functionality across different implementations.

Available mixins:
- ContextAwareMixin: Adds retriever context support to any critic or model
- APIKeyMixin: Standardized API key management for model providers
- ValidationMixin: Common validation result creation patterns
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ConfigurationError, log_error
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ContextAwareMixin:
    """Mixin to add retriever context support to any critic or model.

    This mixin provides standardized methods for handling retrieved context
    from the Thought container. It can be mixed into any critic or model class
    to add context awareness without code duplication.

    Features:
    - Context preparation from pre/post-generation documents
    - Context availability detection
    - Smart template enhancement with context placeholders
    - Context relevance filtering and summarization
    - Consistent context formatting across all critics and models

    Usage for Critics:
        ```python
        class MyCritic(ContextAwareMixin):
            def critique(self, thought: Thought) -> Dict[str, Any]:
                context = self._prepare_context(thought)
                prompt = template.format(..., context=context)
                # ... rest of critique logic
        ```

    Usage for Models:
        ```python
        class MyModel(ContextAwareMixin):
            def generate_with_thought(self, thought: Thought, **options) -> str:
                context = self._prepare_context_for_generation(thought)
                full_prompt = f"{thought.prompt}\n\n{context}"
                return self.generate(full_prompt, **options)
        ```
    """

    def _prepare_context(self, thought: Thought, max_docs: Optional[int] = None) -> str:
        """Prepare context string from retrieved documents.

        This method extracts and formats context from both pre-generation
        and post-generation retrieved documents in the Thought container.

        Args:
            thought: The Thought container with retrieved context.
            max_docs: Maximum number of documents to include (None for all).

        Returns:
            A formatted context string ready for use in prompts.
        """
        context_parts = []
        doc_count = 0

        # Add pre-generation context
        if thought.pre_generation_context:
            for doc in thought.pre_generation_context:
                if max_docs and doc_count >= max_docs:
                    break
                context_parts.append(f"Document {doc_count + 1}: {doc.text}")
                doc_count += 1

        # Add post-generation context
        if thought.post_generation_context:
            for doc in thought.post_generation_context:
                if max_docs and doc_count >= max_docs:
                    break
                context_parts.append(f"Additional Document {doc_count + 1}: {doc.text}")
                doc_count += 1

        if context_parts:
            context_str = "\n\n".join(context_parts)
            logger.debug(
                f"Prepared context with {doc_count} documents, {len(context_str)} characters"
            )
            return context_str
        else:
            logger.debug("No retrieved context available")
            return "No retrieved context available."

    def _has_context(self, thought: Thought) -> bool:
        """Check if the thought has any retrieved context.

        Args:
            thought: The Thought container to check.

        Returns:
            True if context is available, False otherwise.
        """
        has_pre = bool(thought.pre_generation_context and len(thought.pre_generation_context) > 0)
        has_post = bool(
            thought.post_generation_context and len(thought.post_generation_context) > 0
        )
        return has_pre or has_post

    def _get_context_summary(self, thought: Thought) -> str:
        """Get a summary of available context.

        Args:
            thought: The Thought container to summarize.

        Returns:
            A human-readable summary of available context.
        """
        pre_count = len(thought.pre_generation_context) if thought.pre_generation_context else 0
        post_count = len(thought.post_generation_context) if thought.post_generation_context else 0
        total_count = pre_count + post_count

        if total_count == 0:
            return "No context available"

        parts = []
        if pre_count > 0:
            parts.append(f"{pre_count} pre-generation document{'s' if pre_count != 1 else ''}")
        if post_count > 0:
            parts.append(f"{post_count} post-generation document{'s' if post_count != 1 else ''}")

        return f"Context available: {' and '.join(parts)} (total: {total_count})"

    def _enhance_template_with_context(
        self, template: str, context_label: str = "Retrieved context"
    ) -> str:
        """Add context placeholder to existing template if not present.

        This method intelligently adds a context section to existing prompt
        templates without breaking their structure.

        Args:
            template: The original template string.
            context_label: Label to use for the context section.

        Returns:
            Enhanced template with context placeholder.
        """
        # If template already has context placeholder, return as-is
        if "{context}" in template:
            return template

        # Smart insertion strategies
        context_section = f"\n\n{context_label}:\n{{context}}\n"

        # Strategy 1: Insert before "Please provide" or similar instructions
        instruction_patterns = [
            r"(Please provide.*)",
            r"(Provide.*)",
            r"(Your task is.*)",
            r"(Analyze.*)",
            r"(Evaluate.*)",
            r"(Critique.*)",
        ]

        for pattern in instruction_patterns:
            match = re.search(pattern, template, re.IGNORECASE)
            if match:
                insertion_point = match.start()
                return template[:insertion_point] + context_section + template[insertion_point:]

        # Strategy 2: Insert before the last paragraph (if multiple paragraphs)
        paragraphs = template.split("\n\n")
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs[:-1]) + context_section + "\n\n" + paragraphs[-1]

        # Strategy 3: Fallback - add at the end
        return template + context_section

    def _prepare_context_with_relevance(
        self, thought: Thought, query: Optional[str] = None, max_docs: int = 5
    ) -> str:
        """Prepare context with relevance filtering.

        This method prepares context but filters documents based on relevance
        to a query (typically the prompt or generated text).

        Args:
            thought: The Thought container with retrieved context.
            query: Query to use for relevance filtering (defaults to prompt).
            max_docs: Maximum number of documents to include.

        Returns:
            A formatted context string with most relevant documents.
        """
        if not self._has_context(thought):
            return "No retrieved context available."

        # Use prompt as default query for relevance
        if query is None:
            query = thought.prompt

        # Collect all documents with their sources
        all_docs = []

        if thought.pre_generation_context:
            for doc in thought.pre_generation_context:
                all_docs.append((doc, "pre-generation"))

        if thought.post_generation_context:
            for doc in thought.post_generation_context:
                all_docs.append((doc, "post-generation"))

        # Simple relevance scoring (can be enhanced with embeddings later)
        scored_docs = []
        query_lower = query.lower()

        for doc, source in all_docs:
            content_lower = doc.text.lower()

            # Simple keyword overlap scoring
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words.intersection(content_words))
            relevance_score = overlap / len(query_words) if query_words else 0

            scored_docs.append((doc, source, relevance_score))

        # Sort by relevance score (descending)
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        # Take top documents
        top_docs = scored_docs[:max_docs]

        # Format context
        context_parts = []
        for i, (doc, source, score) in enumerate(top_docs):
            label = f"Document {i + 1} ({source}, relevance: {score:.2f})"
            context_parts.append(f"{label}: {doc.text}")

        context_str = "\n\n".join(context_parts)
        logger.debug(
            f"Prepared relevance-filtered context: {len(top_docs)} docs, "
            f"avg relevance: {sum(score for _, _, score in top_docs) / len(top_docs):.2f}"
        )

        return context_str

    def _context_aware_template(
        self, template: str, thought: Thought
    ) -> Tuple[str, Dict[str, str]]:
        """Automatically enhance template and prepare context based on availability.

        This is a convenience method that combines template enhancement and
        context preparation in one call.

        Args:
            template: The original template string.
            thought: The Thought container with potential context.

        Returns:
            A tuple of (enhanced_template, format_kwargs) where format_kwargs
            contains the context value to use in template.format().
        """
        enhanced_template = self._enhance_template_with_context(template)
        context = self._prepare_context(thought)

        format_kwargs = {"context": context}

        logger.debug(
            f"Enhanced template with context: has_context={self._has_context(thought)}, "
            f"context_length={len(context)}"
        )

        return enhanced_template, format_kwargs

    # Model-specific methods
    def _prepare_context_for_generation(
        self,
        thought: Thought,
        max_docs: Optional[int] = None,
        include_post_generation: bool = False,
        include_critic_feedback: bool = True,
        include_validation_results: bool = True,
    ) -> str:
        """Prepare context specifically for text generation.

        This method is optimized for models that need context for generation.
        It focuses on pre-generation context by default but can include
        post-generation context, validation results, and critic feedback from previous iterations.

        Args:
            thought: The Thought container with retrieved context.
            max_docs: Maximum number of documents to include.
            include_post_generation: Whether to include post-generation context.
            include_critic_feedback: Whether to include critic feedback from previous iterations.
            include_validation_results: Whether to include validation results from previous iterations.

        Returns:
            A formatted context string optimized for generation.
        """
        context_parts = []
        doc_count = 0

        # Always include pre-generation context for models
        if thought.pre_generation_context:
            for doc in thought.pre_generation_context:
                if max_docs and doc_count >= max_docs:
                    break
                context_parts.append(f"Reference {doc_count + 1}: {doc.text}")
                doc_count += 1

        # Optionally include post-generation context
        if include_post_generation and thought.post_generation_context:
            for doc in thought.post_generation_context:
                if max_docs and doc_count >= max_docs:
                    break
                context_parts.append(f"Additional Reference {doc_count + 1}: {doc.text}")
                doc_count += 1

        # Include validation results from current thought if available
        if include_validation_results and thought.validation_results:
            validation_parts = []
            for validator_name, validation_result in thought.validation_results.items():
                status = "PASSED" if validation_result.passed else "FAILED"
                validation_summary = f"Validation by {validator_name}: {status}"

                # Include score if available
                if validation_result.score is not None:
                    validation_summary += f" (score: {validation_result.score:.3f})"

                # Include specific issues if validation failed
                if not validation_result.passed and validation_result.issues:
                    validation_summary += "\nIssues found:"
                    for issue in validation_result.issues[:3]:  # Limit to top 3
                        validation_summary += f"\n- {issue}"

                # Include suggestions if available
                if validation_result.suggestions:
                    validation_summary += "\nSuggestions:"
                    for suggestion in validation_result.suggestions[:3]:  # Limit to top 3
                        validation_summary += f"\n- {suggestion}"

                # Include validation message if available and different from issues
                if validation_result.message and validation_result.message not in str(
                    validation_result.issues
                ):
                    validation_summary += f"\nMessage: {validation_result.message}"

                validation_parts.append(validation_summary)

            if validation_parts:
                context_parts.append(
                    "Previous Validation Results:\n" + "\n\n".join(validation_parts)
                )

        # Include critic feedback from current thought if available
        if include_critic_feedback and thought.critic_feedback:
            feedback_parts = []
            for feedback in thought.critic_feedback:
                # Extract key information from critic feedback
                critic_name = feedback.critic_name
                confidence = feedback.confidence

                feedback_summary = f"Feedback from {critic_name} (confidence: {confidence:.2f}):"

                # Include violations if available
                if feedback.violations:
                    feedback_summary += "\nViolations found:"
                    for violation in feedback.violations[:3]:  # Limit to top 3
                        feedback_summary += f"\n- {violation}"

                # Include suggestions if available
                if feedback.suggestions:
                    feedback_summary += "\nSuggestions:"
                    for suggestion in feedback.suggestions[:3]:  # Limit to top 3
                        feedback_summary += f"\n- {suggestion}"

                # Include the main feedback text if no specific violations/suggestions
                if not feedback.violations and not feedback.suggestions and feedback.feedback:
                    # Extract key parts of the feedback (first few lines)
                    feedback_lines = feedback.feedback.split("\n")[:5]  # First 5 lines
                    feedback_summary += "\nFeedback summary:"
                    for line in feedback_lines:
                        if line.strip():
                            feedback_summary += f"\n- {line.strip()}"

                if not feedback.violations and not feedback.suggestions and not feedback.feedback:
                    feedback_summary += "\n- No specific feedback provided"

                feedback_parts.append(feedback_summary)

            if feedback_parts:
                context_parts.append("Previous Critic Feedback:\n" + "\n\n".join(feedback_parts))

        if context_parts:
            context_str = "\n\n".join(context_parts)
            logger.debug(
                f"Prepared generation context with {doc_count} documents, validation results, and critic feedback"
            )
            return f"Context:\n{context_str}\n\nTask:"
        else:
            logger.debug("No context available for generation")
            return ""

    def _build_contextualized_prompt(
        self,
        thought: Thought,
        max_docs: Optional[int] = None,
        context_position: str = "before",  # "before", "after", or "system"
    ) -> str:
        """Build a complete prompt with context for generation.

        This method combines the prompt with retrieved context in an
        optimal format for text generation.

        Args:
            thought: The Thought container with prompt and context.
            max_docs: Maximum number of documents to include.
            context_position: Where to place context ("before", "after", "system").

        Returns:
            A complete prompt ready for generation.
        """
        context = self._prepare_context_for_generation(thought, max_docs)
        prompt = thought.prompt

        if not context:
            return prompt

        if context_position == "before":
            return f"{context}\n\n{prompt}"
        elif context_position == "after":
            return f"{prompt}\n\n{context}"
        elif context_position == "system":
            # Return tuple for system message handling
            return prompt  # Context should be added to system prompt separately
        else:
            # Default to before
            return f"{context}\n\n{prompt}"

    def _get_context_for_system_prompt(
        self, thought: Thought, max_docs: Optional[int] = None
    ) -> str:
        """Get context formatted for system prompts.

        This method prepares context in a format suitable for system prompts,
        which is useful for models that support system messages.

        Args:
            thought: The Thought container with retrieved context.
            max_docs: Maximum number of documents to include.

        Returns:
            Context formatted for system prompts.
        """
        context = self._prepare_context_for_generation(thought, max_docs)

        if not context:
            return ""

        return f"You have access to the following reference information:\n\n{context}\n\nUse this information to inform your response when relevant."

    def _prepare_context_with_embeddings(
        self,
        thought: Thought,
        query: Optional[str] = None,
        max_docs: int = 5,
        similarity_threshold: float = 0.7,
    ) -> str:
        """Prepare context with embedding-based relevance filtering.

        This method uses semantic similarity (when available) to filter
        the most relevant documents for the given query.

        Args:
            thought: The Thought container with retrieved context.
            query: Query to use for relevance filtering (defaults to prompt).
            max_docs: Maximum number of documents to include.
            similarity_threshold: Minimum similarity score to include document.

        Returns:
            A formatted context string with most semantically relevant documents.
        """
        if not self._has_context(thought):
            return "No retrieved context available."

        # Use prompt as default query for relevance
        if query is None:
            query = thought.prompt

        # Collect all documents with their sources
        all_docs = []

        if thought.pre_generation_context:
            for doc in thought.pre_generation_context:
                all_docs.append((doc, "pre-generation"))

        if thought.post_generation_context:
            for doc in thought.post_generation_context:
                all_docs.append((doc, "post-generation"))

        # Try to use embedding-based similarity if available
        try:
            # Check if documents have embedding scores
            scored_docs = []
            for doc, source in all_docs:
                # Use existing score if available, otherwise fall back to keyword overlap
                if hasattr(doc, "score") and doc.score is not None:
                    similarity_score = doc.score
                else:
                    # Fall back to keyword overlap
                    query_lower = query.lower()
                    content_lower = doc.text.lower()
                    query_words = set(query_lower.split())
                    content_words = set(content_lower.split())
                    overlap = len(query_words.intersection(content_words))
                    similarity_score = overlap / len(query_words) if query_words else 0

                # Only include documents above threshold
                if similarity_score >= similarity_threshold:
                    scored_docs.append((doc, source, similarity_score))

            # Sort by similarity score (descending)
            scored_docs.sort(key=lambda x: x[2], reverse=True)

            # Take top documents
            top_docs = scored_docs[:max_docs]

            # Format context
            context_parts = []
            for i, (doc, source, score) in enumerate(top_docs):
                label = f"Document {i + 1} ({source}, similarity: {score:.3f})"
                context_parts.append(f"{label}: {doc.text}")

            context_str = "\n\n".join(context_parts)
            logger.debug(
                f"Prepared embedding-based context: {len(top_docs)} docs, "
                f"avg similarity: {sum(score for _, _, score in top_docs) / len(top_docs):.3f}"
            )

            return context_str

        except Exception as e:
            logger.warning(
                f"Embedding-based filtering failed, falling back to keyword overlap: {e}"
            )
            # Fall back to keyword-based relevance
            return self._prepare_context_with_relevance(thought, query, max_docs)

    def _compress_context(
        self, thought: Thought, max_length: int = 2000, preserve_diversity: bool = True
    ) -> str:
        """Compress context to fit within length limits while preserving key information.

        This method intelligently truncates or summarizes context to fit
        within token/character limits while maintaining diversity and relevance.

        Args:
            thought: The Thought container with retrieved context.
            max_length: Maximum character length for the compressed context.
            preserve_diversity: Whether to preserve diversity across documents.

        Returns:
            A compressed context string that fits within the length limit.
        """
        if not self._has_context(thought):
            return "No retrieved context available."

        # Get all documents
        all_docs = []

        if thought.pre_generation_context:
            for doc in thought.pre_generation_context:
                all_docs.append((doc, "pre-generation"))

        if thought.post_generation_context:
            for doc in thought.post_generation_context:
                all_docs.append((doc, "post-generation"))

        if not all_docs:
            return "No retrieved context available."

        # If total length is already under limit, return as-is
        full_context = self._prepare_context(thought)
        if len(full_context) <= max_length:
            return full_context

        # Strategy 1: Truncate each document proportionally
        if preserve_diversity:
            # Calculate how much space each document gets
            space_per_doc = max_length // len(all_docs)
            # Reserve some space for labels and formatting
            content_per_doc = max(100, space_per_doc - 50)

            context_parts = []
            for i, (doc, source) in enumerate(all_docs):
                label = f"Document {i + 1} ({source})"
                content = doc.text

                if len(content) > content_per_doc:
                    # Truncate and add ellipsis
                    content = content[: content_per_doc - 3] + "..."

                context_parts.append(f"{label}: {content}")

            compressed_context = "\n\n".join(context_parts)

        else:
            # Strategy 2: Take documents in order until we hit the limit
            context_parts = []
            current_length = 0

            for i, (doc, source) in enumerate(all_docs):
                label = f"Document {i + 1} ({source}): "
                content = doc.text

                # Check if we can fit this document
                needed_length = len(label) + len(content) + 2  # +2 for \n\n

                if current_length + needed_length <= max_length:
                    context_parts.append(f"{label}{content}")
                    current_length += needed_length
                else:
                    # Try to fit a truncated version
                    remaining_space = (
                        max_length - current_length - len(label) - 5
                    )  # -5 for "..." and \n\n
                    if remaining_space > 50:  # Only if we have meaningful space
                        truncated_content = content[:remaining_space] + "..."
                        context_parts.append(f"{label}{truncated_content}")
                    break

            compressed_context = "\n\n".join(context_parts)

        logger.debug(
            f"Compressed context from {len(full_context)} to {len(compressed_context)} characters "
            f"({len(all_docs)} docs -> {len(context_parts)} docs)"
        )

        return compressed_context


class APIKeyMixin:
    """Mixin that provides standardized API key management.

    This mixin handles API key retrieval from environment variables,
    validation, and error handling in a consistent way across all model providers.
    """

    def get_api_key(
        self, api_key: Optional[str], env_var_name: str, provider_name: str, required: bool = True
    ) -> Optional[str]:
        """Get API key from parameter or environment variable.

        Args:
            api_key: Explicitly provided API key.
            env_var_name: Environment variable name to check.
            provider_name: Name of the provider for error messages.
            required: Whether the API key is required.

        Returns:
            The API key if found, None if not required and not found.

        Raises:
            ConfigurationError: If API key is required but not found.
        """
        # Use provided key if available
        if api_key:
            return api_key

        # Try environment variable
        env_key = os.getenv(env_var_name)
        if env_key:
            return env_key

        # Handle missing key
        if required:
            raise ConfigurationError(
                f"{provider_name} API key not provided",
                component=provider_name,
                operation="initialization",
                suggestions=[
                    f"Set the {env_var_name} environment variable",
                    f"Pass api_key parameter to the {provider_name} constructor",
                    f"Ensure your API key is valid and has the necessary permissions",
                ],
            )

        return None


class ValidationMixin:
    """Mixin that provides standardized validation result creation.

    This mixin reduces duplication in validation logic by providing
    common patterns for creating ValidationResult objects.
    """

    def create_validation_result(
        self,
        passed: bool,
        message: str,
        score: Optional[float] = None,
        issues: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Create a standardized ValidationResult.

        Args:
            passed: Whether validation passed.
            message: Validation message.
            score: Optional confidence score.
            issues: List of issues found.
            suggestions: List of suggestions for improvement.

        Returns:
            A ValidationResult object.
        """
        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues or [],
            suggestions=suggestions or [],
        )

    def create_empty_text_result(self, validator_name: str) -> ValidationResult:
        """Create a validation result for empty text.

        Args:
            validator_name: Name of the validator.

        Returns:
            A ValidationResult indicating empty text failure.
        """
        return self.create_validation_result(
            passed=False,
            message="No text available for validation",
            issues=["Text is empty or None"],
            suggestions=["Provide text to validate"],
        )

    def create_error_result(
        self, error: Exception, validator_name: str, operation: str = "validation"
    ) -> ValidationResult:
        """Create a validation result for errors.

        Args:
            error: The exception that occurred.
            validator_name: Name of the validator.
            operation: The operation that failed.

        Returns:
            A ValidationResult indicating an error.
        """
        error_message = str(error)
        log_error(error, component=validator_name, operation=operation)

        return self.create_validation_result(
            passed=False,
            message=f"{operation.title()} error: {error_message}",
            issues=[f"{validator_name} error: {error_message}"],
            suggestions=[
                f"Check {validator_name} implementation and input format",
                "Verify that all required dependencies are installed",
                "Check the logs for more detailed error information",
            ],
        )
