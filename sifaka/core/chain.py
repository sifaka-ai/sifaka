"""
Chain implementation for Sifaka.

This module defines the Chain class, which orchestrates the generation, validation,
and improvement of text using models, validators, and critics.
"""

import logging
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Model, PersistenceProvider, Retriever, Validator
from sifaka.core.thought import Thought

logger = logging.getLogger(__name__)


class ChainResult:
    """Result of a chain execution."""

    def __init__(self, thought: Thought, success: bool):
        self.thought = thought
        self.success = success

    @property
    def text(self) -> str:
        """Return the generated text."""
        return self.thought.text

    @property
    def validation_results(self) -> List[Any]:
        """Return the validation results."""
        return self.thought.validation_results

    @property
    def critic_feedback(self) -> List[Any]:
        """Return the critic feedback."""
        return self.thought.critic_feedback

    @property
    def history(self) -> List[Thought]:
        """Return the history of thoughts."""
        return self.thought.history


class Chain:
    """
    Chain orchestrates the generation, validation, and improvement of text.

    A chain consists of:
    - A model that generates text
    - Validators that check if the text meets certain criteria
    - Critics that provide feedback on how to improve the text
    - Retrievers that fetch relevant information
    - A persistence provider that stores thoughts

    The chain executes in the following steps:
    1. Create a thought with the prompt
    2. Use retrievers to fetch relevant information
    3. Generate text using the model
    4. Validate the text using validators
    5. If validation fails and apply_critics_on_validation_failure is True:
       a. Get feedback from critics
       b. Generate improved text using the model
       c. Repeat from step 4 until validation passes or max_iterations is reached
    6. Persist the thought
    7. Return the result
    """

    def __init__(
        self,
        model: Model,
        validators: Optional[List[Validator]] = None,
        critics: Optional[List[Critic]] = None,
        retrievers: Optional[List[Retriever]] = None,
        persistence: Optional[PersistenceProvider] = None,
        max_iterations: int = 3,
        apply_critics_on_validation_failure: bool = True,
    ):
        """
        Initialize a chain.

        Args:
            model: The model to use for text generation.
            validators: Validators to check if the text meets certain criteria.
            critics: Critics to provide feedback on how to improve the text.
            retrievers: Retrievers to fetch relevant information.
            persistence: Provider to persist thoughts.
            max_iterations: Maximum number of iterations to try to improve the text.
            apply_critics_on_validation_failure: Whether to apply critics when validation fails.
        """
        self.model = model
        self.validators = validators or []
        self.critics = critics or []
        self.retrievers = retrievers or []
        self.persistence = persistence
        self.max_iterations = max_iterations
        self.apply_critics_on_validation_failure = apply_critics_on_validation_failure

    def generate(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> ChainResult:
        """
        Generate text based on the prompt.

        Args:
            prompt: The prompt to generate text from.
            metadata: Optional metadata to include in the thought.

        Returns:
            A ChainResult containing the final thought and success status.
        """
        # Create initial thought
        thought = Thought(prompt=prompt, metadata=metadata or {})

        # Retrieve relevant information
        self._retrieve(thought)

        # Generate initial text
        thought.text = self.model.generate(thought)
        logger.info(f"Generated text: {thought.text[:100]}...")

        # Validate and improve
        iterations = 0
        while iterations < self.max_iterations:
            # Validate
            self._validate(thought)

            # Check if validation passed
            if thought.validation_passed:
                logger.info("Validation passed")
                break

            # If validation failed and we should apply critics
            if self.apply_critics_on_validation_failure:
                logger.info(f"Validation failed, applying critics (iteration {iterations + 1})")

                # Get feedback from critics
                self._critique(thought)

                # Generate improved text
                new_text = self.model.generate(thought)

                # Create new version of thought with improved text
                thought = thought.create_new_version(new_text)
                logger.info(f"Generated improved text: {thought.text[:100]}...")

                iterations += 1
            else:
                logger.info("Validation failed, not applying critics")
                break

        # Persist the thought
        if self.persistence:
            thought_id = self.persistence.save(thought)
            thought.metadata["thought_id"] = thought_id
            logger.info(f"Persisted thought with ID: {thought_id}")

        # Return result
        success = thought.validation_passed
        return ChainResult(thought=thought, success=success)

    def _retrieve(self, thought: Thought) -> None:
        """
        Retrieve relevant information for the thought.

        Args:
            thought: The thought to retrieve information for.
        """
        if not self.retrievers:
            return

        for retriever in self.retrievers:
            try:
                results = retriever.retrieve(thought.prompt, thought)
                for result in results:
                    thought.add_retrieved_context(
                        source=retriever.name,
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                        relevance_score=result.get("relevance_score"),
                    )
                logger.info(f"Retrieved {len(results)} documents from {retriever.name}")
            except Exception as e:
                logger.error(f"Error retrieving from {retriever.name}: {e}")

    def _validate(self, thought: Thought) -> None:
        """
        Validate the text in the thought.

        Args:
            thought: The thought containing the text to validate.
        """
        if not self.validators:
            thought.validation_passed = True
            return

        for validator in self.validators:
            try:
                passed = validator.validate(thought)
                # The validator should have added a validation result to the thought
                logger.info(f"Validation {validator.name}: {'passed' if passed else 'failed'}")
            except Exception as e:
                logger.error(f"Error validating with {validator.name}: {e}")
                thought.add_validation_result(
                    validator_name=validator.name,
                    passed=False,
                    message=f"Error: {str(e)}",
                )

        # Update validation status
        thought.update_validation_status()

    def _critique(self, thought: Thought) -> None:
        """
        Critique the text in the thought.

        Args:
            thought: The thought containing the text to critique.
        """
        if not self.critics:
            return

        for critic in self.critics:
            try:
                feedback = critic.critique(thought)
                # The critic should have added feedback to the thought
                logger.info(f"Critic {critic.name} provided feedback")
            except Exception as e:
                logger.error(f"Error critiquing with {critic.name}: {e}")
                thought.add_critic_feedback(
                    critic_name=critic.name,
                    feedback=f"Error: {str(e)}",
                )
