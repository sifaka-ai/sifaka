"""
Chain orchestration for Sifaka.

This module defines the Chain class, which is the main entry point for the Sifaka framework.
It orchestrates the generation, validation, and improvement of text using LLMs.

The Chain class uses a builder pattern to provide a fluent API for configuring and
executing LLM operations. It allows you to specify which model to use, set the prompt
for generation, add validators to check if the generated text meets requirements,
add improvers to enhance the quality of the generated text, and configure model options.

Example:
    ```python
    from sifaka import Chain
    from sifaka.validators import length, clarity

    result = (Chain()
        .with_model("openai:gpt-4")
        .with_prompt("Write a short story about a robot.")
        .validate_with(length(min_words=50, max_words=200))
        .improve_with(clarity())
        .run())

    print(f"Result passed validation: {result.passed}")
    print(result.text)
    ```
"""

from typing import Union, List, Dict, Any, Optional, Callable

from sifaka.interfaces import Model, Validator, Improver
from sifaka.factories import create_model, create_model_from_string
from sifaka.results import Result
from sifaka.errors import ChainError


class Chain:
    """Main orchestrator for the generation, validation, and improvement flow.

    The Chain class is the central component of Sifaka. It coordinates the process of:
    1. Generating text using a language model
    2. Validating the generated text against specified criteria
    3. Improving the text using various improvement strategies

    The class uses a builder pattern to provide a fluent API, allowing method chaining
    for a more readable and intuitive interface.

    Attributes:
        _model: The model to use for generation.
        _prompt: The prompt to use for generation.
        _validators: List of validators to apply to the generated text.
        _improvers: List of improvers to apply to the generated text.
        _options: Options to pass to the model during generation.

    Example:
        ```python
        from sifaka import Chain

        chain = Chain()
        chain.with_model("openai:gpt-4")
        chain.with_prompt("Write a short story about a robot.")
        result = chain.run()

        # Or using method chaining:
        result = (Chain()
            .with_model("openai:gpt-4")
            .with_prompt("Write a short story about a robot.")
            .run())
        ```
    """

    def __init__(self, model_factory: Optional[Callable[[str, str], Model]] = None):
        """Initialize a new Chain instance.

        Args:
            model_factory: Optional factory function for creating models.
                If not provided, the default factory function will be used.
                The factory function should take a provider and model name and return a Model instance.
        """
        self._model: Optional[Model] = None
        self._prompt: Optional[str] = None
        self._validators: List[Validator] = []
        self._improvers: List[Improver] = []
        self._options: Dict[str, Any] = {}

        # Use the provided model factory or use the registry
        if model_factory is not None:
            self._model_factory = model_factory
        else:
            # Use the registry
            self._model_factory = create_model

    def with_model(self, model: Union[str, Model]) -> "Chain":
        """Set the model to use for generation.

        This method configures the model that will be used to generate text when the chain
        is run. You can provide either a model instance or a string in the format
        "provider:model_name", which will be used to create a model instance.

        Args:
            model: Either a model instance or a string in the format "provider:model_name".
                  Supported providers include "openai", "anthropic", "gemini", and "mock".

        Returns:
            The chain instance for method chaining.

        Examples:
            ```python
            # Using a string
            chain = Chain().with_model("openai:gpt-4")

            # Using a model instance
            from sifaka.models import OpenAIModel
            model = OpenAIModel("gpt-4", api_key="your-api-key")
            chain = Chain().with_model(model)

            # Using dependency injection
            from sifaka.factories import create_model
            chain = Chain(model_factory=create_model).with_model("openai:gpt-4")
            ```

        Raises:
            FactoryError: If the provider or model is not found.
            ValueError: If the model string is not in the correct format.
        """
        if isinstance(model, str):
            # Use create_model_from_string to handle the model string
            self._model = create_model_from_string(model)
        else:
            self._model = model
        return self

    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt to use for generation.

        Args:
            prompt: The prompt to use for generation.

        Returns:
            The chain instance for method chaining.
        """
        self._prompt = prompt
        return self

    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain.

        Args:
            validator: The validator to add.

        Returns:
            The chain instance for method chaining.
        """
        self._validators.append(validator)
        return self

    def improve_with(self, improver: Improver) -> "Chain":
        """Add an improver to the chain.

        Args:
            improver: The improver to add.

        Returns:
            The chain instance for method chaining.
        """
        self._improvers.append(improver)
        return self

    def with_options(self, **options: Any) -> "Chain":
        """Set options for the model.

        Args:
            **options: Options to pass to the model.

        Returns:
            The chain instance for method chaining.
        """
        self._options.update(options)
        return self

    def run(self) -> Result:
        """Execute the chain and return the result.

        This method runs the complete chain execution process:
        1. Checks that the chain is properly configured (has a model and prompt)
        2. Generates text using the model and prompt
        3. Validates the generated text using all configured validators
        4. If all validations pass, improves the text using all configured improvers
        5. Returns a Result object containing the final text and all validation/improvement results

        The validation process stops at the first validator that fails, returning a failed result.
        Improvers are applied in sequence, with each improver receiving the text from the previous one.

        Returns:
            The result of the chain execution, containing:
            - The final text after all validations and improvements
            - Whether all validations passed
            - The results of all validations
            - The results of all improvements

        Raises:
            ChainError: If the chain is not properly configured (missing model or prompt).
            ModelError: If there is an error with the model during generation.

        Example:
            ```python
            from sifaka import Chain
            from sifaka.validators import length, clarity

            result = (Chain()
                .with_model("openai:gpt-4")
                .with_prompt("Write a short story about a robot.")
                .validate_with(length(min_words=50, max_words=200))
                .improve_with(clarity())
                .run())

            if result.passed:
                print("Chain execution succeeded")
                print(result.text)
            else:
                print("Chain execution failed validation")
                print(result.validation_results[0].message)
            ```
        """
        import logging
        import time
        from sifaka.utils.error_handling import chain_context, log_error

        logger = logging.getLogger(__name__)

        # Check configuration
        if not self._model:
            raise ChainError(
                message="Model not specified",
                component="Chain",
                operation="run",
                suggestions=["Use with_model() to specify a model before running the chain"],
                metadata={"prompt_specified": self._prompt is not None},
            )

        if not self._prompt:
            raise ChainError(
                message="Prompt not specified",
                component="Chain",
                operation="run",
                suggestions=["Use with_prompt() to specify a prompt before running the chain"],
                metadata={"model_specified": self._model is not None},
            )

        # Start timing
        start_time = time.time()

        # Log chain execution start
        logger.debug(
            f"Starting chain execution with model={self._model.__class__.__name__}, "
            f"prompt_length={len(self._prompt)}, "
            f"validators={len(self._validators)}, "
            f"improvers={len(self._improvers)}"
        )

        try:
            # Generate initial text
            with chain_context(
                component="Chain",
                operation="generation",
                message_prefix="Failed to generate text",
                suggestions=["Check the model configuration", "Verify the prompt format"],
                metadata={
                    "model_type": self._model.__class__.__name__,
                    "prompt_length": len(self._prompt),
                    "options": self._options,
                },
            ):
                text = self._model.generate(self._prompt, **self._options)
                logger.debug(f"Generated text of length {len(text)}")

            # Validate text
            validation_results = []
            for i, validator in enumerate(self._validators):
                try:
                    with chain_context(
                        component="Chain",
                        operation=f"validation_{i}",
                        message_prefix=f"Failed to validate text with {validator.__class__.__name__}",
                        suggestions=["Check the validator configuration", "Verify the text format"],
                        metadata={
                            "validator_type": validator.__class__.__name__,
                            "text_length": len(text),
                            "validator_index": i,
                        },
                    ):
                        result = validator.validate(text)
                        validation_results.append(result)
                        logger.debug(
                            f"Validation {i} with {validator.__class__.__name__}: "
                            f"passed={result.passed}"
                        )

                        if not result.passed:
                            # Log validation failure
                            logger.info(
                                f"Chain execution stopped at validator {i} ({validator.__class__.__name__}): "
                                f"{result.message}"
                            )

                            # Return early with failed result
                            execution_time = time.time() - start_time
                            return Result(
                                text=text,
                                passed=False,
                                validation_results=validation_results,
                                improvement_results=[],
                                execution_time_ms=execution_time * 1000,
                            )
                except Exception as e:
                    # Log the error
                    log_error(e, logger, component="Chain", operation=f"validation_{i}")

                    # Add the error to validation results
                    from sifaka.results import ValidationResult

                    error_result = ValidationResult(
                        passed=False,
                        message=f"Validation error: {str(e)}",
                        details={"error_type": type(e).__name__},
                        issues=[
                            f"Validator {validator.__class__.__name__} failed with error: {str(e)}"
                        ],
                        suggestions=["Check the validator configuration", "Verify the text format"],
                    )
                    validation_results.append(error_result)

                    # Return early with failed result
                    execution_time = time.time() - start_time
                    return Result(
                        text=text,
                        passed=False,
                        validation_results=validation_results,
                        improvement_results=[],
                        execution_time_ms=execution_time * 1000,
                    )

            # Improve text
            improvement_results = []
            for i, improver in enumerate(self._improvers):
                try:
                    with chain_context(
                        component="Chain",
                        operation=f"improvement_{i}",
                        message_prefix=f"Failed to improve text with {improver.__class__.__name__}",
                        suggestions=["Check the improver configuration", "Verify the text format"],
                        metadata={
                            "improver_type": improver.__class__.__name__,
                            "text_length": len(text),
                            "improver_index": i,
                        },
                    ):
                        improved_text, result = improver.improve(text)
                        improvement_results.append(result)
                        logger.debug(
                            f"Improvement {i} with {improver.__class__.__name__}: "
                            f"changes_made={result.changes_made}, "
                            f"text_length_before={len(text)}, "
                            f"text_length_after={len(improved_text)}"
                        )
                        text = improved_text
                except Exception as e:
                    # Log the error
                    log_error(e, logger, component="Chain", operation=f"improvement_{i}")

                    # Add the error to improvement results
                    from sifaka.results import ImprovementResult

                    error_result = ImprovementResult(
                        original_text=text,
                        improved_text=text,
                        changes_made=False,
                        message=f"Improvement error: {str(e)}",
                        details={"error_type": type(e).__name__},
                    )
                    improvement_results.append(error_result)

                    # Continue with the next improver
                    continue

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log successful chain execution
            logger.info(
                f"Chain execution completed successfully in {execution_time:.2f}s: "
                f"validators_passed={len(validation_results)}, "
                f"improvements_applied={len(improvement_results)}, "
                f"final_text_length={len(text)}"
            )

            # Return successful result
            return Result(
                text=text,
                passed=True,
                validation_results=validation_results,
                improvement_results=improvement_results,
                execution_time_ms=execution_time * 1000,
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="Chain", operation="run")

            # Calculate execution time
            execution_time = time.time() - start_time

            # Raise as ChainError with more context
            raise ChainError(
                message=f"Chain execution failed: {str(e)}",
                component="Chain",
                operation="run",
                suggestions=[
                    "Check the model configuration",
                    "Verify the prompt format",
                    "Check the validator and improver configurations",
                ],
                metadata={
                    "model_type": self._model.__class__.__name__,
                    "prompt_length": len(self._prompt),
                    "validators_count": len(self._validators),
                    "improvers_count": len(self._improvers),
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                },
            )
