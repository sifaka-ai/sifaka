"""Chain orchestration for Sifaka.

This module defines the Chain class, which is the main entry point for the Sifaka framework.
It orchestrates the process of generating text using language models, validating the text
against specified criteria, and improving the text using specialized critics when validation fails.

The Chain class uses a builder pattern to provide a fluent API for configuring and
executing LLM operations. It allows you to specify which model to use, set the prompt
for generation, add validators to check if the generated text meets requirements,
add critics to provide feedback for improvement when validation fails, and configure model options.

Example:
    ```python
    from sifaka import Chain
    from sifaka.validators import length, prohibited_content
    from sifaka.critics.reflexion import create_reflexion_critic
    from sifaka.models.openai import OpenAIModel

    # Create a model
    model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create a chain with validators and critics
    chain = (Chain()
        .with_model(model)
        .with_prompt("Write a short story about a robot.")
        .validate_with(length(min_words=50, max_words=500))
        .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
        .improve_with(create_reflexion_critic(model=model))
    )

    # Run the chain
    result = chain.run()

    if result.passed:
        print("Chain execution succeeded!")
        print(result.text)
    else:
        print("Chain execution failed validation")
        print(result.validation_results[0].message)
    ```

    Using configuration:
    ```python
    from sifaka import Chain
    from sifaka.config import SifakaConfig, ModelConfig
    from sifaka.models.openai import OpenAIModel

    # Create a custom configuration
    config = SifakaConfig(
        model=ModelConfig(temperature=0.8, max_tokens=500),
        debug=True
    )

    # Create a model
    model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Use with a chain
    chain = (Chain(config)
        .with_model(model)
        .with_prompt("Write a short story about a robot.")
        .run())
    ```
"""

from typing import Any, Callable, Dict, List, Optional, Union

from sifaka.config import SifakaConfig
from sifaka.errors import ChainError
from sifaka.factories import create_model, create_model_from_string
from sifaka.interfaces import Improver, Model, Validator
from sifaka.results import Result


class Chain:
    """Main orchestrator for text generation, validation, and improvement.

    The Chain class is the central component of the Sifaka framework, coordinating
    the process of generating text using language models, validating it against
    specified criteria, and improving it using specialized critics when validation fails.

    It follows a fluent interface pattern (builder pattern) for easy configuration,
    allowing you to chain method calls to set up the desired behavior.

    The typical workflow is:
    1. Create a Chain instance
    2. Configure it with a model, prompt, validators, and critics
    3. Run the chain to generate and validate text, with critics providing feedback for improvement when validation fails
    4. Process the result

    Attributes:
        _model (Optional[Model]): The model used for text generation.
        _prompt (Optional[str]): The prompt used for text generation.
        _validators (List[Validator]): Validators used to check text quality.
        _improvers (List[Improver]): Critics used to improve text quality.
        _config (SifakaConfig): Configuration for the chain and its components.
        _options (Dict[str, Any]): Additional options for backward compatibility.
        _model_factory (Callable): Factory function for creating models from strings.

    Example:
        ```python
        from sifaka import Chain
        from sifaka.models.openai import OpenAIModel
        from sifaka.validators import length, prohibited_content
        from sifaka.critics.reflexion import create_reflexion_critic

        # Create a model
        model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

        # Create a chain with validators and critics
        chain = (Chain()
            .with_model(model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(length(min_words=100, max_words=500))
            .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
            .improve_with(create_reflexion_critic(model=model))
        )

        # Run the chain
        result = chain.run()

        # Check the result
        if result.passed:
            print("Chain execution succeeded!")
            print(result.text)
        else:
            print("Chain execution failed validation")
            print(result.validation_results[0].message)

        # Using string-based model specification
        result = (Chain()
            .with_model("openai:gpt-4")  # Will use OPENAI_API_KEY environment variable
            .with_prompt("Write a short story about a robot.")
            .run())

        # With configuration:
        from sifaka.config import SifakaConfig, ModelConfig
        config = SifakaConfig(model=ModelConfig(temperature=0.8))
        result = (Chain(config)
            .with_model(model)
            .with_prompt("Write a short story about a robot.")
            .run())
        ```
    """

    def __init__(
        self,
        config: Optional[SifakaConfig] = None,
        model_factory: Optional[Callable[[str, str], Model]] = None,
        max_improvement_iterations: int = 3,
    ):
        """Initialize a new Chain instance with optional configuration and model factory.

        This constructor creates a new Chain instance that can be configured with a model,
        prompt, validators, and improvers. The chain is not ready to run until at least
        a model and prompt have been specified.

        Args:
            config (Optional[SifakaConfig]): Optional configuration for the chain and its components.
                If not provided, a default configuration will be used. The configuration can
                include settings for the model, validators, and critics.
            model_factory (Optional[Callable[[str, str], Model]]): Optional factory function for
                creating models from strings. If not provided, the default factory function from
                the registry system will be used. The factory function should take a provider
                and model name and return a Model instance.
            max_improvement_iterations (int): Maximum number of improvement iterations when
                applying the feedback loop between validators and improvers. Default is 3.

        Example:
            ```python
            # Create a chain with default configuration
            chain1 = Chain()

            # Create a chain with custom configuration
            from sifaka.config import SifakaConfig, ModelConfig
            config = SifakaConfig(
                model=ModelConfig(temperature=0.7, max_tokens=500),
                debug=True
            )
            chain2 = Chain(config)

            # Create a chain with custom model factory
            def my_model_factory(provider: str, model_name: str) -> Model:
                # Custom logic to create models
                return MyCustomModel(model_name)

            chain3 = Chain(model_factory=my_model_factory)

            # Create a chain with custom improvement iterations
            chain4 = Chain(max_improvement_iterations=5)
            ```
        """
        self._model: Optional[Model] = None
        self._prompt: Optional[str] = None
        self._validators: List[Validator] = []
        self._improvers: List[Improver] = []
        self._config: SifakaConfig = config or SifakaConfig()
        # For backward compatibility
        self._options: Dict[str, Any] = {}
        # Maximum number of improvement iterations
        self._max_improvement_iterations: int = max_improvement_iterations

        # Use the provided model factory or use the registry
        if model_factory is not None:
            self._model_factory = model_factory
        else:
            # Use the registry
            self._model_factory = create_model

    def with_config(self, config: SifakaConfig) -> "Chain":
        """Set the configuration for the chain.

        This method allows you to update the configuration for the chain and its components.

        Args:
            config: The configuration to use.

        Returns:
            The chain instance for method chaining.

        Examples:
            ```python
            from sifaka.config import SifakaConfig, ModelConfig

            # Create a custom configuration
            config = SifakaConfig(
                model=ModelConfig(temperature=0.8, max_tokens=1000),
                debug=True
            )

            # Update the chain configuration
            chain = Chain().with_config(config)
            ```
        """
        self._config = config
        return self

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

            # Using configuration
            from sifaka.config import SifakaConfig, ModelConfig
            config = SifakaConfig(model=ModelConfig(temperature=0.8))
            chain = Chain(config).with_model("openai:gpt-4")
            ```

        Raises:
            FactoryError: If the provider or model is not found.
            ValueError: If the model string is not in the correct format.
        """
        if isinstance(model, str):
            # Use create_model_from_string to handle the model string
            # Pass model options from configuration
            model_options = self._config.get_model_options()
            self._model = create_model_from_string(model, **model_options)
        else:
            self._model = model
        return self

    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt to use for text generation.

        This method sets the prompt that will be sent to the language model when
        the chain is run. The prompt is the input text that guides the model's
        generation process.

        Args:
            prompt (str): The prompt text to use for generation. This should be
                a clear instruction or context for the model to generate from.

        Returns:
            Chain: The chain instance for method chaining.

        Example:
            ```python
            # Simple prompt
            chain = Chain().with_prompt("Write a short story about a robot.")

            # More detailed prompt
            chain = Chain().with_prompt(
                "Write a short story about a robot who discovers emotions. "
                "The story should be heartwarming and suitable for children. "
                "Include themes of friendship and self-discovery."
            )

            # Prompt with specific formatting
            chain = Chain().with_prompt(
                "Generate a JSON object with the following structure:\n"
                "{\n"
                "  \"name\": \"string\",\n"
                "  \"age\": number,\n"
                "  \"interests\": [\"string\", \"string\"]\n"
                "}"
            )
            ```
        """
        self._prompt = prompt
        return self

    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain.

        This method adds a validator to the chain, which will be used to validate
        the generated text. Validators check if the text meets certain criteria,
        such as length, content, or format requirements.

        If the validator has a `configure` method, it will be called with the
        validator options from the configuration.

        Args:
            validator: The validator to add.

        Returns:
            The chain instance for method chaining.

        Examples:
            ```python
            from sifaka import Chain
            from sifaka.validators import length
            from sifaka.critics import self_refine

            chain = Chain()
                .with_model("openai:gpt-4")
                .with_prompt("Write a short story about a robot.")
                .validate_with(length(min_words=50, max_words=200))
                .validate_with(self_refine())
            ```
        """
        # Configure the validator with options from the configuration
        validator_options = self._config.get_validator_options()
        if hasattr(validator, "configure"):
            validator.configure(**validator_options)

        self._validators.append(validator)
        return self

    def improve_with(self, improver: Improver) -> "Chain":
        """Add an improver to the chain.

        This method adds an improver to the chain, which will be used to provide feedback
        when validation fails. Critics provide suggestions on how to improve the text,
        which are then sent back to the model to generate improved text.

        If the improver has a `configure` method, it will be called with the
        critic options from the configuration.

        Args:
            improver: The improver to add.

        Returns:
            The chain instance for method chaining.

        Examples:
            ```python
            from sifaka import Chain
            from sifaka.critics import self_refine
            from sifaka.validators import length

            chain = Chain()
                .with_model("openai:gpt-4")
                .with_prompt("Write a short story about a robot.")
                .validate_with(length(min_words=50, max_words=200))
                .improve_with(self_refine())  # Used when validation fails
                .with_options(apply_improvers_on_validation_failure=True)
            ```
        """
        # Configure the improver with options from the configuration
        critic_options = self._config.get_critic_options()
        if hasattr(improver, "configure"):
            improver.configure(**critic_options)

        self._improvers.append(improver)
        return self

    def with_options(self, **options: Any) -> "Chain":
        """Set options to pass to the model during generation.

        This method allows you to configure options that will be passed to the model
        during text generation. These options can include parameters like temperature,
        max_tokens, and top_p, which control the behavior of the model.

        Note: These options will override any options set in the configuration.

        Args:
            **options: Keyword arguments to pass to the model during generation.
                Special options:
                - apply_improvers_on_validation_failure (bool): If True, improvers will be applied
                  when validation fails. Default is True.

        Returns:
            The chain instance for method chaining.

        Examples:
            ```python
            chain = Chain()
                .with_model("openai:gpt-4")
                .with_prompt("Write a short story about a robot.")
                .with_options(
                    temperature=0.7,
                    max_tokens=500,
                    top_p=0.9,
                    apply_improvers_on_validation_failure=True
                )
            ```
        """
        # Handle special options
        if "apply_improvers_on_validation_failure" in options:
            self._options["apply_improvers_on_validation_failure"] = options.pop(
                "apply_improvers_on_validation_failure"
            )
        else:
            # Default to True for apply_improvers_on_validation_failure
            self._options["apply_improvers_on_validation_failure"] = True

        # Update the model configuration with the provided options
        for key, value in options.items():
            if hasattr(self._config.model, key):
                setattr(self._config.model, key, value)
            else:
                # Store in custom options
                self._config.model.custom[key] = value

        return self

    def run_with_text(self, text: str) -> Result:
        """Execute the chain with pre-generated text and return the result.

        This method runs the chain with pre-generated text, skipping the text generation step.
        It's useful when you already have text that you want to validate and improve.

        The method follows the same process as run(), but starts with the provided text
        instead of generating new text from the prompt:
        1. Validates the provided text using all configured validators
        2. If validation fails and apply_improvers_on_validation_failure is True:
           a. Gets feedback from critics on how to improve the text
           b. Sends the original text + feedback to the model to generate improved text
           c. Re-validates the improved text
           d. Repeats steps 2a-2c until validation passes or max iterations is reached
        3. If validation passes, returns the result immediately
        4. Returns a Result object with the final text and all validation/improvement results

        Args:
            text (str): The pre-generated text to validate and improve

        Returns:
            Result: A Result object containing the final text, validation results, and improvement results

        Example:
            ```python
            from sifaka import Chain
            from sifaka.validators import length
            from sifaka.critics.reflexion import create_reflexion_critic

            # Generate text separately
            text = "This is some pre-generated text that needs validation and improvement."

            # Create a chain and run it with the pre-generated text
            chain = (Chain()
                .with_model("openai:gpt-4")
                .validate_with(length(min_words=50, max_words=200))
                .improve_with(create_reflexion_critic(model="openai:gpt-4"))
            )

            # Run the chain with pre-generated text
            result = chain.run_with_text(text)

            if result.passed:
                print("Chain execution succeeded")
                print(f"Initial text: {result.initial_text}")
                print(f"Final text: {result.text}")
            else:
                print("Chain execution failed validation")
                print(result.validation_results[0].message)
            ```
        """
        import logging
        import time

        from sifaka.utils.error_handling import log_error

        logger = logging.getLogger(__name__)

        # Check configuration
        if not self._model:
            raise ChainError(
                message="Model not specified",
                component="Chain",
                operation="run_with_text",
                suggestions=["Use with_model() to specify a model before running the chain"],
                metadata={"text_length": len(text)},
            )

        # Start timing
        start_time = time.time()

        # Log chain execution start
        logger.debug(
            f"Starting chain execution with pre-generated text, model={self._model.__class__.__name__}, "
            f"text_length={len(text)}, "
            f"validators={len(self._validators)}, "
            f"improvers={len(self._improvers)}"
        )

        try:
            # Initialize results lists
            from sifaka.results import ImprovementResult as ResultsImprovementResult
            from sifaka.results import ValidationResult as ResultsValidationResult

            validation_results: List[ResultsValidationResult] = []
            improvement_results: List[ResultsImprovementResult] = []

            # Main improvement loop
            current_text = text
            initial_text = text  # Store the initial text

            for iteration in range(self._max_improvement_iterations):
                logger.debug(
                    f"Starting improvement iteration {iteration+1}/{self._max_improvement_iterations}"
                )

                # Validate current text
                validation_passed, validation_results, validation_feedback = self._validate_text(
                    current_text
                )

                # If validation fails and we should apply improvers on validation failure
                if not validation_passed and self._options.get(
                    "apply_improvers_on_validation_failure", True
                ):
                    logger.info(f"Validation failed in iteration {iteration+1}, applying improvers")

                    # Get feedback from critics
                    critic_feedback = self._get_critic_feedback(current_text, validation_feedback)

                    # Generate improved text using the model with feedback
                    improved_text = self._generate_improved_text(current_text, critic_feedback)

                    # Create improvement result
                    from sifaka.results import ImprovementResult as SifakaImprovementResult

                    improvement_result = SifakaImprovementResult(
                        _original_text=current_text,
                        _improved_text=improved_text,
                        _changes_made=improved_text != current_text,
                        message=f"Improvement iteration {iteration+1}",
                        _details={
                            "validation_feedback": validation_feedback,
                            "critic_feedback": critic_feedback,
                            "iteration": iteration + 1,
                        },
                    )
                    improvement_results.append(improvement_result)

                    # Update current text for next iteration
                    current_text = improved_text

                    # If no changes were made, break the loop
                    if not improvement_result.changes_made:
                        logger.debug(f"No changes made in iteration {iteration+1}, breaking loop")
                        break

                    # Continue to next iteration to validate the improved text
                    continue

                # If validation fails and we should not apply improvers on validation failure
                elif not validation_passed:
                    logger.info(f"Chain execution stopped at validation in iteration {iteration+1}")
                    execution_time = time.time() - start_time
                    return Result(
                        text=current_text,
                        initial_text=initial_text,
                        passed=False,
                        validation_results=validation_results,
                        improvement_results=improvement_results,
                        execution_time_ms=execution_time * 1000,
                    )

                # If validation passes, return the result immediately
                else:
                    logger.debug("Text passed validation, returning result")
                    break

            # Final validation to ensure the text still passes after all improvements
            if current_text != text:
                validation_passed, validation_results, _ = self._validate_text(current_text)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log chain execution result
            if validation_passed:
                logger.info(
                    f"Chain execution completed successfully in {execution_time:.2f}s: "
                    f"validators_passed={len(validation_results)}, "
                    f"improvements_applied={len(improvement_results)}, "
                    f"final_text_length={len(current_text)}"
                )
            else:
                logger.info(
                    f"Chain execution failed validation in {execution_time:.2f}s: "
                    f"validators_failed={len(validation_results)}, "
                    f"improvements_attempted={len(improvement_results)}, "
                    f"final_text_length={len(current_text)}"
                )

            # Return result
            return Result(
                text=current_text,
                initial_text=initial_text,
                passed=validation_passed,
                validation_results=validation_results,
                improvement_results=improvement_results,
                execution_time_ms=execution_time * 1000,
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="Chain", operation="run_with_text")

            # Calculate execution time
            execution_time = time.time() - start_time

            # Raise as ChainError with more context
            raise ChainError(
                message=f"Chain execution failed: {str(e)}",
                component="Chain",
                operation="run_with_text",
                suggestions=[
                    "Check the model configuration",
                    "Verify the text format",
                    "Check the validator and improver configurations",
                ],
                metadata={
                    "model_type": self._model.__class__.__name__,
                    "text_length": len(text),
                    "validators_count": len(self._validators),
                    "improvers_count": len(self._improvers),
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                    "debug_mode": self._config.debug,
                    "model_options": self._config.get_model_options(),
                },
            )

    def _validate_text(self, text: str) -> tuple[bool, List[Any], Dict[str, Any]]:
        """Validate text and collect feedback from validators.

        Args:
            text: The text to validate.

        Returns:
            A tuple of (passed, validation_results, validation_feedback).
        """
        import logging

        logger = logging.getLogger(__name__)

        from sifaka.results import ValidationResult as ResultsValidationResult
        from sifaka.utils.error_handling import chain_context, log_error

        validation_results: List[ResultsValidationResult] = []
        validation_feedback: Dict[str, Any] = {
            "issues": [],
            "suggestions": [],
        }

        # Run all validators
        for i, validator in enumerate(self._validators):
            try:
                with chain_context(
                    operation=f"validation_{i}",
                    message_prefix=f"Failed to validate text with {validator.__class__.__name__}",
                    suggestions=[
                        "Check the validator configuration",
                        "Verify the text format",
                    ],
                    metadata={
                        "validator_type": validator.__class__.__name__,
                        "text_length": len(text),
                        "validator_index": i,
                    },
                ):
                    result = validator.validate(text)

                    # Cast the result to the expected type
                    from sifaka.results import ValidationResult

                    typed_result: ResultsValidationResult = ValidationResult(
                        passed=result.passed,
                        message=(result.message if hasattr(result, "message") else ""),
                        _details=(result.details if hasattr(result, "details") else {}),
                        issues=result.issues if hasattr(result, "issues") else None,
                        suggestions=(
                            result.suggestions if hasattr(result, "suggestions") else None
                        ),
                    )
                    validation_results.append(typed_result)
                    logger.debug(
                        f"Validation {i} with {validator.__class__.__name__}: "
                        f"passed={result.passed}"
                    )

                    # Add issues and suggestions to feedback
                    if hasattr(result, "issues") and result.issues:
                        validation_feedback["issues"].extend(result.issues)
                    if hasattr(result, "suggestions") and result.suggestions:
                        validation_feedback["suggestions"].extend(result.suggestions)

                    # If validation failed, return early
                    if not result.passed:
                        return False, validation_results, validation_feedback

            except Exception as e:
                # Log the error
                log_error(e, logger, component="Chain", operation=f"validation_{i}")

                # Add the error to validation results
                from sifaka.results import ValidationResult

                error_result: ResultsValidationResult = ValidationResult(
                    passed=False,
                    message=f"Validation error: {str(e)}",
                    _details={"error_type": type(e).__name__},
                    issues=[
                        f"Validator {validator.__class__.__name__} failed with error: {str(e)}"
                    ],
                    suggestions=[
                        "Check the validator configuration",
                        "Verify the text format",
                    ],
                )
                validation_results.append(error_result)

                # Add error to feedback
                validation_feedback["issues"].append(
                    f"Validator {validator.__class__.__name__} failed with error: {str(e)}"
                )

                # Return early with failed result
                return False, validation_results, validation_feedback

        # All validators passed
        return True, validation_results, validation_feedback

    def _get_critic_feedback(
        self, text: str, validation_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get feedback from critics on how to improve the text.

        Args:
            text: The text to get feedback on.
            validation_feedback: Feedback from validators.

        Returns:
            Feedback from critics on how to improve the text.
        """
        import logging

        logger = logging.getLogger(__name__)

        from sifaka.utils.error_handling import chain_context, log_error

        critic_feedback: Dict[str, Any] = {
            "issues": [],
            "suggestions": [],
        }

        # Run all critics to get feedback
        for i, improver in enumerate(self._improvers):
            try:
                with chain_context(
                    operation=f"critic_feedback_{i}",
                    message_prefix=f"Failed to get feedback from {improver.__class__.__name__}",
                    suggestions=[
                        "Check the critic configuration",
                        "Verify the text format",
                    ],
                    metadata={
                        "critic_type": improver.__class__.__name__,
                        "text_length": len(text),
                        "critic_index": i,
                    },
                ):
                    # Use the improver to get feedback
                    # Note: This is a temporary approach until critics are updated to provide feedback
                    improved_text, result = improver.improve(text)

                    # Extract feedback from the result
                    if hasattr(result, "details") and result.details:
                        # If the improver provides a critique, use it
                        if "critique" in result.details:
                            critique = result.details["critique"]
                            if isinstance(critique, str):
                                critic_feedback["suggestions"].append(critique)
                            elif isinstance(critique, dict):
                                # Handle structured critique
                                if "issues" in critique:
                                    critic_feedback["issues"].extend(critique["issues"])
                                if "suggestions" in critique:
                                    critic_feedback["suggestions"].extend(critique["suggestions"])
                            elif isinstance(critique, list):
                                # Assume list of suggestions
                                critic_feedback["suggestions"].extend(critique)

                    # If no critique is available, infer feedback from changes
                    if result.changes_made and improved_text != text:
                        critic_feedback["suggestions"].append(
                            f"Consider improvements suggested by {improver.__class__.__name__}"
                        )

            except Exception as e:
                # Log the error
                log_error(e, logger, component="Chain", operation=f"critic_feedback_{i}")

                # Add error to feedback
                critic_feedback["issues"].append(
                    f"Critic {improver.__class__.__name__} failed with error: {str(e)}"
                )

        # Combine validation and critic feedback
        combined_feedback: Dict[str, Any] = {
            "validation_feedback": validation_feedback,
            "critic_feedback": critic_feedback,
            "issues": validation_feedback.get("issues", []) + critic_feedback.get("issues", []),
            "suggestions": validation_feedback.get("suggestions", [])
            + critic_feedback.get("suggestions", []),
        }

        return combined_feedback

    def _generate_improved_text(self, text: str, feedback: Dict[str, Any]) -> str:
        """Generate improved text using the model with feedback.

        Args:
            text: The text to improve.
            feedback: Feedback from validators and critics.

        Returns:
            The improved text.
        """
        import logging

        logger = logging.getLogger(__name__)

        from sifaka.utils.error_handling import chain_context, log_error

        # Create a prompt for the model to improve the text
        improvement_prompt = self._create_improvement_prompt(text, feedback)

        try:
            with chain_context(
                operation="improvement_generation",
                message_prefix="Failed to generate improved text",
                suggestions=[
                    "Check the model configuration",
                    "Verify the feedback format",
                ],
                metadata={
                    "model_type": self._model.__class__.__name__,
                    "text_length": len(text),
                    "feedback_issues": len(feedback.get("issues", [])),
                    "feedback_suggestions": len(feedback.get("suggestions", [])),
                },
            ):
                # Get model options from configuration
                model_options = self._config.get_model_options()
                # We know self._model is not None here because we check in the run method
                assert self._model is not None, "Model should not be None at this point"
                improved_text = self._model.generate(improvement_prompt, **model_options)
                logger.debug(f"Generated improved text of length {len(improved_text)}")

                return improved_text

        except Exception as e:
            # Log the error
            log_error(e, logger, component="Chain", operation="improvement_generation")

            # Return the original text if improvement fails
            logger.warning("Failed to generate improved text, returning original text")
            return text

    def _create_improvement_prompt(self, text: str, feedback: Dict[str, Any]) -> str:
        """Create a prompt for the model to improve the text.

        Args:
            text: The text to improve.
            feedback: Feedback from validators and critics.

        Returns:
            A prompt for the model to improve the text.
        """
        # Format issues and suggestions
        issues_text = ""
        if feedback.get("issues"):
            issues_text = "Issues that need to be addressed:\n"
            for i, issue in enumerate(feedback.get("issues", [])):
                issues_text += f"{i+1}. {issue}\n"

        suggestions_text = ""
        if feedback.get("suggestions"):
            suggestions_text = "Suggestions for improvement:\n"
            for i, suggestion in enumerate(feedback.get("suggestions", [])):
                suggestions_text += f"{i+1}. {suggestion}\n"

        # Create the prompt
        prompt = f"""
        You are tasked with improving the following text based on feedback from validators and critics.

        Original text:
        ```
        {text}
        ```

        {issues_text}

        {suggestions_text}

        IMPORTANT INSTRUCTIONS:
        1. Preserve the core narrative elements (characters, plot, setting, theme)
        2. Address all the issues mentioned in the feedback
        3. Incorporate the suggestions where appropriate
        4. Maintain the original style and tone
        5. Do NOT completely rewrite the text - make targeted improvements
        6. If PII (Personally Identifiable Information) is mentioned in the issues, replace it with fictional equivalents

        Improved text:
        """

        return prompt

    def run(self) -> Result:
        """Execute the chain and return the result.

        This method runs the complete chain execution process with a feedback loop:
        1. Checks that the chain is properly configured (has a model and prompt)
        2. Generates text using the model and prompt
        3. Validates the generated text using all configured validators
        4. If validation fails and apply_improvers_on_validation_failure is True:
           a. Gets feedback from critics on how to improve the text
           b. Sends the original text + feedback to the model to generate improved text
           c. Re-validates the improved text
           d. Repeats steps 4a-4c until validation passes or max iterations is reached
        5. If validation passes, returns the result immediately
        6. Returns a Result object with the final text and all validation/improvement results

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
                .with_options(apply_improvers_on_validation_failure=True)
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
                operation="generation",
                message_prefix="Failed to generate text",
                suggestions=[
                    "Check the model configuration",
                    "Verify the prompt format",
                ],
                metadata={
                    "model_type": self._model.__class__.__name__,
                    "prompt_length": len(self._prompt),
                    "options": self._config.get_model_options(),
                },
            ):
                # Get model options from configuration
                model_options = self._config.get_model_options()
                text = self._model.generate(self._prompt, **model_options)
                logger.debug(f"Generated text of length {len(text)}")

            # Initialize results lists
            from sifaka.results import ImprovementResult as ResultsImprovementResult
            from sifaka.results import ValidationResult as ResultsValidationResult

            validation_results: List[ResultsValidationResult] = []
            improvement_results: List[ResultsImprovementResult] = []

            # Main improvement loop
            current_text = text

            for iteration in range(self._max_improvement_iterations):
                logger.debug(
                    f"Starting improvement iteration {iteration+1}/{self._max_improvement_iterations}"
                )

                # Validate current text
                validation_passed, validation_results, validation_feedback = self._validate_text(
                    current_text
                )

                # If validation fails and we should apply improvers on validation failure
                if not validation_passed and self._options.get(
                    "apply_improvers_on_validation_failure", True
                ):
                    logger.info(f"Validation failed in iteration {iteration+1}, applying improvers")

                    # Get feedback from critics
                    critic_feedback = self._get_critic_feedback(current_text, validation_feedback)

                    # Generate improved text using the model with feedback
                    improved_text = self._generate_improved_text(current_text, critic_feedback)

                    # Create improvement result
                    from sifaka.results import ImprovementResult as SifakaImprovementResult

                    improvement_result = SifakaImprovementResult(
                        _original_text=current_text,
                        _improved_text=improved_text,
                        _changes_made=improved_text != current_text,
                        message=f"Improvement iteration {iteration+1}",
                        _details={
                            "validation_feedback": validation_feedback,
                            "critic_feedback": critic_feedback,
                            "iteration": iteration + 1,
                        },
                    )
                    improvement_results.append(improvement_result)

                    # Update current text for next iteration
                    current_text = improved_text

                    # If no changes were made, break the loop
                    if not improvement_result.changes_made:
                        logger.debug(f"No changes made in iteration {iteration+1}, breaking loop")
                        break

                    # Continue to next iteration to validate the improved text
                    continue

                # If validation fails and we should not apply improvers on validation failure
                elif not validation_passed:
                    logger.info(f"Chain execution stopped at validation in iteration {iteration+1}")
                    execution_time = time.time() - start_time
                    return Result(
                        text=current_text,
                        initial_text=text,  # Include the initial text
                        passed=False,
                        validation_results=validation_results,
                        improvement_results=improvement_results,
                        execution_time_ms=execution_time * 1000,
                    )

                # If validation passes, return the result immediately
                else:
                    logger.debug("Text passed validation, returning result")
                    break

            # Final validation to ensure the text still passes after all improvements
            if current_text != text:
                validation_passed, validation_results, _ = self._validate_text(current_text)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log chain execution result
            if validation_passed:
                logger.info(
                    f"Chain execution completed successfully in {execution_time:.2f}s: "
                    f"validators_passed={len(validation_results)}, "
                    f"improvements_applied={len(improvement_results)}, "
                    f"final_text_length={len(current_text)}"
                )
            else:
                logger.info(
                    f"Chain execution failed validation in {execution_time:.2f}s: "
                    f"validators_failed={len(validation_results)}, "
                    f"improvements_attempted={len(improvement_results)}, "
                    f"final_text_length={len(current_text)}"
                )

            # Return result
            return Result(
                text=current_text,
                initial_text=text,  # Include the initial text
                passed=validation_passed,
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
                    "debug_mode": self._config.debug,
                    "model_options": self._config.get_model_options(),
                },
            )
