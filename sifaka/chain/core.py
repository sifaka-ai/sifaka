"""
Chain Core Module

A brief description of the module's purpose and functionality.

## Overview
This module provides the ChainCore class which is the main interface for
chains, delegating to specialized components. ChainCore is the central
orchestration component that coordinates model providers, rules, critics,
and other components to generate and validate text. It implements the core
chain execution logic and manages the lifecycle of chain components.

## Components
1. **ChainCore**: Central orchestrator that delegates to specialized components
2. **Generator**: Handles text generation using model providers
3. **ValidationManager**: Manages validation against rules
4. **PromptManager**: Handles prompt creation and modification
5. **RetryStrategy**: Implements retry logic for validation failures
6. **ResultFormatter**: Formats results and feedback
7. **CriticCore**: Optional component for improving outputs

## Usage Examples
```python
from sifaka.chain.core import ChainCore
from sifaka.chain.managers.validation import ValidationManager
from sifaka.chain.managers.prompt import PromptManager
from sifaka.chain.strategies.retry import SimpleRetryStrategy
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.models import create_openai_chat_provider
from sifaka.rules import create_length_rule, create_toxicity_rule

# Create model provider
model_provider = create_openai_chat_provider(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Create components
validation_manager = ValidationManager(
    rules=[
        create_length_rule(min_length=10, max_length=1000),
        create_toxicity_rule(threshold=0.7)
    ]
)
prompt_manager = PromptManager()
retry_strategy = SimpleRetryStrategy(max_attempts=3)
result_formatter = ResultFormatter()

# Create chain core
chain_core = ChainCore(
    model=model_provider,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    retry_strategy=retry_strategy,
    result_formatter=result_formatter
)

# Run the chain
prompt = "Write a short story about a robot learning to paint."
result = chain_core.run(prompt)

# Check the result
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

## Error Handling
- ChainError: Raised when chain execution fails
- ValidationError: Raised when validation fails
- CriticError: Raised when critic refinement fails
- ModelError: Raised when model generation fails

## Configuration
- model: The model provider for text generation
- validation_manager: Manager for rule validation
- prompt_manager: Manager for prompt handling
- retry_strategy: Strategy for retry logic
- result_formatter: Formatter for results
- critic: Optional critic for text improvement
"""

from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..critics import CriticCore
from ..generation import Generator
from ..core.interfaces import Component, Configurable, Identifiable
from ..models.base import ModelProvider
from .formatters.result import ResultFormatter
from .interfaces.chain import Chain as ChainInterface
from .interfaces.critic import CriticProtocol
from .interfaces.formatter import ResultFormatterProtocol
from .interfaces.manager import PromptManagerProtocol, ValidationManagerProtocol
from .interfaces.strategy import RetryStrategyProtocol
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .result import ChainResult
from .strategies.retry import RetryStrategy
from ..utils.logging import get_logger
from ..utils.state import create_chain_state

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ChainCore(Generic[OutputType], BaseModel):
    """
    Core chain implementation that delegates to specialized components.

    Detailed description of the class's purpose, functionality, and usage.

    ## Architecture
    ChainCore follows a component-based architecture:
    1. **Core Components**: Essential components
       - Model provider: Text generation
       - Validation manager: Rule validation
       - Prompt manager: Prompt handling
       - Retry strategy: Retry logic
       - Result formatter: Result formatting
    2. **Optional Components**: Additional components
       - Critic: Text improvement
       - Custom validators
       - Custom formatters

    ## Lifecycle
    1. **Initialization**: Set up components
       - Store components
       - Create generator
       - Initialize state
    2. **Execution**: Run chain
       - Process prompt
       - Generate output
       - Validate output
       - Improve if needed
    3. **Cleanup**: Clean up resources
       - Release resources
       - Reset state
       - Log completion

    ## Error Handling
    - ChainError: Raised when chain execution fails
    - ValidationError: Raised when validation fails
    - CriticError: Raised when critic refinement fails
    - ModelError: Raised when model generation fails

    ## Examples
    ```python
    from sifaka.chain.core import ChainCore
    from sifaka.chain.managers.validation import ValidationManager
    from sifaka.chain.managers.prompt import PromptManager
    from sifaka.chain.strategies.retry import SimpleRetryStrategy
    from sifaka.chain.formatters.result import ResultFormatter
    from sifaka.models import create_openai_chat_provider
    from sifaka.rules import create_length_rule

    # Create model provider
    model_provider = create_openai_chat_provider(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )

    # Create components
    validation_manager = ValidationManager(
        rules=[create_length_rule(min_length=10, max_length=1000)]
    )
    prompt_manager = PromptManager()
    retry_strategy = SimpleRetryStrategy(max_attempts=3)
    result_formatter = ResultFormatter()

    # Create chain core
    chain_core = ChainCore(
        model=model_provider,
        validation_manager=validation_manager,
        prompt_manager=prompt_manager,
        retry_strategy=retry_strategy,
        result_formatter=result_formatter
    )

    # Run the chain
    prompt = "Write a short story about a robot learning to paint."
    result = chain_core.run(prompt)

    # Check the result
    print(f"Output: {result.output}")
    print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
    ```

    Attributes:
        model (ModelProvider): The model provider for text generation
        validation_manager (ValidationManager): Manager for rule validation
        prompt_manager (PromptManager): Manager for prompt handling
        retry_strategy (RetryStrategy): Strategy for retry logic
        result_formatter (ResultFormatter): Formatter for results
        critic (Optional[CriticCore]): Optional critic for text improvement
    """

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_chain_state)

    def __init__(
        self,
        model: ModelProvider,
        validation_manager: ValidationManager[OutputType],
        prompt_manager: PromptManager,
        retry_strategy: RetryStrategy[OutputType],
        result_formatter: ResultFormatter[OutputType],
        critic: Optional[CriticCore] = None,
    ):
        """
        Initialize a ChainCore instance.

        ## Overview
        This method initializes a ChainCore instance with the required components.
        It stores the components in the state object and creates a generator
        using the model provider.

        ## Lifecycle
        1. **Component Storage**: Store all components in the state object
           - Store model provider
           - Store validation manager
           - Store prompt manager
           - Store retry strategy
           - Store result formatter
           - Store critic (if provided)
        2. **Generator Creation**: Create a generator using the model provider
           - Create a Generator instance with the model provider
           - Store it in the state object

        Args:
            model: The model provider to use for text generation
            validation_manager: The validation manager to use for validating outputs
            prompt_manager: The prompt manager to use for creating and modifying prompts
            retry_strategy: The retry strategy to use for handling validation failures
            result_formatter: The result formatter to use for formatting results
            critic: Optional critic to use for improving outputs

        Raises:
            ValueError: When required components are missing or invalid
            ChainError: When initialization fails
        """

        # Store components in state
        self._state_manager.set("model", model)
        self._state_manager.set("validation_manager", validation_manager)
        self._state_manager.set("prompt_manager", prompt_manager)
        self._state_manager.set("retry_strategy", retry_strategy)
        self._state_manager.set("result_formatter", result_formatter)
        if critic:
            self._state_manager.set("critic", critic)

        # Create generator
        self._state_manager.set("generator", Generator[OutputType](model))

    @property
    def model(self) -> ModelProvider:
        """
        Get the model provider.

        ## Overview
        This property returns the model provider used for text generation.

        ## Lifecycle
        1. **State Access**: Access state
           - Get model from state
           - Return model

        Returns:
            ModelProvider: The model provider

        Raises:
            ValueError: When model is not found in state
        """
        return self._state_manager.get("model")

    @property
    def validation_manager(self) -> ValidationManager[OutputType]:
        """
        Get the validation manager.

        ## Overview
        This property returns the validation manager used for rule validation.

        ## Lifecycle
        1. **State Access**: Access state
           - Get validation manager from state
           - Return validation manager

        Returns:
            ValidationManager[OutputType]: The validation manager

        Raises:
            ValueError: When validation manager is not found in state
        """
        return self._state_manager.get("validation_manager")

    @property
    def prompt_manager(self) -> PromptManager:
        """
        Get the prompt manager.

        ## Overview
        This property returns the prompt manager used for prompt handling.

        ## Lifecycle
        1. **State Access**: Access state
           - Get prompt manager from state
           - Return prompt manager

        Returns:
            PromptManager: The prompt manager

        Raises:
            ValueError: When prompt manager is not found in state
        """
        return self._state_manager.get("prompt_manager")

    @property
    def retry_strategy(self) -> RetryStrategy[OutputType]:
        """
        Get the retry strategy.

        ## Overview
        This property returns the retry strategy used for handling validation failures.

        ## Lifecycle
        1. **State Access**: Access state
           - Get retry strategy from state
           - Return retry strategy

        Returns:
            RetryStrategy[OutputType]: The retry strategy

        Raises:
            ValueError: When retry strategy is not found in state
        """
        return self._state_manager.get("retry_strategy")

    @property
    def result_formatter(self) -> ResultFormatter[OutputType]:
        """
        Get the result formatter.

        ## Overview
        This property returns the result formatter used for formatting results.

        ## Lifecycle
        1. **State Access**: Access state
           - Get result formatter from state
           - Return result formatter

        Returns:
            ResultFormatter[OutputType]: The result formatter

        Raises:
            ValueError: When result formatter is not found in state
        """
        return self._state_manager.get("result_formatter")

    @property
    def critic(self) -> Optional[CriticCore]:
        """
        Get the critic.

        ## Overview
        This property returns the optional critic used for text improvement.

        ## Lifecycle
        1. **State Access**: Access state
           - Get critic from state
           - Return critic

        Returns:
            Optional[CriticCore]: The critic, if any

        Raises:
            ValueError: When critic is not found in state
        """
        return self._state_manager.get("critic")

    @property
    def generator(self) -> Generator[OutputType]:
        """
        Get the generator.

        ## Overview
        This property returns the generator used for text generation.

        ## Lifecycle
        1. **State Access**: Access state
           - Get generator from state
           - Return generator

        Returns:
            Generator[OutputType]: The generator

        Raises:
            ValueError: When generator is not found in state
        """
        return self._state_manager.get("generator")

    @property
    def name(self) -> str:
        """
        Get the name of the chain.

        ## Overview
        This property returns the name of the chain.

        ## Lifecycle
        1. **State Access**: Access state
           - Get name from state
           - Return name

        Returns:
            str: The name of the chain

        Raises:
            ValueError: When name is not found in state
        """
        return self._state_manager.get("name")

    @property
    def description(self) -> str:
        """
        Get the description of the chain.

        ## Overview
        This property returns the description of the chain.

        ## Lifecycle
        1. **State Access**: Access state
           - Get description from state
           - Return description

        Returns:
            str: The description of the chain

        Raises:
            ValueError: When description is not found in state
        """
        return self._state_manager.get("description")

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the configuration of the chain.

        ## Overview
        This property returns the configuration of the chain.

        ## Lifecycle
        1. **State Access**: Access state
           - Get config from state
           - Return config

        Returns:
            Dict[str, Any]: The configuration of the chain

        Raises:
            ValueError: When config is not found in state
        """
        return self._state_manager.get("config")

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the chain.

        ## Overview
        This method updates the configuration of the chain.

        ## Lifecycle
        1. **State Update**: Update state
           - Update config in state
           - Validate config
           - Apply config

        Args:
            config (Dict[str, Any]): The new configuration

        Raises:
            ValueError: When config is invalid
            ChainError: When config update fails
        """
        self._state_manager.set("config", config)

    def initialize(self) -> None:
        """
        Initialize the chain.

        ## Overview
        This method initializes the chain, setting up all components and
        preparing for execution.

        ## Lifecycle
        1. **Component Initialization**: Initialize components
           - Initialize model
           - Initialize validation manager
           - Initialize prompt manager
           - Initialize retry strategy
           - Initialize result formatter
           - Initialize critic (if any)
        2. **State Initialization**: Initialize state
           - Set initial state
           - Validate state
           - Log initialization

        Raises:
            ChainError: When initialization fails
        """
        # Initialize components
        self.model.initialize()
        self.validation_manager.initialize()
        self.prompt_manager.initialize()
        self.retry_strategy.initialize()
        self.result_formatter.initialize()
        if self.critic:
            self.critic.initialize()

        # Initialize state
        self._state_manager.initialize()

    def cleanup(self) -> None:
        """
        Clean up the chain.

        ## Overview
        This method cleans up the chain, releasing resources and resetting state.

        ## Lifecycle
        1. **Component Cleanup**: Clean up components
           - Clean up model
           - Clean up validation manager
           - Clean up prompt manager
           - Clean up retry strategy
           - Clean up result formatter
           - Clean up critic (if any)
        2. **State Cleanup**: Clean up state
           - Reset state
           - Release resources
           - Log cleanup

        Raises:
            ChainError: When cleanup fails
        """
        # Clean up components
        self.model.cleanup()
        self.validation_manager.cleanup()
        self.prompt_manager.cleanup()
        self.retry_strategy.cleanup()
        self.result_formatter.cleanup()
        if self.critic:
            self.critic.cleanup()

        # Clean up state
        self._state_manager.cleanup()

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the chain with the given prompt.

        ## Overview
        This method runs the chain with the given prompt, generating and
        validating text according to the configured components.

        ## Lifecycle
        1. **Prompt Processing**: Process prompt
           - Format prompt
           - Generate output
           - Validate output
        2. **Improvement Loop**: Improve if needed
           - Check validation
           - Apply critic
           - Retry if needed
        3. **Result Creation**: Create result
           - Format output
           - Format feedback
           - Return result

        Args:
            prompt (str): The prompt to process

        Returns:
            ChainResult[OutputType]: The result of running the chain

        Raises:
            ChainError: When chain execution fails
            ValidationError: When validation fails
            CriticError: When critic refinement fails
            ModelError: When model generation fails

        Examples:
            ```python
            # Run the chain
            result = chain_core.run("Write a short story about a robot learning to paint.")

            # Check the result
            print(f"Output: {result.output}")
            print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
            ```
        """
        # Process prompt
        formatted_prompt = self.prompt_manager.format_prompt(prompt)
        output = self.generator.generate(formatted_prompt)
        validation_result = self.validation_manager.validate(output)

        # Improve if needed
        if not validation_result.passed and self.critic:
            improved_output = self.critic.improve(output, validation_result)
            validation_result = self.validation_manager.validate(improved_output)

        # Create result
        return self.result_formatter.format(
            output=output,
            validation_result=validation_result,
            critique_details=self.critic.get_details() if self.critic else None,
        )
