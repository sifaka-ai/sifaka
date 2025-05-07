"""
Core chain module for Sifaka.

This module provides the ChainCore class which is the main interface for
chains, delegating to specialized components. ChainCore is the central
orchestration component that coordinates model providers, rules, critics,
and other components to generate and validate text.

## Architecture Overview

The chain core system follows a component-based architecture:

1. **ChainCore**: Central orchestrator that delegates to specialized components
2. **Generator**: Handles text generation using model providers
3. **ValidationManager**: Manages validation against rules
4. **PromptManager**: Handles prompt creation and modification
5. **RetryStrategy**: Implements retry logic for validation failures
6. **ResultFormatter**: Formats results and feedback
7. **CriticCore**: Optional component for improving outputs

## Component Lifecycle

### ChainCore
1. **Initialization**: Set up with required components
   - Model provider for text generation
   - ValidationManager for rule validation
   - PromptManager for prompt handling
   - RetryStrategy for retry logic
   - ResultFormatter for result formatting
   - Optional CriticCore for text improvement

2. **Execution**: Process prompts with run() method
   - Delegate to RetryStrategy for execution flow
   - Generate text with Generator
   - Validate with ValidationManager
   - Improve with CriticCore if validation fails
   - Format results with ResultFormatter

3. **Result Handling**: Return standardized ChainResult objects
   - Include generated output
   - Include validation results
   - Include critique details if available

## Error Handling Patterns

The chain core system implements several error handling patterns:

1. **Component Delegation**: Delegates error handling to specialized components
   - RetryStrategy handles retry logic and validation failures
   - ValidationManager handles rule validation errors
   - Generator handles model provider errors
   - CriticCore handles improvement errors

2. **Exception Propagation**: Propagates exceptions with context
   - Preserves original exceptions
   - Adds context information
   - Provides meaningful error messages

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

## Integration with Other Components

ChainCore is typically not used directly but through higher-level components
like ChainOrchestrator or factory functions like create_simple_chain and
create_backoff_chain, which provide simpler interfaces for common use cases.

```python
from sifaka.chain import create_simple_chain
from sifaka.models import create_openai_chat_provider
from sifaka.rules import create_length_rule, create_toxicity_rule

# Create a simple chain
chain = create_simple_chain(
    model=create_openai_chat_provider(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    ),
    rules=[
        create_length_rule(min_length=10, max_length=1000),
        create_toxicity_rule(threshold=0.7)
    ],
    max_attempts=3
)

# Run the chain
result = chain.run("Write a short story about a robot learning to paint.")
```
"""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..critics import CriticCore
from ..generation import Generator
from ..models.base import ModelProvider
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .result import ChainResult
from .strategies.retry import RetryStrategy
from ..utils.logging import get_logger
from ..utils.state import create_chain_state

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ChainCore(BaseModel, Generic[OutputType]):
    """
    Core chain implementation that delegates to specialized components.

    This class implements the Chain interface but delegates most of its
    functionality to specialized components for better separation of concerns.
    It serves as the central orchestration component that coordinates the
    interaction between model providers, rules, critics, and other components
    to generate and validate text.

    ## Architecture

    ChainCore follows a component-based architecture where each responsibility
    is delegated to a specialized component. This design pattern promotes:

    1. **Separation of Concerns**: Each component has a single responsibility
       - **Generator**: Handles text generation using model providers
       - **ValidationManager**: Manages validation against rules
       - **PromptManager**: Handles prompt creation and modification
       - **RetryStrategy**: Implements retry logic for validation failures
       - **ResultFormatter**: Formats results and feedback
       - **CriticCore**: Optional component for improving outputs

    2. **Dependency Injection**: Components are injected at initialization
       - Components are created externally and passed to ChainCore
       - This allows for easy testing and customization
       - Components can be swapped out without changing ChainCore

    3. **Composition Over Inheritance**: Uses composition for flexibility
       - ChainCore composes multiple specialized components
       - New functionality can be added by creating new components
       - Existing components can be extended without modifying ChainCore

    4. **Delegation Pattern**: Delegates operations to specialized components
       - ChainCore acts as a facade for the underlying components
       - Each operation is delegated to the appropriate component
       - Components can be modified independently

    5. **Flow Control**: Orchestrates the flow between components
       - Manages the sequence of operations (generate, validate, improve)
       - Handles the flow of data between components
       - Ensures proper error handling and recovery

    6. **State Management**: Uses a dedicated state object
       - Stores all components in a central state object
       - Provides property accessors for components
       - Ensures proper initialization before use

    ## State Management

    ChainCore uses the standardized state management approach:

    1. **State Initialization**
       - Creates a state object using ChainState
       - Stores all components in the state object
       - Accesses components through the state object

    2. **State Access**
       - Uses state directly to access components
       - Updates state through the state object
       - Maintains clear separation between configuration and state

    3. **State Components**
       - model: The model provider
       - generator: The text generator
       - validation_manager: The validation manager
       - prompt_manager: The prompt manager
       - retry_strategy: The retry strategy
       - result_formatter: The result formatter
       - critic: The optional critic
       - initialized: Whether the chain is initialized

    ## Lifecycle

    1. **Initialization**: Set up with required components
       - Create with all required components
       - Initialize internal state
       - Set up generator with model provider

    2. **Execution**: Process prompts with run() method
       - Delegate to RetryStrategy for execution flow
       - Generate text with Generator
       - Validate with ValidationManager
       - Improve with CriticCore if validation fails
       - Format results with ResultFormatter

    3. **Result Handling**: Return standardized ChainResult objects
       - Include generated output
       - Include validation results
       - Include critique details if available

    ## Error Handling

    The class implements these error handling patterns:
    - Delegates error handling to specialized components
    - Propagates exceptions with context
    - Provides meaningful error messages

    ## Examples

    Creating and using a ChainCore instance:

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

    Using with a critic for improvement:

    ```python
    from sifaka.chain.core import ChainCore
    from sifaka.critics import create_prompt_critic

    # Create components (as in previous example)
    # ...

    # Create critic
    critic = create_prompt_critic(
        llm_provider=model_provider,
        system_prompt="You are an expert editor that improves text."
    )

    # Create chain core with critic
    chain_core = ChainCore(
        model=model_provider,
        validation_manager=validation_manager,
        prompt_manager=prompt_manager,
        retry_strategy=retry_strategy,
        result_formatter=result_formatter,
        critic=critic
    )

    # Run the chain
    result = chain_core.run("Write a poem about autumn.")

    # Check critique details
    if result.critique_details:
        print(f"Critique feedback: {result.critique_details.get('feedback', '')}")
    ```

    Type parameters:
        OutputType: The type of output generated by the chain
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

        ## Examples

        Creating a ChainCore instance with all required components:

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
        ```

        Args:
            model: The model provider to use for text generation
            validation_manager: The validation manager to use for validating outputs
            prompt_manager: The prompt manager to use for creating and modifying prompts
            retry_strategy: The retry strategy to use for handling validation failures
            result_formatter: The result formatter to use for formatting results
            critic: Optional critic to use for improving outputs
        """
        # Initialize the base class
        super().__init__()

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Store components in state
        state.model = model
        state.validation_manager = validation_manager
        state.prompt_manager = prompt_manager
        state.retry_strategy = retry_strategy
        state.result_formatter = result_formatter
        state.critic = critic

        # Create generator
        state.generator = Generator[OutputType](model)

        # Mark as initialized
        state.initialized = True

    @property
    def model(self) -> ModelProvider:
        """
        Get the model provider.

        The model provider is responsible for generating text based on prompts.
        It's used by the generator to produce outputs that are then validated
        against rules.

        Returns:
            The model provider used by this chain
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.model

    @property
    def validation_manager(self) -> ValidationManager[OutputType]:
        """
        Get the validation manager.

        The validation manager is responsible for validating outputs against rules.
        It manages the rules and provides methods for validation and error message
        generation.

        Returns:
            The validation manager used by this chain
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.validation_manager

    @property
    def prompt_manager(self) -> PromptManager:
        """
        Get the prompt manager.

        The prompt manager is responsible for creating and modifying prompts.
        It provides methods for adding feedback, history, context, and examples
        to prompts, which is especially useful during retry attempts.

        Returns:
            The prompt manager used by this chain
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.prompt_manager

    @property
    def retry_strategy(self) -> RetryStrategy[OutputType]:
        """
        Get the retry strategy.

        The retry strategy is responsible for implementing retry logic when
        validation fails. It handles the execution flow, including text generation,
        validation, improvement, and retries with feedback. Different strategies
        can implement different retry behaviors, such as simple retries or
        exponential backoff.

        Returns:
            The retry strategy used by this chain
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.retry_strategy

    @property
    def result_formatter(self) -> ResultFormatter[OutputType]:
        """
        Get the result formatter.

        The result formatter is responsible for formatting results and feedback.
        It provides methods for creating ChainResult objects, formatting feedback
        from validation results, and formatting feedback from critique details.

        Returns:
            The result formatter used by this chain
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.result_formatter

    @property
    def critic(self) -> Optional[CriticCore]:
        """
        Get the critic.

        The critic is an optional component responsible for improving outputs
        when validation fails. It provides methods for critiquing text and
        generating feedback, which can be used to improve the output in
        subsequent attempts.

        Returns:
            The critic used by this chain, or None if not set
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.critic

    @property
    def generator(self) -> Generator[OutputType]:
        """
        Get the generator.

        The generator is responsible for generating text using the model provider.
        It's a wrapper around the model provider that provides a consistent
        interface for text generation.

        Returns:
            The generator used by this chain
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")
        return state.generator

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the chain with the given prompt.

        This method processes the given prompt through the chain's components,
        generating text, validating it against rules, and improving it with
        critics if necessary. It delegates the execution flow to the retry
        strategy, which handles the retry logic and validation failures.

        ## Lifecycle

        1. **Delegation**: Delegate to retry strategy
           - Pass prompt and all components to retry strategy
           - Let retry strategy handle the execution flow

        2. **Processing**: The retry strategy typically:
           - Generates text with the generator
           - Validates it with the validation manager
           - Improves it with the critic if validation fails
           - Retries with feedback if necessary

        3. **Result**: Return the chain result
           - Contains the generated output
           - Contains validation results
           - Contains critique details if available

        ## Error Handling

        This method handles these error cases:
        - Delegates error handling to the retry strategy
        - Propagates exceptions from the retry strategy
        - Preserves the original exception context

        ## Examples

        Basic usage:

        ```python
        # Assuming chain_core is a properly configured ChainCore instance
        prompt = "Write a short story about a robot learning to paint."
        result = chain_core.run(prompt)

        print(f"Output: {result.output}")
        print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
        ```

        Error handling:

        ```python
        try:
            result = chain_core.run("Write a very short story.")
        except ValueError as e:
            print(f"Validation failed: {e}")
            # Handle the error appropriately
        ```

        Args:
            prompt: The prompt to process

        Returns:
            The chain result with output, validation results, and critique details

        Raises:
            ValueError: If validation fails after max attempts
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ChainCore not properly initialized")

        # Delegate to retry strategy
        return state.retry_strategy.run(
            prompt=prompt,
            generator=state.generator,
            validation_manager=state.validation_manager,
            prompt_manager=state.prompt_manager,
            result_formatter=state.result_formatter,
            critic=state.critic,
        )
