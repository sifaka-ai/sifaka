"""
Chain Orchestrator Module

A brief description of the module's purpose and functionality.

## Overview
This module provides the ChainOrchestrator class which implements the
standard chain orchestration pattern, providing a simplified interface
for creating and running chains with validation and improvement flows.
The orchestrator manages the lifecycle of chain execution, including
validation, retry logic, and result formatting.

## Components
1. **ChainOrchestrator**: Main orchestrator class that manages the chain flow
2. **ChainCore**: Core implementation that handles the actual execution
3. **ValidationManager**: Manages validation against rules
4. **PromptManager**: Handles prompt creation and modification
5. **RetryStrategy**: Implements retry logic for validation failures
6. **ResultFormatter**: Formats results and feedback

## Usage Examples
```python
from sifaka.chain import ChainOrchestrator
from sifaka.models import create_openai_chat_provider
from sifaka.rules import create_length_rule, create_toxicity_rule
from sifaka.critics import create_prompt_critic

# Create model provider
model_provider = create_openai_chat_provider(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Create rules
rules = [
    create_length_rule(min_length=10, max_length=1000),
    create_toxicity_rule(threshold=0.7)
]

# Create critic
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create orchestrator
orchestrator = ChainOrchestrator(
    model=model_provider,
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Run the chain
result = orchestrator.run("Write a short story about a robot learning to paint.")

# Check the result
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

## Error Handling
- ValueError: Raised when validation fails after max attempts
- ChainError: Raised when chain execution fails
- ValidationError: Raised when validation fails
- CriticError: Raised when critic refinement fails
- ModelError: Raised when model generation fails

## Configuration
- model: The model provider for text generation
- rules: List of rules to validate outputs against
- critic: Optional critic for improving outputs
- max_attempts: Maximum number of validation attempts
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import CriticCore
from ..models.core import ModelProviderCore
from ..rules.base import Rule
from .core import ChainCore
from .formatters.result import ResultFormatter
from sifaka.core.managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .result import ChainResult
from .strategies.retry import RetryStrategy

OutputType = TypeVar("OutputType")


class ChainOrchestrator(Generic[OutputType]):
    """
    Orchestrates the execution of a validation and improvement flow.

    Detailed description of the class's purpose, functionality, and usage.

    ## Architecture
    ChainOrchestrator follows a component-based architecture:
    1. **Core Components**: Essential components
       - ChainCore: Handles execution
       - ValidationManager: Manages validation
       - PromptManager: Handles prompts
       - RetryStrategy: Implements retry logic
       - ResultFormatter: Formats results
    2. **Optional Components**: Additional components
       - Critic: Improves outputs
       - Custom validators
       - Custom formatters

    ## Lifecycle
    1. **Initialization**: Set up components
       - Create managers
       - Configure strategy
       - Initialize core
    2. **Execution**: Run chain
       - Process prompt
       - Validate output
       - Improve if needed
    3. **Result Handling**: Format results
       - Format output
       - Format feedback
       - Return result

    ## Error Handling
    - ValueError: Raised when validation fails after max attempts
    - ChainError: Raised when chain execution fails
    - ValidationError: Raised when validation fails
    - CriticError: Raised when critic refinement fails
    - ModelError: Raised when model generation fails

    ## Examples
    ```python
    from sifaka.chain import ChainOrchestrator
    from sifaka.models import create_openai_chat_provider
    from sifaka.rules import create_length_rule, create_toxicity_rule

    # Create model provider
    model_provider = create_openai_chat_provider(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )

    # Create rules
    rules = [
        create_length_rule(min_length=10, max_length=1000),
        create_toxicity_rule(threshold=0.7)
    ]

    # Create orchestrator
    orchestrator = ChainOrchestrator(
        model=model_provider,
        rules=rules,
        max_attempts=3
    )

    # Run the chain
    result = orchestrator.run("Write a short story about a robot learning to paint.")

    # Check the result
    print(f"Output: {result.output}")
    print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
    ```

    Attributes:
        model (ModelProviderCore): The model provider for text generation
        rules (List[Rule]): List of rules to validate outputs against
        critic (Optional[CriticCore]): Optional critic for improving outputs
        max_attempts (int): Maximum number of validation attempts
    """

    def __init__(
        self,
        model: ModelProviderCore,
        rules: List[Rule],
        critic: Optional[CriticCore] = None,
        max_attempts: int = 3,
    ):
        """
        Initialize a ChainOrchestrator instance.

        ## Overview
        This method creates and configures specialized components, sets up the core
        chain with all components, and initializes validation and retry logic.

        ## Lifecycle
        1. **Component Creation**: Create components
           - Create validation manager
           - Create prompt manager
           - Create retry strategy
           - Create result formatter
        2. **Core Setup**: Set up core chain
           - Configure core
           - Set up components
           - Initialize state

        Args:
            model (ModelProviderCore): The model provider to use
            rules (List[Rule]): The rules to validate outputs against
            critic (Optional[CriticCore]): Optional critic for improving outputs
            max_attempts (int): Maximum number of attempts

        Raises:
            ValueError: When validation fails after max attempts
            ChainError: When chain execution fails
        """
        # Create components
        validation_manager = ValidationManager[OutputType](rules)
        prompt_manager = PromptManager()
        retry_strategy = RetryStrategy(max_attempts=max_attempts)
        result_formatter = ResultFormatter[OutputType]()

        # Create core chain
        self._core = ChainCore[OutputType](
            model=model,
            validation_manager=validation_manager,
            prompt_manager=prompt_manager,
            retry_strategy=retry_strategy,
            result_formatter=result_formatter,
            critic=critic,
        )

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the prompt through the orchestration flow.

        ## Overview
        This method processes the prompt through the chain, validates the output
        against rules, improves the output if validation fails, and returns a
        standardized result.

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
            ValueError: When validation fails after max attempts
            ChainError: When chain execution fails
            ValidationError: When validation fails
            CriticError: When critic refinement fails
            ModelError: When model generation fails

        Examples:
            ```python
            # Run the chain
            result = orchestrator.run("Write a short story about a robot learning to paint.")

            # Check the result
            print(f"Output: {result.output}")
            print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
            ```
        """
        return self._core.run(prompt)
