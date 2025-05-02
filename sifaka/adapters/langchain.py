"""
LangChain adapter for Sifaka.

This module provides adapter classes and functions to integrate LangChain with Sifaka's
reflection and reliability features.

## Architecture Overview

The LangChain adapter follows a component-based architecture:

1. **Protocol Definitions**: Define interfaces for validation and processing
2. **Configuration**: Standardized configuration for chain components
3. **Core Components**: Wrapper classes for chains, memory, and validators
4. **Factory Functions**: Simple creation patterns for common use cases

## Component Lifecycle

### SifakaChain
1. **Initialization**: Set up with chain and configuration
2. **Execution**: Run the chain with inputs
3. **Validation**: Validate the chain's output
4. **Processing**: Process the output based on validation results

### Validators
1. **Initialization**: Set up with validation rules
2. **Validation**: Validate chain outputs against rules
3. **Result**: Return standardized validation results

## Usage Examples

### Basic Chain Wrapping

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from sifaka.adapters import wrap_chain
from sifaka.rules.content import create_sentiment_rule

# Create a LangChain LLMChain
llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Create rules for validation
rule = create_sentiment_rule(valid_labels=["positive", "neutral"])

# Wrap the chain with Sifaka's features
sifaka_chain = wrap_chain(
    chain=chain,
    rules=[rule],
    critique=True
)

# Run the chain
output = sifaka_chain.run("What is the capital of France?")
```

### Memory Integration

```python
from langchain.memory import ConversationBufferMemory
from sifaka.adapters import wrap_memory, wrap_chain
from sifaka.rules.formatting import create_length_rule

# Create a memory component
memory = ConversationBufferMemory()

# Create a rule
rule = create_length_rule(max_chars=500)

# Wrap the memory with validation
sifaka_memory = wrap_memory(
    memory=memory,
    rules=[rule]
)

# Use the memory in a chain
sifaka_chain = wrap_chain(
    chain=chain,
    memory=sifaka_memory
)
```
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from langchain.chains import LLMChain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence

from sifaka.rules.base import Rule, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.chain.managers.validation import ValidationManager
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)
T = TypeVar("T")  # Generic type parameter
InputType = TypeVar("InputType")  # Input type for chain
OutputType = TypeVar("OutputType")  # Output type for chain
ChainType = Union[LLMChain, RunnableSequence]  # Supported chain types


@runtime_checkable
class ChainValidator(Protocol[OutputType]):
    """
    Protocol for chain output validation.

    Classes implementing this protocol can validate chain outputs
    and return standardized results.

    Type Parameters:
        OutputType: The type of output to validate

    Lifecycle:
    1. Initialization: Configure validation parameters
    2. Validation: Receive output and apply validation logic
    3. Result: Return standardized validation results

    Examples:
        ```python
        from sifaka.adapters.langchain import ChainValidator

        class ContentValidator(ChainValidator[str]):
            def validate(self, output: str) -> RuleResult:
                # Apply validation logic
                is_valid = len(output) > 10
                return RuleResult(
                    passed=is_valid,
                    message="Content validation " +
                            ("passed" if is_valid else "failed")
                )

            def can_validate(self, output: str) -> bool:
                return isinstance(output, str)
        ```
    """

    def validate(self, output: OutputType) -> RuleResult:
        """
        Validate a chain output.

        Args:
            output: The output to validate

        Returns:
            Validation result
        """
        ...

    def can_validate(self, output: OutputType) -> bool:
        """
        Check if this validator can validate the output.

        Args:
            output: The output to check

        Returns:
            True if this validator can validate the output
        """
        ...


@runtime_checkable
class ChainOutputProcessor(Protocol[OutputType]):
    """
    Protocol for chain output processing.

    Classes implementing this protocol can process chain outputs
    to enhance or modify them.

    Type Parameters:
        OutputType: The type of output to process

    Lifecycle:
    1. Initialization: Configure processing parameters
    2. Processing: Receive output and apply processing logic
    3. Result: Return modified output

    Examples:
        ```python
        from sifaka.adapters.langchain import ChainOutputProcessor

        class TextFormatter(ChainOutputProcessor[str]):
            def process(self, output: str) -> str:
                # Format the output
                return output.strip().capitalize()

            def can_process(self, output: str) -> bool:
                return isinstance(output, str)
        ```
    """

    def process(self, output: OutputType) -> OutputType:
        """
        Process a chain output.

        Args:
            output: The output to process

        Returns:
            The processed output
        """
        ...

    def can_process(self, output: OutputType) -> bool:
        """
        Check if this processor can process the output.

        Args:
            output: The output to check

        Returns:
            True if this processor can process the output
        """
        ...


@runtime_checkable
class ChainMemory(Protocol):
    """
    Protocol for chain memory components.

    Classes implementing this protocol can store and retrieve
    context for chain executions.

    Lifecycle:
    1. Initialization: Set up memory storage
    2. Loading: Retrieve stored context for chain execution
    3. Saving: Store new context from chain execution
    4. Clearing: Remove stored context when needed

    Examples:
        ```python
        from sifaka.adapters.langchain import ChainMemory

        class SimpleMemory(ChainMemory):
            def __init__(self):
                self._memory = {}

            def load_memory_variables(self, inputs):
                return {"history": self._memory.get("history", "")}

            def save_context(self, inputs, outputs):
                history = self._memory.get("history", "")
                self._memory["history"] = history + f"\nQ: {inputs.get('input')}\nA: {outputs.get('output')}"

            def clear(self):
                self._memory = {}

            @property
            def memory_variables(self):
                return ["history"]
        ```
    """

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables for chain execution.

        Args:
            inputs: The chain inputs

        Returns:
            Dictionary of memory variables
        """
        ...

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save context from chain execution.

        Args:
            inputs: The chain inputs
            outputs: The chain outputs
        """
        ...

    def clear(self) -> None:
        """Clear all stored memory."""
        ...

    @property
    def memory_variables(self) -> List[str]:
        """
        Get the memory variables provided by this memory.

        Returns:
            List of memory variable names
        """
        ...


@dataclass
class ChainConfig(Generic[OutputType]):
    """
    Configuration for Sifaka chains.

    This class provides a standardized way to configure LangChain
    integrations with Sifaka features.

    Type Parameters:
        OutputType: The type of chain output

    Lifecycle:
    1. Creation: Instantiated with configuration options
    2. Validation: Post-init validation of configuration values
    3. Usage: Accessed by chain components during setup and execution

    Examples:
        ```python
        from sifaka.adapters.langchain import ChainConfig, ChainValidator

        # Create validators
        validators = [MyValidator(), AnotherValidator()]

        # Create processors
        processors = [MyProcessor(), AnotherProcessor()]

        # Create memory
        memory = ConversationBufferMemory()

        # Create configuration
        config = ChainConfig(
            validators=validators,
            processors=processors,
            memory=memory,
            critique=True
        )
        ```
    """

    validators: List[ChainValidator[OutputType]] = field(default_factory=list)
    processors: List[ChainOutputProcessor[OutputType]] = field(default_factory=list)
    memory: Optional[ChainMemory] = None
    callbacks: List[BaseCallbackHandler] = field(default_factory=list)
    output_parser: Optional[BaseOutputParser[OutputType]] = None
    critique: bool = True
    tracer: Optional[Tracer] = None

    def __post_init__(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.output_parser and not isinstance(self.output_parser, BaseOutputParser):
            raise ValueError("output_parser must be an instance of BaseOutputParser")
        if self.memory and not isinstance(self.memory, (BaseMemory, ChainMemory)):
            raise ValueError("memory must implement ChainMemory protocol")


class SifakaChain(Generic[OutputType]):
    """
    A LangChain chain that integrates Sifaka's reflection and reliability features.

    This class follows the component-based architecture pattern by delegating to
    specialized components for validation, processing, and error handling.

    Type Parameters:
        OutputType: The type of chain output

    Lifecycle:
    1. Initialization: Set up with chain and configuration
    2. Execution: Run the chain with inputs
    3. Validation: Validate the chain's output
    4. Processing: Process the output based on validation results

    Examples:
        ```python
        from langchain.chains import LLMChain
        from sifaka.adapters.langchain import SifakaChain, ChainConfig

        # Create a LangChain chain
        chain = LLMChain(...)

        # Create configuration
        config = ChainConfig(
            validators=[MyValidator()],
            processors=[MyProcessor()],
            critique=True
        )

        # Create Sifaka chain
        sifaka_chain = SifakaChain(chain=chain, config=config)

        # Run the chain
        output = sifaka_chain.run("What is the capital of France?")
        ```
    """

    def __init__(
        self,
        chain: ChainType,
        config: ChainConfig[OutputType],
    ) -> None:
        """
        Initialize a Sifaka chain.

        Args:
            chain: The LangChain chain to wrap (either LLMChain or RunnableSequence)
            config: Chain configuration
        """
        self._chain = chain
        self._config = config

        # Configure the chain
        if isinstance(chain, LLMChain):
            if config.memory:
                self._chain.memory = cast(BaseMemory, config.memory)
            if config.callbacks:
                self._chain.callbacks = config.callbacks

        # Create validation manager
        self._validation_manager = self._create_validation_manager()

        # Create result formatter
        self._result_formatter = self._create_result_formatter()

    def _create_validation_manager(self):
        """
        Create a validation manager for this chain.

        Returns:
            ValidationManager for this chain
        """
        return ValidationManager(self._config.validators)

    def _create_result_formatter(self):
        """
        Create a result formatter for this chain.

        Returns:
            ResultFormatter for this chain
        """
        return ResultFormatter()

    @property
    def has_validators(self) -> bool:
        """
        Return whether the chain has any validators.

        Returns:
            True if the chain has validators, False otherwise
        """
        return bool(self._config.validators)

    @property
    def has_processors(self) -> bool:
        """
        Return whether the chain has any processors.

        Returns:
            True if the chain has processors, False otherwise
        """
        return bool(self._config.processors)

    @property
    def has_memory(self) -> bool:
        """
        Return whether the chain has memory.

        Returns:
            True if the chain has memory, False otherwise
        """
        return self._config.memory is not None

    @property
    def has_output_parser(self) -> bool:
        """
        Return whether the chain has an output parser.

        Returns:
            True if the chain has an output parser, False otherwise
        """
        return self._config.output_parser is not None

    @property
    def chain(self) -> ChainType:
        """
        Return the wrapped chain.

        Returns:
            The wrapped LangChain chain
        """
        return self._chain

    @property
    def config(self) -> ChainConfig[OutputType]:
        """
        Return the chain configuration.

        Returns:
            The chain configuration
        """
        return self._config

    def _validate_output(self, output: OutputType) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the output using the chain's validators.

        This method applies all validators to the output and collects
        any validation violations.

        Args:
            output: The output to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the output passed all validators
                - violations: List of validation violations
        """
        violations = []
        for validator in self._config.validators:
            if validator.can_validate(output):
                result = validator.validate(output)
                if not result.passed:
                    violations.append(
                        {
                            "validator": validator.__class__.__name__,
                            "message": result.message,
                            "metadata": result.metadata,
                        }
                    )

        return not violations, violations

    def _process_output(self, output: OutputType) -> OutputType:
        """
        Process the output using the chain's processors.

        This method applies all processors to the output in sequence.

        Args:
            output: The output to process

        Returns:
            The processed output
        """
        processed = output
        for processor in self._config.processors:
            if processor.can_process(processed):
                processed = processor.process(processed)
        return processed

    def run(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> OutputType:
        """
        Run the chain with Sifaka's reflection and reliability features.

        This method runs the chain, processes the output,
        validates it, and handles any violations.

        Args:
            inputs: The inputs to the chain
            **kwargs: Additional arguments for the chain

        Returns:
            The chain's output, optionally parsed and processed

        Raises:
            ValueError: If the output fails validation and critique is disabled
        """
        # Run the chain
        if isinstance(self._chain, LLMChain):
            output = self._chain.run(inputs, **kwargs)
        else:
            # For RunnableSequence
            if isinstance(inputs, str):
                inputs = {"human_input": inputs}
            output = self._chain.invoke(inputs)
            if isinstance(output, dict):
                output = output.get("text", output.get("output", output))

        logger.debug("Chain output: %s", output)

        # Parse the output if configured
        if self.has_output_parser:
            output = cast(BaseOutputParser[OutputType], self._config.output_parser).parse(output)

        # Process the output
        if self.has_processors:
            output = self._process_output(output)

        # Validate the output
        passed, violations = self._validate_output(output)
        if not passed:
            if self._config.critique:
                # TODO: Implement critique with standardized error handling
                logger.warning("Output validation failed: %s", violations)
            else:
                error_message = "\n".join([f"{v['validator']}: {v['message']}" for v in violations])
                raise ValueError(f"Output validation failed:\n{error_message}")

        # Add trace event if tracing is enabled
        if self._config.tracer:
            self._config.tracer.add_event(
                "chain_execution",
                "complete",
                {
                    "inputs": str(inputs),
                    "output": str(output),
                    "validation_passed": passed,
                    "validation_violations": violations if not passed else [],
                }
            )

        return output

    def __call__(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> OutputType:
        """
        Run the chain (callable interface).

        Args:
            inputs: The inputs to the chain
            **kwargs: Additional arguments for the chain

        Returns:
            The chain's output
        """
        return self.run(inputs, **kwargs)


class RuleBasedValidator(ChainValidator[str]):
    """
    A validator that uses Sifaka rules to validate chain outputs.

    This validator applies a list of Sifaka rules to string outputs
    from a chain.

    Lifecycle:
    1. Initialization: Set up with rules
    2. Validation: Apply rules to output text
    3. Result: Return validation result

    Examples:
        ```python
        from sifaka.adapters.langchain import RuleBasedValidator
        from sifaka.rules.content import create_sentiment_rule

        # Create rules
        rule = create_sentiment_rule(valid_labels=["positive", "neutral"])

        # Create validator
        validator = RuleBasedValidator([rule])

        # Use validator
        result = validator.validate("This is great!")
        ```
    """

    def __init__(self, rules: List[Rule]) -> None:
        """
        Initialize with rules.

        Args:
            rules: The Sifaka rules to use for validation
        """
        self._rules = rules

    def validate(self, output: str) -> RuleResult:
        """
        Validate using rules.

        This method applies all rules to the output text in sequence
        until one fails or all pass.

        Args:
            output: The output to validate

        Returns:
            RuleResult indicating whether validation passed
        """
        for rule in self._rules:
            result = rule.validate(output)
            if not result.passed:
                return result
        return RuleResult(passed=True, message="All rules passed")

    def can_validate(self, output: str) -> bool:
        """
        Check if can validate the output.

        Args:
            output: The output to check

        Returns:
            True if the output is a string, False otherwise
        """
        return isinstance(output, str)


class SifakaMemory(ChainMemory):
    """
    A memory component that integrates Sifaka's rules with LangChain's memory system.

    This class adds validation and processing capabilities to LangChain memory
    components, ensuring that stored context meets quality standards.

    Lifecycle:
    1. Initialization: Set up with memory component and validators/processors
    2. Loading: Retrieve and validate stored context for chain execution
    3. Saving: Validate and store new context from chain execution
    4. Clearing: Remove stored context when needed

    Examples:
        ```python
        from langchain.memory import ConversationBufferMemory
        from sifaka.adapters.langchain import SifakaMemory
        from sifaka.rules.formatting import create_length_rule

        # Create a LangChain memory component
        memory = ConversationBufferMemory()

        # Create validators
        length_rule = create_length_rule(max_chars=500)
        validator = RuleBasedValidator([length_rule])

        # Create Sifaka memory
        sifaka_memory = SifakaMemory(
            memory=memory,
            validators=[validator]
        )
        ```
    """

    def __init__(
        self,
        memory: BaseMemory,
        validators: Optional[List[ChainValidator[Any]]] = None,
        processors: Optional[List[ChainOutputProcessor[Any]]] = None,
    ) -> None:
        """
        Initialize with memory and optional validators/processors.

        Args:
            memory: The LangChain memory component to wrap
            validators: Optional list of validators
            processors: Optional list of processors
        """
        self._memory = memory
        self._validators = validators or []
        self._processors = processors or []

    @property
    def memory_variables(self) -> List[str]:
        """
        Return the memory variables.

        Returns:
            List of memory variable names
        """
        return self._memory.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and validate memory variables.

        This method loads memory variables from the wrapped memory component,
        processes them, and validates them before returning.

        Args:
            inputs: The chain inputs

        Returns:
            Dictionary of memory variables
        """
        variables = self._memory.load_memory_variables(inputs)

        # Process variables
        for processor in self._processors:
            for key, value in variables.items():
                if processor.can_process(value):
                    variables[key] = processor.process(value)

        # Validate variables
        for validator in self._validators:
            for value in variables.values():
                if validator.can_validate(value):
                    result = validator.validate(value)
                    if not result.passed:
                        logger.warning("Memory validation failed: %s", result.message)

        return variables

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save context after validation.

        This method validates the inputs and outputs before saving them
        to the wrapped memory component.

        Args:
            inputs: The chain inputs
            outputs: The chain outputs
        """
        # Validate inputs/outputs before saving
        for validator in self._validators:
            for value in {**inputs, **outputs}.values():
                if validator.can_validate(value):
                    result = validator.validate(value)
                    if not result.passed:
                        logger.warning("Context validation failed: %s", result.message)
                        return

        self._memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """
        Clear the memory.

        This method clears all stored context in the wrapped memory component.
        """
        self._memory.clear()


def create_simple_langchain(
    chain: ChainType,
    rules: Optional[List[Rule]] = None,
    validators: Optional[List[ChainValidator[OutputType]]] = None,
    processors: Optional[List[ChainOutputProcessor[OutputType]]] = None,
    memory: Optional[ChainMemory] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    output_parser: Optional[BaseOutputParser[OutputType]] = None,
    critique: bool = True,
    tracer: Optional[Tracer] = None,
) -> SifakaChain[OutputType]:
    """
    Create a simple LangChain integration with Sifaka's features.

    This factory function creates a SifakaChain with the specified components.
    If rules are provided, they are wrapped in a RuleBasedValidator.

    Args:
        chain: The LangChain chain to wrap (either LLMChain or RunnableSequence)
        rules: Optional list of Sifaka rules (converted to validators)
        validators: Optional list of validators
        processors: Optional list of processors
        memory: Optional memory component
        callbacks: Optional list of callbacks
        output_parser: Optional output parser
        critique: Whether to enable critique
        tracer: Optional tracer for debugging

    Returns:
        A configured SifakaChain

    Examples:
        ```python
        from langchain.chains import LLMChain
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from sifaka.adapters.langchain import create_simple_langchain
        from sifaka.rules.content import create_sentiment_rule

        # Create a LangChain chain
        llm = OpenAI()
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer the following question: {question}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Create rules
        rule = create_sentiment_rule(valid_labels=["positive", "neutral"])

        # Create Sifaka chain
        sifaka_chain = create_simple_langchain(
            chain=chain,
            rules=[rule],
            critique=True
        )
        ```
    """
    # Convert rules to validator if provided
    all_validators = validators or []
    if rules:
        all_validators.append(RuleBasedValidator(rules))

    config = ChainConfig(
        validators=all_validators,
        processors=processors or [],
        memory=memory,
        callbacks=callbacks or [],
        output_parser=output_parser,
        critique=critique,
        tracer=tracer,
    )

    return SifakaChain(chain=chain, config=config)


def wrap_chain(
    chain: ChainType,
    rules: Optional[List[Rule]] = None,
    validators: Optional[List[ChainValidator[OutputType]]] = None,
    processors: Optional[List[ChainOutputProcessor[OutputType]]] = None,
    memory: Optional[ChainMemory] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    output_parser: Optional[BaseOutputParser[OutputType]] = None,
    critique: bool = True,
    tracer: Optional[Tracer] = None,
) -> SifakaChain[OutputType]:
    """
    Wrap a LangChain chain with Sifaka's reflection and reliability features.

    This factory function is a more descriptive alias for create_simple_langchain.
    It wraps a chain with validation, processing, and critique capabilities.

    Args:
        chain: The LangChain chain to wrap (either LLMChain or RunnableSequence)
        rules: Optional list of Sifaka rules
        validators: Optional list of validators
        processors: Optional list of processors
        memory: Optional memory component
        callbacks: Optional list of callbacks
        output_parser: Optional output parser
        critique: Whether to enable critique
        tracer: Optional tracer for debugging

    Returns:
        The wrapped chain
    """
    return create_simple_langchain(
        chain=chain,
        rules=rules,
        validators=validators,
        processors=processors,
        memory=memory,
        callbacks=callbacks,
        output_parser=output_parser,
        critique=critique,
        tracer=tracer,
    )


def wrap_memory(
    memory: BaseMemory,
    rules: Optional[List[Rule]] = None,
    validators: Optional[List[ChainValidator[Any]]] = None,
    processors: Optional[List[ChainOutputProcessor[Any]]] = None,
) -> SifakaMemory:
    """
    Wrap a LangChain memory component with Sifaka's features.

    This factory function creates a SifakaMemory with validation and processing
    capabilities. If rules are provided, they are wrapped in a RuleBasedValidator.

    Args:
        memory: The memory to wrap
        rules: Optional list of Sifaka rules
        validators: Optional list of validators
        processors: Optional list of processors

    Returns:
        The wrapped memory

    Examples:
        ```python
        from langchain.memory import ConversationBufferMemory
        from sifaka.adapters import wrap_memory
        from sifaka.rules.formatting import create_length_rule

        # Create a memory component
        memory = ConversationBufferMemory()

        # Create rules
        rule = create_length_rule(max_chars=500)

        # Wrap the memory
        sifaka_memory = wrap_memory(
            memory=memory,
            rules=[rule]
        )
        ```
    """
    # Convert rules to validator if provided
    all_validators = validators or []
    if rules:
        all_validators.append(RuleBasedValidator(rules))

    return SifakaMemory(
        memory=memory,
        validators=all_validators,
        processors=processors
    )


# Export public classes and functions
__all__ = [
    # Protocols
    "ChainValidator",
    "ChainOutputProcessor",
    "ChainMemory",
    # Configuration
    "ChainConfig",
    # Core components
    "SifakaChain",
    "RuleBasedValidator",
    "SifakaMemory",
    # Factory functions
    "create_simple_langchain",
    "wrap_chain",
    "wrap_memory",
]
