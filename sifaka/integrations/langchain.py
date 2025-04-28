"""
LangChain integration for Sifaka.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
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

logger = get_logger(__name__)
T = TypeVar("T")
OutputType = TypeVar("OutputType")
ChainType = Union[LLMChain, RunnableSequence]

@runtime_checkable
class ChainValidator(Protocol[OutputType]):
    """Protocol for chain output validation."""

    def validate(self, output: OutputType) -> RuleResult: ...
    def can_validate(self, output: OutputType) -> bool: ...

@runtime_checkable
class ChainOutputProcessor(Protocol[OutputType]):
    """Protocol for chain output processing."""

    def process(self, output: OutputType) -> OutputType: ...
    def can_process(self, output: OutputType) -> bool: ...

@runtime_checkable
class ChainMemory(Protocol):
    """Protocol for chain memory components."""

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None: ...
    def clear(self) -> None: ...
    @property
    def memory_variables(self) -> List[str]: ...

@dataclass
class ChainConfig(Generic[OutputType]):
    """Configuration for Sifaka chains."""

    validators: List[ChainValidator[OutputType]] = field(default_factory=list)
    processors: List[ChainOutputProcessor[OutputType]] = field(default_factory=list)
    memory: Optional[ChainMemory] = None
    callbacks: List[BaseCallbackHandler] = field(default_factory=list)
    output_parser: Optional[BaseOutputParser[OutputType]] = None
    critique: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.output_parser and not isinstance(self.output_parser, BaseOutputParser):
            raise ValueError("output_parser must be an instance of BaseOutputParser")
        if self.memory and not isinstance(self.memory, (BaseMemory, ChainMemory)):
            raise ValueError("memory must implement ChainMemory protocol")

class SifakaChain(Generic[OutputType]):
    """
    A LangChain chain that integrates Sifaka's reflection and reliability features.
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

        if isinstance(chain, LLMChain):
            if config.memory:
                self._chain.memory = cast(BaseMemory, config.memory)
            if config.callbacks:
                self._chain.callbacks = config.callbacks

    @property
    def has_validators(self) -> bool:
        """Return whether the chain has any validators."""
        return bool(self._config.validators)

    @property
    def has_processors(self) -> bool:
        """Return whether the chain has any processors."""
        return bool(self._config.processors)

    @property
    def has_memory(self) -> bool:
        """Return whether the chain has memory."""
        return self._config.memory is not None

    @property
    def has_output_parser(self) -> bool:
        """Return whether the chain has an output parser."""
        return self._config.output_parser is not None

    def _validate_output(self, output: OutputType) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the output using the chain's validators.

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
                # TODO: Implement critique
                pass
            else:
                raise ValueError(f"Output validation failed: {violations}")

        return output

    def __call__(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> OutputType:
        """Alias for run."""
        return self.run(inputs, **kwargs)

def wrap_chain(
    chain: ChainType,
    config: Optional[ChainConfig[OutputType]] = None,
) -> SifakaChain[OutputType]:
    """
    Wrap a LangChain chain with Sifaka's reflection and reliability features.

    Args:
        chain: The LangChain chain to wrap (either LLMChain or RunnableSequence)
        config: Chain configuration

    Returns:
        A Sifaka chain
    """
    return SifakaChain(chain=chain, config=config or ChainConfig())

class RuleBasedValidator(ChainValidator[str]):
    """A validator that uses Sifaka rules."""

    def __init__(self, rules: List[Rule]) -> None:
        """Initialize with rules."""
        self._rules = rules

    def validate(self, output: str) -> RuleResult:
        """Validate using rules."""
        for rule in self._rules:
            result = rule.validate(output)
            if not result.passed:
                return result
        return RuleResult(passed=True, message="All rules passed")

    def can_validate(self, output: str) -> bool:
        """Check if can validate the output."""
        return isinstance(output, str)

class SifakaMemory(ChainMemory):
    """
    A memory component that integrates Sifaka's rules with LangChain's memory system.
    """

    def __init__(
        self,
        memory: BaseMemory,
        validators: Optional[List[ChainValidator[Any]]] = None,
        processors: Optional[List[ChainOutputProcessor[Any]]] = None,
    ) -> None:
        """Initialize with memory and optional validators/processors."""
        self._memory = memory
        self._validators = validators or []
        self._processors = processors or []

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return self._memory.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate memory variables."""
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
        """Save context after validation."""
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
        """Clear the memory."""
        self._memory.clear()

def wrap_memory(
    memory: BaseMemory,
    validators: Optional[List[ChainValidator[Any]]] = None,
    processors: Optional[List[ChainOutputProcessor[Any]]] = None,
) -> SifakaMemory:
    """
    Wrap a LangChain memory component with Sifaka's features.

    Args:
        memory: The memory to wrap
        validators: Optional list of validators
        processors: Optional list of processors

    Returns:
        The wrapped memory
    """
    return SifakaMemory(memory=memory, validators=validators, processors=processors)

# Export public classes and functions
__all__ = [
    "ChainValidator",
    "ChainOutputProcessor",
    "ChainMemory",
    "ChainConfig",
    "SifakaChain",
    "RuleBasedValidator",
    "SifakaMemory",
    "wrap_chain",
    "wrap_memory",
]
