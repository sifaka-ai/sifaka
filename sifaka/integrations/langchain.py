"""
LangChain integration for Sifaka.
"""

from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from pydantic import Field
from langchain.chains import LLMChain
from langchain.memory import BaseMemory
from langchain.callbacks import BaseCallbackHandler
from langchain.output_parsers import BaseOutputParser
from sifaka.rules.base import Rule, RuleResult
from sifaka.models.base import ModelProvider
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


class SifakaChain(Generic[T]):
    """
    A LangChain chain that integrates Sifaka's reflection and reliability features.

    Attributes:
        chain: The underlying LangChain chain
        rules: List of rules to apply to chain outputs
        critique: Whether to enable critique
        memory: Optional memory to use with the chain
        callbacks: Optional callbacks to use with the chain
        output_parser: Optional output parser to use with the chain
    """

    chain: LLMChain
    rules: List[Rule] = Field(default_factory=list)
    critique: bool = Field(default=True)
    memory: Optional[BaseMemory] = Field(default=None)
    callbacks: List[BaseCallbackHandler] = Field(default_factory=list)
    output_parser: Optional[BaseOutputParser[T]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(
        self,
        chain: LLMChain,
        rules: Optional[List[Rule]] = None,
        critique: bool = True,
        memory: Optional[BaseMemory] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        output_parser: Optional[BaseOutputParser[T]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Sifaka chain.

        Args:
            chain: The LangChain chain to wrap
            rules: List of rules to apply to chain outputs
            critique: Whether to enable critique
            memory: Optional memory to use with the chain
            callbacks: Optional callbacks to use with the chain
            output_parser: Optional output parser to use with the chain
            **kwargs: Additional arguments for the chain
        """
        self.chain = chain
        self.rules = rules or []
        self.critique = critique
        self.memory = memory
        self.callbacks = callbacks or []
        self.output_parser = output_parser

        if memory:
            self.chain.memory = memory

        if callbacks:
            self.chain.callbacks = callbacks

    @property
    def has_rules(self) -> bool:
        """Return whether the chain has any rules."""
        return bool(self.rules)

    @property
    def has_memory(self) -> bool:
        """Return whether the chain has memory."""
        return self.memory is not None

    @property
    def has_output_parser(self) -> bool:
        """Return whether the chain has an output parser."""
        return self.output_parser is not None

    def _validate_output(self, output: str) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the output using the chain's rules.

        Args:
            output: The output to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the output passed all rules
                - violations: List of rule violations
        """
        violations = []
        for rule in self.rules:
            result = rule.validate(output)
            if not result.passed:
                violations.append(
                    {"rule": rule.name, "message": result.message, "metadata": result.metadata}
                )

        return not violations, violations

    def run(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> Union[str, T]:
        """
        Run the chain with Sifaka's reflection and reliability features.

        Args:
            inputs: The inputs to the chain
            **kwargs: Additional arguments for the chain

        Returns:
            The chain's output, optionally parsed by the output parser

        Raises:
            ValueError: If the output fails validation and critique is disabled
        """
        # Run the chain
        output = self.chain.run(inputs, **kwargs)
        logger.debug("Chain output: %s", output)

        # Validate the output
        passed, violations = self._validate_output(output)
        if not passed:
            if self.critique:
                # TODO: Implement critique
                pass
            else:
                raise ValueError(f"Chain output failed validation: {violations}")

        # Parse the output if an output parser is configured
        if self.has_output_parser:
            return self.output_parser.parse(output)

        return output

    def __call__(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> Union[str, T]:
        """
        Run the chain with Sifaka's reflection and reliability features.

        This allows the chain to be called like a function.

        Args:
            inputs: The inputs to the chain
            **kwargs: Additional arguments for the chain

        Returns:
            The chain's output, optionally parsed by the output parser
        """
        return self.run(inputs, **kwargs)


def wrap_chain(chain: LLMChain, **kwargs) -> SifakaChain:
    """
    Wrap a LangChain chain with Sifaka's features.

    Args:
        chain: The chain to wrap
        **kwargs: Additional arguments for SifakaChain

    Returns:
        The wrapped chain
    """
    return SifakaChain(chain=chain, **kwargs)


class SifakaMemory(BaseMemory):
    """
    A memory component that integrates Sifaka's rules with LangChain's memory system.

    Attributes:
        memory (BaseMemory): The underlying memory component
        rules (List[Rule]): List of Sifaka rules to apply to memory contents
    """

    memory: BaseMemory
    rules: List[Rule] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return self.memory.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables with Sifaka rule validation.

        Args:
            inputs (Dict[str, Any]): Input dictionary

        Returns:
            Dict[str, Any]: Memory variables dictionary
        """
        variables = self.memory.load_memory_variables(inputs)

        # Apply rules to memory contents
        for key, value in variables.items():
            if isinstance(value, str):
                for rule in self.rules:
                    result = rule.validate(value)
                    if not result.passed:
                        # Log the validation failure
                        print(f"Memory validation failed for {key}: {result.message}")

        return variables

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory."""
        self.memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear memory."""
        self.memory.clear()


class SifakaDocumentProcessor:
    """
    A document processor that integrates Sifaka's rules with LangChain's document processing.

    Attributes:
        text_splitter (TextSplitter): The text splitter to use
        rules (List[Rule]): List of Sifaka rules to apply to documents
        vector_store (Optional[VectorStore]): Optional vector store for document storage
    """

    text_splitter: TextSplitter
    rules: List[Rule] = Field(default_factory=list)
    vector_store: Optional[VectorStore] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def process_document(self, document: Document) -> List[Document]:
        """
        Process a document with Sifaka's rules.

        Args:
            document (Document): The document to process

        Returns:
            List[Document]: List of processed documents
        """
        # Split the document
        chunks = self.text_splitter.split_documents([document])

        # Apply rules to each chunk
        processed_chunks = []
        for chunk in chunks:
            valid = True
            for rule in self.rules:
                result = rule.validate(chunk.page_content)
                if not result.passed:
                    valid = False
                    break

            if valid:
                processed_chunks.append(chunk)

        # Store in vector store if available
        if self.vector_store and processed_chunks:
            self.vector_store.add_documents(processed_chunks)

        return processed_chunks


class SifakaCallbackHandler(BaseCallbackHandler):
    """
    A callback handler that integrates Sifaka's rules with LangChain's callback system.

    Attributes:
        rules (List[Rule]): List of Sifaka rules to apply to callback events
    """

    rules: List[Rule] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Handle chain start event."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Handle chain end event."""
        # Apply rules to chain outputs
        for key, value in outputs.items():
            if isinstance(value, str):
                for rule in self.rules:
                    result = rule.validate(value)
                    if not result.passed:
                        print(f"Chain output validation failed for {key}: {result.message}")

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Handle chain error event."""
        pass


def wrap_memory(memory: BaseMemory, **kwargs) -> SifakaMemory:
    """
    Wrap a LangChain memory component with Sifaka's features.

    Args:
        memory (BaseMemory): The memory component to wrap
        **kwargs: Additional arguments for SifakaMemory

    Returns:
        SifakaMemory: The wrapped memory component
    """
    return SifakaMemory(memory=memory, **kwargs)


def wrap_callback_handler(handler: BaseCallbackHandler, **kwargs) -> SifakaCallbackHandler:
    """
    Wrap a LangChain callback handler with Sifaka's features.

    Args:
        handler (BaseCallbackHandler): The callback handler to wrap
        **kwargs: Additional arguments for SifakaCallbackHandler

    Returns:
        SifakaCallbackHandler: The wrapped callback handler
    """
    return SifakaCallbackHandler(**kwargs)
