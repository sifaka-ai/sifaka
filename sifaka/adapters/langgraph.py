"""
LangGraph adapter for Sifaka.

This module provides adapter classes and functions to integrate LangGraph with Sifaka's
reflection and reliability features.

Example usage:
    ```python
    from langgraph.graph import Graph
    from sifaka.adapters.langgraph import wrap_graph
    from sifaka.rules.base import Rule
    from sifaka.critique.base import Critique
    from sifaka.utils.tracing import Tracer

    # Create a LangGraph graph
    graph = Graph()

    # Create rules for validation
    rules = [Rule(...)]

    # Create a critic for improving outputs
    critic = Critique(...)

    # Create a tracer for debugging
    tracer = Tracer()

    # Wrap the graph with Sifaka's features
    sifaka_graph = wrap_graph(
        graph=graph,
        rules=rules,
        critique=True,
        critic=critic,
        tracer=tracer
    )

    # Run the graph
    output = sifaka_graph({"input": "Hello, world!"})
    ```

Example with state graph:
    ```python
    from langgraph.graph import StateGraph
    from sifaka.adapters.langgraph import wrap_state_graph

    # Create a state graph
    state_graph = StateGraph()

    # Wrap the state graph
    sifaka_state_graph = wrap_state_graph(
        graph=state_graph,
        rules=rules,
        critique=True,
        critic=critic,
        tracer=tracer
    )

    # Run the state graph
    output = sifaka_state_graph({"input": "Hello, world!"})
    ```

Example with tool node:
    ```python
    from langgraph.prebuilt import ToolNode
    from sifaka.adapters.langgraph import wrap_tool_node

    # Create a tool node
    node = ToolNode(...)

    # Wrap the node
    sifaka_node = wrap_tool_node(
        node=node,
        rules=rules,
        critique=True,
        critic=critic,
        tracer=tracer
    )

    # Use the node
    output = sifaka_node.invoke({"tool": "example", "input": "Hello"})
    ```

Example with channel:
    ```python
    from langgraph.channels import AnyValue
    from sifaka.adapters.langgraph import wrap_channel

    # Create a channel
    channel = AnyValue()

    # Wrap the channel
    sifaka_channel = wrap_channel(
        channel=channel,
        rules=rules,
        critique=True,
        critic=critic,
        tracer=tracer
    )

    # Use the channel
    sifaka_channel.send("Hello, world!")
    message = sifaka_channel.receive()
    ```
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from langgraph.graph import StateGraph

from sifaka.rules.base import Rule, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.chain.managers.validation import ValidationManager

logger = get_logger(__name__)
StateType = TypeVar("StateType")
NodeType = TypeVar("NodeType")


@runtime_checkable
class GraphValidator(Protocol[StateType]):
    """Protocol for graph state validation."""

    def validate(self, state: StateType) -> RuleResult: ...
    def can_validate(self, state: StateType) -> bool: ...


@runtime_checkable
class GraphProcessor(Protocol[StateType]):
    """Protocol for graph state processing."""

    def process(self, state: StateType) -> StateType: ...
    def can_process(self, state: StateType) -> bool: ...


@runtime_checkable
class GraphNode(Protocol[StateType, NodeType]):
    """Protocol for graph nodes."""

    def run(self, state: StateType) -> NodeType: ...
    def can_run(self, state: StateType) -> bool: ...


@dataclass
class GraphConfig(Generic[StateType]):
    """Configuration for Sifaka graphs."""

    validators: List[GraphValidator[StateType]] = field(default_factory=list)
    processors: List[GraphProcessor[StateType]] = field(default_factory=list)
    critique: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.validators, list):
            raise ValueError("validators must be a list")
        if not isinstance(self.processors, list):
            raise ValueError("processors must be a list")


class SifakaGraph(Generic[StateType, NodeType]):
    """
    A LangGraph graph that integrates Sifaka's reflection and reliability features.

    This class follows the component-based architecture pattern by delegating to
    specialized components for validation, processing, and error handling.
    """

    def __init__(
        self,
        graph: StateGraph,
        config: GraphConfig[StateType],
    ) -> None:
        """
        Initialize a Sifaka graph.

        Args:
            graph: The LangGraph graph to wrap
            config: Graph configuration
        """
        self._graph = graph
        self._config = config

        # Create validation manager
        self._validation_manager = self._create_validation_manager()

        # Create result formatter
        self._result_formatter = self._create_result_formatter()

    def _create_validation_manager(self):
        """Create a validation manager for this graph."""
        return ValidationManager(self._config.validators)

    def _create_result_formatter(self):
        """Create a result formatter for this graph."""
        return ResultFormatter()

    @property
    def has_validators(self) -> bool:
        """Return whether the graph has any validators."""
        return bool(self._config.validators)

    @property
    def has_processors(self) -> bool:
        """Return whether the graph has any processors."""
        return bool(self._config.processors)

    @property
    def graph(self) -> StateGraph:
        """Return the wrapped graph."""
        return self._graph

    @property
    def config(self) -> GraphConfig[StateType]:
        """Return the graph configuration."""
        return self._config

    def _validate_state(self, state: StateType) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the state using the graph's validators.

        Args:
            state: The state to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the state passed all validators
                - violations: List of validation violations
        """
        violations = []
        for validator in self._config.validators:
            if validator.can_validate(state):
                result = validator.validate(state)
                if not result.passed:
                    violations.append(
                        {
                            "validator": validator.__class__.__name__,
                            "message": result.message,
                            "metadata": result.metadata,
                        }
                    )

        return not violations, violations

    def _process_state(self, state: StateType) -> StateType:
        """
        Process the state using the graph's processors.

        Args:
            state: The state to process

        Returns:
            The processed state
        """
        processed = state
        for processor in self._config.processors:
            if processor.can_process(processed):
                processed = processor.process(processed)
        return processed

    def run(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the graph with Sifaka's reflection and reliability features.

        Args:
            inputs: The inputs to the graph
            **kwargs: Additional arguments for the graph

        Returns:
            The graph's output state

        Raises:
            ValueError: If the state fails validation and critique is disabled
        """
        # Run the graph
        state = self._graph.run(inputs, **kwargs)
        logger.debug("Graph state: %s", state)

        # Process the state
        if self.has_processors:
            state = self._process_state(state)

        # Validate the state
        passed, violations = self._validate_state(state)
        if not passed:
            if self._config.critique:
                # TODO: Implement critique with standardized error handling
                logger.warning("Graph state validation failed: %s", violations)
            else:
                error_message = "\n".join([f"{v['validator']}: {v['message']}" for v in violations])
                raise ValueError(f"Graph state validation failed:\n{error_message}")

        return state

    def __call__(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the graph (callable interface)."""
        return self.run(inputs, **kwargs)


class RuleBasedValidator(GraphValidator[Dict[str, Any]]):
    """A validator that uses Sifaka rules."""

    def __init__(self, rules: List[Rule]) -> None:
        """Initialize with rules."""
        self._rules = rules

    def validate(self, state: Dict[str, Any]) -> RuleResult:
        """
        Validate using rules.

        Args:
            state: The state to validate

        Returns:
            RuleResult indicating whether validation passed
        """
        # Iterate through all values in the state dictionary
        for _, value in state.items():
            if isinstance(value, str):
                for rule in self._rules:
                    result = rule.validate(value)
                    if not result.passed:
                        return result
        return RuleResult(passed=True, message="All rules passed")

    def can_validate(self, state: Dict[str, Any]) -> bool:
        """Check if can validate the state."""
        return isinstance(state, dict)


class SifakaNode(GraphNode[StateType, NodeType]):
    """
    A node in a Sifaka graph that integrates validation and processing.
    """

    def __init__(
        self,
        node: Callable[[StateType], NodeType],
        validators: Optional[List[GraphValidator[StateType]]] = None,
        processors: Optional[List[GraphProcessor[StateType]]] = None,
    ) -> None:
        """
        Initialize a Sifaka node.

        Args:
            node: The node function to wrap
            validators: Optional list of validators
            processors: Optional list of processors
        """
        self._node = node
        self._validators = validators or []
        self._processors = processors or []

    def run(self, state: StateType) -> NodeType:
        """Run the node with validation and processing."""
        # Process state before running node
        processed_state = state
        for processor in self._processors:
            if processor.can_process(processed_state):
                processed_state = processor.process(processed_state)

        # Validate state before running node
        for validator in self._validators:
            if validator.can_validate(processed_state):
                result = validator.validate(processed_state)
                if not result.passed:
                    logger.warning("Node validation failed: %s", result.message)

        # Run the node
        return self._node(processed_state)

    def can_run(self, state: StateType) -> bool:
        """
        Check if the node can run on the state.

        Args:
            state: The state to check

        Returns:
            Always returns True as this node can run on any state
        """
        # We don't use the state parameter but it's required by the protocol
        return True


def create_simple_graph(
    graph: StateGraph,
    validators: Optional[List[GraphValidator[StateType]]] = None,
    processors: Optional[List[GraphProcessor[StateType]]] = None,
    critique: bool = True,
) -> SifakaGraph[StateType, Any]:
    """
    Create a simple LangGraph integration with Sifaka's features.

    This factory function creates a SifakaGraph with the specified components.

    Args:
        graph: The graph to wrap
        validators: Optional list of validators
        processors: Optional list of processors
        critique: Whether to enable critique

    Returns:
        A configured SifakaGraph
    """
    config = GraphConfig(
        validators=validators or [],
        processors=processors or [],
        critique=critique,
    )

    return SifakaGraph(graph=graph, config=config)


def wrap_graph(
    graph: StateGraph,
    config: Optional[GraphConfig[StateType]] = None,
) -> SifakaGraph[StateType, Any]:
    """
    Wrap a LangGraph graph with Sifaka's features.

    Args:
        graph: The graph to wrap
        config: Optional graph configuration

    Returns:
        The wrapped graph
    """
    return SifakaGraph(graph=graph, config=config or GraphConfig())


def wrap_node(
    node: Callable[[StateType], NodeType],
    validators: Optional[List[GraphValidator[StateType]]] = None,
    processors: Optional[List[GraphProcessor[StateType]]] = None,
) -> SifakaNode[StateType, NodeType]:
    """
    Wrap a graph node with Sifaka's features.

    Args:
        node: The node function to wrap
        validators: Optional list of validators
        processors: Optional list of processors

    Returns:
        The wrapped node
    """
    return SifakaNode(node=node, validators=validators, processors=processors)


# Export public classes and functions
__all__ = [
    # Protocols
    "GraphValidator",
    "GraphProcessor",
    "GraphNode",
    # Configuration
    "GraphConfig",
    # Core components
    "SifakaGraph",
    "RuleBasedValidator",
    "SifakaNode",
    # Factory functions
    "create_simple_graph",
    "wrap_graph",
    "wrap_node",
]
