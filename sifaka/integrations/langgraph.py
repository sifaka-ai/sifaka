"""
LangGraph integration for Sifaka.

Example usage:
    ```python
    from langgraph.graph import Graph
    from sifaka.integrations.langgraph import wrap_graph
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
    from sifaka.integrations.langgraph import wrap_state_graph

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
    from sifaka.integrations.langgraph import wrap_tool_node

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
    from sifaka.integrations.langgraph import wrap_channel

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

    @property
    def has_validators(self) -> bool:
        """Return whether the graph has any validators."""
        return bool(self._config.validators)

    @property
    def has_processors(self) -> bool:
        """Return whether the graph has any processors."""
        return bool(self._config.processors)

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
                # TODO: Implement critique
                pass
            else:
                raise ValueError(f"Graph state failed validation: {violations}")

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
        """Validate using rules."""
        for key, value in state.items():
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
        """Check if the node can run on the state."""
        return True

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
    "GraphValidator",
    "GraphProcessor",
    "GraphNode",
    "GraphConfig",
    "SifakaGraph",
    "RuleBasedValidator",
    "SifakaNode",
    "wrap_graph",
    "wrap_node",
]
