"""
LangGraph adapter for Sifaka.

This module provides adapter classes and functions to integrate LangGraph with Sifaka's
reflection and reliability features.

## Architecture Overview

The LangGraph adapter follows a component-based architecture:

1. **Protocol Definitions**: Define interfaces for validation and processing
2. **Configuration**: Standardized configuration for graph components
3. **Core Components**: Wrapper classes for graphs, nodes, and validators
4. **Factory Functions**: Simple creation patterns for common use cases

## Component Lifecycle

### SifakaGraph
1. **Initialization**: Set up with graph and configuration
2. **Execution**: Run the graph with inputs
3. **Validation**: Validate the graph's output state
4. **Processing**: Process the state based on validation results

### Validators
1. **Initialization**: Set up with validation rules
2. **Validation**: Validate graph states against rules
3. **Result**: Return standardized validation results

## Usage Examples

### Basic Graph Wrapping

```python
from langgraph.graph import StateGraph
from sifaka.adapters.langgraph import wrap_graph
from sifaka.rules.base import Rule
from sifaka.utils.tracing import Tracer

# Create a LangGraph graph
graph = StateGraph()
# ... define graph nodes and edges

# Create rules for validation
rules = [Rule(...)]

# Create a tracer for debugging
tracer = Tracer()

# Wrap the graph with Sifaka's features
sifaka_graph = wrap_graph(
    graph=graph,
    rules=rules,
    critique=True,
    tracer=tracer
)

# Run the graph
output = sifaka_graph({"input": "Hello, world!"})
```

### Tool Node Integration

```python
from langgraph.prebuilt import ToolNode
from sifaka.adapters.langgraph import wrap_node

# Create a tool node
node = ToolNode(...)

# Wrap the node
sifaka_node = wrap_node(
    node=node,
    rules=rules
)

# Use the node
output = sifaka_node.invoke({"tool": "example", "input": "Hello"})
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
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from langgraph.graph import StateGraph

from sifaka.rules.base import Rule, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.chain.managers.validation import ValidationManager
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)
StateType = TypeVar("StateType")  # Type of graph state
NodeType = TypeVar("NodeType")    # Type of node output
ResultType = TypeVar("ResultType") # Type of validation result


@runtime_checkable
class GraphValidator(Protocol[StateType, ResultType]):
    """
    Protocol for graph state validation.

    Classes implementing this protocol can validate graph states
    and return standardized results.

    Type Parameters:
        StateType: The type of state to validate
        ResultType: The type of validation result

    Lifecycle:
    1. Initialization: Configure validation parameters
    2. Validation: Receive state and apply validation logic
    3. Result: Return standardized validation results

    Examples:
        ```python
        from sifaka.adapters.langgraph import GraphValidator

        class ContentValidator(GraphValidator[Dict[str, Any], RuleResult]):
            def validate(self, state: Dict[str, Any]) -> RuleResult:
                # Extract text from state
                text = state.get("output", "")
                if not isinstance(text, str):
                    return RuleResult(passed=False, message="Output is not a string")

                # Apply validation logic
                is_valid = len(text) > 10
                return RuleResult(
                    passed=is_valid,
                    message="Content validation " +
                            ("passed" if is_valid else "failed")
                )

            def can_validate(self, state: Dict[str, Any]) -> bool:
                return "output" in state
        ```
    """

    def validate(self, state: StateType) -> ResultType:
        """
        Validate a graph state.

        Args:
            state: The state to validate

        Returns:
            Validation result
        """
        ...

    def can_validate(self, state: StateType) -> bool:
        """
        Check if this validator can validate the state.

        Args:
            state: The state to check

        Returns:
            True if this validator can validate the state
        """
        ...


@runtime_checkable
class GraphProcessor(Protocol[StateType]):
    """
    Protocol for graph state processing.

    Classes implementing this protocol can process graph states
    to enhance or modify them.

    Type Parameters:
        StateType: The type of state to process

    Lifecycle:
    1. Initialization: Configure processing parameters
    2. Processing: Receive state and apply processing logic
    3. Result: Return modified state

    Examples:
        ```python
        from sifaka.adapters.langgraph import GraphProcessor

        class OutputFormatter(GraphProcessor[Dict[str, Any]]):
            def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
                # Extract and format text
                if "output" in state and isinstance(state["output"], str):
                    state["output"] = state["output"].strip().capitalize()
                return state

            def can_process(self, state: Dict[str, Any]) -> bool:
                return "output" in state and isinstance(state["output"], str)
        ```
    """

    def process(self, state: StateType) -> StateType:
        """
        Process a graph state.

        Args:
            state: The state to process

        Returns:
            The processed state
        """
        ...

    def can_process(self, state: StateType) -> bool:
        """
        Check if this processor can process the state.

        Args:
            state: The state to check

        Returns:
            True if this processor can process the state
        """
        ...


@runtime_checkable
class GraphNode(Protocol[StateType, NodeType]):
    """
    Protocol for graph nodes.

    Classes implementing this protocol can act as nodes in a graph,
    processing states and returning outputs.

    Type Parameters:
        StateType: The type of state to process
        NodeType: The type of node output

    Lifecycle:
    1. Initialization: Configure node parameters
    2. Execution: Receive state and apply node logic
    3. Result: Return node output

    Examples:
        ```python
        from sifaka.adapters.langgraph import GraphNode

        class TextProcessingNode(GraphNode[Dict[str, Any], str]):
            def run(self, state: Dict[str, Any]) -> str:
                # Extract and process text
                text = state.get("input", "")
                return text.upper()

            def can_run(self, state: Dict[str, Any]) -> bool:
                return "input" in state and isinstance(state["input"], str)
        ```
    """

    def run(self, state: StateType) -> NodeType:
        """
        Run the node with the given state.

        Args:
            state: The state to process

        Returns:
            The node output
        """
        ...

    def can_run(self, state: StateType) -> bool:
        """
        Check if this node can run with the given state.

        Args:
            state: The state to check

        Returns:
            True if this node can run with the state
        """
        ...


@dataclass
class GraphConfig(Generic[StateType]):
    """
    Configuration for Sifaka graphs.

    This class provides a standardized way to configure LangGraph
    integrations with Sifaka features.

    Type Parameters:
        StateType: The type of graph state

    Lifecycle:
    1. Creation: Instantiated with configuration options
    2. Validation: Post-init validation of configuration values
    3. Usage: Accessed by graph components during setup and execution

    Examples:
        ```python
        from sifaka.adapters.langgraph import GraphConfig, GraphValidator

        # Create validators
        validators = [MyValidator(), AnotherValidator()]

        # Create processors
        processors = [MyProcessor(), AnotherProcessor()]

        # Create configuration
        config = GraphConfig(
            validators=validators,
            processors=processors,
            critique=True
        )
        ```
    """

    validators: List[GraphValidator[StateType, Any]] = field(default_factory=list)
    processors: List[GraphProcessor[StateType]] = field(default_factory=list)
    critique: bool = True
    tracer: Optional[Tracer] = None

    def __post_init__(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If validators or processors are not lists
        """
        if not isinstance(self.validators, list):
            raise ValueError("validators must be a list")
        if not isinstance(self.processors, list):
            raise ValueError("processors must be a list")


class SifakaGraph(Generic[StateType, NodeType]):
    """
    A LangGraph graph that integrates Sifaka's reflection and reliability features.

    This class follows the component-based architecture pattern by delegating to
    specialized components for validation, processing, and error handling.

    Type Parameters:
        StateType: The type of graph state
        NodeType: The type of node output

    Lifecycle:
    1. Initialization: Set up with graph and configuration
    2. Execution: Run the graph with inputs
    3. Validation: Validate the graph's output state
    4. Processing: Process the state based on validation results

    Examples:
        ```python
        from langgraph.graph import StateGraph
        from sifaka.adapters.langgraph import SifakaGraph, GraphConfig

        # Create a LangGraph graph
        graph = StateGraph()
        # ... define graph nodes and edges

        # Create configuration
        config = GraphConfig(
            validators=[MyValidator()],
            processors=[MyProcessor()],
            critique=True
        )

        # Create Sifaka graph
        sifaka_graph = SifakaGraph(graph=graph, config=config)

        # Run the graph
        output = sifaka_graph.run({"input": "Hello, world!"})
        ```
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
        """
        Create a validation manager for this graph.

        Returns:
            ValidationManager for this graph
        """
        return ValidationManager(self._config.validators)

    def _create_result_formatter(self):
        """
        Create a result formatter for this graph.

        Returns:
            ResultFormatter for this graph
        """
        return ResultFormatter()

    @property
    def has_validators(self) -> bool:
        """
        Return whether the graph has any validators.

        Returns:
            True if the graph has validators, False otherwise
        """
        return bool(self._config.validators)

    @property
    def has_processors(self) -> bool:
        """
        Return whether the graph has any processors.

        Returns:
            True if the graph has processors, False otherwise
        """
        return bool(self._config.processors)

    @property
    def graph(self) -> StateGraph:
        """
        Return the wrapped graph.

        Returns:
            The wrapped LangGraph graph
        """
        return self._graph

    @property
    def config(self) -> GraphConfig[StateType]:
        """
        Return the graph configuration.

        Returns:
            The graph configuration
        """
        return self._config

    def _validate_state(self, state: StateType) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the state using the graph's validators.

        This method applies all validators to the state and collects
        any validation violations.

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

        This method applies all processors to the state in sequence.

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

        This method runs the graph, processes the output state,
        validates it, and handles any violations.

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

        # Add trace event if tracing is enabled
        if self._config.tracer:
            self._config.tracer.add_event(
                "graph_execution",
                "complete",
                {
                    "inputs": inputs,
                    "outputs": state,
                    "validation_passed": passed,
                    "validation_violations": violations if not passed else [],
                }
            )

        return state

    def __call__(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the graph (callable interface).

        Args:
            inputs: The inputs to the graph
            **kwargs: Additional arguments for the graph

        Returns:
            The graph's output state
        """
        return self.run(inputs, **kwargs)


class RuleBasedValidator(GraphValidator[Dict[str, Any], RuleResult]):
    """
    A validator that uses Sifaka rules to validate graph states.

    This validator applies a list of Sifaka rules to string values
    in a graph state.

    Lifecycle:
    1. Initialization: Set up with rules
    2. Validation: Extract strings from state and apply rules
    3. Result: Return validation result

    Examples:
        ```python
        from sifaka.adapters.langgraph import RuleBasedValidator
        from sifaka.rules.content import create_sentiment_rule

        # Create rules
        rule = create_sentiment_rule(valid_labels=["positive", "neutral"])

        # Create validator
        validator = RuleBasedValidator([rule])

        # Use validator
        result = validator.validate({"output": "This is great!"})
        ```
    """

    def __init__(self, rules: List[Rule]) -> None:
        """
        Initialize with rules.

        Args:
            rules: The Sifaka rules to use for validation
        """
        self._rules = rules

    def validate(self, state: Dict[str, Any]) -> RuleResult:
        """
        Validate using rules.

        This method extracts string values from the state and
        applies rules to them.

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
        """
        Check if can validate the state.

        Args:
            state: The state to check

        Returns:
            True if the state is a dictionary, False otherwise
        """
        return isinstance(state, dict)


class SifakaNode(GraphNode[StateType, NodeType]):
    """
    A node in a Sifaka graph that integrates validation and processing.

    This class wraps a node function with validation and processing
    capabilities from Sifaka.

    Type Parameters:
        StateType: The type of state to process
        NodeType: The type of node output

    Lifecycle:
    1. Initialization: Set up with node function and validators/processors
    2. Execution: Process state, validate, and run node function
    3. Result: Return node output

    Examples:
        ```python
        from sifaka.adapters.langgraph import SifakaNode

        # Create a node function
        def process_text(state: Dict[str, Any]) -> str:
            return state.get("input", "").upper()

        # Create validators and processors
        validators = [MyValidator()]
        processors = [MyProcessor()]

        # Create Sifaka node
        node = SifakaNode(
            node=process_text,
            validators=validators,
            processors=processors
        )

        # Use the node
        output = node.run({"input": "Hello, world!"})
        ```
    """

    def __init__(
        self,
        node: Callable[[StateType], NodeType],
        validators: Optional[List[GraphValidator[StateType, Any]]] = None,
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
        """
        Run the node with validation and processing.

        This method processes the state, validates it, and then
        runs the node function.

        Args:
            state: The state to process

        Returns:
            The node output
        """
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
    rules: Optional[List[Rule]] = None,
    validators: Optional[List[GraphValidator[StateType, Any]]] = None,
    processors: Optional[List[GraphProcessor[StateType]]] = None,
    critique: bool = True,
    tracer: Optional[Tracer] = None,
) -> SifakaGraph[StateType, Any]:
    """
    Create a simple LangGraph integration with Sifaka's features.

    This factory function creates a SifakaGraph with the specified components.
    If rules are provided, they are wrapped in a RuleBasedValidator.

    Args:
        graph: The graph to wrap
        rules: Optional list of Sifaka rules (converted to validators)
        validators: Optional list of validators
        processors: Optional list of processors
        critique: Whether to enable critique
        tracer: Optional tracer for debugging

    Returns:
        A configured SifakaGraph

    Examples:
        ```python
        from langgraph.graph import StateGraph
        from sifaka.adapters.langgraph import create_simple_graph
        from sifaka.rules.content import create_sentiment_rule

        # Create a graph
        graph = StateGraph()
        # ... define graph nodes and edges

        # Create rules
        rules = [create_sentiment_rule(valid_labels=["positive", "neutral"])]

        # Create integrated graph
        sifaka_graph = create_simple_graph(
            graph=graph,
            rules=rules,
            critique=True
        )
        ```
    """
    # Convert rules to validator if provided
    all_validators = validators or []
    if rules:
        all_validators.append(RuleBasedValidator(rules))

    # Create configuration
    config = GraphConfig(
        validators=all_validators,
        processors=processors or [],
        critique=critique,
        tracer=tracer,
    )

    return SifakaGraph(graph=graph, config=config)


def wrap_graph(
    graph: StateGraph,
    rules: Optional[List[Rule]] = None,
    validators: Optional[List[GraphValidator[StateType, Any]]] = None,
    processors: Optional[List[GraphProcessor[StateType]]] = None,
    critique: bool = True,
    tracer: Optional[Tracer] = None,
) -> SifakaGraph[StateType, Any]:
    """
    Wrap a LangGraph graph with Sifaka's reflection and reliability features.

    This factory function creates a SifakaGraph with validation, processing, and critique
    capabilities. If rules are provided, they are wrapped in a RuleBasedValidator.

    ## Lifecycle

    1. **Initialization**: Set up wrapped graph with validation and processing capabilities
       - Configure validators from rules and explicit validators
       - Set up processors for state transformation
       - Configure tracing if enabled

    2. **Execution**: Run graph with enhanced capabilities
       - Process input state
       - Execute wrapped graph
       - Validate output state
       - Apply post-processing
       - Handle validation failures

    3. **Maintenance**: Monitor and debug graph execution
       - Trace execution steps
       - Report validation issues
       - Apply critique for improvement

    ## Error Handling

    This wrapper implements several error handling patterns:

    - **Input Validation**: Checks input state before processing
    - **Output Validation**: Validates output state against rules
    - **Critique Mode**: Provides feedback on validation failures
    - **Tracing**: Records execution details for debugging
    - **Graceful Degradation**: Returns original results with warnings when validation fails

    Args:
        graph: The LangGraph graph to wrap
        rules: Optional list of Sifaka rules
        validators: Optional list of graph validators
        processors: Optional list of graph processors
        critique: Whether to enable critique mode (defaults to True)
        tracer: Optional tracer for debugging

    Returns:
        The wrapped graph

    Raises:
        ValueError: If graph is None or not a StateGraph
        ValueError: If rules and validators are both None

    ## Examples

    ```python
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from sifaka.adapters.langgraph import wrap_graph
    from sifaka.rules.content import create_toxicity_rule
    from sifaka.rules.formatting import create_length_rule
    from sifaka.utils.tracing import Tracer

    # Create a LangGraph graph
    builder = StateGraph()

    # Add nodes
    builder.add_node("generate", lambda state: {"output": f"Generated response to: {state['input']}"})
    builder.add_node("process", lambda state: {"output": state["output"].upper()})

    # Add edges
    builder.set_entry_point("generate")
    builder.add_edge("generate", "process")
    builder.set_finish_point("process")

    # Compile graph
    graph = builder.compile()

    # Create rules for validation
    rules = [
        create_length_rule(min_chars=10, max_chars=1000),
        create_toxicity_rule(threshold=0.8)
    ]

    # Create a tracer for debugging
    tracer = Tracer()

    # Wrap the graph with Sifaka's features
    sifaka_graph = wrap_graph(
        graph=graph,
        rules=rules,
        critique=True,
        tracer=tracer
    )

    # Use try/except for robust error handling
    try:
        # Run the graph
        output = sifaka_graph({"input": "Hello, world!"})
        print(f"Output: {output['output']}")

    except ValueError as e:
        # Handle validation errors
        print(f"Validation error: {e}")
        # Fall back to original graph
        output = graph({"input": "Hello, world!"})
        print(f"Fallback output: {output['output']}")

    except Exception as e:
        # Handle other errors
        print(f"Unexpected error: {e}")
        # Implement appropriate fallback strategy
    ```

    ### Advanced Usage with Custom Components

    ```python
    from sifaka.adapters.langgraph import wrap_graph, GraphValidator, GraphProcessor

    # Custom validator for specific state structure
    class StateStructureValidator(GraphValidator[Dict[str, Any], RuleResult]):
        def validate(self, state: Dict[str, Any]) -> RuleResult:
            required_keys = ["input", "output", "metadata"]
            missing_keys = [k for k in required_keys if k not in state]

            if missing_keys:
                return RuleResult(
                    passed=False,
                    message=f"State missing required keys: {missing_keys}",
                    metadata={"missing_keys": missing_keys}
                )
            return RuleResult(passed=True, message="State structure valid")

        def can_validate(self, state: Dict[str, Any]) -> bool:
            return isinstance(state, dict)

    # Custom processor for adding timestamps
    class TimestampProcessor(GraphProcessor[Dict[str, Any]]):
        def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
            from datetime import datetime
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["timestamp"] = datetime.now().isoformat()
            return state

        def can_process(self, state: Dict[str, Any]) -> bool:
            return isinstance(state, dict)

    # Wrap graph with custom components
    enhanced_graph = wrap_graph(
        graph=graph,
        rules=rules,
        validators=[StateStructureValidator()],
        processors=[TimestampProcessor()],
        critique=True,
        tracer=tracer
    )
    ```
    """
    return create_simple_graph(
        graph=graph,
        rules=rules,
        validators=validators,
        processors=processors,
        critique=critique,
        tracer=tracer,
    )


def wrap_node(
    node: Callable[[StateType], NodeType],
    rules: Optional[List[Rule]] = None,
    validators: Optional[List[GraphValidator[StateType, Any]]] = None,
    processors: Optional[List[GraphProcessor[StateType]]] = None,
) -> SifakaNode[StateType, NodeType]:
    """
    Wrap a graph node with Sifaka's features.

    This factory function creates a SifakaNode with validation
    and processing capabilities. If rules are provided, they are
    wrapped in a RuleBasedValidator.

    Args:
        node: The node function to wrap
        rules: Optional list of Sifaka rules
        validators: Optional list of validators
        processors: Optional list of processors

    Returns:
        The wrapped node

    Examples:
        ```python
        from sifaka.adapters.langgraph import wrap_node
        from sifaka.rules.content import create_sentiment_rule

        # Create a node function
        def process_text(state: Dict[str, Any]) -> str:
            return state.get("input", "").upper()

        # Create rules
        rules = [create_sentiment_rule(valid_labels=["positive", "neutral"])]

        # Create processors
        processors = [MyProcessor()]

        # Wrap the node
        sifaka_node = wrap_node(
            node=process_text,
            rules=rules,
            processors=processors
        )
        ```
    """
    # Convert rules to validator if provided
    all_validators = validators or []
    if rules:
        all_validators.append(RuleBasedValidator(rules))

    return SifakaNode(
        node=node,
        validators=all_validators,
        processors=processors
    )


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
