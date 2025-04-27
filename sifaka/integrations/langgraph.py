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

# Standard library imports
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

# Third-party imports
from langgraph.channels import AnyValue
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict

# Local imports
from sifaka.critique.base import Critique
from sifaka.models.base import ModelProvider
from sifaka.rules.base import Rule, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer, TraceEvent

logger = get_logger(__name__)
T = TypeVar("T")


class SifakaGraph(BaseModel, Generic[T]):
    """
    A LangGraph graph that integrates Sifaka's reflection and reliability features.

    Attributes:
        graph: The underlying LangGraph graph
        rules: List of rules to apply to graph outputs
        critique: Whether to enable critique
        tracer: Optional tracer for debugging
        critic: Optional critique system for improving outputs
    """

    graph: Graph
    rules: List[Rule] = []
    critique: bool = True
    tracer: Optional[Tracer] = None
    critic: Optional[Critique] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        graph: Graph,
        rules: Optional[List[Rule]] = None,
        critique: bool = True,
        tracer: Optional[Tracer] = None,
        critic: Optional[Critique] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Sifaka graph.

        Args:
            graph: The LangGraph graph to wrap
            rules: List of rules to apply to graph outputs
            critique: Whether to enable critique
            tracer: Optional tracer for debugging
            critic: Optional critique system for improving outputs
            **kwargs: Additional arguments for the graph
        """
        super().__init__(
            graph=graph,
            rules=rules or [],
            critique=critique,
            tracer=tracer,
            critic=critic,
            **kwargs,
        )

    @property
    def has_rules(self) -> bool:
        """Return whether the graph has any rules."""
        return bool(self.rules)

    @property
    def has_tracer(self) -> bool:
        """Return whether the graph has a tracer."""
        return self.tracer is not None

    @property
    def has_critic(self) -> bool:
        """Return whether the graph has a critic."""
        return self.critic is not None

    def _validate_output(self, output: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the output using the graph's rules.

        Args:
            output: The output to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the output passed all rules
                - violations: List of rule violations
        """
        violations = []
        for rule in self.rules:
            result = rule.validate(str(output))
            if not result.passed:
                violations.append(
                    {"rule": rule.name, "message": result.message, "metadata": result.metadata}
                )

        return not violations, violations

    def _improve_output(
        self, output: Dict[str, Any], violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Improve the output using the critique system.

        Args:
            output: The output to improve
            violations: List of rule violations

        Returns:
            The improved output

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if not self.has_critic:
            raise ValueError(
                "Critique is enabled but no critic is configured. "
                "Please provide a critic when initializing the graph."
            )

        # Create a critique prompt
        critique_prompt = f"""
        The following output failed validation with the following violations:
        {violations}

        Original output:
        {output}

        Please provide an improved version that addresses these issues.
        """

        # Get the improved output from the critic
        improved_output = self.critic.critique(critique_prompt)
        logger.debug("Improved output: %s", improved_output)

        # Validate the improved output
        passed, new_violations = self._validate_output(improved_output)
        if not passed:
            logger.warning("Improved output still has violations: %s", new_violations)

        return improved_output

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Trace an event if a tracer is configured.

        Args:
            event_type: The type of event
            data: The event data
        """
        if self.has_tracer:
            self.tracer.trace(event_type, data)

    def run(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the graph with Sifaka's reflection and reliability features.

        Args:
            inputs: The inputs to the graph
            **kwargs: Additional arguments for the graph

        Returns:
            The graph's output

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if self.has_tracer:
            self._trace_event("run", {"inputs": inputs})

        # Validate the graph
        self.graph.validate()

        # Run the graph
        output = self.graph.compile(inputs)

        # Validate the output
        if self.has_rules:
            passed, violations = self._validate_output(output)
            if not passed:
                if self.critique and self.has_critic:
                    output = self._improve_output(output, violations)
                else:
                    logger.warning("Output validation failed: %s", violations)

        if self.has_tracer:
            self._trace_event("run_complete", {"output": output})

        return output

    def __call__(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Call the graph with Sifaka's reflection and reliability features.

        Args:
            inputs: The inputs to the graph
            **kwargs: Additional arguments for the graph

        Returns:
            The graph's output
        """
        return self.run(inputs, **kwargs)


class SifakaStateGraph(SifakaGraph[T]):
    """
    A LangGraph state graph that integrates Sifaka's reflection and reliability features.

    Attributes:
        graph: The underlying LangGraph state graph
        rules: List of rules to apply to graph outputs
        critique: Whether to enable critique
        tracer: Optional tracer for debugging
        critic: Optional critique system for improving outputs
    """

    graph: StateGraph

    def __init__(
        self,
        graph: StateGraph,
        rules: Optional[List[Rule]] = None,
        critique: bool = True,
        tracer: Optional[Tracer] = None,
        critic: Optional[Critique] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Sifaka state graph.

        Args:
            graph: The LangGraph state graph to wrap
            rules: List of rules to apply to graph outputs
            critique: Whether to enable critique
            tracer: Optional tracer for debugging
            critic: Optional critique system for improving outputs
            **kwargs: Additional arguments for the graph
        """
        super().__init__(
            graph=graph,
            rules=rules,
            critique=critique,
            tracer=tracer,
            critic=critic,
            **kwargs,
        )

    def run(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the state graph with Sifaka's reflection and reliability features.

        Args:
            inputs: The inputs to the graph
            **kwargs: Additional arguments for the graph

        Returns:
            The graph's output

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if self.has_tracer:
            self._trace_event("run", {"inputs": inputs})

        # Validate the graph
        self.graph.validate()

        # Run the graph
        output = self.graph.compile(inputs)

        # Validate the output
        if self.has_rules:
            passed, violations = self._validate_output(output)
            if not passed:
                if self.critique and self.has_critic:
                    output = self._improve_output(output, violations)
                else:
                    logger.warning("Output validation failed: %s", violations)

        if self.has_tracer:
            self._trace_event("run_complete", {"output": output})

        return output


class SifakaToolNode(BaseModel):
    """
    A LangGraph tool node that integrates Sifaka's reflection and reliability features.

    Attributes:
        node: The underlying tool node
        rules: List of rules to apply to tool outputs
        critique: Whether to enable critique
        tracer: Optional tracer for debugging
        critic: Optional critique system for improving outputs
    """

    node: ToolNode
    rules: List[Rule] = []
    critique: bool = True
    tracer: Optional[Tracer] = None
    critic: Optional[Critique] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        node: ToolNode,
        rules: Optional[List[Rule]] = None,
        critique: bool = True,
        tracer: Optional[Tracer] = None,
        critic: Optional[Critique] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Sifaka tool node.

        Args:
            node: The LangGraph tool node to wrap
            rules: List of rules to apply to tool outputs
            critique: Whether to enable critique
            tracer: Optional tracer for debugging
            critic: Optional critique system for improving outputs
            **kwargs: Additional arguments for the node
        """
        super().__init__(
            node=node,
            rules=rules or [],
            critique=critique,
            tracer=tracer,
            critic=critic,
            **kwargs,
        )

    @property
    def has_rules(self) -> bool:
        """Return whether the node has any rules."""
        return bool(self.rules)

    @property
    def has_tracer(self) -> bool:
        """Return whether the node has a tracer."""
        return self.tracer is not None

    @property
    def has_critic(self) -> bool:
        """Return whether the node has a critic."""
        return self.critic is not None

    def _validate_output(self, output: Any) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the output using the node's rules.

        Args:
            output: The output to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the output passed all rules
                - violations: List of rule violations
        """
        violations = []
        for rule in self.rules:
            result = rule.validate(str(output))
            if not result.passed:
                violations.append(
                    {"rule": rule.name, "message": result.message, "metadata": result.metadata}
                )

        return not violations, violations

    def _improve_output(self, output: Any, violations: List[Dict[str, Any]]) -> Any:
        """
        Improve the output using the critique system.

        Args:
            output: The output to improve
            violations: List of rule violations

        Returns:
            The improved output

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if not self.has_critic:
            raise ValueError(
                "Critique is enabled but no critic is configured. "
                "Please provide a critic when initializing the node."
            )

        # Create a critique prompt
        critique_prompt = f"""
        The following output failed validation with the following violations:
        {violations}

        Original output:
        {output}

        Please provide an improved version that addresses these issues.
        """

        # Get the improved output from the critic
        improved_output = self.critic.critique(critique_prompt)
        logger.debug("Improved output: %s", improved_output)

        # Validate the improved output
        passed, new_violations = self._validate_output(improved_output)
        if not passed:
            logger.warning("Improved output still has violations: %s", new_violations)

        return improved_output

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Trace an event if a tracer is configured.

        Args:
            event_type: The type of event
            data: The event data
        """
        if self.has_tracer:
            self.tracer.trace(event_type, data)

    def invoke(self, tool_input: Dict[str, Any]) -> Any:
        """
        Invoke the tool node with the given input.

        Args:
            tool_input: The input to the tool

        Returns:
            The output of the tool

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if self.has_tracer:
            self._trace_event("invoke", {"input": tool_input})

        # Invoke the tool
        output = self.node.invoke(tool_input)

        # Validate the output
        if self.has_rules:
            passed, violations = self._validate_output(output)
            if not passed:
                if self.critique and self.has_critic:
                    output = self._improve_output(output, violations)
                else:
                    logger.warning("Output validation failed: %s", violations)

        if self.has_tracer:
            self._trace_event("invoke_complete", {"output": output})

        return output


class SifakaChannel(BaseModel):
    """
    A LangGraph channel that integrates Sifaka's reflection and reliability features.

    Attributes:
        channel: The underlying channel
        rules: List of rules to apply to channel messages
        critique: Whether to enable critique
        tracer: Optional tracer for debugging
        critic: Optional critique system for improving messages
    """

    channel: AnyValue
    rules: List[Rule] = []
    critique: bool = True
    tracer: Optional[Tracer] = None
    critic: Optional[Critique] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def has_rules(self) -> bool:
        """Return whether the channel has any rules."""
        return bool(self.rules)

    @property
    def has_tracer(self) -> bool:
        """Return whether the channel has a tracer."""
        return self.tracer is not None

    @property
    def has_critic(self) -> bool:
        """Return whether the channel has a critic."""
        return self.critic is not None

    def _validate_message(self, message: Any) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a message using the channel's rules.

        Args:
            message: The message to validate

        Returns:
            Tuple of (passed, violations) where:
                - passed: Whether the message passed all rules
                - violations: List of rule violations
        """
        violations = []
        for rule in self.rules:
            result = rule.validate(str(message))
            if not result.passed:
                violations.append(
                    {"rule": rule.name, "message": result.message, "metadata": result.metadata}
                )

        return not violations, violations

    def _improve_message(self, message: Any, violations: List[Dict[str, Any]]) -> Any:
        """
        Improve a message using the critique system.

        Args:
            message: The message to improve
            violations: List of rule violations

        Returns:
            The improved message

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if not self.has_critic:
            raise ValueError(
                "Critique is enabled but no critic is configured. "
                "Please provide a critic when initializing the channel."
            )

        # Create a critique prompt
        critique_prompt = f"""
        The following message failed validation with the following violations:
        {violations}

        Original message:
        {message}

        Please provide an improved version that addresses these issues.
        """

        # Get the improved message from the critic
        improved_message = self.critic.critique(critique_prompt)
        logger.debug("Improved message: %s", improved_message)

        # Validate the improved message
        passed, new_violations = self._validate_message(improved_message)
        if not passed:
            logger.warning("Improved message still has violations: %s", new_violations)

        return improved_message

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Trace an event if a tracer is configured.

        Args:
            event_type: The type of event
            data: The event data
        """
        if self.has_tracer:
            self.tracer.trace(event_type, data)

    def send(self, message: Any) -> None:
        """
        Send a message through the channel.

        Args:
            message: The message to send

        Raises:
            ValueError: If critique is enabled but no critic is configured
        """
        if self.has_tracer:
            self._trace_event("send", {"message": message})

        # Validate the message
        if self.has_rules:
            passed, violations = self._validate_message(message)
            if not passed:
                if self.critique and self.has_critic:
                    message = self._improve_message(message, violations)
                else:
                    logger.warning("Message validation failed: %s", violations)

        # Send the message
        self.channel.update(message)

    def receive(self) -> Any:
        """
        Receive a message from the channel.

        Returns:
            The received message
        """
        if self.has_tracer:
            self._trace_event("receive", {})

        # Receive the message
        message = self.channel.get()

        # Validate the message
        if self.has_rules:
            passed, violations = self._validate_message(message)
            if not passed:
                if self.critique and self.has_critic:
                    message = self._improve_message(message, violations)
                else:
                    logger.warning("Message validation failed: %s", violations)

        return message


def wrap_graph(graph: Graph, **kwargs) -> SifakaGraph:
    """
    Wrap a LangGraph graph with Sifaka's features.

    Args:
        graph: The graph to wrap
        **kwargs: Additional arguments for SifakaGraph

    Returns:
        The wrapped graph
    """
    return SifakaGraph(graph=graph, **kwargs)


def wrap_state_graph(graph: StateGraph, **kwargs) -> SifakaStateGraph:
    """
    Wrap a LangGraph state graph with Sifaka's features.

    Args:
        graph: The state graph to wrap
        **kwargs: Additional arguments for SifakaStateGraph

    Returns:
        The wrapped state graph
    """
    return SifakaStateGraph(graph=graph, **kwargs)


def wrap_tool_node(node: ToolNode, **kwargs) -> SifakaToolNode:
    """
    Wrap a LangGraph tool node with Sifaka's features.

    Args:
        node: The node to wrap
        **kwargs: Additional arguments for the node

    Returns:
        A wrapped node with Sifaka's features
    """
    return SifakaToolNode(node=node, **kwargs)


def wrap_channel(channel: AnyValue, **kwargs) -> SifakaChannel:
    """
    Wrap a LangGraph channel with Sifaka's features.

    Args:
        channel: The channel to wrap
        **kwargs: Additional arguments for the channel

    Returns:
        A wrapped channel with Sifaka's features
    """
    return SifakaChannel(channel=channel, **kwargs)
