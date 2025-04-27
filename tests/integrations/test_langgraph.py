"""
Tests for the LangGraph integration.
"""

from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import Graph, StateGraph
from langgraph.channels import AnyValue
from langgraph.prebuilt import ToolNode

from sifaka.integrations.langgraph import (
    SifakaGraph,
    SifakaStateGraph,
    SifakaToolNode,
    SifakaChannel,
    wrap_graph,
    wrap_state_graph,
    wrap_tool_node,
    wrap_channel,
)
from sifaka.rules.base import Rule, RuleResult
from sifaka.critique.base import Critique
from sifaka.utils.tracing import Tracer


class MockRule(Rule):
    """Mock rule for testing."""

    def __init__(self) -> None:
        """Initialize the mock rule."""
        super().__init__(name="mock_rule", description="A mock rule for testing")

    def validate(self, output: str) -> RuleResult:
        """Validate the output."""
        if "error" in output:
            return RuleResult(
                passed=False, message="Output contains error", metadata={"error": "error"}
            )
        return RuleResult(passed=True)


class MockCritique(Critique):
    """Mock critique for testing."""

    def __init__(self) -> None:
        """Initialize the mock critique."""
        super().__init__(name="mock_critique", description="A mock critique for testing")

    def critique(self, prompt: str) -> Dict[str, Any]:
        """Critique the prompt."""
        return {"output": "fixed"}


@pytest.fixture
def mock_graph() -> Graph:
    """Create a mock graph."""
    graph = MagicMock(spec=Graph)
    graph.compile.return_value = {"output": "Hello, world!"}
    graph.validate.return_value = True
    return graph


@pytest.fixture
def mock_state_graph() -> StateGraph:
    """Create a mock state graph."""
    graph = MagicMock(spec=StateGraph)
    graph.compile.return_value = {"output": "Hello, world!"}
    graph.validate.return_value = True
    return graph


@pytest.fixture
def mock_node() -> ToolNode:
    """Create a mock tool node."""
    node = MagicMock(spec=ToolNode)
    node.invoke.return_value = "Hello, world!"
    return node


@pytest.fixture
def mock_channel() -> AnyValue:
    """Create a mock channel."""
    channel = MagicMock(spec=AnyValue)
    channel.get.return_value = "Hello, world!"
    return channel


@pytest.fixture
def mock_tracer() -> Tracer:
    """Create a mock tracer."""
    return MagicMock(spec=Tracer)


def test_sifaka_graph_init(mock_graph: Graph) -> None:
    """Test SifakaGraph initialization."""
    graph = SifakaGraph(graph=mock_graph)
    assert graph.graph == mock_graph
    assert not graph.rules
    assert graph.critique
    assert graph.tracer is None
    assert graph.critic is None


def test_sifaka_graph_with_rules(mock_graph: Graph) -> None:
    """Test SifakaGraph with rules."""
    rules = [MockRule()]
    graph = SifakaGraph(graph=mock_graph, rules=rules)
    assert graph.rules == rules


def test_sifaka_graph_with_critic(mock_graph: Graph) -> None:
    """Test SifakaGraph with critic."""
    critic = MockCritique()
    graph = SifakaGraph(graph=mock_graph, critic=critic)
    assert graph.critic == critic


def test_sifaka_graph_with_tracer(mock_graph: Graph, mock_tracer: Tracer) -> None:
    """Test SifakaGraph with tracer."""
    graph = SifakaGraph(graph=mock_graph, tracer=mock_tracer)
    assert graph.tracer == mock_tracer


def test_sifaka_graph_run(mock_graph: Graph) -> None:
    """Test SifakaGraph run method."""
    graph = SifakaGraph(graph=mock_graph)
    output = graph.run({"input": "Hello"})
    assert output == {"output": "Hello, world!"}
    mock_graph.compile.assert_called_once_with({"input": "Hello"})
    mock_graph.validate.assert_called_once()


def test_sifaka_graph_run_with_validation(mock_graph: Graph) -> None:
    """Test SifakaGraph run method with validation."""
    rules = [MockRule()]
    graph = SifakaGraph(graph=mock_graph, rules=rules)
    output = graph.run({"input": "Hello"})
    assert output == {"output": "Hello, world!"}
    mock_graph.compile.assert_called_once_with({"input": "Hello"})
    mock_graph.validate.assert_called_once()


def test_sifaka_graph_run_with_critique(mock_graph: Graph) -> None:
    """Test SifakaGraph run method with critique."""
    rules = [MockRule()]
    critic = MockCritique()
    graph = SifakaGraph(graph=mock_graph, rules=rules, critic=critic)
    mock_graph.compile.return_value = {"output": "error"}
    output = graph.run({"input": "Hello"})
    assert output == {"output": "fixed"}
    mock_graph.compile.assert_called_once_with({"input": "Hello"})
    mock_graph.validate.assert_called_once()


def test_sifaka_state_graph_init(mock_state_graph: StateGraph) -> None:
    """Test SifakaStateGraph initialization."""
    graph = SifakaStateGraph(graph=mock_state_graph)
    assert graph.graph == mock_state_graph
    assert not graph.rules
    assert graph.critique
    assert graph.tracer is None
    assert graph.critic is None


def test_sifaka_state_graph_run(mock_state_graph: StateGraph) -> None:
    """Test SifakaStateGraph run method."""
    graph = SifakaStateGraph(graph=mock_state_graph)
    output = graph.run({"input": "Hello"})
    assert output == {"output": "Hello, world!"}
    mock_state_graph.compile.assert_called_once_with({"input": "Hello"})
    mock_state_graph.validate.assert_called_once()


def test_sifaka_tool_node_init(mock_node: ToolNode) -> None:
    """Test SifakaToolNode initialization."""
    node = SifakaToolNode(node=mock_node)
    assert node.node == mock_node
    assert not node.rules
    assert node.critique
    assert node.tracer is None
    assert node.critic is None


def test_sifaka_tool_node_invoke(mock_node: ToolNode) -> None:
    """Test SifakaToolNode invoke method."""
    node = SifakaToolNode(node=mock_node)
    output = node.invoke({"tool": "example", "input": "Hello"})
    assert output == "Hello, world!"
    mock_node.invoke.assert_called_once_with({"tool": "example", "input": "Hello"})


def test_sifaka_channel_init(mock_channel: AnyValue) -> None:
    """Test SifakaChannel initialization."""
    channel = SifakaChannel(channel=mock_channel)
    assert channel.channel == mock_channel
    assert not channel.rules
    assert channel.critique
    assert channel.tracer is None
    assert channel.critic is None


def test_sifaka_channel_send(mock_channel: AnyValue) -> None:
    """Test SifakaChannel send method."""
    channel = SifakaChannel(channel=mock_channel)
    channel.send("Hello, world!")
    mock_channel.update.assert_called_once_with("Hello, world!")


def test_sifaka_channel_receive(mock_channel: AnyValue) -> None:
    """Test SifakaChannel receive method."""
    channel = SifakaChannel(channel=mock_channel)
    message = channel.receive()
    assert message == "Hello, world!"
    mock_channel.get.assert_called_once()


def test_wrap_graph(mock_graph: Graph) -> None:
    """Test wrap_graph function."""
    graph = wrap_graph(mock_graph)
    assert isinstance(graph, SifakaGraph)
    assert graph.graph == mock_graph


def test_wrap_state_graph(mock_state_graph: StateGraph) -> None:
    """Test wrap_state_graph function."""
    graph = wrap_state_graph(mock_state_graph)
    assert isinstance(graph, SifakaStateGraph)
    assert graph.graph == mock_state_graph


def test_wrap_tool_node(mock_node: ToolNode) -> None:
    """Test wrap_tool_node function."""
    node = wrap_tool_node(mock_node)
    assert isinstance(node, SifakaToolNode)
    assert node.node == mock_node


def test_wrap_channel(mock_channel: AnyValue) -> None:
    """Test wrap_channel function."""
    channel = wrap_channel(mock_channel)
    assert isinstance(channel, SifakaChannel)
    assert channel.channel == mock_channel
