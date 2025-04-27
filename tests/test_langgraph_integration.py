"""Tests for langgraph integration."""

from typing import Dict, Any, Optional
from pydantic import Field

import pytest
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from sifaka.rules.base import Rule, RuleResult
from sifaka.rules.domain import MedicalRule
from sifaka.rules.legal import LegalCitationRule
from sifaka.rules.format import FormatRule
from sifaka.rules.safety import ToxicityRule
from sifaka.rules.content import ToneConsistencyRule
from sifaka.rules.wrapper import wrap_rule, wrap_graph


class MockRule(Rule):
    """Mock rule for testing."""

    validate_called: bool = Field(default=False, description="Whether validate was called")
    validate_args: Optional[Dict[str, Any]] = Field(
        default=None, description="Arguments passed to validate"
    )

    def __init__(
        self,
        name: str = "mock_rule",
        description: str = "Mock rule for testing",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize mock rule."""
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            **kwargs,
        )

    def _validate_impl(self, text: str, **kwargs) -> RuleResult:
        """Mock validation implementation."""
        self.validate_called = True
        self.validate_args = {"text": text, **kwargs}
        return RuleResult(
            passed=True,
            score=1.0,
            message="Mock validation passed",
            feedback="Mock validation passed",
            metadata={"mock": True},
        )


@pytest.fixture
def mock_rule():
    """Create a mock rule instance."""
    return MockRule()


def test_wrap_rule(mock_rule):
    """Test wrapping a rule."""
    wrapped = wrap_rule(mock_rule)

    # Test with string input
    result = wrapped("test text")
    assert mock_rule.validate_called
    assert mock_rule.validate_args["text"] == "test text"
    assert result.passed
    assert result.score == 1.0
    assert "mock" in result.metadata

    # Test with message input
    message = AIMessage(content="test message")
    result = wrapped(message)
    assert mock_rule.validate_args["text"] == "test message"
    assert result.passed


def test_wrap_graph():
    """Test wrapping a graph with rules."""
    rules = [
        MockRule(name="medical"),
        MockRule(name="legal"),
        MockRule(name="format"),
        MockRule(name="toxicity"),
        MockRule(name="tone"),
    ]

    graph = wrap_graph(rules)

    # Test with string input
    state = {"input": "test text"}
    result = graph.invoke(state)
    assert isinstance(result, dict)
    assert "results" in result
    assert len(result["results"]) == len(rules)

    # Test with message input
    state = {"input": BaseMessage(content="test text")}
    result = graph.invoke(state)
    assert isinstance(result, dict)
    assert "results" in result
    assert len(result["results"]) == len(rules)
