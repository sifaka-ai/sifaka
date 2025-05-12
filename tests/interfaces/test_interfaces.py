"""
Tests for the interfaces.
"""

import pytest
from typing import Any, Dict, List, Optional

from sifaka.interfaces import (
    Chain,
    Validator,
    Improver,
    Retriever,
    Critic,
    Rule,
)


class TestChainImplementation(Chain):
    """Test implementation of the Chain interface."""

    def __init__(self, name_value: str = "test_chain"):
        self._name = name_value
        self.calls = []
        self._state_manager_obj = {}

    @property
    def name(self) -> str:
        """Get the chain name."""
        return self._name

    @property
    def _state_manager(self) -> Any:
        """Get the state manager."""
        return self._state_manager_obj

    @property
    def description(self) -> str:
        """Get the chain description."""
        return "Test chain implementation"

    @property
    def config(self) -> Any:
        """Get the chain configuration."""
        return {}

    def update_config(self, config: Any) -> None:
        """Update the chain configuration."""
        pass

    def initialize(self) -> None:
        """Initialize the chain."""
        pass

    def cleanup(self) -> None:
        """Clean up the chain."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the current state."""
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state."""
        pass

    def reset_state(self) -> None:
        """Reset the state to its initial values."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {"calls": len(self.calls)}

    def run(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Run the chain."""
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return {
            "output": f"Output for {prompt}",
            "all_passed": True,
            "validation_results": [],
        }


class TestValidatorImplementation(Validator):
    """Test implementation of the Validator interface."""

    def __init__(self, name_value: str = "test_validator"):
        self._name = name_value
        self.calls = []
        self._description = "Test validator implementation"

    @property
    def name(self) -> str:
        """Get the validator name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the validator description."""
        return self._description

    def validate(self, text: str) -> Dict[str, Any]:
        """Validate the text."""
        self.calls.append(text)
        return {
            "passed": True,
            "message": "Validation passed",
            "metadata": {"validator": self.name},
        }


class TestImproverImplementation(Improver):
    """Test implementation of the Improver interface."""

    def __init__(self, name_value: str = "test_improver"):
        self._name = name_value
        self.calls = []
        self._description = "Test improver implementation"

    @property
    def name(self) -> str:
        """Get the improver name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the improver description."""
        return self._description

    def improve(self, text: str, validation_results: List[Dict[str, Any]] = None) -> str:
        """Improve the text."""
        self.calls.append({"text": text, "issues": validation_results})
        return f"Improved: {text}"


class TestRetrieverImplementation(Retriever):
    """Test implementation of the Retriever interface."""

    def __init__(self, name_value: str = "test_retriever"):
        self._name = name_value
        self.calls = []
        self._state_manager_obj = {}
        self._description = "Test retriever implementation"
        self._config = {}

    @property
    def name(self) -> str:
        """Get the retriever name."""
        return self._name

    @property
    def _state_manager(self) -> Any:
        """Get the state manager."""
        return self._state_manager_obj

    @property
    def description(self) -> str:
        """Get the retriever description."""
        return self._description

    @property
    def config(self) -> Any:
        """Get the retriever configuration."""
        return self._config

    def update_config(self, config: Any) -> None:
        """Update the retriever configuration."""
        pass

    def initialize(self) -> None:
        """Initialize the retriever."""
        pass

    def cleanup(self) -> None:
        """Clean up the retriever."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the current state."""
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state."""
        pass

    def reset_state(self) -> None:
        """Reset the state to its initial values."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {"calls": len(self.calls)}

    def retrieve(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Retrieve documents."""
        self.calls.append(query)
        return {
            "query": query,
            "results": [
                {"id": "doc1", "content": f"Content for {query}", "score": 0.9},
            ],
        }


class TestCriticImplementation(Critic):
    """Test implementation of the Critic interface."""

    def __init__(self, name_value: str = "test_critic"):
        self._name = name_value
        self.calls = []
        self._state_manager_obj = {}
        self._description = "Test critic implementation"
        self._config = {}

    @property
    def name(self) -> str:
        """Get the critic name."""
        return self._name

    @property
    def _state_manager(self) -> Any:
        """Get the state manager."""
        return self._state_manager_obj

    @property
    def description(self) -> str:
        """Get the critic description."""
        return self._description

    @property
    def config(self) -> Any:
        """Get the critic configuration."""
        return self._config

    def update_config(self, config: Any) -> None:
        """Update the critic configuration."""
        pass

    def reset_state(self) -> None:
        """Reset the state to its initial values."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {"calls": len(self.calls)}

    def process(self, text: str) -> Dict[str, Any]:
        """Process the text."""
        return self.critique(text)

    def critique(self, text: str) -> Dict[str, Any]:
        """Critique the text."""
        self.calls.append(text)
        return {
            "score": 0.8,
            "feedback": f"Feedback for {text}",
        }

    def validate(self, text: str) -> Dict[str, Any]:
        """Validate the text."""
        critique_result = self.critique(text)
        return {
            "passed": critique_result["score"] >= 0.7,
            "message": critique_result["feedback"],
            "metadata": {"critic": self.name},
        }

    def improve(self, text: str, validation_results: List[Dict[str, Any]] = None) -> str:
        """Improve the text."""
        return f"Improved by critic: {text}"


class TestRuleImplementation(Rule):
    """Test implementation of the Rule interface."""

    def __init__(self, name_value: str = "test_rule"):
        self._name = name_value
        self.calls = []
        self._state_manager_obj = {}
        self._description = "Test rule implementation"
        self._config = {}

    @property
    def name(self) -> str:
        """Get the rule name."""
        return self._name

    @property
    def _state_manager(self) -> Any:
        """Get the state manager."""
        return self._state_manager_obj

    @property
    def description(self) -> str:
        """Get the rule description."""
        return self._description

    @property
    def config(self) -> Any:
        """Get the rule configuration."""
        return self._config

    def update_config(self, config: Any) -> None:
        """Update the rule configuration."""
        pass

    def reset_state(self) -> None:
        """Reset the state to its initial values."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {"calls": len(self.calls)}

    def model_validate(self, text: str) -> Dict[str, Any]:
        """Validate the text."""
        self.calls.append(text)
        return {
            "passed": True,
            "message": "Rule passed",
            "metadata": {"rule": self.name},
        }


def test_chain_interface():
    """Test that the Chain interface works correctly."""
    chain = TestChainImplementation()

    result = chain.run("Test prompt")

    assert result is not None
    assert "output" in result
    assert "all_passed" in result
    assert "validation_results" in result
    assert result["output"] == "Output for Test prompt"
    assert result["all_passed"] is True
    assert len(chain.calls) == 1
    assert chain.calls[0]["prompt"] == "Test prompt"


def test_validator_interface():
    """Test that the Validator interface works correctly."""
    validator = TestValidatorImplementation()

    result = validator.validate("Test text")

    assert result is not None
    assert "passed" in result
    assert "message" in result
    assert "metadata" in result
    assert result["passed"] is True
    assert len(validator.calls) == 1
    assert validator.calls[0] == "Test text"


def test_improver_interface():
    """Test that the Improver interface works correctly."""
    improver = TestImproverImplementation()

    result = improver.improve("Test text")

    assert result is not None
    assert result == "Improved: Test text"
    assert len(improver.calls) == 1
    assert improver.calls[0]["text"] == "Test text"


def test_retriever_interface():
    """Test that the Retriever interface works correctly."""
    retriever = TestRetrieverImplementation()

    result = retriever.retrieve("Test query")

    assert result is not None
    assert "query" in result
    assert "results" in result
    assert result["query"] == "Test query"
    assert len(result["results"]) > 0
    assert "content" in result["results"][0]
    assert len(retriever.calls) == 1
    assert retriever.calls[0] == "Test query"


def test_critic_interface():
    """Test that the Critic interface works correctly."""
    critic = TestCriticImplementation()

    result = critic.critique("Test text")

    assert result is not None
    assert "score" in result
    assert "feedback" in result
    assert result["score"] == 0.8
    assert result["feedback"] == "Feedback for Test text"
    assert len(critic.calls) == 1
    assert critic.calls[0] == "Test text"


def test_rule_interface():
    """Test that the Rule interface works correctly."""
    rule = TestRuleImplementation()

    result = rule.model_validate("Test text")

    assert result is not None
    assert "passed" in result
    assert "message" in result
    assert "metadata" in result
    assert result["passed"] is True
    assert result["message"] == "Rule passed"
    assert result["metadata"]["rule"] == "test_rule"
    assert len(rule.calls) == 1
    assert rule.calls[0] == "Test text"
