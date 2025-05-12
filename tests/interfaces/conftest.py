"""
Pytest fixtures for interface tests.
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


@pytest.fixture
def test_chain_implementation():
    """Fixture for a test Chain implementation."""
    
    class TestChainImplementation(Chain):
        """Test implementation of the Chain interface."""
        
        def __init__(self, name: str = "test_chain"):
            self.name = name
            self.calls = []
        
        def run(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
            """Run the chain."""
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            return {
                "output": f"Output for {prompt}",
                "all_passed": True,
                "validation_results": [],
            }
    
    return TestChainImplementation()


@pytest.fixture
def test_validator_implementation():
    """Fixture for a test Validator implementation."""
    
    class TestValidatorImplementation(Validator):
        """Test implementation of the Validator interface."""
        
        def __init__(self, name: str = "test_validator"):
            self.name = name
            self.calls = []
        
        def validate(self, text: str) -> Dict[str, Any]:
            """Validate the text."""
            self.calls.append(text)
            return {
                "passed": True,
                "message": "Validation passed",
                "metadata": {"validator": self.name},
            }
    
    return TestValidatorImplementation()


@pytest.fixture
def test_improver_implementation():
    """Fixture for a test Improver implementation."""
    
    class TestImproverImplementation(Improver):
        """Test implementation of the Improver interface."""
        
        def __init__(self, name: str = "test_improver"):
            self.name = name
            self.calls = []
        
        def improve(self, text: str, issues: Optional[List[Dict[str, Any]]] = None) -> str:
            """Improve the text."""
            self.calls.append({"text": text, "issues": issues})
            return f"Improved: {text}"
    
    return TestImproverImplementation()


@pytest.fixture
def test_retriever_implementation():
    """Fixture for a test Retriever implementation."""
    
    class TestRetrieverImplementation(Retriever):
        """Test implementation of the Retriever interface."""
        
        def __init__(self, name: str = "test_retriever"):
            self.name = name
            self.calls = []
        
        def retrieve(self, query: str) -> Dict[str, Any]:
            """Retrieve documents."""
            self.calls.append(query)
            return {
                "query": query,
                "results": [
                    {"id": "doc1", "content": f"Content for {query}", "score": 0.9},
                ],
            }
    
    return TestRetrieverImplementation()


@pytest.fixture
def test_critic_implementation():
    """Fixture for a test Critic implementation."""
    
    class TestCriticImplementation(Critic):
        """Test implementation of the Critic interface."""
        
        def __init__(self, name: str = "test_critic"):
            self.name = name
            self.calls = []
        
        def critique(self, text: str) -> Dict[str, Any]:
            """Critique the text."""
            self.calls.append(text)
            return {
                "score": 0.8,
                "feedback": f"Feedback for {text}",
            }
    
    return TestCriticImplementation()


@pytest.fixture
def test_rule_implementation():
    """Fixture for a test Rule implementation."""
    
    class TestRuleImplementation(Rule):
        """Test implementation of the Rule interface."""
        
        def __init__(self, name: str = "test_rule"):
            self.name = name
            self.calls = []
        
        def validate(self, text: str) -> Dict[str, Any]:
            """Validate the text."""
            self.calls.append(text)
            return {
                "passed": True,
                "message": "Rule passed",
                "metadata": {"rule": self.name},
            }
    
    return TestRuleImplementation()
