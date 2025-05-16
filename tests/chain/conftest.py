"""
Pytest fixtures for chain tests.
"""

import pytest
from typing import List, Optional

from sifaka.chain import Chain
from sifaka.interfaces import ValidatorProtocol as Validator, ImproverProtocol as Improver

from tests.utils.mock_provider import MockProvider


@pytest.fixture
def mock_provider():
    """Fixture for a mock provider."""
    provider = MockProvider(model_name="test-model", default_response="Generated text")
    return provider


@pytest.fixture
def mock_validator(request):
    """Fixture for a mock validator."""
    should_pass = getattr(request, "param", True)

    class MockValidator(Validator):
        """Mock validator for testing."""

        def __init__(self, should_pass: bool = should_pass, name: str = "mock_validator"):
            """Initialize the mock validator."""
            self.should_pass = should_pass
            self.name = name
            self.calls = []

        def validate(self, text: str) -> dict:
            """Validate the text."""
            self.calls.append(text)
            return {
                "passed": self.should_pass,
                "message": "Validation passed" if self.should_pass else "Validation failed",
                "metadata": {"validator": self.name},
            }

        def reset_calls(self) -> None:
            """Reset the list of calls."""
            self.calls = []

    return MockValidator()


@pytest.fixture
def mock_improver(request):
    """Fixture for a mock improver."""
    improved_text = getattr(request, "param", "Improved text")

    class MockImprover(Improver):
        """Mock improver for testing."""

        def __init__(self, improved_text: str = improved_text, name: str = "mock_improver"):
            """Initialize the mock improver."""
            self.improved_text = improved_text
            self.name = name
            self.calls = []

        def improve(self, text: str, issues: Optional[List[dict]] = None) -> str:
            """Improve the text."""
            self.calls.append({"text": text, "issues": issues})
            return self.improved_text

        def reset_calls(self) -> None:
            """Reset the list of calls."""
            self.calls = []

    return MockImprover()


@pytest.fixture
def simple_chain(mock_provider, mock_validator):
    """Fixture for a simple chain."""
    chain = Chain(
        model=mock_provider,
        validators=[mock_validator],
        improver=None,
        max_attempts=3,
    )
    return chain


@pytest.fixture
def chain_with_improver(mock_provider, mock_validator, mock_improver):
    """Fixture for a chain with an improver."""
    chain = Chain(
        model=mock_provider,
        validators=[mock_validator],
        improver=mock_improver,
        max_attempts=3,
    )
    return chain
