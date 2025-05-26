"""Test fixtures for Sifaka testing framework.

This module provides reusable test fixtures for creating consistent
test data and components across the test suite.
"""

from datetime import datetime
from typing import Any, List, Optional
from uuid import uuid4

import pytest

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator


def create_test_thought(
    prompt: str = "Test prompt",
    text: Optional[str] = "Test generated text",
    iteration: int = 0,
    **kwargs: Any,
) -> Thought:
    """Create a test Thought with sensible defaults."""
    return Thought(
        prompt=prompt,
        text=text,
        iteration=iteration,
        id=str(uuid4()),
        timestamp=datetime.now(),
        chain_id=str(uuid4()),
        **kwargs,
    )


def create_test_chain(
    prompt: str = "Test prompt", model: Optional[Any] = None, **kwargs: Any
) -> Chain:
    """Create a test Chain with sensible defaults."""
    if model is None:
        model = MockModel(model_name="test-model")

    return Chain(model=model, prompt=prompt, **kwargs)


def create_mock_model(
    model_name: str = "test-model", response_text: str = "Mock generated response", **kwargs: Any
) -> MockModel:
    """Create a MockModel with customizable responses."""
    from tests.utils.mocks import MockModelFactory

    return MockModelFactory.create_standard(model_name=model_name, response_text=response_text)


def create_test_validators() -> List[Any]:
    """Create a standard set of test validators."""
    return [
        LengthValidator(min_length=10, max_length=1000),
        RegexValidator(required_patterns=[r"\w+"]),
        ContentValidator(prohibited=["bad", "evil"], name="Safety Filter"),
    ]


def create_test_critics(model: Optional[Any] = None) -> List[Any]:
    """Create a standard set of test critics."""
    if model is None:
        model = MockModel(model_name="critic-model")

    return [
        ReflexionCritic(model=model),
        SelfRefineCritic(model=model),
    ]


def create_test_storage() -> MemoryStorage:
    """Create a test storage instance."""
    return MemoryStorage()


@pytest.fixture
def test_thought():
    """Pytest fixture for a test Thought."""
    return create_test_thought()


@pytest.fixture
def test_chain():
    """Pytest fixture for a test Chain."""
    return create_test_chain()


@pytest.fixture
def mock_model():
    """Pytest fixture for a MockModel."""
    return create_mock_model()


@pytest.fixture
def test_validators():
    """Pytest fixture for test validators."""
    return create_test_validators()


@pytest.fixture
def test_critics():
    """Pytest fixture for test critics."""
    return create_test_critics()


@pytest.fixture
def test_storage():
    """Pytest fixture for test storage."""
    return create_test_storage()


# Performance test fixtures
@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance tests."""
    return {
        "small_prompt": "Write a short sentence.",
        "medium_prompt": "Write a paragraph about artificial intelligence and its applications in modern technology.",
        "large_prompt": "Write a comprehensive essay about the history, current state, and future prospects of artificial intelligence, covering machine learning, deep learning, natural language processing, computer vision, and ethical considerations."
        * 3,
        "expected_max_time": {
            "small": 2.0,  # seconds
            "medium": 5.0,
            "large": 15.0,
        },
        "expected_max_memory": {
            "small": 50,  # MB
            "medium": 100,
            "large": 200,
        },
    }


# Integration test fixtures
@pytest.fixture
def integration_test_config():
    """Fixture providing configuration for integration tests."""
    return {
        "test_prompts": [
            "Write a helpful guide about Python programming.",
            "Explain the concept of machine learning in simple terms.",
            "Create a story about a robot learning to paint.",
        ],
        "model_providers": ["mock"],  # Start with mock, can add real providers
        "storage_backends": ["memory"],  # Start with memory, can add Redis/Milvus
        "validator_combinations": [
            ["length"],
            ["length", "content"],
            ["length", "regex", "content"],
        ],
        "critic_combinations": [
            ["reflexion"],
            ["self_refine"],
            ["reflexion", "self_refine"],
        ],
    }


# Error scenario fixtures
@pytest.fixture
def error_scenarios():
    """Fixture providing error scenarios for testing."""
    return {
        "validation_failures": {
            "too_short": {
                "text": "Hi",
                "validator": LengthValidator(min_length=100, max_length=1000),
            },
            "too_long": {
                "text": "x" * 2000,
                "validator": LengthValidator(min_length=10, max_length=100),
            },
            "prohibited_content": {
                "text": "This contains bad words",
                "validator": ContentValidator(prohibited=["bad"]),
            },
        },
        "model_failures": {
            "empty_response": "",
            "none_response": None,
            "invalid_response": 12345,
        },
    }
