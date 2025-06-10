#!/usr/bin/env python3
"""Fix critical test issues identified in the comprehensive test run.

This script addresses the major API mismatches and environment issues:
1. Graph execution result structure (result.output vs result)
2. Environment variable mocking for tests
3. ValidationResult field name mismatches
4. Event loop issues in sync tests
5. Mock configuration issues
"""

import os
import sys
from pathlib import Path


def fix_graph_execution_result():
    """Fix the graph execution result structure issue."""
    print("🔧 Fixing graph execution result structure...")

    # The issue is in engine.py line 151: final_thought = result.output
    # But the graph returns the thought directly, not wrapped in a result object
    engine_file = Path("sifaka/core/engine.py")

    if engine_file.exists():
        content = engine_file.read_text()

        # Fix the result.output issue
        if "final_thought = result.output" in content:
            content = content.replace("final_thought = result.output", "final_thought = result")

        if "return result.output" in content:
            content = content.replace("return result.output", "return result")

        engine_file.write_text(content)
        print("✅ Fixed graph execution result structure")
    else:
        print("⚠️  Engine file not found")


def create_test_env_setup():
    """Create environment setup for tests."""
    print("🔧 Creating test environment setup...")

    env_setup = '''"""Test environment setup for Sifaka tests.

This module provides environment configuration and mocking
for tests that require external API keys or services.
"""

import os
import pytest
from unittest.mock import patch, Mock

# Mock API keys for testing
TEST_API_KEYS = {
    "GEMINI_API_KEY": "test-gemini-key-12345",
    "OPENAI_API_KEY": "test-openai-key-12345",
    "ANTHROPIC_API_KEY": "test-anthropic-key-12345",
}

@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Automatically mock environment variables for all tests."""
    with patch.dict(os.environ, TEST_API_KEYS):
        yield

@pytest.fixture
def mock_pydantic_agent():
    """Mock PydanticAI agent for testing."""
    mock_agent = Mock()
    mock_agent.model = "test:mock-model"

    # Mock run_async method
    async def mock_run_async(*args, **kwargs):
        result = Mock()
        result.data = "Mock generated text"
        result.all_messages = [
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Mock generated text"}
        ]
        result.cost = 0.001
        result.usage = {"tokens": 100}
        return result

    mock_agent.run_async = mock_run_async
    return mock_agent

@pytest.fixture
def mock_dependencies():
    """Create mock SifakaDependencies for testing."""
    from sifaka.graph.dependencies import SifakaDependencies
    from unittest.mock import Mock

    mock_deps = Mock(spec=SifakaDependencies)
    mock_deps.generator_agent = Mock()
    mock_deps.generator_agent.model = "test:mock-model"
    mock_deps.validators = []
    mock_deps.critics = {}
    mock_deps.retrievers = {}

    return mock_deps
'''

    test_env_file = Path("tests/test_env_setup.py")
    test_env_file.write_text(env_setup)
    print("✅ Created test environment setup")


def fix_validation_result_issues():
    """Fix ValidationResult field name mismatches."""
    print("🔧 Fixing ValidationResult field issues...")

    # Files that need ValidationResult fixes
    test_files = [
        "tests/unit_tests/validators/test_validators_comprehensive.py",
        "tests/unit_tests/graph/test_nodes_comprehensive.py",
    ]

    for file_path in test_files:
        test_file = Path(file_path)
        if test_file.exists():
            content = test_file.read_text()

            # Fix field name mismatches
            content = content.replace("result.details", "result.metadata")
            content = content.replace("validation.validator_name", "validation.validator")
            content = content.replace("critique.critic_name", "critique.critic")

            # Fix ValidatorResult references
            content = content.replace("ValidatorResult", "ValidationResult")

            test_file.write_text(content)
            print(f"✅ Fixed ValidationResult issues in {file_path}")


def fix_event_loop_issues():
    """Fix event loop issues in sync tests."""
    print("🔧 Fixing event loop issues...")

    # Add pytest-asyncio configuration
    conftest_content = '''"""Pytest configuration for Sifaka tests."""

import pytest
import asyncio
from unittest.mock import patch
import os

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_api_keys():
    """Mock API keys for all tests."""
    test_keys = {
        "GEMINI_API_KEY": "test-gemini-key",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }
    with patch.dict(os.environ, test_keys):
        yield

@pytest.fixture
def mock_sync_validator():
    """Create a validator that works in sync context."""
    from sifaka.validators.base import BaseValidator, ValidationResult
    from sifaka.core.thought import SifakaThought

    class SyncTestValidator(BaseValidator):
        def __init__(self, name="sync_test"):
            super().__init__(name, "Sync test validator")

        async def validate_async(self, thought: SifakaThought) -> ValidationResult:
            return ValidationResult(
                passed=True,
                message="Sync test passed",
                validator_name=self.name,
                processing_time_ms=1.0
            )

    return SyncTestValidator()
'''

    conftest_file = Path("tests/conftest.py")
    conftest_file.write_text(conftest_content)
    print("✅ Created pytest configuration with event loop fixes")


def fix_mock_configuration_issues():
    """Fix mock configuration issues."""
    print("🔧 Fixing mock configuration issues...")

    # Create a mock utilities module
    mock_utils_content = '''"""Mock utilities for Sifaka tests."""

from unittest.mock import Mock, AsyncMock
from sifaka.core.thought import SifakaThought
from sifaka.validators.base import ValidationResult

class MockGraphRunContext:
    """Mock GraphRunContext for testing nodes."""

    def __init__(self, state: SifakaThought, deps):
        self.state = state
        self.deps = deps

class MockValidator:
    """Mock validator for testing."""

    def __init__(self, name: str = "mock", should_pass: bool = True):
        self.name = name
        self.should_pass = should_pass
        self.call_count = 0

    async def validate_async(self, text: str) -> dict:
        """Mock validation that returns dict format expected by interfaces."""
        self.call_count += 1
        return {
            "passed": self.should_pass,
            "details": {"mock_validation": True},
            "score": 0.8 if self.should_pass else 0.4
        }

class MockCritic:
    """Mock critic for testing."""

    def __init__(self, name: str = "mock", needs_improvement: bool = False):
        self.name = name
        self.needs_improvement = needs_improvement
        self.call_count = 0

    async def critique_async(self, thought: SifakaThought) -> dict:
        """Mock critique that returns dict format."""
        self.call_count += 1
        return {
            "feedback": f"Mock feedback from {self.name}",
            "suggestions": ["Mock suggestion"] if self.needs_improvement else [],
            "needs_improvement": self.needs_improvement
        }

def create_mock_dependencies():
    """Create properly configured mock dependencies."""
    from sifaka.graph.dependencies import SifakaDependencies

    mock_deps = Mock(spec=SifakaDependencies)

    # Mock generator agent
    mock_agent = Mock()
    mock_agent.model = "test:mock-model"

    async def mock_run_async(*args, **kwargs):
        result = Mock()
        result.data = "Mock generated text"
        result.all_messages = []
        return result

    mock_agent.run_async = mock_run_async
    mock_deps.generator_agent = mock_agent

    # Mock validators
    mock_deps.validators = [MockValidator("length"), MockValidator("coherence")]

    # Mock critics
    mock_deps.critics = {
        "constitutional": MockCritic("constitutional"),
        "reflexion": MockCritic("reflexion")
    }

    # Mock retrievers
    mock_deps.retrievers = {}

    return mock_deps
'''

    mock_utils_file = Path("tests/mock_utils.py")
    mock_utils_file.write_text(mock_utils_content)
    print("✅ Created mock utilities")


def main():
    """Run all fixes."""
    print("🔧 Fixing Sifaka Test Issues")
    print("=" * 50)

    try:
        # Change to the correct directory
        os.chdir("/Users/evanvolgas/Documents/not_beam/sifaka")

        fix_graph_execution_result()
        create_test_env_setup()
        fix_validation_result_issues()
        fix_event_loop_issues()
        fix_mock_configuration_issues()

        print("\n🎉 All test fixes applied successfully!")
        print("\nNext steps:")
        print("1. Run tests again to verify fixes")
        print("2. Address any remaining specific test failures")
        print("3. Update test documentation")

        return True

    except Exception as e:
        print(f"\n❌ Error applying fixes: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
