"""Mock utilities for Sifaka tests."""

from unittest.mock import Mock

from sifaka.core.thought import SifakaThought


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
            "score": 0.8 if self.should_pass else 0.4,
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
            "needs_improvement": self.needs_improvement,
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
        "reflexion": MockCritic("reflexion"),
    }

    # Mock retrievers
    mock_deps.retrievers = {}

    return mock_deps
