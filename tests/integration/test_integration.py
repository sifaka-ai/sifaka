"""Tests for integration components."""

import unittest
from unittest.mock import patch, MagicMock
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from sifaka.integrations.langchain import (
    ChainValidator,
    ChainOutputProcessor,
    ChainMemory,
    ChainConfig,
    SifakaChain,
    RuleBasedValidator,
    SifakaMemory,
    wrap_chain,
    wrap_memory,
)
from sifaka.integrations.langgraph import (
    GraphValidator,
    GraphProcessor,
    GraphNode,
    GraphConfig,
    SifakaGraph,
    SifakaNode,
    wrap_graph,
    wrap_node,
)
from sifaka.rules.base import Rule, RuleResult


class MockChainValidator(ChainValidator[str]):
    """Mock chain validator for testing."""

    def validate(self, output: str) -> RuleResult:
        """Mock implementation of validate."""
        return RuleResult(passed=True, message="Mock validation passed")

    def can_validate(self, output: str) -> bool:
        """Mock implementation of can_validate."""
        return True


class MockChainProcessor(ChainOutputProcessor[str]):
    """Mock chain processor for testing."""

    def process(self, output: str) -> str:
        """Mock implementation of process."""
        return f"Processed: {output}"

    def can_process(self, output: str) -> bool:
        """Mock implementation of can_process."""
        return True


class MockGraphValidator(GraphValidator[Dict[str, Any]]):
    """Mock graph validator for testing."""

    def validate(self, state: Dict[str, Any]) -> RuleResult:
        """Mock implementation of validate."""
        return RuleResult(passed=True, message="Mock validation passed")

    def can_validate(self, state: Dict[str, Any]) -> bool:
        """Mock implementation of can_validate."""
        return True


class MockGraphProcessor(GraphProcessor[Dict[str, Any]]):
    """Mock graph processor for testing."""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation of process."""
        return {"processed": True, **state}

    def can_process(self, state: Dict[str, Any]) -> bool:
        """Mock implementation of can_process."""
        return True


class TestLangChainIntegration(unittest.TestCase):
    """Tests for LangChain integration."""

    def setUp(self):
        """Set up test fixtures."""
        load_dotenv()
        self.validator = MockChainValidator()
        self.processor = MockChainProcessor()
        self.config = ChainConfig[str](
            validators=[self.validator],
            processors=[self.processor],
            critique=True
        )

    def test_chain_config_initialization(self):
        """Test chain configuration initialization."""
        self.assertIsNotNone(self.config)
        self.assertIsInstance(self.config, ChainConfig)
        self.assertEqual(len(self.config.validators), 1)
        self.assertEqual(len(self.config.processors), 1)
        self.assertTrue(self.config.critique)

    def test_chain_config_validation(self):
        """Test chain configuration validation."""
        # Test valid config
        config = ChainConfig[str](
            validators=[self.validator],
            processors=[self.processor],
            critique=True
        )
        self.assertIsNotNone(config)

        # Test invalid config - output_parser must be an instance of BaseOutputParser
        with self.assertRaises(ValueError):
            ChainConfig[str](
                validators=[self.validator],
                processors=[self.processor],
                output_parser=42,  # Not a BaseOutputParser instance
                critique=True
            )

    def test_chain_wrapper(self):
        """Test chain wrapper functionality."""
        # Mock a LangChain chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Generated text"

        # Wrap the chain
        wrapped_chain = wrap_chain(mock_chain, config=self.config)

        # Test the wrapped chain
        result = wrapped_chain.run("Test input")
        self.assertEqual(result, "Processed: Generated text")
        mock_chain.invoke.assert_called_once()


class TestLangGraphIntegration(unittest.TestCase):
    """Tests for LangGraph integration."""

    def setUp(self):
        """Set up test fixtures."""
        load_dotenv()
        self.validator = MockGraphValidator()
        self.processor = MockGraphProcessor()
        self.config = GraphConfig[Dict[str, Any]](
            validators=[self.validator],
            processors=[self.processor],
            critique=True
        )

    def test_graph_config_initialization(self):
        """Test graph configuration initialization."""
        self.assertIsNotNone(self.config)
        self.assertIsInstance(self.config, GraphConfig)
        self.assertEqual(len(self.config.validators), 1)
        self.assertEqual(len(self.config.processors), 1)
        self.assertTrue(self.config.critique)

    def test_graph_config_validation(self):
        """Test graph configuration validation."""
        # Test valid config
        config = GraphConfig[Dict[str, Any]](
            validators=[self.validator],
            processors=[self.processor],
            critique=True
        )
        self.assertIsNotNone(config)

        # Test invalid config
        with self.assertRaises(ValueError):
            GraphConfig[Dict[str, Any]](
                validators="not a list",  # Invalid validators
                processors=[self.processor],
                critique=True
            )

    def test_graph_wrapper(self):
        """Test graph wrapper functionality."""
        # Mock a LangGraph graph
        mock_graph = MagicMock()
        mock_graph.run.return_value = {"output": "Generated text"}

        # Wrap the graph
        wrapped_graph = wrap_graph(mock_graph, config=self.config)

        # Test the wrapped graph
        result = wrapped_graph.run({"input": "Test input"})
        self.assertEqual(result, {"processed": True, "output": "Generated text"})
        mock_graph.run.assert_called_once_with({"input": "Test input"})


class TestIntegrationCombination(unittest.TestCase):
    """Tests for combining multiple integrations."""

    def setUp(self):
        """Set up test fixtures."""
        load_dotenv()
        self.chain_validator = MockChainValidator()
        self.chain_processor = MockChainProcessor()
        self.chain_config = ChainConfig[str](
            validators=[self.chain_validator],
            processors=[self.chain_processor],
            critique=True
        )

        self.graph_validator = MockGraphValidator()
        self.graph_processor = MockGraphProcessor()
        self.graph_config = GraphConfig[Dict[str, Any]](
            validators=[self.graph_validator],
            processors=[self.graph_processor],
            critique=True
        )

    def test_integration_combination(self):
        """Test combining multiple integrations."""
        # Mock chains and graphs
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Chain output"
        mock_graph = MagicMock()
        mock_graph.run.return_value = {"output": "Graph output"}

        # Wrap them
        wrapped_chain = wrap_chain(mock_chain, config=self.chain_config)
        wrapped_graph = wrap_graph(mock_graph, config=self.graph_config)

        # Test both integrations
        chain_result = wrapped_chain.run("Test input")
        graph_result = wrapped_graph.run({"input": "Test input"})

        self.assertEqual(chain_result, "Processed: Chain output")
        self.assertEqual(graph_result, {"processed": True, "output": "Graph output"})
        mock_chain.invoke.assert_called_once()
        mock_graph.run.assert_called_once_with({"input": "Test input"})


if __name__ == "__main__":
    unittest.main()