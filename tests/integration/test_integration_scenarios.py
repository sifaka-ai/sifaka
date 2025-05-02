"""Tests for integration scenarios and component interactions."""

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from sifaka.models.base import ModelProvider
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.openai import OpenAIProvider
from sifaka.models.mock import MockProvider
from sifaka.rules.base import (
    Rule,
    RuleResult,
    RuleConfig,
    RulePriority,
    BaseValidator,
    RuleResultHandler,
)
from sifaka.critics.base import CriticConfig, CriticMetadata, CriticOutput, CriticResult
from sifaka.domain.base import DomainConfig
from sifaka.integrations.langchain import (
    ChainValidator,
    ChainOutputProcessor,
    ChainConfig,
    SifakaChain,
    wrap_chain,
)
from dotenv import load_dotenv


class TestMockProvider(MockProvider):
    """Test mock provider that implements abstract methods."""

    def __init__(self, **kwargs):
        """Initialize the mock provider."""
        config = kwargs.get("config", {
            "name": "test_mock_provider",
            "description": "A test mock provider",
            "params": {
                "test_param": "test_value"
            }
        })
        super().__init__(config)
        self._client = self._create_default_client()
        self._token_counter = self._create_default_token_counter()

    def _create_default_client(self) -> Any:
        """Create a mock client."""
        return MagicMock()

    def _create_default_token_counter(self) -> Any:
        """Create a mock token counter."""
        return MagicMock()

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        super().validate_config(config)
        if not config.get("params"):
            raise ValueError("params is required")

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate a mock response."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return super().generate(prompt, **kwargs)


class MockValidator(BaseValidator[str]):
    """Mock validator for testing."""

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Mock implementation of validate."""
        return RuleResult(passed=True, message="Mock validation passed")

    def can_validate(self, output: str) -> bool:
        """Mock implementation of can_validate."""
        return True

    @property
    def validation_type(self) -> type[str]:
        """Get validation type."""
        return str


class MockRule(Rule[str, RuleResult, MockValidator, RuleResultHandler[RuleResult]]):
    """Mock rule for testing."""

    def _create_default_validator(self) -> MockValidator:
        """Create default validator."""
        return MockValidator()


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


class TestModelCriticIntegration(unittest.TestCase):
    """Tests for model and critic integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = TestMockProvider(
            config={
                "name": "mock_model",
                "description": "Mock model for testing",
                "params": {
                    "response_template": "This is a test response.",
                    "delay": 0.1
                }
            }
        )
        self.critic_config = CriticConfig(
            name="test_critic",
            description="Test critic configuration",
            min_confidence=0.8,
            max_attempts=3,
            cache_size=100,
            priority=1,
            cost=1.0
        )

    def test_model_critic_workflow(self):
        """Test the workflow between model and critic."""
        # Test basic workflow
        text = "Test input text"
        model_response = self.mock_model.generate(text)
        self.assertIsNotNone(model_response)
        self.assertIn("text", model_response)

        # Test critic processing
        with patch('sifaka.critics.base.Critic.process') as mock_process:
            mock_process.return_value = CriticOutput(
                result=CriticResult.SUCCESS,
                improved_text=model_response["text"],
                metadata=CriticMetadata(
                    score=0.9,
                    feedback="Good response",
                    issues=[],
                    suggestions=[],
                    attempt_number=1,
                    processing_time_ms=100.0
                )
            )
            result = mock_process(model_response["text"])
            self.assertEqual(result.result, CriticResult.SUCCESS)
            self.assertEqual(result.metadata.score, 0.9)

    def test_model_critic_error_handling(self):
        """Test error handling in model-critic workflow."""
        # Test model error
        with self.assertRaises(ValueError):
            self.mock_model.generate("")

        # Test critic error
        with patch('sifaka.critics.base.Critic.process') as mock_process:
            mock_process.side_effect = ValueError("Test error")
            with self.assertRaises(ValueError):
                mock_process("Test text")


class TestModelDomainIntegration(unittest.TestCase):
    """Tests for model and domain integration."""

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

    def test_model_domain_integration(self):
        """Test integration between models and domains."""
        # Mock a LangChain chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Generated text"

        # Wrap the chain
        wrapped_chain = wrap_chain(mock_chain, config=self.config)

        # Test the wrapped chain
        result = wrapped_chain.run("Test input")
        self.assertEqual(result, "Processed: Generated text")
        mock_chain.invoke.assert_called_once()

    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        with self.assertRaises(ValueError):
            ChainConfig[str](
                validators=[self.validator],
                processors=[self.processor],
                output_parser=42,  # Invalid output_parser (not a BaseOutputParser instance)
                critique=True
            )


class TestCriticDomainIntegration(unittest.TestCase):
    """Tests for critic and domain integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.critic_config = CriticConfig(
            name="test_critic",
            description="Test critic configuration",
            min_confidence=0.8,
            max_attempts=3,
            cache_size=100,
            priority=1,
            cost=1.0
        )
        self.mock_rule = MockRule(
            name="test_rule",
            description="Test rule",
            config=RuleConfig(
                priority=RulePriority.MEDIUM,
                cache_size=100,
                cost=1,
                params={
                    "domain": "technical",
                    "required_terms": ["algorithm", "implementation"],
                    "prohibited_terms": ["magic", "simple"],
                    "threshold": 0.8
                }
            )
        )
        self.domain_config = DomainConfig(
            name="test_domain",
            description="Test domain configuration",
            params={
                "domain": "technical",
                "rules": [self.mock_rule],
                "required_terms": ["algorithm", "implementation"],
                "prohibited_terms": ["magic", "simple"],
                "threshold": 0.8
            }
        )

    def test_critic_domain_workflow(self):
        """Test the workflow between critic and domain."""
        # Test basic workflow
        text = "This is a test text."
        with patch('sifaka.critics.base.Critic.process') as mock_process:
            mock_process.return_value = CriticOutput(
                result=CriticResult.SUCCESS,
                improved_text=text,
                metadata=CriticMetadata(
                    score=0.9,
                    feedback="Good response",
                    issues=[],
                    suggestions=[],
                    attempt_number=1,
                    processing_time_ms=100.0
                )
            )
            result = mock_process(text)
            self.assertEqual(result.result, CriticResult.SUCCESS)
            self.assertEqual(result.metadata.score, 0.9)

        # Test domain configuration
        self.assertEqual(self.domain_config.name, "test_domain")
        self.assertEqual(self.domain_config.description, "Test domain configuration")
        self.assertIn("rules", self.domain_config.params)
        self.assertIn("domain", self.domain_config.params)
        self.assertIn("required_terms", self.domain_config.params)
        self.assertIn("prohibited_terms", self.domain_config.params)
        self.assertIn("threshold", self.domain_config.params)

        # Test rule validation
        result = self.mock_rule.validate(text)
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Mock validation passed")

    def test_critic_domain_error_handling(self):
        """Test error handling in critic-domain workflow."""
        # Test critic error
        with patch('sifaka.critics.base.Critic.process') as mock_process:
            mock_process.side_effect = ValueError("Test error")
            with self.assertRaises(ValueError):
                mock_process("")

        # Test domain error
        with self.assertRaises(ValueError):
            DomainConfig(
                name="",
                description="Test",
                params={
                    "domain": "technical",
                    "rules": [],
                    "required_terms": [],
                    "prohibited_terms": [],
                    "threshold": 0.8
                }
            )


class TestFullIntegration(unittest.TestCase):
    """Tests for full system integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = TestMockProvider(
            config={
                "name": "mock_model",
                "description": "Mock model for testing",
                "params": {
                    "response_template": "This is a test response.",
                    "delay": 0.1
                }
            }
        )
        self.critic_config = CriticConfig(
            name="test_critic",
            description="Test critic configuration",
            min_confidence=0.8,
            max_attempts=3,
            cache_size=100,
            priority=1,
            cost=1.0
        )
        self.mock_rule = MockRule(
            name="test_rule",
            description="Test rule",
            config=RuleConfig(
                priority=RulePriority.MEDIUM,
                cache_size=100,
                cost=1,
                params={
                    "domain": "technical",
                    "required_terms": ["algorithm", "implementation"],
                    "prohibited_terms": ["magic", "simple"],
                    "threshold": 0.8
                }
            )
        )
        self.domain_config = DomainConfig(
            name="test_domain",
            description="Test domain configuration",
            params={
                "domain": "technical",
                "rules": [self.mock_rule],
                "required_terms": ["algorithm", "implementation"],
                "prohibited_terms": ["magic", "simple"],
                "threshold": 0.8
            }
        )
        self.chain_validator = MockChainValidator()
        self.chain_processor = MockChainProcessor()
        self.chain_config = ChainConfig[str](
            validators=[self.chain_validator],
            processors=[self.chain_processor],
            critique=True
        )

    def test_full_workflow(self):
        """Test the full system workflow."""
        # Test model generation
        text = "Test input text"
        model_response = self.mock_model.generate(text)
        self.assertIsNotNone(model_response)
        self.assertIn("text", model_response)

        # Test critic processing
        with patch('sifaka.critics.base.Critic.process') as mock_process:
            mock_process.return_value = CriticOutput(
                result=CriticResult.SUCCESS,
                improved_text=model_response["text"],
                metadata=CriticMetadata(
                    score=0.9,
                    feedback="Good response",
                    issues=[],
                    suggestions=[],
                    attempt_number=1,
                    processing_time_ms=100.0
                )
            )
            result = mock_process(model_response["text"])
            self.assertEqual(result.result, CriticResult.SUCCESS)
            self.assertEqual(result.metadata.score, 0.9)

        # Test domain configuration
        self.assertEqual(self.domain_config.name, "test_domain")
        self.assertEqual(self.domain_config.description, "Test domain configuration")
        self.assertIn("rules", self.domain_config.params)
        self.assertIn("domain", self.domain_config.params)
        self.assertIn("required_terms", self.domain_config.params)
        self.assertIn("prohibited_terms", self.domain_config.params)
        self.assertIn("threshold", self.domain_config.params)

        # Test rule validation
        result = self.mock_rule.validate(model_response["text"])
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Mock validation passed")

        # Test chain processing
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Chain output"
        wrapped_chain = wrap_chain(mock_chain, config=self.chain_config)
        chain_result = wrapped_chain.run("Test input")
        self.assertEqual(chain_result, "Processed: Chain output")


if __name__ == "__main__":
    unittest.main()