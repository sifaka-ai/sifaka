"""
Tests for the Chain class.
"""

import pytest
from unittest.mock import Mock

from sifaka.chain import Chain
from sifaka.results import ValidationResult, ImprovementResult, Result
from sifaka.errors import ChainError


class TestChain:
    """Tests for the Chain class."""
    
    def test_chain_initialization(self):
        """Test that a Chain can be initialized."""
        chain = Chain()
        assert chain._model is None
        assert chain._prompt is None
        assert chain._validators == []
        assert chain._improvers == []
        assert chain._options == {}
    
    def test_with_model_string(self):
        """Test setting a model with a string."""
        chain = Chain()
        # This will raise NotImplementedError for now, which is expected
        with pytest.raises(NotImplementedError):
            chain.with_model("openai:gpt-4")
    
    def test_with_model_instance(self):
        """Test setting a model with an instance."""
        chain = Chain()
        mock_model = Mock()
        mock_model.generate.return_value = "Generated text"
        mock_model.count_tokens.return_value = 10
        
        chain.with_model(mock_model)
        assert chain._model is mock_model
    
    def test_with_prompt(self):
        """Test setting a prompt."""
        chain = Chain()
        prompt = "Write a short story about a robot."
        chain.with_prompt(prompt)
        assert chain._prompt == prompt
    
    def test_validate_with(self):
        """Test adding a validator."""
        chain = Chain()
        mock_validator = Mock()
        chain.validate_with(mock_validator)
        assert chain._validators == [mock_validator]
    
    def test_improve_with(self):
        """Test adding an improver."""
        chain = Chain()
        mock_improver = Mock()
        chain.improve_with(mock_improver)
        assert chain._improvers == [mock_improver]
    
    def test_with_options(self):
        """Test setting options."""
        chain = Chain()
        options = {"temperature": 0.7, "max_tokens": 100}
        chain.with_options(**options)
        assert chain._options == options
    
    def test_run_without_model(self):
        """Test running without a model."""
        chain = Chain()
        chain.with_prompt("Write a short story about a robot.")
        with pytest.raises(ChainError, match="Model not specified"):
            chain.run()
    
    def test_run_without_prompt(self):
        """Test running without a prompt."""
        chain = Chain()
        mock_model = Mock()
        chain.with_model(mock_model)
        with pytest.raises(ChainError, match="Prompt not specified"):
            chain.run()
    
    def test_run_basic(self):
        """Test running a basic chain."""
        chain = Chain()
        mock_model = Mock()
        mock_model.generate.return_value = "Generated text"
        
        result = (chain
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .run())
        
        assert isinstance(result, Result)
        assert result.text == "Generated text"
        assert result.passed is True
        assert result.validation_results == []
        assert result.improvement_results == []
        
        mock_model.generate.assert_called_once_with(
            "Write a short story about a robot."
        )
    
    def test_run_with_validation(self):
        """Test running a chain with validation."""
        chain = Chain()
        mock_model = Mock()
        mock_model.generate.return_value = "Generated text"
        
        mock_validator = Mock()
        mock_validator.validate.return_value = ValidationResult(
            passed=True,
            message="Validation passed"
        )
        
        result = (chain
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
            .run())
        
        assert result.passed is True
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True
        
        mock_validator.validate.assert_called_once_with("Generated text")
    
    def test_run_with_failed_validation(self):
        """Test running a chain with failed validation."""
        chain = Chain()
        mock_model = Mock()
        mock_model.generate.return_value = "Generated text"
        
        mock_validator = Mock()
        mock_validator.validate.return_value = ValidationResult(
            passed=False,
            message="Validation failed"
        )
        
        result = (chain
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .validate_with(mock_validator)
            .run())
        
        assert result.passed is False
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is False
        assert result.improvement_results == []
    
    def test_run_with_improvement(self):
        """Test running a chain with improvement."""
        chain = Chain()
        mock_model = Mock()
        mock_model.generate.return_value = "Generated text"
        
        mock_improver = Mock()
        mock_improver.improve.return_value = (
            "Improved text",
            ImprovementResult(
                original_text="Generated text",
                improved_text="Improved text",
                changes_made=True,
                message="Improvement applied"
            )
        )
        
        result = (chain
            .with_model(mock_model)
            .with_prompt("Write a short story about a robot.")
            .improve_with(mock_improver)
            .run())
        
        assert result.passed is True
        assert result.text == "Improved text"
        assert len(result.improvement_results) == 1
        assert result.improvement_results[0].changes_made is True
        
        mock_improver.improve.assert_called_once_with("Generated text")
