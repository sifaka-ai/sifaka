#!/usr/bin/env python3
"""Comprehensive tests for Sifaka HuggingFace model integration.

This test suite covers HuggingFace model functionality including model loading,
text generation, tokenization, caching, and error handling.
"""

from unittest.mock import Mock, patch

import pytest

from sifaka.models.huggingface import HuggingFaceModel
from sifaka.utils.error_handling import ModelError


class TestHuggingFaceModel:
    """Test HuggingFace model functionality."""

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_basic(self, mock_import):
        """Test basic HuggingFace model functionality."""
        # Mock transformers components
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "Generated text response"

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        response = model.generate("Test prompt")

        assert isinstance(response, str)
        assert len(response) > 0
        assert response == "Generated text response"

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_with_options(self, mock_import):
        """Test HuggingFace model with generation options."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated with options"

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        response = model.generate(
            "Test prompt", max_length=100, temperature=0.8, top_p=0.9, do_sample=True
        )

        assert isinstance(response, str)
        assert response == "Generated with options"

        # Verify generation was called with options
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_length"] == 100
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["do_sample"] is True

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_token_counting(self, mock_import):
        """Test HuggingFace model token counting."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        count = model.count_tokens("This is a test sentence")

        assert count == 5
        mock_tokenizer.encode.assert_called_with("This is a test sentence")

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_caching(self, mock_import):
        """Test HuggingFace model caching functionality."""
        mock_tokenizer = Mock()
        mock_model = Mock()

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        # Create two instances with same model name
        model1 = HuggingFaceModel(model_name="gpt2")
        model2 = HuggingFaceModel(model_name="gpt2")

        # Should use cached model for second instance
        assert mock_transformers.AutoModelForCausalLM.from_pretrained.call_count <= 2

    def test_huggingface_model_cache_key_generation(self):
        """Test cache key generation for model caching."""
        model = HuggingFaceModel(model_name="gpt2")

        # Test basic cache key
        key1 = model._generate_cache_key("gpt2", "cpu", None)
        key2 = model._generate_cache_key("gpt2", "cpu", None)
        assert key1 == key2

        # Test different parameters produce different keys
        key3 = model._generate_cache_key("gpt2", "cuda", None)
        assert key1 != key3

        # Test with quantization
        key4 = model._generate_cache_key("gpt2", "cpu", "8bit")
        assert key1 != key4

    def test_huggingface_model_cache_key_with_kwargs(self):
        """Test cache key generation with additional kwargs."""
        model = HuggingFaceModel(model_name="gpt2")

        key1 = model._generate_cache_key(
            "gpt2", "cpu", None, torch_dtype="float16", trust_remote_code=True
        )

        key2 = model._generate_cache_key(
            "gpt2", "cpu", None, torch_dtype="float32", trust_remote_code=True
        )

        # Different kwargs should produce different keys
        assert key1 != key2

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_device_handling(self, mock_import):
        """Test HuggingFace model device handling."""
        mock_tokenizer = Mock()
        mock_model = Mock()

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        # Test CPU device
        model_cpu = HuggingFaceModel(model_name="gpt2", device="cpu")
        assert model_cpu.device == "cpu"

        # Test CUDA device
        model_cuda = HuggingFaceModel(model_name="gpt2", device="cuda")
        assert model_cuda.device == "cuda"

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_quantization(self, mock_import):
        """Test HuggingFace model quantization options."""
        mock_tokenizer = Mock()
        mock_model = Mock()

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        # Test 8-bit quantization
        model_8bit = HuggingFaceModel(model_name="gpt2", quantization="8bit")
        assert model_8bit.quantization == "8bit"

        # Test 4-bit quantization
        model_4bit = HuggingFaceModel(model_name="gpt2", quantization="4bit")
        assert model_4bit.quantization == "4bit"

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_error_handling(self, mock_import):
        """Test HuggingFace model error handling."""
        # Test import error
        mock_import.side_effect = ImportError("transformers not installed")

        with pytest.raises(ModelError):
            HuggingFaceModel(model_name="gpt2")

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_loading_error(self, mock_import):
        """Test HuggingFace model loading error handling."""
        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = Exception("Model not found")
        mock_import.return_value = mock_transformers

        with pytest.raises(ModelError):
            HuggingFaceModel(model_name="nonexistent-model")

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_generation_error(self, mock_import):
        """Test HuggingFace model generation error handling."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        mock_model = Mock()
        mock_model.generate.side_effect = Exception("Generation failed")

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")

        with pytest.raises(ModelError):
            model.generate("Test prompt")

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_empty_prompt(self, mock_import):
        """Test HuggingFace model with empty prompt."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = ""

        mock_model = Mock()
        mock_model.generate.return_value = [[]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        response = model.generate("")

        assert response == ""

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_long_prompt(self, mock_import):
        """Test HuggingFace model with very long prompt."""
        mock_tokenizer = Mock()
        # Simulate a very long token sequence
        long_tokens = list(range(2000))  # 2000 tokens
        mock_tokenizer.encode.return_value = long_tokens
        mock_tokenizer.decode.return_value = "Generated response for long prompt"

        mock_model = Mock()
        mock_model.generate.return_value = [long_tokens + [2001, 2002]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        long_prompt = "This is a very long prompt. " * 100
        response = model.generate(long_prompt)

        assert isinstance(response, str)
        assert len(response) > 0

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_special_tokens(self, mock_import):
        """Test HuggingFace model handling of special tokens."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 50256]  # Include EOS token
        mock_tokenizer.decode.return_value = "Response with special tokens"
        mock_tokenizer.eos_token_id = 50256

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 50256]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        response = model.generate("Test prompt")

        assert response == "Response with special tokens"

    def test_huggingface_model_configuration(self):
        """Test HuggingFace model configuration options."""
        # Test default configuration
        model = HuggingFaceModel(model_name="gpt2")
        assert model.model_name == "gpt2"
        assert model.device == "cpu"  # Default device
        assert model.quantization is None  # Default quantization

        # Test custom configuration
        model_custom = HuggingFaceModel(
            model_name="microsoft/DialoGPT-medium",
            device="cuda",
            quantization="8bit",
            torch_dtype="float16",
        )
        assert model_custom.model_name == "microsoft/DialoGPT-medium"
        assert model_custom.device == "cuda"
        assert model_custom.quantization == "8bit"

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_memory_management(self, mock_import):
        """Test HuggingFace model memory management."""
        mock_tokenizer = Mock()
        mock_model = Mock()

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")

        # Test cleanup method if it exists
        if hasattr(model, "cleanup"):
            model.cleanup()

        # Test that model can be garbage collected
        del model

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_batch_processing(self, mock_import):
        """Test HuggingFace model batch processing capabilities."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3] if x else []
        mock_tokenizer.decode.return_value = "Batch response"

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")

        # Test multiple generations
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = []

        for prompt in prompts:
            response = model.generate(prompt)
            responses.append(response)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0


class TestHuggingFaceModelIntegration:
    """Test HuggingFace model integration with other components."""

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_with_chain(self, mock_import):
        """Test HuggingFace model integration with Chain."""
        from sifaka.core.chain import Chain

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Chain integration response"

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        chain = Chain(model=model, prompt="Test chain integration")

        result = chain.run()
        assert result.text == "Chain integration response"

    @patch("sifaka.models.huggingface.HuggingFaceModel._import_transformers")
    def test_huggingface_model_with_validators(self, mock_import):
        """Test HuggingFace model with validators."""
        from sifaka.core.chain import Chain
        from sifaka.validators.base import LengthValidator

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "This is a response that meets length requirements"

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        mock_transformers = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_transformers

        model = HuggingFaceModel(model_name="gpt2")
        validator = LengthValidator(min_length=10, max_length=100)

        chain = Chain(model=model, prompt="Generate text").validate_with(validator)
        result = chain.run()

        assert result.text is not None
        assert len(result.text) >= 10

    def test_huggingface_model_factory_integration(self):
        """Test HuggingFace model creation through factory."""
        from sifaka.models.base import create_model

        with patch("sifaka.models.huggingface.HuggingFaceModel") as mock_hf_model:
            mock_instance = Mock()
            mock_hf_model.return_value = mock_instance

            model = create_model("huggingface:gpt2")

            # Should create HuggingFace model instance
            mock_hf_model.assert_called_once_with(model_name="gpt2")
            assert model == mock_instance
