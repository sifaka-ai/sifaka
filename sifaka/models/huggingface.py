"""HuggingFace model implementation for Sifaka.

This module provides dual-mode HuggingFace model support:
1. Inference API mode: Cloud-based inference using HuggingFace's hosted models
2. Local mode: Local model loading with optimization features

The implementation includes:
- Dual mode architecture (API + local)
- Model loading optimization with LRU caching
- Quantization support for resource efficiency
- Device auto-detection (CPU/GPU/MPS)
- Memory management and model eviction
- Lazy loading with intelligent cache management

Example:
    ```python
    from sifaka.models.huggingface import HuggingFaceModel, create_huggingface_model

    # Create a model using Inference API (cloud)
    api_model = HuggingFaceModel(
        model_name="microsoft/DialoGPT-medium",
        use_inference_api=True,
        api_token="your_hf_token"
    )

    # Create a model for local inference
    local_model = HuggingFaceModel(
        model_name="microsoft/DialoGPT-small",
        use_inference_api=False,
        device="auto"
    )

    # Generate text
    response = api_model.generate("Hello, how are you?")
    print(response)
    ```
"""

import time
from typing import Any, Dict, Optional, Tuple

# Try to import HuggingFace dependencies
try:
    import torch
    from huggingface_hub import InferenceClient
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    InferenceClient = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoConfig = None  # type: ignore
    pipeline = None
    BitsAndBytesConfig = None  # type: ignore
    torch = None  # type: ignore

from sifaka.core.thought import Thought
from sifaka.models.shared import BaseModelImplementation
from sifaka.utils.error_handling import ModelError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceModelLoader:
    """Model loader with caching and optimization for local HuggingFace models.

    This class manages the loading, caching, and optimization of HuggingFace models
    for local inference. It includes LRU caching, quantization support, and
    intelligent memory management.
    """

    def __init__(self, max_cached_models: int = 3):
        """Initialize the model loader.

        Args:
            max_cached_models: Maximum number of models to keep in cache
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "HuggingFace dependencies not available. Please install with: "
                "pip install 'transformers>=4.30.0' 'torch>=2.0.0' 'accelerate>=0.20.0'"
            )

        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._max_cached_models = max_cached_models

    def _generate_cache_key(
        self, model_name: str, device: str, quantization: Optional[str], **kwargs: Any
    ) -> str:
        """Generate a stable cache key for model caching.

        Args:
            model_name: Name of the model
            device: Device to load on
            quantization: Quantization type
            **kwargs: Additional model loading options

        Returns:
            A stable cache key string
        """
        import hashlib
        import json

        # Filter kwargs to only include serializable values
        serializable_kwargs = {
            k: v
            for k, v in sorted(kwargs.items())
            if isinstance(v, (str, int, float, bool, type(None)))
        }

        key_data = {
            "model_name": model_name,
            "device": device,
            "quantization": quantization,
            "kwargs": serializable_kwargs,
        }

        # Create stable JSON representation
        key_str = json.dumps(key_data, sort_keys=True)

        # Use MD5 hash for shorter cache keys
        return hashlib.md5(key_str.encode()).hexdigest()

    def _detect_device(self, device: str = "auto") -> str:
        """Detect the best available device for model inference.

        Args:
            device: Device preference ("auto", "cpu", "cuda", "mps")

        Returns:
            The device string to use
        """
        if device != "auto":
            return device

        # Auto-detect best device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _create_quantization_config(self, quantization: Optional[str] = None) -> Optional[Any]:
        """Create quantization configuration for memory efficiency.

        Args:
            quantization: Quantization type ("4bit", "8bit", None)

        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise
        """
        if not quantization:
            return None

        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            logger.warning(f"Unknown quantization type: {quantization}")
            return None

    def _evict_oldest_model(self) -> None:
        """Evict the oldest model from cache to free memory."""
        if not self._model_cache:
            return

        # Find the model with the oldest last_used timestamp
        oldest_key = min(self._model_cache.keys(), key=lambda k: self._model_cache[k]["last_used"])

        logger.info(f"Evicting model from cache: {oldest_key}")
        del self._model_cache[oldest_key]

        # Force garbage collection to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _detect_model_type(self, model_name: str) -> str:
        """Detect whether a model is causal LM or seq2seq.

        Args:
            model_name: Name of the HuggingFace model

        Returns:
            "causal" for causal language models, "seq2seq" for sequence-to-sequence models
        """
        try:
            # Load model config to determine architecture
            config = AutoConfig.from_pretrained(model_name)

            # Check model architecture type
            model_type = config.model_type.lower()

            # Seq2Seq models
            seq2seq_types = {
                "t5",
                "bart",
                "pegasus",
                "mbart",
                "blenderbot",
                "blenderbot_small",
                "marian",
                "prophetnet",
                "bigbird_pegasus",
                "led",
                "longt5",
                "mt5",
                "switch_transformers",
                "ul2",
                "flan-t5",
            }

            # Causal LM models
            causal_types = {
                "gpt2",
                "gpt_neo",
                "gpt_neox",
                "gptj",
                "opt",
                "bloom",
                "llama",
                "mistral",
                "mixtral",
                "phi",
                "gemma",
                "qwen",
                "falcon",
                "mpt",
                "codegen",
                "starcoder",
                "santacoder",
                "persimmon",
                "stablelm",
            }

            if model_type in seq2seq_types:
                return "seq2seq"
            elif model_type in causal_types:
                return "causal"
            else:
                # Default heuristic: if it has decoder_start_token_id, it's likely seq2seq
                if hasattr(config, "decoder_start_token_id"):
                    return "seq2seq"
                else:
                    return "causal"

        except Exception as e:
            logger.warning(f"Could not detect model type for {model_name}: {e}")
            # Default to causal for backwards compatibility
            return "causal"

    def load_model(
        self,
        model_name: str,
        device: str = "auto",
        quantization: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Load a model with caching and optimization.

        Args:
            model_name: Name of the HuggingFace model
            device: Device to load the model on
            quantization: Quantization type for memory efficiency
            **kwargs: Additional model loading options

        Returns:
            Tuple of (model, tokenizer)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(model_name, device, quantization, **kwargs)

        # Check if model is already cached
        if cache_key in self._model_cache:
            self._model_cache[cache_key]["last_used"] = time.time()
            cached = self._model_cache[cache_key]
            logger.debug(f"Using cached model: {model_name}")
            return cached["model"], cached["tokenizer"]

        # Evict oldest model if cache is full
        if len(self._model_cache) >= self._max_cached_models:
            self._evict_oldest_model()

        # Load the model
        logger.info(f"Loading HuggingFace model: {model_name}")

        try:
            # Detect device
            actual_device = self._detect_device(device)
            logger.debug(f"Using device: {actual_device}")

            # Detect model type
            model_type = self._detect_model_type(model_name)
            logger.debug(f"Detected model type: {model_type}")

            # Create quantization config
            quantization_config = self._create_quantization_config(quantization)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with optimizations
            # Filter out generation-specific parameters that shouldn't go to model constructor
            generation_params = {
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
            }
            model_kwargs = {
                "torch_dtype": torch.float16 if actual_device != "cpu" else torch.float32,
                "device_map": actual_device if actual_device != "cpu" else None,
                **{k: v for k, v in kwargs.items() if k not in generation_params},
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            # Load the appropriate model class based on type
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Move to device if not using device_map
            if actual_device == "cpu":
                model = model.to(actual_device)

            # Cache the loaded model
            self._model_cache[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "last_used": time.time(),
                "device": actual_device,
            }

            logger.info(f"Successfully loaded model: {model_name} on {actual_device}")
            return model, tokenizer

        except Exception as e:
            raise ModelError(f"Failed to load HuggingFace model '{model_name}': {e}")

    def clear_cache(self) -> None:
        """Clear all cached models to free memory."""
        logger.info("Clearing HuggingFace model cache")
        self._model_cache.clear()

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()


class HuggingFaceModel(BaseModelImplementation):
    """HuggingFace model implementation with dual mode support.

    This class provides both cloud-based inference (via Inference API) and
    local model loading with optimization features. It automatically handles
    device detection, quantization, and memory management.

    Features:
    - Dual mode: Inference API (cloud) or local inference
    - Model loading optimization with LRU caching
    - Quantization support (4-bit, 8-bit)
    - Device auto-detection (CPU/GPU/MPS)
    - Memory management and model eviction
    - Lazy loading with intelligent cache management

    Example:
        ```python
        # Cloud inference
        api_model = HuggingFaceModel(
            model_name="microsoft/DialoGPT-medium",
            use_inference_api=True
        )

        # Local inference with optimizations
        local_model = HuggingFaceModel(
            model_name="microsoft/DialoGPT-small",
            use_inference_api=False,
            device="auto",
            quantization="4bit"
        )
        ```
    """

    def __init__(
        self,
        model_name: str,
        use_inference_api: bool = True,
        api_token: Optional[str] = None,
        device: str = "cpu",
        quantization: Optional[str] = None,
        **options: Any,
    ):
        """Initialize the HuggingFace model.

        Args:
            model_name: Name of the HuggingFace model
            use_inference_api: Whether to use Inference API (cloud) or local inference
            api_token: HuggingFace API token (for Inference API)
            device: Device for local inference ("auto", "cpu", "cuda", "mps")
            quantization: Quantization type for local inference ("4bit", "8bit", None)
            retriever: Optional retriever for direct access
            **options: Additional options for the model
        """
        # Initialize base class with HuggingFace-specific configuration
        # For HuggingFace, API key is only required for Inference API mode
        super().__init__(
            model_name=model_name,
            api_key=api_token,
            provider_name="HuggingFace",
            env_var_name="HUGGINGFACE_API_TOKEN",
            required_packages=(
                ["transformers", "torch", "accelerate"] if not HUGGINGFACE_AVAILABLE else None
            ),
            api_key_required=use_inference_api,  # Only require API key for Inference API mode
            **options,
        )

        # Store HuggingFace-specific configuration
        self.use_inference_api = use_inference_api
        self.device = device
        self.quantization = quantization

        # Initialize based on mode
        if use_inference_api:
            # Use new Inference Providers API with correct provider
            self.client = InferenceClient(provider="hf-inference", api_key=self.api_key)
            self.model = None
            self.tokenizer = None
            self.loader = None
        else:
            self.client = None  # type: ignore
            self.loader = HuggingFaceModelLoader()
            self.model = None
            self.tokenizer = None
            # Models will be loaded lazily on first use

        logger.debug(f"Created HuggingFace model: {model_name} (API: {use_inference_api})")

    @staticmethod
    def _import_transformers():
        """Import transformers library for testing compatibility.

        This method exists for backward compatibility with tests that mock
        the transformers import. In the actual implementation, transformers
        is imported at the module level.

        Returns:
            Mock transformers module for testing
        """
        # This is a compatibility method for tests
        # In actual usage, transformers is imported at module level
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers not available")

        # Return a mock-like object that has the expected attributes
        class MockTransformers:
            AutoTokenizer = AutoTokenizer
            AutoModelForCausalLM = AutoModelForCausalLM
            AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
            AutoConfig = AutoConfig
            BitsAndBytesConfig = BitsAndBytesConfig

        return MockTransformers()

    def _generate_cache_key(
        self, model_name: str, device: str, quantization: Optional[str], **kwargs: Any
    ) -> str:
        """Generate a cache key for model caching (for test compatibility).

        Args:
            model_name: Name of the model
            device: Device to load on
            quantization: Quantization type
            **kwargs: Additional model loading options

        Returns:
            A stable cache key string
        """
        if self.loader:
            return self.loader._generate_cache_key(model_name, device, quantization, **kwargs)
        else:
            # Fallback for API mode
            import hashlib
            import json

            key_data = {
                "model_name": model_name,
                "device": device,
                "quantization": quantization,
                "kwargs": {
                    k: v
                    for k, v in sorted(kwargs.items())
                    if isinstance(v, (str, int, float, bool, type(None)))
                },
            }
            key_str = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()

    def _ensure_local_model_loaded(self) -> None:
        """Ensure the local model is loaded for inference."""
        if self.use_inference_api:
            return  # Not needed for API mode

        if self.model is None or self.tokenizer is None:
            if self.loader is None:
                raise ModelError("Model loader not initialized for local inference")
            self.model, self.tokenizer = self.loader.load_model(
                self.model_name, device=self.device, quantization=self.quantization, **self.options
            )
            # Store model type for generation
            self.model_type = self.loader._detect_model_type(self.model_name)

    def _generate_via_api(self, prompt: str, **options: Any) -> str:
        """Generate text using the Inference Providers API.

        Args:
            prompt: The prompt to generate from
            **options: Generation options

        Returns:
            Generated text
        """
        try:
            # Use new Inference Providers API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=options.get("max_tokens", 100),
                temperature=options.get("temperature", 0.7),
                top_p=options.get("top_p", 0.95),  # Must be < 1.0
            )

            return str(completion.choices[0].message.content).strip()

        except Exception as e:
            raise ModelError(f"HuggingFace Inference API generation failed: {e}")

    def _generate_local(self, prompt: str, **options: Any) -> str:
        """Generate text using local model inference.

        Args:
            prompt: The prompt to generate from
            **options: Generation options

        Returns:
            Generated text
        """
        try:
            # Ensure model is loaded
            self._ensure_local_model_loaded()

            # Handle different model types
            if hasattr(self, "model_type") and self.model_type == "seq2seq":
                return self._generate_seq2seq(prompt, **options)
            else:
                return self._generate_causal(prompt, **options)

        except Exception as e:
            raise ModelError(f"Local HuggingFace generation failed: {e}")

    def _generate_causal(self, prompt: str, **options: Any) -> str:
        """Generate text using causal language model."""
        if self.tokenizer is None or self.model is None:
            raise ModelError("Model and tokenizer must be loaded for local inference")

        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": options.get("max_tokens", 100),
            "temperature": options.get("temperature", 0.7),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add other supported parameters
        if "top_p" in options:
            gen_params["top_p"] = options["top_p"]
        if "top_k" in options:
            gen_params["top_k"] = options["top_k"]
        if "repetition_penalty" in options:
            gen_params["repetition_penalty"] = options["repetition_penalty"]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(inputs, **gen_params)

        # Decode only the new tokens
        new_tokens = outputs[0][len(inputs[0]) :]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return str(generated_text).strip()

    def _generate_seq2seq(self, prompt: str, **options: Any) -> str:
        """Generate text using sequence-to-sequence model."""
        if self.tokenizer is None or self.model is None:
            raise ModelError("Model and tokenizer must be loaded for local inference")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Prepare generation parameters for seq2seq
        gen_params = {
            "max_length": options.get("max_tokens", 100),
            "temperature": options.get("temperature", 0.7),
            "do_sample": True,
        }

        # Add other supported parameters
        if "top_p" in options:
            gen_params["top_p"] = options["top_p"]
        if "top_k" in options:
            gen_params["top_k"] = options["top_k"]
        if "repetition_penalty" in options:
            gen_params["repetition_penalty"] = options["repetition_penalty"]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return str(generated_text).strip()

    def _generate_impl(self, prompt: str, **options: Any) -> str:
        """Generate text using HuggingFace models (API or local).

        This is the internal implementation called by the base class generate method.
        It handles both Inference API and local inference modes.

        Args:
            prompt: The prompt to generate text from
            **options: Additional options for generation

        Returns:
            The generated text
        """
        try:
            if self.use_inference_api:
                return self._generate_via_api(prompt, **options)
            else:
                return self._generate_local(prompt, **options)

        except Exception as e:
            # Use base class error handling
            self._handle_api_error(e, "generation")
            raise  # Re-raise after handling

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in

        Returns:
            The number of tokens
        """
        if not text:
            return 0

        try:
            if self.use_inference_api:
                # For API mode, use approximate counting
                return max(1, len(text) // 4)
            else:
                # For local mode, use actual tokenizer
                self._ensure_local_model_loaded()
                if self.tokenizer is None:
                    raise ModelError("Tokenizer not loaded for token counting")
                tokens = self.tokenizer.encode(text)
                return len(tokens)

        except Exception as e:
            logger.warning(f"Token counting failed, using approximation: {e}")
            return max(1, len(text) // 4)

    # Async methods required by Model protocol
    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt asynchronously.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.
        """
        # For now, just call the sync method
        # In a real implementation, you would use async HTTP calls for API mode
        return self.generate(prompt, **options)

    async def _generate_with_thought_async(
        self, thought: Thought, **options: Any
    ) -> tuple[str, str]:
        """Generate text using a Thought container asynchronously.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        # For now, just call the sync method
        # In a real implementation, you would use async HTTP calls for API mode
        return self.generate_with_thought(thought, **options)


# Custom factory function for HuggingFace models to handle dual-mode parameters
def create_huggingface_model(
    model_name: str, use_inference_api: bool = True, **kwargs: Any
) -> HuggingFaceModel:
    """Create a HuggingFace model instance.

    Args:
        model_name: Name of the HuggingFace model
        use_inference_api: Whether to use Inference API or local inference
        **kwargs: Additional arguments for the model

    Returns:
        A HuggingFaceModel instance

    Example:
        ```python
        # Create API model
        api_model = create_huggingface_model(
            "microsoft/DialoGPT-medium",
            use_inference_api=True,
            api_token="your_token"
        )

        # Create local model
        local_model = create_huggingface_model(
            "microsoft/DialoGPT-small",
            use_inference_api=False,
            device="auto",
            quantization="4bit"
        )
        ```
    """
    logger.debug(f"Creating HuggingFace model: {model_name} (API: {use_inference_api})")

    try:
        model = HuggingFaceModel(
            model_name=model_name, use_inference_api=use_inference_api, **kwargs
        )

        logger.debug(f"Successfully created HuggingFace model: {model_name}")
        return model

    except Exception as e:
        logger.error(f"Failed to create HuggingFace model '{model_name}': {e}")
        raise
