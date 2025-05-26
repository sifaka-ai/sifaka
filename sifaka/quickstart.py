"""QuickStart utilities for Sifaka.

This module provides simplified configuration and setup utilities for common
use cases, reducing the boilerplate code needed to get started with Sifaka.
"""

import os
from typing import Any, Dict, List, Optional, Union

from sifaka.core.chain import Chain
from sifaka.core.interfaces import Critic, Model, Validator
from sifaka.storage.protocol import Storage
from sifaka.models.base import create_model
from sifaka.utils.logging import get_logger
from sifaka.utils.error_handling import ConfigurationError

logger = get_logger(__name__)


class QuickStart:
    """Simplified configuration utilities for common Sifaka use cases.

    This class provides one-liner setups for common scenarios, making it easy
    to get started with Sifaka without writing boilerplate configuration code.

    Features:
    - Basic chain creation with sensible defaults
    - Storage-specific configurations (memory, file, Redis, Milvus)
    - Multi-tier storage combinations
    - Validator and critic integration
    - Use case-specific presets (development, production, research)
    - Configuration validation with helpful error messages

    Examples:
        ```python
        from sifaka.quickstart import QuickStart

        # Basic chain with just a model
        chain = QuickStart.basic_chain("openai:gpt-4", "Write a story")
        result = chain.run()

        # Chain with Redis storage
        chain = QuickStart.with_redis("openai:gpt-4", redis_url="redis://localhost:6379")

        # Full stack with multiple storage backends
        chain = QuickStart.full_stack("openai:gpt-4", storage="redis+milvus")

        # Use case-specific configurations
        chain = QuickStart.for_development()  # Fast setup for testing
        chain = QuickStart.for_production("openai:gpt-4", "Your prompt")  # Production-ready
        chain = QuickStart.for_research("anthropic:claude-3", "Research question")  # Research setup

        # Preset-based configuration
        chain = QuickStart.from_preset("content_generation", "openai:gpt-4", "Write content")

        # Add validators and critics easily
        chain = QuickStart.with_validation("openai:gpt-4", ["length", "toxicity"])
        chain = QuickStart.with_critics("openai:gpt-4", ["reflexion", "constitutional"])
        ```
    """

    @staticmethod
    def basic_chain(
        model_spec: str, prompt: str, max_iterations: int = 3, **model_options: Any
    ) -> Chain:
        """Create a basic chain with just a model and prompt.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            prompt: The prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain ready to run.
        """
        logger.debug(f"Creating basic chain with model '{model_spec}'")

        model = create_model(model_spec, **model_options)

        chain = Chain(model=model, prompt=prompt, max_improvement_iterations=max_iterations)

        logger.debug("Basic chain created successfully")
        return chain

    @staticmethod
    def with_memory_storage(
        model_spec: str, prompt: Optional[str] = None, max_iterations: int = 3, **model_options: Any
    ) -> Chain:
        """Create a chain with memory storage (default behavior).

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with memory storage.
        """
        logger.debug(f"Creating chain with memory storage and model '{model_spec}'")

        from sifaka.storage.memory import MemoryStorage

        model = create_model(model_spec, **model_options)
        storage = MemoryStorage()

        chain = Chain(
            model=model, prompt=prompt, storage=storage, max_improvement_iterations=max_iterations
        )

        logger.debug("Chain with memory storage created successfully")
        return chain

    @staticmethod
    def with_file_storage(
        model_spec: str,
        file_path: str,
        prompt: Optional[str] = None,
        max_iterations: int = 3,
        **model_options: Any,
    ) -> Chain:
        """Create a chain with file-based persistence.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            file_path: Path to the JSON file for persistence.
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with file storage.
        """
        logger.debug(f"Creating chain with file storage '{file_path}' and model '{model_spec}'")

        from sifaka.storage.file import FileStorage

        model = create_model(model_spec, **model_options)
        storage = FileStorage(file_path)

        chain = Chain(
            model=model, prompt=prompt, storage=storage, max_improvement_iterations=max_iterations
        )

        logger.debug("Chain with file storage created successfully")
        return chain

    @staticmethod
    def with_redis(
        model_spec: str,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "sifaka",
        prompt: Optional[str] = None,
        max_iterations: int = 3,
        **model_options: Any,
    ) -> Chain:
        """Create a chain with Redis storage.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            redis_url: Redis connection URL.
            key_prefix: Prefix for Redis keys.
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with Redis storage.
        """
        logger.debug(f"Creating chain with Redis storage and model '{model_spec}'")

        from sifaka.mcp import MCPServerConfig, MCPTransportType
        from sifaka.storage.redis import RedisStorage

        model = create_model(model_spec, **model_options)

        # Create Redis MCP configuration
        redis_config = MCPServerConfig(
            name="redis-server",
            transport_type=MCPTransportType.STDIO,
            url="python -m mcp_redis",
        )

        storage = RedisStorage(mcp_config=redis_config, key_prefix=key_prefix)

        chain = Chain(
            model=model, prompt=prompt, storage=storage, max_improvement_iterations=max_iterations
        )

        logger.debug("Chain with Redis storage created successfully")
        return chain

    @staticmethod
    def with_milvus(
        model_spec: str,
        collection_name: str = "sifaka_storage",
        prompt: Optional[str] = None,
        max_iterations: int = 3,
        **model_options: Any,
    ) -> Chain:
        """Create a chain with Milvus vector storage.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            collection_name: Name of the Milvus collection.
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with Milvus storage.
        """
        logger.debug(f"Creating chain with Milvus storage and model '{model_spec}'")

        from sifaka.mcp import MCPServerConfig, MCPTransportType
        from sifaka.storage.milvus import MilvusStorage

        model = create_model(model_spec, **model_options)

        # Create Milvus MCP configuration
        milvus_config = MCPServerConfig(
            name="milvus-server",
            transport_type=MCPTransportType.STDIO,
            url="cd /Users/evanvolgas/Documents/not_beam/sifaka/mcp && python -m main.py",
        )

        storage = MilvusStorage(mcp_config=milvus_config, collection_name=collection_name)

        chain = Chain(
            model=model, prompt=prompt, storage=storage, max_improvement_iterations=max_iterations
        )

        logger.debug("Chain with Milvus storage created successfully")
        return chain

    @staticmethod
    def full_stack(
        model_spec: str,
        storage: str = "memory+redis",
        prompt: Optional[str] = None,
        max_iterations: int = 3,
        **model_options: Any,
    ) -> Chain:
        """Create a chain with multiple storage backends.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            storage: Storage configuration string (e.g., "memory+redis", "redis+milvus").
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with multiple storage backends.
        """
        logger.debug(f"Creating full stack chain with storage '{storage}' and model '{model_spec}'")

        from sifaka.storage.cached import CachedStorage

        model = create_model(model_spec, **model_options)

        # Parse storage configuration
        storage_parts = storage.lower().split("+")

        if len(storage_parts) == 1:
            # Single storage backend
            if storage_parts[0] == "memory":
                storage_instance = QuickStart.with_memory_storage(
                    model_spec, **model_options
                )._config.storage
            elif storage_parts[0] == "redis":
                storage_instance = QuickStart.with_redis(
                    model_spec, **model_options
                )._config.storage
            elif storage_parts[0] == "milvus":
                storage_instance = QuickStart.with_milvus(
                    model_spec, **model_options
                )._config.storage
            else:
                raise ValueError(f"Unknown storage backend: {storage_parts[0]}")

        elif len(storage_parts) == 2:
            # Two-tier storage (cache + persistence)
            cache_type, persist_type = storage_parts

            # Create cache layer
            if cache_type == "memory":
                from sifaka.storage.memory import MemoryStorage

                cache = MemoryStorage()
            else:
                raise ValueError(f"Unsupported cache type: {cache_type}")

            # Create persistence layer
            persistence_layer: Any
            if persist_type == "redis":
                from sifaka.mcp import MCPServerConfig, MCPTransportType
                from sifaka.storage.redis import RedisStorage

                redis_config = MCPServerConfig(
                    name="redis-server",
                    transport_type=MCPTransportType.STDIO,
                    url="python -m mcp_redis",
                )
                persistence_layer = RedisStorage(mcp_config=redis_config)

            elif persist_type == "milvus":
                from sifaka.mcp import MCPServerConfig, MCPTransportType
                from sifaka.storage.milvus import MilvusStorage

                milvus_config = MCPServerConfig(
                    name="milvus-server",
                    transport_type=MCPTransportType.STDIO,
                    url="cd /Users/evanvolgas/Documents/not_beam/sifaka/mcp && python -m main.py",
                )
                persistence_layer = MilvusStorage(mcp_config=milvus_config)

            else:
                raise ValueError(f"Unsupported persistence type: {persist_type}")

            combined_storage: Any = CachedStorage(cache=cache, persistence=persistence_layer)

        else:
            raise ValueError(f"Too many storage backends specified: {storage}")

        chain = Chain(
            model=model,
            prompt=prompt,
            storage=combined_storage if len(storage_parts) == 2 else storage_instance,
            max_improvement_iterations=max_iterations,
        )

        logger.debug("Full stack chain created successfully")
        return chain

    @staticmethod
    def with_validation(
        model_spec: str,
        validators: List[Union[str, Validator]],
        prompt: Optional[str] = None,
        max_iterations: int = 3,
        **model_options: Any,
    ) -> Chain:
        """Create a chain with common validators.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            validators: List of validator names or instances.
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with validators.
        """
        logger.debug(f"Creating chain with validators and model '{model_spec}'")

        chain = QuickStart.basic_chain(model_spec, prompt or "", max_iterations, **model_options)

        # Add validators
        for validator in validators:
            if isinstance(validator, str):
                # Create validator from string specification
                if validator == "length":
                    from sifaka.validators.base import LengthValidator

                    chain.validate_with(LengthValidator(min_length=50, max_length=1000))
                elif validator == "toxicity":
                    from sifaka.classifiers.toxicity import ToxicityClassifier
                    from sifaka.validators.classifier import create_classifier_validator

                    classifier = ToxicityClassifier()
                    chain.validate_with(create_classifier_validator(classifier, threshold=0.8))
                else:
                    logger.warning(f"Unknown validator type: {validator}")
            else:
                # Use validator instance directly
                chain.validate_with(validator)

        logger.debug("Chain with validators created successfully")
        return chain

    @staticmethod
    def with_critics(
        model_spec: str,
        critics: List[Union[str, Critic]],
        prompt: Optional[str] = None,
        max_iterations: int = 3,
        **model_options: Any,
    ) -> Chain:
        """Create a chain with common critics.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            critics: List of critic names or instances.
            prompt: Optional prompt to use for generation.
            max_iterations: Maximum improvement iterations.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain with critics.
        """
        logger.debug(f"Creating chain with critics and model '{model_spec}'")

        chain = QuickStart.basic_chain(model_spec, prompt or "", max_iterations, **model_options)

        # Add critics
        for critic in critics:
            if isinstance(critic, str):
                # Create critic from string specification
                if critic == "reflexion":
                    from sifaka.critics.reflexion import ReflexionCritic

                    model = create_model(model_spec, **model_options)
                    chain.improve_with(ReflexionCritic(model=model))
                elif critic == "constitutional":
                    from sifaka.critics.constitutional import ConstitutionalCritic

                    model = create_model(model_spec, **model_options)
                    chain.improve_with(ConstitutionalCritic(model=model))
                elif critic == "self_rag":
                    from sifaka.critics.self_rag import SelfRAGCritic

                    model = create_model(model_spec, **model_options)
                    chain.improve_with(SelfRAGCritic(model=model))
                else:
                    logger.warning(f"Unknown critic type: {critic}")
            else:
                # Use critic instance directly
                chain.improve_with(critic)

        logger.debug("Chain with critics created successfully")
        return chain

    @staticmethod
    def for_development(
        model_spec: str = "mock:test-model",
        prompt: Optional[str] = None,
        **model_options: Any,
    ) -> Chain:
        """Create a chain optimized for development and testing.

        Uses memory storage and minimal iterations for fast feedback.

        Args:
            model_spec: Model specification (defaults to mock model).
            prompt: Optional prompt to use for generation.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain for development.
        """
        logger.debug("Creating development chain")

        return QuickStart.basic_chain(
            model_spec=model_spec,
            prompt=prompt or "Test prompt for development",
            max_iterations=1,
            **model_options,
        )

    @staticmethod
    def for_production(
        model_spec: str,
        prompt: str,
        storage: str = "memory+redis",
        validators: Optional[List[str]] = None,
        critics: Optional[List[str]] = None,
        **model_options: Any,
    ) -> Chain:
        """Create a chain optimized for production use.

        Uses persistent storage, validation, and improvement by default.

        Args:
            model_spec: Model specification (e.g., "openai:gpt-4").
            prompt: The prompt to use for generation.
            storage: Storage configuration (default: "memory+redis").
            validators: Optional list of validator names.
            critics: Optional list of critic names.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain for production.
        """
        logger.debug("Creating production chain")

        # Validate required parameters
        if not prompt:
            raise ConfigurationError(
                "Prompt is required for production chains",
                suggestions=[
                    "Provide a specific prompt for your use case",
                    "Use QuickStart.for_development() for testing without a prompt",
                ],
            )

        # Create base chain with storage
        chain = QuickStart.full_stack(
            model_spec=model_spec, storage=storage, prompt=prompt, max_iterations=3, **model_options
        )

        # Add default validators if none specified
        if validators is None:
            validators = ["length"]

        # Add validators
        for validator in validators:
            if validator == "length":
                from sifaka.validators.base import LengthValidator

                chain.validate_with(LengthValidator(min_length=50, max_length=2000))
            elif validator == "toxicity":
                from sifaka.classifiers.toxicity import ToxicityClassifier
                from sifaka.validators.classifier import create_classifier_validator

                classifier = ToxicityClassifier()
                chain.validate_with(create_classifier_validator(classifier, threshold=0.8))

        # Add default critics if none specified
        if critics is None:
            critics = ["reflexion"]

        # Add critics
        for critic in critics:
            if critic == "reflexion":
                from sifaka.critics.reflexion import ReflexionCritic

                model = create_model(model_spec, **model_options)
                chain.improve_with(ReflexionCritic(model=model))

        logger.debug("Production chain created successfully")
        return chain

    @staticmethod
    def for_research(
        model_spec: str,
        prompt: str,
        storage: str = "memory+redis+milvus",
        retrievers: bool = True,
        **model_options: Any,
    ) -> Chain:
        """Create a chain optimized for research use.

        Uses full storage stack and retrievers for comprehensive analysis.

        Args:
            model_spec: Model specification (e.g., "anthropic:claude-3-sonnet").
            prompt: The research prompt to use for generation.
            storage: Storage configuration (default: full 3-tier).
            retrievers: Whether to include retrievers for context.
            **model_options: Additional options for the model.

        Returns:
            A configured Chain for research.
        """
        logger.debug("Creating research chain")

        # Create base chain with full storage
        chain = QuickStart.full_stack(
            model_spec=model_spec,
            storage=storage,
            prompt=prompt,
            max_iterations=5,  # More iterations for thorough research
            **model_options,
        )

        # Add retrievers if requested
        if retrievers:
            try:
                from sifaka.retrievers.simple import InMemoryRetriever

                model_retriever = InMemoryRetriever()
                critic_retriever = InMemoryRetriever()

                # Add some research context
                model_retriever.add_document("research_context", "Academic research context")
                critic_retriever.add_document("evaluation_criteria", "Research evaluation criteria")

                chain._config.model_retrievers.append(model_retriever)
                chain._config.critic_retrievers.append(critic_retriever)
            except ImportError:
                logger.warning("Retrievers not available, skipping retriever setup")

        # Add research-focused validators
        from sifaka.validators.base import LengthValidator

        chain.validate_with(LengthValidator(min_length=200, max_length=5000))

        # Add research-focused critics
        from sifaka.critics.reflexion import ReflexionCritic

        model = create_model(model_spec, **model_options)
        chain.improve_with(ReflexionCritic(model=model, reflection_depth=3))

        logger.debug("Research chain created successfully")
        return chain


class ConfigPresets:
    """Predefined configuration presets for common Sifaka use cases.

    This class provides ready-to-use configurations that combine models,
    storage, validators, and critics for specific scenarios.

    Examples:
        ```python
        from sifaka.quickstart import ConfigPresets

        # Get a preset configuration
        config = ConfigPresets.content_generation()
        chain = Chain(**config)

        # Or use with QuickStart
        chain = QuickStart.from_preset("content_generation", model_spec="openai:gpt-4")
        ```
    """

    @staticmethod
    def development() -> Dict[str, Any]:
        """Configuration preset for development and testing.

        Returns:
            Configuration dictionary for development use.
        """
        return {
            "max_improvement_iterations": 1,
            "apply_improvers_on_validation_failure": False,
            "always_apply_critics": False,
            "storage_type": "memory",
            "validators": [],
            "critics": [],
        }

    @staticmethod
    def content_generation() -> Dict[str, Any]:
        """Configuration preset for content generation tasks.

        Returns:
            Configuration dictionary for content generation.
        """
        return {
            "max_improvement_iterations": 2,
            "apply_improvers_on_validation_failure": True,
            "always_apply_critics": True,
            "storage_type": "memory+redis",
            "validators": ["length", "toxicity"],
            "critics": ["constitutional"],
        }

    @staticmethod
    def fact_checking() -> Dict[str, Any]:
        """Configuration preset for fact-checking and verification tasks.

        Returns:
            Configuration dictionary for fact-checking.
        """
        return {
            "max_improvement_iterations": 3,
            "apply_improvers_on_validation_failure": True,
            "always_apply_critics": True,
            "storage_type": "memory+redis+milvus",
            "validators": ["length"],
            "critics": ["self_rag", "reflexion"],
            "retrievers": True,
        }

    @staticmethod
    def research_analysis() -> Dict[str, Any]:
        """Configuration preset for research and analysis tasks.

        Returns:
            Configuration dictionary for research.
        """
        return {
            "max_improvement_iterations": 5,
            "apply_improvers_on_validation_failure": True,
            "always_apply_critics": True,
            "storage_type": "memory+redis+milvus",
            "validators": ["length"],
            "critics": ["reflexion"],
            "retrievers": True,
            "length_min": 200,
            "length_max": 5000,
        }

    @staticmethod
    def production_safe() -> Dict[str, Any]:
        """Configuration preset for production environments with safety checks.

        Returns:
            Configuration dictionary for safe production use.
        """
        return {
            "max_improvement_iterations": 3,
            "apply_improvers_on_validation_failure": True,
            "always_apply_critics": True,
            "storage_type": "memory+redis",
            "validators": ["length", "toxicity"],
            "critics": ["constitutional", "reflexion"],
            "length_min": 50,
            "length_max": 2000,
        }


class ConfigWizard:
    """Interactive configuration wizard for complex Sifaka setups.

    This class provides guided setup for users who need help configuring
    complex chains with multiple components.

    Examples:
        ```python
        from sifaka.quickstart import ConfigWizard

        # Interactive setup
        wizard = ConfigWizard()
        chain = wizard.setup_interactive()

        # Guided setup for specific use case
        chain = wizard.setup_for_use_case("content_generation", model_spec="openai:gpt-4")
        ```
    """

    def __init__(self) -> None:
        """Initialize the configuration wizard."""
        self.logger = get_logger(self.__class__.__name__)

    def validate_environment(self, model_spec: str) -> Dict[str, bool]:
        """Validate that the environment is properly configured.

        Args:
            model_spec: Model specification to validate.

        Returns:
            Dictionary of validation results.
        """
        results = {}

        # Check model provider API keys
        provider = model_spec.split(":")[0] if ":" in model_spec else "unknown"

        if provider == "openai":
            results["openai_api_key"] = bool(os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            results["anthropic_api_key"] = bool(os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "huggingface":
            results["huggingface_token"] = bool(os.getenv("HUGGINGFACE_API_TOKEN"))

        # Check for Redis availability (basic check)
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            results["redis_available"] = True
        except Exception:
            results["redis_available"] = False

        # Check for required packages
        try:
            import sifaka.critics.reflexion

            results["critics_available"] = True
        except ImportError:
            results["critics_available"] = False

        try:
            import sifaka.validators.base

            results["validators_available"] = True
        except ImportError:
            results["validators_available"] = False

        return results

    def get_recommendations(self, use_case: str, model_spec: str) -> Dict[str, Any]:
        """Get configuration recommendations for a specific use case.

        Args:
            use_case: The intended use case.
            model_spec: Model specification.

        Returns:
            Recommended configuration.
        """
        # Validate environment first
        env_status = self.validate_environment(model_spec)

        # Get base preset
        if use_case == "development":
            config = ConfigPresets.development()
        elif use_case == "content_generation":
            config = ConfigPresets.content_generation()
        elif use_case == "fact_checking":
            config = ConfigPresets.fact_checking()
        elif use_case == "research":
            config = ConfigPresets.research_analysis()
        elif use_case == "production":
            config = ConfigPresets.production_safe()
        else:
            config = ConfigPresets.development()

        # Adjust based on environment
        if not env_status.get("redis_available", False):
            # Fallback to memory storage if Redis not available
            if "redis" in config.get("storage_type", ""):
                config["storage_type"] = "memory"
                config["warnings"] = config.get("warnings", [])
                config["warnings"].append("Redis not available, using memory storage")

        if not env_status.get("critics_available", False):
            config["critics"] = []
            config["warnings"] = config.get("warnings", [])
            config["warnings"].append("Critics not available, disabled")

        config["environment_status"] = env_status
        return config

    def setup_for_use_case(
        self, use_case: str, model_spec: str, prompt: Optional[str] = None
    ) -> Chain:
        """Set up a chain for a specific use case with validation.

        Args:
            use_case: The intended use case.
            model_spec: Model specification.
            prompt: Optional prompt.

        Returns:
            Configured Chain.
        """
        self.logger.info(f"Setting up chain for use case: {use_case}")

        # Get recommendations
        config = self.get_recommendations(use_case, model_spec)

        # Log any warnings
        for warning in config.get("warnings", []):
            self.logger.warning(warning)

        # Create chain based on use case
        if use_case == "development":
            return QuickStart.for_development(model_spec, prompt)
        elif use_case == "production":
            if not prompt:
                raise ConfigurationError(
                    "Prompt is required for production use case",
                    suggestions=["Provide a specific prompt for your production use case"],
                )
            return QuickStart.for_production(
                model_spec,
                prompt,
                storage=config["storage_type"],
                validators=config["validators"],
                critics=config["critics"],
            )
        elif use_case == "research":
            if not prompt:
                raise ConfigurationError(
                    "Prompt is required for research use case",
                    suggestions=["Provide a research question or topic"],
                )
            return QuickStart.for_research(
                model_spec,
                prompt,
                storage=config["storage_type"],
                retrievers=config.get("retrievers", False),
            )
        else:
            # Default to basic chain
            return QuickStart.basic_chain(model_spec, prompt or "Default prompt")


# Add convenience method to QuickStart for using presets
def _add_preset_method() -> None:
    """Add from_preset method to QuickStart class."""

    def from_preset(
        preset_name: str, model_spec: str, prompt: Optional[str] = None, **overrides: Any
    ) -> Chain:
        """Create a chain from a configuration preset.

        Args:
            preset_name: Name of the preset to use.
            model_spec: Model specification.
            prompt: Optional prompt.
            **overrides: Configuration overrides.

        Returns:
            Configured Chain.
        """
        wizard = ConfigWizard()
        return wizard.setup_for_use_case(preset_name, model_spec, prompt)

    # Add the method to QuickStart
    QuickStart.from_preset = staticmethod(from_preset)  # type: ignore[attr-defined]


# Apply the method addition
_add_preset_method()
