"""
Centralized configuration system for Sifaka.

This module provides a comprehensive configuration system for all Sifaka components,
including models, validators, critics, and retrievers. It uses dataclasses for
type safety and provides utilities for loading and saving configuration.

Example:
    ```python
    from sifaka.config import SifakaConfig, ModelConfig, load_config_from_json

    # Create a custom configuration
    config = SifakaConfig(
        model=ModelConfig(temperature=0.8, max_tokens=1000),
        debug=True
    )

    # Use with a chain
    from sifaka import Chain
    result = (Chain(config)
        .with_model("openai:gpt-4")
        .with_prompt("Write a short story about a robot.")
        .run())

    # Load configuration from a file
    config = load_config_from_json("config.json")
    ```
"""

from typing import Dict, Any, Optional, List, Union, TypeVar
from dataclasses import dataclass, field, asdict
import json
import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Configuration for model components.

    This class defines configuration options for model components, including
    general options like temperature and max_tokens, as well as provider-specific
    options like API keys.

    Attributes:
        temperature: Controls randomness in generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
        frequency_penalty: Penalizes repeated tokens (0.0 to 2.0)
        presence_penalty: Penalizes tokens already present (0.0 to 2.0)
        stop_sequences: List of sequences that stop generation
        api_key: API key for the model provider
        api_base: Base URL for the model provider's API
        organization: Organization ID for the model provider
        custom: Dictionary of additional custom options
    """

    # General model options
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)

    # Provider-specific options
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None

    # Custom options
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatorConfig:
    """Configuration for validator components.

    This class defines configuration options for validator components, including
    options for length validators, content validators, format validators, and
    classifier validators.

    Attributes:
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        min_words: Minimum number of words
        max_words: Maximum number of words
        prohibited_content: List of prohibited content patterns
        required_content: List of required content patterns
        format_type: Type of format to validate (e.g., "json", "yaml")
        format_schema: Schema for format validation
        threshold: Threshold for classifier validators
        guardrails_api_key: API key for GuardrailsAI
        custom: Dictionary of additional custom options
    """

    # Length validator options
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None

    # Content validator options
    prohibited_content: List[str] = field(default_factory=list)
    required_content: List[str] = field(default_factory=list)

    # Format validator options
    format_type: Optional[str] = None
    format_schema: Optional[Dict[str, Any]] = None

    # Classifier validator options
    threshold: float = 0.5

    # GuardrailsAI options
    guardrails_api_key: Optional[str] = None

    # Custom options
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticConfig:
    """Configuration for critic components.

    This class defines configuration options for critic components, including
    options for various critic types like self-refine, n-critics, and
    constitutional critics.

    Attributes:
        temperature: Controls randomness in critic operations (0.0 to 1.0)
        system_prompt: System prompt for the critic
        refinement_rounds: Number of refinement rounds for self-refine critics
        num_critics: Number of critics for n-critics
        principles: List of principles for constitutional critics
        max_passages: Maximum number of passages for retrieval-enhanced critics
        include_passages_in_critique: Whether to include passages in critique
        include_passages_in_improve: Whether to include passages in improve
        custom: Dictionary of additional custom options
    """

    # General critic options
    temperature: float = 0.7
    system_prompt: Optional[str] = None

    # Self-refine options
    refinement_rounds: int = 2

    # N-critics options
    num_critics: int = 3

    # Constitutional critic options
    principles: List[str] = field(default_factory=list)

    # Retrieval options
    max_passages: int = 5
    include_passages_in_critique: bool = True
    include_passages_in_improve: bool = True

    # Custom options
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for retriever components.

    This class defines configuration options for retriever components, including
    options for various retriever types like Elasticsearch and Milvus.

    Attributes:
        top_k: Number of top results to retrieve
        es_host: Elasticsearch host URL
        es_index: Elasticsearch index name
        es_username: Elasticsearch username
        es_password: Elasticsearch password
        milvus_host: Milvus host URL
        milvus_port: Milvus port
        milvus_collection: Milvus collection name
        custom: Dictionary of additional custom options
    """

    # General retriever options
    top_k: int = 3

    # Elasticsearch options
    es_host: Optional[str] = None
    es_index: Optional[str] = None
    es_username: Optional[str] = None
    es_password: Optional[str] = None

    # Milvus options
    milvus_host: Optional[str] = None
    milvus_port: int = 19530
    milvus_collection: Optional[str] = None

    # Custom options
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SifakaConfig:
    """Centralized configuration for all Sifaka components.

    This class provides a centralized configuration for all Sifaka components,
    including models, validators, critics, and retrievers. It also includes
    global options like debug mode and logging level.

    Attributes:
        model: Configuration for model components
        validator: Configuration for validator components
        critic: Configuration for critic components
        retriever: Configuration for retriever components
        debug: Whether to enable debug mode
        log_level: Logging level
        custom: Dictionary of additional custom options
    """

    # Component-specific configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)

    # Global options
    debug: bool = False
    log_level: str = "INFO"

    # Custom options
    custom: Dict[str, Any] = field(default_factory=dict)

    def get_model_options(self) -> Dict[str, Any]:
        """Get all options relevant to models.

        Returns:
            A dictionary of options relevant to models.
        """
        options = {
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "top_p": self.model.top_p,
            "frequency_penalty": self.model.frequency_penalty,
            "presence_penalty": self.model.presence_penalty,
            "stop_sequences": self.model.stop_sequences,
            "api_key": self.model.api_key,
            "api_base": self.model.api_base,
            "organization": self.model.organization,
            "debug": self.debug,
        }
        # Add custom options
        options.update(self.model.custom)
        return {k: v for k, v in options.items() if v is not None}

    def get_validator_options(self) -> Dict[str, Any]:
        """Get all options relevant to validators.

        Returns:
            A dictionary of options relevant to validators.
        """
        options = {
            "min_length": self.validator.min_length,
            "max_length": self.validator.max_length,
            "min_words": self.validator.min_words,
            "max_words": self.validator.max_words,
            "prohibited_content": self.validator.prohibited_content,
            "required_content": self.validator.required_content,
            "format_type": self.validator.format_type,
            "format_schema": self.validator.format_schema,
            "threshold": self.validator.threshold,
            "guardrails_api_key": self.validator.guardrails_api_key,
            "debug": self.debug,
        }
        # Add custom options
        options.update(self.validator.custom)
        return {k: v for k, v in options.items() if v is not None}

    def get_critic_options(self) -> Dict[str, Any]:
        """Get all options relevant to critics.

        Returns:
            A dictionary of options relevant to critics.
        """
        options = {
            "temperature": self.critic.temperature,
            "system_prompt": self.critic.system_prompt,
            "refinement_rounds": self.critic.refinement_rounds,
            "num_critics": self.critic.num_critics,
            "principles": self.critic.principles,
            "max_passages": self.critic.max_passages,
            "include_passages_in_critique": self.critic.include_passages_in_critique,
            "include_passages_in_improve": self.critic.include_passages_in_improve,
            "debug": self.debug,
        }
        # Add custom options
        options.update(self.critic.custom)
        return {k: v for k, v in options.items() if v is not None}

    def get_retriever_options(self) -> Dict[str, Any]:
        """Get all options relevant to retrievers.

        Returns:
            A dictionary of options relevant to retrievers.
        """
        options = {
            "top_k": self.retriever.top_k,
            "es_host": self.retriever.es_host,
            "es_index": self.retriever.es_index,
            "es_username": self.retriever.es_username,
            "es_password": self.retriever.es_password,
            "milvus_host": self.retriever.milvus_host,
            "milvus_port": self.retriever.milvus_port,
            "milvus_collection": self.retriever.milvus_collection,
            "debug": self.debug,
        }
        # Add custom options
        options.update(self.retriever.custom)
        return {k: v for k, v in options.items() if v is not None}


def load_config_from_json(file_path: Union[str, Path]) -> SifakaConfig:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A SifakaConfig object.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        config_dict = json.load(f)
    return _create_config_from_dict(config_dict)


def load_config_from_yaml(file_path: Union[str, Path]) -> SifakaConfig:
    """Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        A SifakaConfig object.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return _create_config_from_dict(config_dict)


def load_config_from_env(prefix: str = "SIFAKA_") -> SifakaConfig:
    """Load configuration from environment variables.

    Environment variables should be named with the prefix followed by the
    section and option, separated by underscores. For example, to set the
    model temperature, use SIFAKA_MODEL_TEMPERATURE=0.8.

    Args:
        prefix: Prefix for environment variables.

    Returns:
        A SifakaConfig object.
    """
    config = SifakaConfig()
    # Example: SIFAKA_MODEL_TEMPERATURE -> config.model.temperature
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix) :].lower().split("_")
            if len(parts) >= 2:
                section, option = parts[0], "_".join(parts[1:])
                _set_config_value(config, section, option, value)
    return config


def save_config_to_json(config: SifakaConfig, file_path: Union[str, Path]) -> None:
    """Save configuration to a JSON file.

    Args:
        config: The configuration to save.
        file_path: Path to the JSON file.

    Raises:
        IOError: If the file cannot be written.
    """
    file_path = Path(file_path)
    config_dict = asdict(config)
    with open(file_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def save_config_to_yaml(config: SifakaConfig, file_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.

    Args:
        config: The configuration to save.
        file_path: Path to the YAML file.

    Raises:
        IOError: If the file cannot be written.
    """
    file_path = Path(file_path)
    config_dict = asdict(config)
    with open(file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def _create_config_from_dict(config_dict: Dict[str, Any]) -> SifakaConfig:
    """Create a SifakaConfig from a dictionary.

    Args:
        config_dict: Dictionary containing configuration values.

    Returns:
        A SifakaConfig object.
    """
    # Extract component configs
    model_config = config_dict.pop("model", {})
    validator_config = config_dict.pop("validator", {})
    critic_config = config_dict.pop("critic", {})
    retriever_config = config_dict.pop("retriever", {})

    # Create component configs
    model = ModelConfig(**model_config) if model_config else ModelConfig()
    validator = ValidatorConfig(**validator_config) if validator_config else ValidatorConfig()
    critic = CriticConfig(**critic_config) if critic_config else CriticConfig()
    retriever = RetrieverConfig(**retriever_config) if retriever_config else RetrieverConfig()

    # Create main config
    return SifakaConfig(
        model=model, validator=validator, critic=critic, retriever=retriever, **config_dict
    )


def _set_config_value(config: SifakaConfig, section: str, option: str, value: str) -> None:
    """Set a configuration value from an environment variable.

    Args:
        config: The configuration object.
        section: The section name (e.g., "model", "validator").
        option: The option name (e.g., "temperature", "max_tokens").
        value: The value as a string.
    """
    if not hasattr(config, section):
        logger.warning(f"Unknown configuration section: {section}")
        return

    section_obj = getattr(config, section)
    if not hasattr(section_obj, option):
        logger.warning(f"Unknown configuration option: {section}.{option}")
        return

    # Convert value to appropriate type
    attr = getattr(section_obj, option)
    converted_value: Any = value  # Default to string

    if attr is None:
        # Try to infer type from the field's type annotation
        from typing import get_type_hints

        type_hints = get_type_hints(type(section_obj))
        if option in type_hints:
            hint = type_hints[option]
            # Handle Optional types
            if hasattr(hint, "__origin__") and hint.__origin__ is Union:
                hint_args = hint.__args__
                if type(None) in hint_args:
                    # Get the non-None type
                    hint = next(arg for arg in hint_args if arg is not type(None))

            # Convert based on the type hint
            if hint is bool:
                converted_value = value.lower() in ("true", "yes", "1", "on")
            elif hint is int:
                converted_value = int(value)
            elif hint is float:
                converted_value = float(value)
            elif hint is str:
                converted_value = value  # Already a string
            elif hasattr(hint, "__origin__") and hint.__origin__ is list:
                # Handle lists
                value_list = value.split(",")
                # Convert list elements if needed
                if hint.__args__[0] is int:
                    converted_value = [int(v) for v in value_list]
                elif hint.__args__[0] is float:
                    converted_value = [float(v) for v in value_list]
                else:
                    converted_value = value_list
    else:
        # Convert based on the existing value's type
        if isinstance(attr, bool):
            converted_value = value.lower() in ("true", "yes", "1", "on")
        elif isinstance(attr, int):
            converted_value = int(value)
        elif isinstance(attr, float):
            converted_value = float(value)
        elif isinstance(attr, list):
            value_list = value.split(",")
            # Try to convert list elements to the same type as the first element in the existing list
            if attr and value_list:
                if isinstance(attr[0], int):
                    converted_value = [int(v) for v in value_list]
                elif isinstance(attr[0], float):
                    converted_value = [float(v) for v in value_list]
                else:
                    converted_value = value_list

    setattr(section_obj, option, converted_value)
