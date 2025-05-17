# Sifaka Configuration Management Plan

## Current Problem

The Sifaka codebase currently lacks a centralized configuration approach, which leads to several issues:

1. **Inconsistent Configuration Passing**: Options are passed through multiple layers (Chain → Model → Critics), creating potential for inconsistency.
2. **Scattered Validation**: Configuration validation happens at different levels throughout the codebase.
3. **Poor Documentation**: There's no clear documentation of which options apply to which components.
4. **Duplication**: The same configuration options may be defined and validated in multiple places.
5. **Difficult Maintenance**: Changes to configuration options require updates in multiple places.

## Implementation Plan

### Phase 1: Create Core Configuration System

1. Create a centralized configuration module with dataclasses for different component types:
   - `ModelConfig`: Configuration for model components
   - `ValidatorConfig`: Configuration for validator components
   - `CriticConfig`: Configuration for critic components
   - `SifakaConfig`: Top-level configuration that contains all component configs

2. Implement configuration loading and saving utilities:
   - Load from JSON/YAML files
   - Load from environment variables
   - Save to JSON/YAML files

### Phase 2: Update Components to Use Centralized Configuration

1. Update the `Chain` class to use the centralized configuration system:
   - Accept a `SifakaConfig` in the constructor
   - Pass appropriate configuration to components

2. Add configuration support to all component types:
   - Models
   - Validators
   - Critics
   - Retrievers

3. Implement a `configure()` method in all components to update configuration after initialization

### Phase 3: Documentation and Examples

1. Create comprehensive configuration documentation:
   - Document all available configuration options
   - Provide examples of common configuration scenarios
   - Explain how configuration is passed to components

2. Update examples to demonstrate configuration usage:
   - Basic configuration
   - Loading from files
   - Environment variable configuration

### Phase 4: Testing and Validation

1. Implement configuration validation:
   - Type checking
   - Range validation
   - Dependency validation

2. Add tests for configuration system:
   - Test loading/saving
   - Test validation
   - Test component configuration

## Implementation Details

### 1. Core Configuration Classes

```python
# sifaka/config.py
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import json
import yaml
import os
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model components."""
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
    """Configuration for validator components."""
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
    """Configuration for critic components."""
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
    """Configuration for retriever components."""
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
    """Centralized configuration for all Sifaka components."""
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
        """Get all options relevant to models."""
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
        return options
    
    def get_validator_options(self) -> Dict[str, Any]:
        """Get all options relevant to validators."""
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
        return options
    
    def get_critic_options(self) -> Dict[str, Any]:
        """Get all options relevant to critics."""
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
        return options
    
    def get_retriever_options(self) -> Dict[str, Any]:
        """Get all options relevant to retrievers."""
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
        return options
```

### 2. Configuration Loading and Saving

```python
# Additional methods for sifaka/config.py

def load_config_from_json(file_path: Union[str, Path]) -> SifakaConfig:
    """Load configuration from a JSON file."""
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return _create_config_from_dict(config_dict)

def load_config_from_yaml(file_path: Union[str, Path]) -> SifakaConfig:
    """Load configuration from a YAML file."""
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return _create_config_from_dict(config_dict)

def load_config_from_env(prefix: str = "SIFAKA_") -> SifakaConfig:
    """Load configuration from environment variables."""
    config = SifakaConfig()
    # Example: SIFAKA_MODEL_TEMPERATURE -> config.model.temperature
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix):].lower().split('_')
            if len(parts) >= 2:
                section, option = parts[0], '_'.join(parts[1:])
                _set_config_value(config, section, option, value)
    return config

def save_config_to_json(config: SifakaConfig, file_path: Union[str, Path]) -> None:
    """Save configuration to a JSON file."""
    file_path = Path(file_path)
    config_dict = _config_to_dict(config)
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def save_config_to_yaml(config: SifakaConfig, file_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    file_path = Path(file_path)
    config_dict = _config_to_dict(config)
    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def _create_config_from_dict(config_dict: Dict[str, Any]) -> SifakaConfig:
    """Create a SifakaConfig from a dictionary."""
    # Extract component configs
    model_config = config_dict.pop('model', {})
    validator_config = config_dict.pop('validator', {})
    critic_config = config_dict.pop('critic', {})
    retriever_config = config_dict.pop('retriever', {})
    
    # Create component configs
    model = ModelConfig(**model_config) if model_config else ModelConfig()
    validator = ValidatorConfig(**validator_config) if validator_config else ValidatorConfig()
    critic = CriticConfig(**critic_config) if critic_config else CriticConfig()
    retriever = RetrieverConfig(**retriever_config) if retriever_config else RetrieverConfig()
    
    # Create main config
    return SifakaConfig(
        model=model,
        validator=validator,
        critic=critic,
        retriever=retriever,
        **config_dict
    )

def _config_to_dict(config: SifakaConfig) -> Dict[str, Any]:
    """Convert a SifakaConfig to a dictionary."""
    from dataclasses import asdict
    return asdict(config)

def _set_config_value(config: SifakaConfig, section: str, option: str, value: str) -> None:
    """Set a configuration value from an environment variable."""
    if not hasattr(config, section):
        return
    
    section_obj = getattr(config, section)
    if not hasattr(section_obj, option):
        return
    
    # Convert value to appropriate type
    attr = getattr(section_obj, option)
    if attr is None:
        # Try to infer type from the field's type annotation
        import inspect
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
                value = value.lower() in ('true', 'yes', '1', 'on')
            elif hint is int:
                value = int(value)
            elif hint is float:
                value = float(value)
            elif hint is str:
                pass  # Already a string
            elif hasattr(hint, "__origin__") and hint.__origin__ is list:
                # Handle lists
                value = value.split(',')
                # Convert list elements if needed
                if hint.__args__[0] is int:
                    value = [int(v) for v in value]
                elif hint.__args__[0] is float:
                    value = [float(v) for v in value]
    else:
        # Convert based on the existing value's type
        if isinstance(attr, bool):
            value = value.lower() in ('true', 'yes', '1', 'on')
        elif isinstance(attr, int):
            value = int(value)
        elif isinstance(attr, float):
            value = float(value)
        elif isinstance(attr, list):
            value = value.split(',')
            # Try to convert list elements to the same type as the first element in the existing list
            if attr and value:
                if isinstance(attr[0], int):
                    value = [int(v) for v in value]
                elif isinstance(attr[0], float):
                    value = [float(v) for v in value]
    
    setattr(section_obj, option, value)
```

## Execution Plan

1. **Create Configuration Module**: Implement the `sifaka/config.py` module with all configuration classes and utilities.

2. **Update Chain Class**: Modify the `Chain` class to use the centralized configuration system.

3. **Update Component Base Classes**: Add configuration support to base classes for models, validators, critics, and retrievers.

4. **Update Factory Functions**: Modify factory functions to accept and pass configuration options.

5. **Create Documentation**: Create comprehensive documentation for the configuration system.

6. **Update Examples**: Update examples to demonstrate configuration usage.

7. **Add Tests**: Add tests for the configuration system.

## Timeline

- **Day 1**: Implement core configuration classes and utilities
- **Day 2**: Update Chain class and component base classes
- **Day 3**: Update factory functions and create documentation
- **Day 4**: Update examples and add tests
- **Day 5**: Review and refine implementation

## Success Criteria

The configuration management system will be considered successful if:

1. All components can be configured through a single, centralized configuration object
2. Configuration can be loaded from files and environment variables
3. Configuration options are well-documented
4. Examples demonstrate proper configuration usage
5. Tests verify that configuration works correctly

## Future Enhancements

After the initial implementation, we can consider:

1. **Schema Validation**: Add JSON Schema validation for configuration files
2. **Configuration UI**: Create a web UI for editing configuration
3. **Configuration Profiles**: Support for different configuration profiles
4. **Dynamic Reconfiguration**: Allow components to be reconfigured at runtime
5. **Configuration Inheritance**: Support for configuration inheritance and overrides
