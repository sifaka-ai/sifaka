"""Configuration templates for Sifaka.

This module provides ready-to-use configuration templates for common scenarios,
making it easy to get started with Sifaka without understanding all the options.

Templates include:
- Simple configurations for quick start
- Advanced configurations for production use
- Domain-specific configurations for different use cases
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path


def get_simple_config() -> Dict[str, Any]:
    """Get a simple configuration for quick start.
    
    Returns:
        Simple configuration dictionary
    """
    return {
        "model": "openai:gpt-4o-mini",
        "max_rounds": 2,
        "critics": ["reflexion"],
        "enable_caching": True,
        "enable_logging": False,
        "enable_timing": False
    }


def get_production_config() -> Dict[str, Any]:
    """Get a production-ready configuration.
    
    Returns:
        Production configuration dictionary
    """
    return {
        "model": "openai:gpt-4",
        "max_rounds": 5,
        "critics": ["reflexion", "constitutional", "self_refine"],
        "enable_caching": True,
        "enable_logging": True,
        "enable_timing": True,
        "min_length": 200,
        "max_length": 2000
    }


def get_high_quality_config() -> Dict[str, Any]:
    """Get a high-quality configuration for best results.
    
    Returns:
        High-quality configuration dictionary
    """
    return {
        "model": "openai:gpt-4",
        "max_rounds": 7,
        "critics": ["reflexion", "constitutional", "self_refine", "self_consistency"],
        "enable_caching": True,
        "enable_logging": True,
        "enable_timing": True,
        "min_length": 400,
        "max_length": 3000
    }


def get_academic_config() -> Dict[str, Any]:
    """Get configuration optimized for academic writing.
    
    Returns:
        Academic writing configuration dictionary
    """
    return {
        "model": "openai:gpt-4",
        "max_rounds": 6,
        "critics": ["reflexion", "constitutional", "self_refine"],
        "enable_caching": True,
        "enable_logging": True,
        "enable_timing": True,
        "min_length": 500,
        "max_length": 5000
    }


def get_creative_config() -> Dict[str, Any]:
    """Get configuration optimized for creative writing.
    
    Returns:
        Creative writing configuration dictionary
    """
    return {
        "model": "anthropic:claude-3-5-sonnet-20241022",
        "max_rounds": 4,
        "critics": ["constitutional", "self_consistency"],
        "enable_caching": True,
        "enable_logging": False,
        "enable_timing": True,
        "max_length": 1500
    }


def get_technical_config() -> Dict[str, Any]:
    """Get configuration optimized for technical documentation.
    
    Returns:
        Technical documentation configuration dictionary
    """
    return {
        "model": "openai:gpt-4",
        "max_rounds": 4,
        "critics": ["reflexion", "self_refine"],
        "enable_caching": True,
        "enable_logging": True,
        "enable_timing": True,
        "min_length": 300,
        "max_length": 2000
    }


def get_business_config() -> Dict[str, Any]:
    """Get configuration optimized for business writing.
    
    Returns:
        Business writing configuration dictionary
    """
    return {
        "model": "openai:gpt-4o-mini",
        "max_rounds": 3,
        "critics": ["constitutional"],
        "enable_caching": True,
        "enable_logging": False,
        "enable_timing": False,
        "max_length": 800
    }


def save_config_template(config: Dict[str, Any], filename: str) -> None:
    """Save a configuration template to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        filename: Name of the file to save (without extension)
    """
    templates_dir = Path(__file__).parent
    filepath = templates_dir / f"{filename}.json"
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config_template(filename: str) -> Dict[str, Any]:
    """Load a configuration template from a JSON file.
    
    Args:
        filename: Name of the file to load (without extension)
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    templates_dir = Path(__file__).parent
    filepath = templates_dir / f"{filename}.json"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Template '{filename}' not found")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def list_available_templates() -> List[str]:
    """List all available configuration templates.
    
    Returns:
        List of template names (without .json extension)
    """
    templates_dir = Path(__file__).parent
    json_files = templates_dir.glob("*.json")
    return [f.stem for f in json_files]


def create_custom_config(
    model: str = "openai:gpt-4o-mini",
    max_rounds: int = 3,
    critics: Optional[List[str]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    enable_caching: bool = True,
    enable_logging: bool = False,
    enable_timing: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """Create a custom configuration with specified parameters.
    
    Args:
        model: Model to use
        max_rounds: Maximum improvement rounds
        critics: List of critics to use
        min_length: Minimum text length
        max_length: Maximum text length
        enable_caching: Whether to enable caching
        enable_logging: Whether to enable logging
        enable_timing: Whether to enable timing
        **kwargs: Additional configuration options
        
    Returns:
        Custom configuration dictionary
    """
    if critics is None:
        critics = ["reflexion"]
    
    config = {
        "model": model,
        "max_rounds": max_rounds,
        "critics": critics,
        "enable_caching": enable_caching,
        "enable_logging": enable_logging,
        "enable_timing": enable_timing
    }
    
    if min_length is not None:
        config["min_length"] = min_length
    
    if max_length is not None:
        config["max_length"] = max_length
    
    # Add any additional options
    config.update(kwargs)
    
    return config


# Pre-defined template configurations
TEMPLATES = {
    "simple": get_simple_config,
    "production": get_production_config,
    "high_quality": get_high_quality_config,
    "academic": get_academic_config,
    "creative": get_creative_config,
    "technical": get_technical_config,
    "business": get_business_config
}


def get_template(name: str) -> Dict[str, Any]:
    """Get a configuration template by name.
    
    Args:
        name: Name of the template
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If template name is not found
    """
    if name not in TEMPLATES:
        available = list(TEMPLATES.keys())
        raise ValueError(f"Template '{name}' not found. Available: {available}")
    
    return TEMPLATES[name]()
