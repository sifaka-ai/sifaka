#!/usr/bin/env python3
"""Configuration management example for Sifaka.

This example demonstrates:
1. Creating configurations programmatically
2. Loading from environment variables
3. Loading from files (YAML/JSON)
4. Configuration validation
5. Error handling for invalid configurations

Run this example to see configuration utilities in action:
    python examples/config_example.py
"""

import json
import os
import tempfile
from pathlib import Path

# Add the project root to the path so we can import sifaka
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sifaka.utils import (
    SifakaConfig,
    ConfigurationError,
    configure_for_development,
    get_logger,
)

# Setup logging
configure_for_development()
logger = get_logger(__name__)


def demonstrate_basic_config():
    """Demonstrate basic configuration creation and validation."""
    print("\n1. Basic Configuration Creation")
    print("-" * 40)
    
    # Create default configuration
    config = SifakaConfig()
    print(f"Default config:")
    print(f"  Model: {config.default_model}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Critics enabled: {config.enable_critics}")
    print(f"  Timeout: {config.timeout_seconds}s")
    
    # Create custom configuration
    custom_config = SifakaConfig(
        default_model="anthropic:claude-3-sonnet",
        max_iterations=5,
        enable_critics=True,
        timeout_seconds=60.0,
        log_level="DEBUG"
    )
    print(f"\nCustom config:")
    print(f"  Model: {custom_config.default_model}")
    print(f"  Max iterations: {custom_config.max_iterations}")
    print(f"  Timeout: {custom_config.timeout_seconds}s")
    print(f"  Log level: {custom_config.log_level}")


def demonstrate_env_config():
    """Demonstrate loading configuration from environment variables."""
    print("\n2. Environment Variable Configuration")
    print("-" * 40)
    
    # Set some environment variables
    os.environ["SIFAKA_DEFAULT_MODEL"] = "openai:gpt-4o-mini"
    os.environ["SIFAKA_MAX_ITERATIONS"] = "4"
    os.environ["SIFAKA_ENABLE_CRITICS"] = "true"
    os.environ["SIFAKA_LOG_LEVEL"] = "INFO"
    os.environ["SIFAKA_TIMEOUT_SECONDS"] = "45.0"
    
    # Load from environment
    env_config = SifakaConfig.from_env()
    print(f"Config from environment:")
    print(f"  Model: {env_config.default_model}")
    print(f"  Max iterations: {env_config.max_iterations}")
    print(f"  Critics enabled: {env_config.enable_critics}")
    print(f"  Log level: {env_config.log_level}")
    print(f"  Timeout: {env_config.timeout_seconds}s")
    
    # Clean up environment variables
    for key in ["SIFAKA_DEFAULT_MODEL", "SIFAKA_MAX_ITERATIONS", 
                "SIFAKA_ENABLE_CRITICS", "SIFAKA_LOG_LEVEL", "SIFAKA_TIMEOUT_SECONDS"]:
        os.environ.pop(key, None)


def demonstrate_file_config():
    """Demonstrate loading configuration from files."""
    print("\n3. File-based Configuration")
    print("-" * 40)
    
    # Create temporary JSON config file
    json_config = {
        "default_model": "gemini-1.5-flash",
        "max_iterations": 3,
        "enable_critics": True,
        "enable_validators": True,
        "timeout_seconds": 30.0,
        "log_level": "WARNING",
        "storage_backend": "memory",
        "critic_models": {
            "reflexion": "openai:gpt-4o-mini",
            "constitutional": "anthropic:claude-3-haiku"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_config, f, indent=2)
        json_file = f.name
    
    try:
        # Load from JSON file
        file_config = SifakaConfig.from_file(json_file)
        print(f"Config from JSON file:")
        print(f"  Model: {file_config.default_model}")
        print(f"  Max iterations: {file_config.max_iterations}")
        print(f"  Storage backend: {file_config.storage_backend}")
        print(f"  Critic models: {file_config.critic_models}")
        
    finally:
        # Clean up temporary file
        os.unlink(json_file)
    
    # Try YAML if available
    try:
        import yaml
        
        yaml_config_content = """
default_model: "anthropic:claude-3-5-sonnet"
max_iterations: 2
enable_critics: true
enable_validators: false
timeout_seconds: 120.0
log_level: "ERROR"
storage_backend: "file"
validator_config:
  min_length: 50
  max_length: 1000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_config_content)
            yaml_file = f.name
        
        try:
            yaml_config = SifakaConfig.from_file(yaml_file)
            print(f"\nConfig from YAML file:")
            print(f"  Model: {yaml_config.default_model}")
            print(f"  Max iterations: {yaml_config.max_iterations}")
            print(f"  Validators enabled: {yaml_config.enable_validators}")
            print(f"  Validator config: {yaml_config.validator_config}")
            
        finally:
            os.unlink(yaml_file)
            
    except ImportError:
        print("\nYAML support not available (install PyYAML to enable)")


def demonstrate_validation_errors():
    """Demonstrate configuration validation and error handling."""
    print("\n4. Configuration Validation and Error Handling")
    print("-" * 40)
    
    # Test various invalid configurations
    test_cases = [
        {
            "name": "Invalid max_iterations (too low)",
            "config": {"max_iterations": 0},
        },
        {
            "name": "Invalid max_iterations (too high)",
            "config": {"max_iterations": 25},
        },
        {
            "name": "Invalid timeout",
            "config": {"timeout_seconds": -5.0},
        },
        {
            "name": "Invalid log level",
            "config": {"log_level": "INVALID"},
        },
        {
            "name": "Invalid storage backend",
            "config": {"storage_backend": "nonexistent"},
        },
    ]
    
    for test_case in test_cases:
        try:
            SifakaConfig(**test_case["config"])
            print(f"  ✗ {test_case['name']}: Should have failed but didn't")
        except ConfigurationError as e:
            print(f"  ✓ {test_case['name']}: Correctly caught error")
            print(f"    Error: {e.message}")
            if e.suggestions:
                print(f"    Suggestions: {', '.join(e.suggestions[:2])}")


def demonstrate_file_errors():
    """Demonstrate file loading error handling."""
    print("\n5. File Loading Error Handling")
    print("-" * 40)
    
    # Test non-existent file
    try:
        SifakaConfig.from_file("nonexistent_file.json")
        print("  ✗ Non-existent file: Should have failed")
    except ConfigurationError as e:
        print(f"  ✓ Non-existent file: {e.message}")
    
    # Test invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json content}')
        invalid_json_file = f.name
    
    try:
        SifakaConfig.from_file(invalid_json_file)
        print("  ✗ Invalid JSON: Should have failed")
    except ConfigurationError as e:
        print(f"  ✓ Invalid JSON: {e.message}")
    finally:
        os.unlink(invalid_json_file)
    
    # Test invalid configuration keys
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"invalid_key": "value", "another_invalid": 123}, f)
        invalid_config_file = f.name
    
    try:
        SifakaConfig.from_file(invalid_config_file)
        print("  ✗ Invalid config keys: Should have failed")
    except ConfigurationError as e:
        print(f"  ✓ Invalid config keys: {e.message}")
    finally:
        os.unlink(invalid_config_file)


def main():
    """Run all configuration examples."""
    logger.info("Starting configuration examples")
    
    print("Sifaka Configuration Management Examples")
    print("=" * 50)
    
    try:
        demonstrate_basic_config()
        demonstrate_env_config()
        demonstrate_file_config()
        demonstrate_validation_errors()
        demonstrate_file_errors()
        
        print("\n" + "=" * 50)
        print("All configuration examples completed successfully!")
        
        logger.info("Configuration examples completed successfully")
        
    except Exception as e:
        logger.error(
            "Configuration examples failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True
        )
        print(f"\nExamples failed: {type(e).__name__} - {str(e)}")
        raise


if __name__ == "__main__":
    main()
