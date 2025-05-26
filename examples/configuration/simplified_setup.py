#!/usr/bin/env python3
"""Simplified Configuration Setup Example.

This example demonstrates the new configuration simplification features:
- Configuration presets for common use cases
- Enhanced QuickStart methods with better defaults
- Configuration validation with helpful error messages
- Setup wizards for guided configuration
"""

import os
from dotenv import load_dotenv

from sifaka.quickstart import QuickStart, ConfigPresets, ConfigWizard
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


def demo_basic_quickstart():
    """Demonstrate basic QuickStart usage."""
    print("\n" + "=" * 60)
    print("BASIC QUICKSTART DEMO")
    print("=" * 60)
    
    # Simple one-liner setup
    chain = QuickStart.basic_chain("mock:test-model", "Write a short story about AI")
    result = chain.run()
    
    print(f"Generated text: {result.text[:100]}...")
    print(f"Iterations: {result.iteration}")


def demo_preset_configurations():
    """Demonstrate configuration presets."""
    print("\n" + "=" * 60)
    print("CONFIGURATION PRESETS DEMO")
    print("=" * 60)
    
    # Show available presets
    presets = [
        ("development", ConfigPresets.development()),
        ("content_generation", ConfigPresets.content_generation()),
        ("fact_checking", ConfigPresets.fact_checking()),
        ("research_analysis", ConfigPresets.research_analysis()),
        ("production_safe", ConfigPresets.production_safe()),
    ]
    
    for name, config in presets:
        print(f"\n{name.upper()} PRESET:")
        print(f"  Max iterations: {config['max_improvement_iterations']}")
        print(f"  Storage: {config['storage_type']}")
        print(f"  Validators: {config.get('validators', [])}")
        print(f"  Critics: {config.get('critics', [])}")


def demo_enhanced_quickstart():
    """Demonstrate enhanced QuickStart methods."""
    print("\n" + "=" * 60)
    print("ENHANCED QUICKSTART DEMO")
    print("=" * 60)
    
    # Development setup
    print("\nDevelopment setup:")
    dev_chain = QuickStart.for_development()
    print(f"  Model: {dev_chain.config.model}")
    print(f"  Max iterations: {dev_chain.config.options['max_improvement_iterations']}")
    
    # Production setup (would require real API key)
    print("\nProduction setup (mock):")
    try:
        prod_chain = QuickStart.for_production(
            "mock:production-model",
            "Generate a professional report about AI trends",
            storage="memory",  # Use memory for demo
            validators=["length"],
            critics=["reflexion"]
        )
        print(f"  Model: {prod_chain.config.model}")
        print(f"  Validators: {len(prod_chain.config.validators)}")
        print(f"  Critics: {len(prod_chain.config.critics)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Research setup
    print("\nResearch setup (mock):")
    try:
        research_chain = QuickStart.for_research(
            "mock:research-model",
            "Analyze the impact of AI on scientific research",
            storage="memory",  # Use memory for demo
            retrievers=False  # Disable retrievers for demo
        )
        print(f"  Model: {research_chain.config.model}")
        print(f"  Max iterations: {research_chain.config.options['max_improvement_iterations']}")
    except Exception as e:
        print(f"  Error: {e}")


def demo_configuration_wizard():
    """Demonstrate configuration wizard."""
    print("\n" + "=" * 60)
    print("CONFIGURATION WIZARD DEMO")
    print("=" * 60)
    
    wizard = ConfigWizard()
    
    # Environment validation
    print("\nEnvironment validation for mock model:")
    env_status = wizard.validate_environment("mock:test-model")
    for key, status in env_status.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {key}: {status}")
    
    # Get recommendations
    print("\nRecommendations for content generation:")
    recommendations = wizard.get_recommendations("content_generation", "mock:test-model")
    print(f"  Storage: {recommendations['storage_type']}")
    print(f"  Validators: {recommendations['validators']}")
    print(f"  Critics: {recommendations['critics']}")
    
    if "warnings" in recommendations:
        print("  Warnings:")
        for warning in recommendations["warnings"]:
            print(f"    - {warning}")
    
    # Setup for use case
    print("\nSetting up chain for development use case:")
    try:
        chain = wizard.setup_for_use_case("development", "mock:test-model", "Test prompt")
        print(f"  Chain created successfully")
        print(f"  Model: {chain.config.model}")
        print(f"  Storage: {type(chain.config.storage).__name__}")
    except Exception as e:
        print(f"  Error: {e}")


def demo_preset_usage():
    """Demonstrate using presets with QuickStart."""
    print("\n" + "=" * 60)
    print("PRESET USAGE DEMO")
    print("=" * 60)
    
    # Use preset through QuickStart
    print("\nUsing development preset:")
    try:
        chain = QuickStart.from_preset("development", "mock:test-model", "Test prompt")
        result = chain.run()
        print(f"  Generated text: {result.text[:50]}...")
        print(f"  Iterations: {result.iteration}")
    except Exception as e:
        print(f"  Error: {e}")


def demo_configuration_validation():
    """Demonstrate configuration validation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION DEMO")
    print("=" * 60)
    
    from sifaka.core.chain import Chain
    from sifaka.models.base import create_model
    
    # Valid configuration
    print("\nValid configuration:")
    try:
        model = create_model("mock:test-model")
        chain = Chain(model=model, prompt="Test prompt")
        chain.config.validate()
        print("  ✓ Configuration is valid")
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
    
    # Invalid configuration (missing prompt)
    print("\nInvalid configuration (missing prompt):")
    try:
        model = create_model("mock:test-model")
        chain = Chain(model=model)  # No prompt
        chain.config.validate()
        print("  ✓ Configuration is valid")
    except Exception as e:
        print(f"  ✗ Validation error: {e}")


def main():
    """Run all configuration demos."""
    print("SIFAKA CONFIGURATION SIMPLIFICATION DEMO")
    print("=" * 80)
    
    try:
        demo_basic_quickstart()
        demo_preset_configurations()
        demo_enhanced_quickstart()
        demo_configuration_wizard()
        demo_preset_usage()
        demo_configuration_validation()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")


if __name__ == "__main__":
    main()
