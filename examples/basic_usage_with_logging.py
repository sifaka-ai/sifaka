#!/usr/bin/env python3
"""Basic usage example for Sifaka with integrated logging.

This example demonstrates:
1. Setting up logging for development
2. Creating a SifakaEngine with default configuration
3. Processing a simple thought
4. Observing the complete workflow with logging output

Run this example to see the utilities in action:
    python examples/basic_usage_with_logging.py
"""

import asyncio
import os
from pathlib import Path

# Add the project root to the path so we can import sifaka
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sifaka.core.engine import SifakaEngine
from sifaka.utils import (
    configure_for_development,
    get_logger,
    SifakaConfig,
    validate_prompt,
)

# Setup logging for development
configure_for_development()
logger = get_logger(__name__)


async def main():
    """Main example function demonstrating Sifaka with logging."""
    
    logger.info("Starting Sifaka basic usage example")
    
    try:
        # Validate the prompt using our utilities
        prompt = validate_prompt("Write a brief explanation of renewable energy sources")
        
        logger.info(
            "Prompt validated successfully",
            extra={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            }
        )
        
        # Create configuration (demonstrating config utilities)
        config = SifakaConfig(
            max_iterations=2,  # Keep it short for demo
            enable_critics=True,
            enable_validators=True,
            log_level="DEBUG"
        )
        
        logger.info(
            "Configuration created",
            extra={
                "max_iterations": config.max_iterations,
                "enable_critics": config.enable_critics,
                "enable_validators": config.enable_validators,
            }
        )
        
        # Create engine with default dependencies
        logger.info("Creating SifakaEngine with default dependencies")
        engine = SifakaEngine()
        
        # Process the thought
        logger.info("Starting thought processing")
        with logger.performance_timer("complete_thought_processing"):
            thought = await engine.think(prompt, max_iterations=config.max_iterations)
        
        # Display results
        logger.info(
            "Thought processing completed successfully",
            extra={
                "thought_id": thought.id,
                "final_iteration": thought.iteration,
                "techniques_applied": thought.techniques_applied,
                "generations_count": len(thought.generations),
                "validations_count": len(thought.validations),
                "critiques_count": len(thought.critiques),
                "final_text_length": len(thought.final_text) if thought.final_text else 0,
            }
        )
        
        print("\n" + "="*80)
        print("SIFAKA THOUGHT PROCESSING RESULTS")
        print("="*80)
        
        print(f"\nThought ID: {thought.id}")
        print(f"Prompt: {thought.prompt}")
        print(f"Final Iteration: {thought.iteration}")
        print(f"Techniques Applied: {', '.join(thought.techniques_applied)}")
        
        print(f"\nFinal Text:")
        print("-" * 40)
        print(thought.final_text or thought.current_text or "No text generated")
        
        print(f"\nProcessing Summary:")
        print(f"- Generations: {len(thought.generations)}")
        print(f"- Validations: {len(thought.validations)}")
        print(f"- Critiques: {len(thought.critiques)}")
        print(f"- Tool Calls: {len(thought.tool_calls)}")
        
        # Show validation results
        if thought.validations:
            print(f"\nValidation Results:")
            for validation in thought.validations:
                status = "✓ PASS" if validation.passed else "✗ FAIL"
                print(f"  {status} {validation.validator} (iteration {validation.iteration})")
        
        # Show critique summary
        if thought.critiques:
            print(f"\nCritique Summary:")
            for critique in thought.critiques:
                suggestions_count = len(critique.suggestions)
                print(f"  {critique.critic}: {suggestions_count} suggestions (iteration {critique.iteration})")
        
        print("\n" + "="*80)
        
        # Demonstrate error handling
        logger.info("Testing error handling with invalid input")
        try:
            # This should raise a ValidationError
            await engine.think("", max_iterations=1)
        except Exception as e:
            logger.info(
                "Error handling working correctly",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
            print(f"\nError handling test: {type(e).__name__} - {str(e)}")
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(
            "Example failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True
        )
        print(f"\nExample failed: {type(e).__name__} - {str(e)}")
        raise


if __name__ == "__main__":
    # Set environment variables for API keys if needed
    # You can set these in your environment or uncomment and set them here:
    # os.environ["OPENAI_API_KEY"] = "your-openai-key"
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    # os.environ["GOOGLE_API_KEY"] = "your-google-key"
    # os.environ["GROQ_API_KEY"] = "your-groq-key"
    
    print("Sifaka Basic Usage Example with Integrated Logging")
    print("=" * 60)
    print("This example demonstrates the utilities integration:")
    print("- Logging setup and structured output")
    print("- Configuration management")
    print("- Input validation")
    print("- Error handling")
    print("- Performance tracking")
    print("=" * 60)
    
    asyncio.run(main())
