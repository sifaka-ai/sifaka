#!/usr/bin/env python3
"""Basic PydanticAI integration example.

This example demonstrates the basic usage of PydanticAI with Sifaka,
showing how to create a hybrid chain that combines PydanticAI's agent
capabilities with Sifaka's validation and criticism framework.
"""

import sys

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run the basic PydanticAI integration example."""
    try:
        # Import PydanticAI (will fail gracefully if not installed)
        from pydantic_ai import Agent

        # Import Sifaka components
        from sifaka.agents import create_pydantic_chain
        from sifaka.critics import SelfRefineCritic
        from sifaka.models import create_model
        from sifaka.storage import FileStorage
        from sifaka.validators import LengthValidator

        print("🤖 Basic PydanticAI + Sifaka Integration Example")
        print("=" * 50)

        # Create a PydanticAI agent with DuckDuckGo search (using their exact pattern)
        print("\n1. Creating PydanticAI agent with web search...")

        # Import DuckDuckGo search tool - using their exact import and usage
        from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

        agent = Agent(
            "openai:gpt-4o-mini",
            tools=[duckduckgo_search_tool()],  # Note: they call it as a function
            system_prompt="You are a helpful research assistant. Search for information about reflective intelligence and write an informative article based on your findings.",
        )
        print(f"✓ Created agent with DuckDuckGo search tool")

        # Create Sifaka components
        print("\n2. Creating Sifaka components...")

        # Validator to ensure reasonable length for an informative article
        validator = LengthValidator(min_length=300, max_length=1200)
        print("✓ Created length validator (300-1200 chars)")

        # Critic to provide feedback (using a cheaper model)
        critic_model = create_model("openai:gpt-3.5-turbo")
        critic = SelfRefineCritic(model=critic_model)
        print("✓ Created self-refine critic")

        # Create storage for thoughts

        # Create file storage in thoughts directory (using relative path)
        from pathlib import Path

        thoughts_dir = Path("thoughts")
        thoughts_dir.mkdir(exist_ok=True)
        storage = FileStorage(
            thoughts_dir / "pydantic_ai_basic_example.json",
            overwrite=True,  # Overwrite existing file instead of appending
        )
        print("✓ Created file storage in thoughts directory")

        # Create the hybrid chain
        print("\n3. Creating hybrid PydanticAI chain...")
        chain = create_pydantic_chain(
            agent=agent,
            validators=[validator],
            critics=[critic],
            storage=storage,
            max_improvement_iterations=2,
        )
        print("✓ Created hybrid chain")

        # Run the chain
        print("\n4. Running the chain...")
        prompt = "Research and write an informative article about reflective intelligence. Use web search to gather current information and insights."
        print(f"Prompt: {prompt}")

        result = chain.run(prompt)

        # Display results
        print("\n" + "=" * 50)
        print("📊 RESULTS")
        print("=" * 50)

        print(f"\n📝 Generated Text ({len(result.text)} characters):")
        print("-" * 30)
        print(result.text)

        print(f"\n🔄 Iterations: {result.iteration}")

        # Show validation results
        if hasattr(result, "validation_results") and result.validation_results:
            print(f"\n✅ Validation Results:")
            for name, validation_result in result.validation_results.items():
                status = "PASSED" if validation_result.passed else "FAILED"
                print(f"  - {name}: {status}")
                if not validation_result.passed and validation_result.message:
                    print(f"    Message: {validation_result.message}")

        # Show critic feedback
        if hasattr(result, "critic_feedback") and result.critic_feedback:
            print(f"\n💭 Critic Feedback:")
            for feedback in result.critic_feedback:
                print(f"  - {feedback.critic_name}: {feedback.feedback[:100]}...")
                print(f"    Confidence: {feedback.confidence:.2f}")

        print(
            f"\n🎯 Final validation status: {'PASSED' if chain._validation_passed(result) else 'FAILED'}"
        )

        # Show storage information
        print(f"\n💾 Thought saved to: {storage.file_path}")
        print(f"🆔 Chain ID: {result.chain_id}")

        print("\n✨ Example completed successfully!")

    except ImportError as e:
        if "pydantic_ai" in str(e):
            print("❌ PydanticAI is not installed.")
            print("Please install it with: pip install pydantic-ai")
            print("Or with uv: uv add pydantic-ai")
        else:
            print(f"❌ Import error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error running example: {e}")
        logger.exception("Example failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
