"""Example of using N-Critics for multi-perspective evaluation.

N-Critics uses multiple critical perspectives to provide comprehensive feedback.
"""

import asyncio
import os

from sifaka import improve
from sifaka.core.config import Config, CriticConfig, LLMConfig
from sifaka.core.types import CriticType
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run N-Critics improvement example."""

    # Business proposal that needs multiple perspectives
    text = """
    We should invest all our money in cryptocurrency because it's the future.
    Traditional investments are outdated. Bitcoin will definitely go to a million
    dollars, so we can't lose. We should act fast before it's too late.
    """

    print("👥 N-Critics Example - Multiple perspectives")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Two approaches: default perspectives or custom perspectives

    # Approach 1: Use default N-Critics with OpenAI
    print("\n1️⃣ Using default perspectives with OpenAI GPT-4o-mini...")

    if os.getenv("OPENAI_API_KEY"):
        result1 = await improve(
            text,
            critics=[CriticType.N_CRITICS],
            max_iterations=2,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini", critic_model="gpt-4o-mini", temperature=0.7
                ),
                critic=CriticConfig(critics=[CriticType.N_CRITICS]),
            ),
            storage=FileStorage(),
        )
        print(f"Improved: {result1.final_text[:100]}...")
    else:
        print("   ⏭️  Skipping - no OPENAI_API_KEY")
        result1 = None

    # Approach 2: Different configuration
    print("\n2️⃣ Using different model configuration...")

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        # Use N_CRITICS with different model config
        result2 = await improve(
            text,
            critics=[CriticType.N_CRITICS],
            max_iterations=2,
            config=Config(
                llm=LLMConfig(
                    model="claude-3-haiku-20240307"
                    if os.getenv("ANTHROPIC_API_KEY")
                    else "gpt-4o-mini",
                    critic_model="gpt-3.5-turbo",  # Specify critic model here
                    temperature=0.6,
                ),
                critic=CriticConfig(critics=[CriticType.N_CRITICS]),
            ),
            storage=FileStorage(),
        )

        print(f"✅ Final text ({len(result2.final_text.split())} words):")
        print(result2.final_text.strip())

        if result1:
            print(f"\n📊 Total iterations: {result1.iteration + result2.iteration}")
            print(
                f"⏱️  Total time: {result1.processing_time + result2.processing_time:.2f}s"
            )
        else:
            print(f"\n📊 Iterations: {result2.iteration}")
            print(f"⏱️  Time: {result2.processing_time:.2f}s")
    else:
        print("   ⏭️  Skipping - no ANTHROPIC_API_KEY")

    # Approach 3: Using Gemini for speed
    print("\n3️⃣ Fast iteration with Google Gemini...")

    if os.getenv("GEMINI_API_KEY"):
        result3 = await improve(
            text,
            critics=[CriticType.N_CRITICS],
            max_iterations=1,  # Just one fast iteration
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini", critic_model="gpt-4o-mini", temperature=0.7
                ),
                critic=CriticConfig(critics=[CriticType.N_CRITICS]),
            ),
            storage=FileStorage(),
        )
        print(f"✅ Quick improvement in {result3.processing_time:.2f}s")
        print(f"   Result preview: {result3.final_text[:150]}...")
    else:
        print("   ⏭️  Skipping - no GEMINI_API_KEY")


if __name__ == "__main__":
    # Note: Can use OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY
    available_providers = []
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("Anthropic")
    if os.getenv("GEMINI_API_KEY"):
        available_providers.append("Google")

    if not available_providers:
        print("❌ No API keys found. Please set at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GEMINI_API_KEY")
    else:
        print(f"✅ Available providers: {', '.join(available_providers)}")
        asyncio.run(main())
