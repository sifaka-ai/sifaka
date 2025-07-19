"""Example of using Self-Consistency critic for consensus-based evaluation.

Self-Consistency evaluates multiple outputs to ensure consistent quality.
This example uses Google Gemini for fast parallel consistency checks.
"""

import asyncio
import os

from sifaka import improve
from sifaka.core.config import Config, CriticConfig, LLMConfig
from sifaka.core.types import CriticType
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run Self-Consistency improvement example with Gemini."""

    # Complex reasoning that benefits from consistency checking
    text = """
    To solve climate change, we need to stop using all fossil fuels immediately.
    This is the only solution that will work. Nothing else matters - not renewable
    energy, not carbon capture, just complete cessation of fossil fuel use today.
    """

    print("🔄 Self-Consistency Example - Consensus evaluation")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Self-Consistency critic using Gemini for speed
    if os.getenv("GOOGLE_API_KEY"):
        print("🌐 Using Google Gemini for fast consistency checks...")
        result = await improve(
            text,
            critics=[CriticType.SELF_CONSISTENCY],
            max_iterations=2,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini",  # Fast for parallel evaluations
                    critic_model="gpt-4o-mini",
                    temperature=0.5,  # Lower for consistency
                ),
                critic=CriticConfig(critics=[CriticType.SELF_CONSISTENCY]),
            ),
            storage=FileStorage(),
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("🔧 Using OpenAI as fallback...")
        result = await improve(
            text,
            critics=[CriticType.SELF_CONSISTENCY],
            max_iterations=2,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini", critic_model="gpt-4o-mini", temperature=0.5
                ),
                critic=CriticConfig(critics=[CriticType.SELF_CONSISTENCY]),
            ),
            storage=FileStorage(),
        )
    else:
        print("❌ No API keys found. Please set GOOGLE_API_KEY or OPENAI_API_KEY")
        return

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")
    print("\n💡 Self-Consistency ensures balanced, nuanced perspectives!")


if __name__ == "__main__":
    # Note: Prefers GOOGLE_API_KEY for speed, falls back to OPENAI_API_KEY
    asyncio.run(main())
