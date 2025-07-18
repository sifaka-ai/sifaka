"""Example of using Meta-Rewarding critic for self-evaluation.

Meta-Rewarding uses a two-stage process to evaluate and improve text quality.
This example uses Anthropic Claude for nuanced meta-evaluation.
"""

import asyncio
import os

from sifaka import improve
from sifaka.core.config import Config, CriticConfig, LLMConfig
from sifaka.core.types import CriticType
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run Meta-Rewarding improvement example with Anthropic."""

    # Technical explanation that needs quality evaluation
    text = """
    Machine learning is when computers learn from data. They use algorithms
    to find patterns. Deep learning is a type of machine learning that uses
    neural networks. It's very powerful and can do many things.
    """

    print("🏆 Meta-Rewarding Example - Self-evaluation")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Meta-Rewarding critic using Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("🤖 Using Anthropic Claude for meta-evaluation...")
        result = await improve(
            text,
            critics=[CriticType.META_REWARDING],
            max_iterations=2,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini",
                    critic_model="gpt-4o-mini",
                    temperature=0.6,  # Lower for consistent evaluation
                ),
                critic=CriticConfig(critics=[CriticType.META_REWARDING]),
            ),
            storage=FileStorage(),
        )
    elif os.getenv("GEMINI_API_KEY"):
        print("🌐 Using Google Gemini as fallback...")
        result = await improve(
            text,
            critics=[CriticType.META_REWARDING],
            max_iterations=2,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini",  # Pro model for complex meta-evaluation
                    critic_model="gpt-4o-mini",
                    temperature=0.6,
                ),
                critic=CriticConfig(critics=[CriticType.META_REWARDING]),
            ),
            storage=FileStorage(),
        )
    else:
        print("❌ No API keys found. Please set ANTHROPIC_API_KEY or GEMINI_API_KEY")
        return

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")
    print("\n💡 Meta-Rewarding evaluates its own critique quality!")


if __name__ == "__main__":
    # Note: Prefers ANTHROPIC_API_KEY for meta-evaluation, falls back to GEMINI_API_KEY
    asyncio.run(main())
