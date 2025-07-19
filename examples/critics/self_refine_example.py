"""Example of using Self-Refine critic for general text improvement.

Self-Refine provides iterative refinement for overall quality enhancement.
This example uses Google Gemini for fast, cost-effective refinement.
"""

import asyncio
import os

from sifaka import improve
from sifaka.core.config import Config, CriticConfig, LLMConfig
from sifaka.core.types import CriticType
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run Self-Refine improvement example with Google Gemini."""

    # Generic text that needs polish
    text = """
    Email marketing is good for businesses. You can send emails to customers
    and they might buy things. It's cheaper than other marketing. You should
    collect email addresses and send newsletters.
    """

    print("✨ Self-Refine Example - General improvement")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Self-Refine critic
    if os.getenv("OPENAI_API_KEY"):
        print("🤖 Using OpenAI GPT-4o-mini for refinement...")
        result = await improve(
            text,
            critics=[CriticType.SELF_REFINE],
            max_iterations=3,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini", critic_model="gpt-3.5-turbo", temperature=0.7
                ),
                critic=CriticConfig(critics=[CriticType.SELF_REFINE]),
            ),
            storage=FileStorage(storage_dir="./thoughts"),
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        print("🤖 Using Anthropic Claude as fallback...")
        result = await improve(
            text,
            critics=[CriticType.SELF_REFINE],
            max_iterations=3,
            config=Config(
                llm=LLMConfig(
                    model="gpt-4o-mini", critic_model="gpt-4o-mini", temperature=0.7
                ),
                critic=CriticConfig(critics=[CriticType.SELF_REFINE]),
            ),
            storage=FileStorage(),
        )
    else:
        print("❌ No API keys found. Please set GOOGLE_API_KEY or ANTHROPIC_API_KEY")
        return

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Prefers GOOGLE_API_KEY, falls back to ANTHROPIC_API_KEY
    asyncio.run(main())
