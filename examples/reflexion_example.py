"""Example of using Reflexion critic for iterative text improvement.

Reflexion uses self-reflection on previous attempts to identify and correct mistakes.
This example shows how to use different providers for optimal results.
"""

import asyncio
import os

from sifaka import Config, improve
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run Reflexion improvement example with multiple providers."""

    # Original text to improve
    text = """
    Climate change is a big problem. It's getting hotter and the ice is melting.
    We should probably do something about it soon or things will get worse.
    """

    print("🔍 Reflexion Example - Learning from iterations")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Try different providers to show variety
    providers_tested = []

    # Option 1: Google Gemini (fastest, good for quick iterations)
    if os.getenv("GOOGLE_API_KEY"):
        print("\n1️⃣ Using Google Gemini 1.5 Flash for rapid Reflexion...")
        result = await improve(
            text,
            critics=["reflexion"],
            max_iterations=3,
            config=Config(
                model="gpt-4o-mini", critic_model="gpt-4o-mini", temperature=0.7
            ),
            storage=FileStorage(),
        )

        print(
            f"✅ Improved in {result.processing_time:.2f}s ({len(result.final_text.split())} words)"
        )
        print(f"   Preview: {result.final_text[:150]}...")
        providers_tested.append("Gemini")

    # Option 2: Anthropic Claude (excellent self-reflection)
    elif os.getenv("ANTHROPIC_API_KEY"):
        print("\n2️⃣ Using Anthropic Claude for deep Reflexion...")
        result = await improve(
            text,
            critics=["reflexion"],
            max_iterations=3,
            config=Config(
                model="gpt-3.5-turbo", critic_model="gpt-3.5-turbo", temperature=0.6
            ),
            storage=FileStorage(),
        )

        print(f"✅ Improved text ({len(result.final_text.split())} words):")
        print(result.final_text.strip())
        print(f"\n📊 Iterations: {result.iteration}")
        print(f"⏱️  Time: {result.processing_time:.2f}s")
        providers_tested.append("Claude")

    # Option 3: OpenAI (fallback)
    elif os.getenv("OPENAI_API_KEY"):
        print("\n3️⃣ Using OpenAI GPT-4o-mini...")
        result = await improve(
            text,
            critics=["reflexion"],
            max_iterations=3,
            config=Config(
                model="gpt-4o-mini", critic_model="gpt-4o-mini", temperature=0.7
            ),
            storage=FileStorage(),
        )

        print(f"✅ Improved text ({len(result.final_text.split())} words):")
        print(result.final_text.strip())
        print(f"\n📊 Iterations: {result.iteration}")
        print(f"⏱️  Time: {result.processing_time:.2f}s")
        providers_tested.append("OpenAI")

    else:
        print("❌ No API keys found. Please set one of:")
        print("   - GOOGLE_API_KEY (recommended for speed)")
        print("   - ANTHROPIC_API_KEY (recommended for quality)")
        print("   - OPENAI_API_KEY")
        return

    # Show how Reflexion learns across iterations
    print("\n📚 How Reflexion learns:")
    print("   - Iteration 1: Initial improvements")
    print("   - Iteration 2: Reflects on what worked/didn't work")
    print("   - Iteration 3: Applies lessons learned")
    print(f"\n✅ Tested with: {', '.join(providers_tested)}")


async def reflexion_comparison():
    """Compare Reflexion performance across providers."""

    if not any(
        [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    ):
        return

    print("\n\n🔬 Provider Comparison for Reflexion")
    print("=" * 50)

    test_text = "AI will replace all jobs. This is bad and we can't stop it."

    results = {}

    # Test each available provider
    if os.getenv("GOOGLE_API_KEY"):
        start = asyncio.get_event_loop().time()
        result = await improve(
            test_text,
            critics=["reflexion"],
            max_iterations=2,
            config=Config(model="gpt-4o-mini", critic_model="gpt-4o-mini"),
        )
        results["Gemini Flash"] = {
            "time": asyncio.get_event_loop().time() - start,
            "length": len(result.final_text.split()),
        }

    if os.getenv("ANTHROPIC_API_KEY"):
        start = asyncio.get_event_loop().time()
        result = await improve(
            test_text,
            critics=["reflexion"],
            max_iterations=2,
            config=Config(model="gpt-3.5-turbo", critic_model="gpt-3.5-turbo"),
        )
        results["Claude Haiku"] = {
            "time": asyncio.get_event_loop().time() - start,
            "length": len(result.final_text.split()),
        }

    # Print comparison
    print("\n📊 Results:")
    for provider, data in results.items():
        print(f"   {provider}: {data['time']:.2f}s, {data['length']} words")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(reflexion_comparison())
