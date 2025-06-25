"""Example of using Reflexion critic for iterative text improvement.

Reflexion uses self-reflection on previous attempts to identify and correct mistakes.
"""

import asyncio
from sifaka import improve


async def main():
    """Run Reflexion improvement example."""

    # Original text to improve
    text = """
    Climate change is a big problem. It's getting hotter and the ice is melting.
    We should probably do something about it soon or things will get worse.
    """

    print("Original text:")
    print(text)
    print("\n" + "=" * 80 + "\n")

    try:
        # Run improvement with Reflexion critic
        result = await improve(text, critics=["reflexion"], max_iterations=3)

        print("Improved text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")

        # Show critique feedback
        print("\nCritique feedback:")
        for critique in result.critiques:
            print(f"\n- {critique.critic} (confidence: {critique.confidence:.2f}):")
            print(f"  {critique.feedback}")
            if critique.suggestions:
                print("  Suggestions:")
                for suggestion in critique.suggestions:
                    print(f"    * {suggestion}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
