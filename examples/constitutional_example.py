"""Example of using Constitutional AI critic for principle-based evaluation.

Constitutional AI evaluates text against a set of principles for safety and quality.
"""

import asyncio
from sifaka import improve


async def main():
    """Run Constitutional AI improvement example."""

    # Text that might need safety improvements
    text = """
    To get rid of pests in your garden, you should use the strongest chemicals
    available. Just spray everything heavily and don't worry about the instructions
    on the label - more is always better when dealing with bugs.
    """

    print("Original text:")
    print(text)
    print("\n" + "=" * 80 + "\n")

    try:
        # Run improvement with Constitutional critic
        result = await improve(text, critics=["constitutional"], max_iterations=3)

        print("Improved text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")

        # Show how the text was made safer
        print("\nSafety improvements:")
        for critique in result.critiques:
            if critique.critic == "constitutional" and critique.needs_improvement:
                print(f"\n- Issue identified: {critique.feedback}")
                if critique.suggestions:
                    print("  Corrections made:")
                    for suggestion in critique.suggestions:
                        print(f"    * {suggestion}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
