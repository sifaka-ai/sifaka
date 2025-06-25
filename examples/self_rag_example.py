"""Example of using Self-RAG critic for retrieval-augmented critique.

Self-RAG uses retrieval-augmented generation to verify factual accuracy.
"""

import asyncio
from sifaka import improve, FileStorage


async def main():
    """Run Self-RAG improvement example with file storage."""

    # Text with potential factual issues
    text = """
    The Amazon rainforest produces 50% of the world's oxygen. It's often called
    the lungs of the Earth. The forest is home to over 10 million species and
    covers an area of 2 million square miles.
    """

    print("Original text:")
    print(text)
    print("\n" + "=" * 80 + "\n")

    try:
        # Run improvement with Self-RAG critic and file storage
        storage = FileStorage("./rag_thoughts")

        result = await improve(
            text, critics=["self_rag"], max_iterations=3, storage=storage
        )

        print("Fact-checked text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Result saved with ID: {result.id}")

        # Show fact-checking results
        print("\nFact-checking critique:")
        for critique in result.critiques:
            if critique.critic == "self_rag":
                print(f"\n- Feedback: {critique.feedback}")
                if critique.suggestions:
                    print("  Fact corrections:")
                    for suggestion in critique.suggestions:
                        print(f"    * {suggestion}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
