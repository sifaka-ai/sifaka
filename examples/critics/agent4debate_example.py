"""Example demonstrating the Agent4Debate critic's competitive debate dynamics.

This critic simulates a multi-agent debate where different agents champion
competing text improvements through structured argumentation.
"""

import asyncio

from sifaka import improve
from sifaka.critics import Agent4DebateCritic


async def main():
    # Example text with clear trade-offs for debate
    text = """
    Our new AI assistant helps users complete tasks quickly. It uses advanced
    technology to understand requests and provide accurate responses. The system
    is designed to be helpful while maintaining user privacy and security.
    """

    print("Original text:")
    print(text)
    print("\n" + "=" * 60 + "\n")

    # Method 1: Direct critic usage to see full debate
    print("Method 1: Direct critic usage (full debate visible)")
    print("-" * 40)

    critic = Agent4DebateCritic(model="gpt-4o-mini", temperature=0.7)

    from sifaka.core.models import SifakaResult

    result = SifakaResult(original_text=text, final_text=text)

    critique = await critic.critique(text, result)

    print(f"Overall feedback: {critique.feedback}\n")

    # Show debate content from simplified structure
    if "opening_positions" in critique.metadata:
        print("\nOPENING POSITIONS:")
        for i, position in enumerate(critique.metadata["opening_positions"], 1):
            print(f"\nPosition {i}: {position}")

    if "key_arguments" in critique.metadata:
        print("\nKEY DEBATE ARGUMENTS:")
        for arg in critique.metadata["key_arguments"]:
            print(f"  • {arg}")

    if "winning_approach" in critique.metadata:
        print(f"\nWINNING APPROACH: {critique.metadata['winning_approach']}")

    if "debate_insights" in critique.metadata:
        print(f"\nDEBATE INSIGHTS: {critique.metadata['debate_insights']}")

    print("\nSuggestions from debate:")
    for i, suggestion in enumerate(critique.suggestions[:3], 1):
        print(f"{i}. {suggestion}")

    print("\n" + "=" * 60 + "\n")

    # Method 2: Using improve() to see iterative debate refinement
    print("Method 2: Using improve() with agent4debate")
    print("-" * 40)

    result = await improve(text, critics=["agent4debate"], max_iterations=2)

    print("\nFinal improved text:")
    print(result.final_text)

    print(f"\nIterations: {result.iteration}")

    # Show how debates evolved
    print("\nDebate evolution:")
    for i, critique in enumerate(result.critiques):
        if critique.critic == "agent4debate" and "debate_summary" in critique.metadata:
            print(f"\nIteration {i + 1}: {critique.metadata['debate_summary']}")

    # Example with controversial text to see stronger debates
    print("\n" + "=" * 60 + "\n")
    print("Example 2: Text with clear trade-offs")
    print("-" * 40)

    controversial_text = """
    This report presents our findings concisely. We analyzed the data and found
    significant results. More details are available in the appendix.
    """

    print(f"Text: {controversial_text}")

    result2 = await improve(
        controversial_text, critics=["agent4debate"], max_iterations=1
    )

    # Show the key debate contention
    if result2.critiques and "debate_rounds" in result2.critiques[0].metadata:
        rounds = result2.critiques[0].metadata["debate_rounds"]
        if rounds and "key_contentions" in rounds[0]:
            print("\nKey debate contentions:")
            for contention in rounds[0]["key_contentions"]:
                print(f"  • {contention}")

    print(
        f"\nWinning approach: {result2.critiques[0].suggestions[0] if result2.critiques[0].suggestions else 'None'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
