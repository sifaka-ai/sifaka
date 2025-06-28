import asyncio
from sifaka import improve, Config


text = """The Style critic helps transform text to match a specific writing style, oice, or tone by analyzing reference text and applying its characteristics.

This critic is useful for:
- Matching brand voice and tone
- Adapting content to specific audiences
- Maintaining consistent writing style across documents
- Emulating author styles or publication standards
"""

comments = """Implements N-Critics ensemble approach for comprehensive evaluation.
        When to Use This Critic:
        - ✅ Need multiple viewpoints on text quality
        - ✅ Want comprehensive evaluation across dimensions
        - ✅ Dealing with complex or multifaceted content
        - ✅ Need to identify blind spots single critics might miss
        - ❌ Quick single-dimension checks
        - ❌ When perspectives would be redundant
        - 🎯 Best for: Comprehensive reviews, final quality checks, complex documents
"""


async def main():
    result = await improve(
        text,
        critics=["style"],
        max_iterations=3,
        config=Config(
            model="gpt-4",
            critic_model="gpt-4o-mini",
            temperature=0.9,
            style_description="Code comment style",
            style_reference_text=comments,
            force_improvements=True,  # Ensure it makes changes
        ),
    )
    print(f"Original: {text}")
    print(f"Styled: {result.final_text}")


asyncio.run(main())
