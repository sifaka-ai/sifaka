"""Example of using validators to ensure text quality.

Validators enforce specific requirements on the improved text.
"""

import asyncio
import os

from sifaka import Config, improve
from sifaka.storage.file import FileStorage
from sifaka.validators.composable import Validator


async def main() -> None:
    """Run validator examples."""

    print("🎯 Validators Example - Quality control")
    print("=" * 50)

    # Example 1: Tweet validator
    print("\n1️⃣ Twitter Post Example")
    tweet_text = "Check out our new product"

    tweet_validator = (
        Validator.create("tweet")
        .length(max_length=280)
        .contains(["#", "@"], mode="any")
        .build()
    )

    result1 = await improve(
        tweet_text,
        validators=[tweet_validator],
        max_iterations=3,
        config=Config(
            model="gpt-4o-mini",
            critic_model="gpt-3.5-turbo",
            temperature=0.8,  # Higher for creative tweets
        ),
        storage=FileStorage(),
    )

    print(f"Original: {tweet_text}")
    print(f"Twitter-ready: {result1.final_text}")
    print(f"Length: {len(result1.final_text)} chars")

    # Example 2: Blog post validator
    print("\n\n2️⃣ Blog Post Example")
    blog_text = "AI is changing the world. It's important to understand how it works."

    blog_validator = (
        Validator.create("blog_post")
        .length(300, 1000)
        .sentences(5, 20)
        .contains(["example", "consider", "learn"], mode="any")
        .build()
    )

    result2 = await improve(
        blog_text,
        validators=[blog_validator],
        critics=["self_refine"],
        max_iterations=3,
        config=Config(
            model="gpt-4o-mini",
            critic_model="gpt-3.5-turbo",
            temperature=0.7,
        ),
        storage=FileStorage(),
    )

    print(f"Original: {blog_text} ({len(blog_text.split())} words)")
    print(
        f"Blog-ready: {result2.final_text[:150]}... ({len(result2.final_text.split())} words)"
    )

    # Example 3: Academic abstract
    print("\n\n3️⃣ Academic Abstract Example")
    abstract_text = "This paper presents research on AI."

    academic_validator = (
        Validator.create("abstract")
        .length(150, 300)
        .contains(["objective", "method", "results", "conclusion"], mode="any")
        .sentences(4, 8)
        .build()
    )

    result3 = await improve(
        abstract_text,
        validators=[academic_validator],
        critics=["constitutional", "self_refine"],
        max_iterations=3,
        config=Config(
            model="gpt-4o-mini",
            critic_model="gpt-3.5-turbo",
            temperature=0.5,  # Lower for academic writing
        ),
        storage=FileStorage(),
    )

    print(f"Original: {abstract_text}")
    print(f"Academic abstract: {result3.final_text[:100]}...")


if __name__ == "__main__":
    # Note: Works with GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY
    if not any(
        [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    ):
        print("❌ No API keys found. Please set at least one of:")
        print("   - GOOGLE_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - OPENAI_API_KEY")
    else:
        asyncio.run(main())
