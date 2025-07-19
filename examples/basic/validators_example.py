"""Example of using validators to ensure text quality.

Validators enforce specific requirements on the improved text.
"""

import asyncio
import os

from sifaka import improve
from sifaka.core.config import Config, LLMConfig, ValidationConfig
from sifaka.core.types import CriticType, ValidatorType
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

    # Configure with new structured config
    config = Config(
        llm=LLMConfig(
            model="gpt-4o-mini",
            temperature=0.8,  # Higher for creative tweets
        ),
        validation=ValidationConfig(
            validators=[ValidatorType.LENGTH, ValidatorType.CONTENT]
        ),
    )

    result1 = await improve(
        tweet_text,
        validators=[tweet_validator],
        max_iterations=3,
        config=config,
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

    blog_config = Config(llm=LLMConfig(model="gpt-4o-mini", temperature=0.7))

    result2 = await improve(
        blog_text,
        validators=[blog_validator],
        critics=[CriticType.SELF_REFINE],
        max_iterations=3,
        config=blog_config,
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

    academic_config = Config(
        llm=LLMConfig(
            model="gpt-4o-mini",
            temperature=0.5,  # Lower for academic writing
        )
    )

    result3 = await improve(
        abstract_text,
        validators=[academic_validator],
        critics=[CriticType.CONSTITUTIONAL, CriticType.SELF_REFINE],
        max_iterations=3,
        config=academic_config,
        storage=FileStorage(),
    )

    print(f"Original: {abstract_text}")
    print(f"Academic abstract: {result3.final_text[:100]}...")

    # Example 4: BATCH PROCESSING - New Feature!
    print("\n\n4️⃣ Batch Processing Example - Multiple Texts at Once")

    # Process multiple texts in parallel
    batch_texts = [
        "Social media changed everything.",
        "Machine learning is complex.",
        "Climate change requires action.",
    ]

    # Create a shared validator for all texts
    batch_validator = (
        Validator.create("social_post").length(50, 200).sentences(2, 4).build()
    )

    # Process all texts concurrently for better performance
    batch_tasks = [
        improve(
            text,
            validators=[batch_validator],
            critics=[CriticType.SELF_REFINE],
            max_iterations=2,
            config=config,  # Reuse the first config
        )
        for text in batch_texts
    ]

    # Wait for all improvements to complete
    batch_results = await asyncio.gather(*batch_tasks)

    print("Batch Results:")
    for i, (original, result) in enumerate(zip(batch_texts, batch_results)):
        print(f"  {i+1}. '{original}' → '{result.final_text}'")


async def batch_improve_demo():
    """Demonstrate advanced batch processing with different configs."""
    print("\n\n🚀 Advanced Batch Processing Demo")
    print("=" * 50)

    # Different text types requiring different processing
    mixed_content = {
        "tweet": "Check out our app",
        "email": "Thanks for your interest in our product",
        "blog": "AI is transforming industries",
    }

    # Different configs for different content types
    configs = {
        "tweet": Config(llm=LLMConfig(model="gpt-4o-mini", temperature=0.8)),
        "email": Config(llm=LLMConfig(model="gpt-4o-mini", temperature=0.6)),
        "blog": Config(llm=LLMConfig(model="gpt-4o-mini", temperature=0.7)),
    }

    # Different validators for each type
    validators = {
        "tweet": Validator.create("tweet").length(max_length=280).build(),
        "email": Validator.create("email").length(50, 500).build(),
        "blog": Validator.create("blog").length(200, 800).sentences(3, 15).build(),
    }

    # Process different content types with appropriate configs
    batch_tasks = [
        improve(
            text,
            validators=[validators[content_type]],
            critics=[CriticType.SELF_REFINE],
            max_iterations=2,
            config=configs[content_type],
        )
        for content_type, text in mixed_content.items()
    ]

    results = await asyncio.gather(*batch_tasks)

    print("Mixed Content Batch Results:")
    for (content_type, original), result in zip(mixed_content.items(), results):
        print(f"  {content_type.upper()}: '{original}' → '{result.final_text[:50]}...'")


if __name__ == "__main__":
    # Note: Works with GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY
    if not any(
        [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    ):
        print("❌ No API keys found. Please set at least one of:")
        print("   - GEMINI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - OPENAI_API_KEY")
    else:
        print("Running basic validator examples...")
        asyncio.run(main())

        print("\n" + "=" * 60)
        print("Running advanced batch processing demo...")
        asyncio.run(batch_improve_demo())
