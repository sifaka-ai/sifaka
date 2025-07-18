"""Example of using the Style critic to match writing styles.

This demonstrates how to transform text to match specific styles,
voices, and tones using reference text and style descriptions.
"""

import asyncio
import os

from sifaka import Config, improve
from sifaka.core.types import CriticType


async def casual_transformation() -> None:
    """Transform formal business text to casual style."""
    print("🎨 Formal → Casual Transformation")
    print("=" * 50)

    formal_text = """
    Dear Valued Customer,

    We wish to inform you that your recent order (#12345) has been
    successfully processed and will be dispatched within 2-3 business days.
    Should you require any further assistance, please do not hesitate to
    contact our customer service department.

    Sincerely,
    Customer Service Team
    """

    config = Config(
        style_description="""
        Very casual, friendly email style. Use contractions (you're, we've).
        Conversational tone like texting a friend. Short, punchy sentences.
        Informal greetings and sign-offs.
        """,
        style_examples=[
            "Hey there!",
            "Just wanted to let you know...",
            "You're all set!",
            "Hit us up if you need anything!",
            "Cheers!",
        ],
        temperature=0.9,  # Higher for more creative transformation
    )

    print("📧 Original formal email:")
    print(formal_text.strip())

    result = await improve(
        formal_text, critics=[CriticType.STYLE], max_iterations=3, config=config
    )

    print("\n📱 Casual version:")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")


async def technical_to_executive() -> None:
    """Adapt technical content for executive audience."""
    print("\n\n👔 Technical → Executive Summary")
    print("=" * 50)

    technical_text = """
    The API utilizes RESTful architecture with JSON payloads for request
    and response bodies. Authentication is handled via OAuth 2.0 with JWT
    tokens. Rate limiting is implemented at 1000 requests per hour per
    API key. The system supports both synchronous and asynchronous operations
    through webhook callbacks, with a 99.9% uptime SLA.
    """

    config = Config(
        style_description="""
        Executive-friendly language. Focus on business value and outcomes,
        not technical details. Use analogies. Short paragraphs. Clear benefits.
        Avoid jargon. Professional but accessible.
        """,
        style_reference_text="""
        Our platform seamlessly integrates with your existing tools, just like
        adding a new app to your smartphone. It's secure, reliable, and scales
        automatically as your business grows. Your team can start using it
        immediately, with no technical expertise required.
        """,
        temperature=0.8,
    )

    print("🔧 Technical version:")
    print(technical_text.strip())

    result = await improve(
        technical_text, critics=[CriticType.STYLE], max_iterations=2, config=config
    )

    print("\n💼 Executive version:")
    print(result.final_text.strip())


async def match_brand_voice() -> None:
    """Apply consistent brand voice to generic content."""
    print("\n\n🏢 Generic → Brand Voice")
    print("=" * 50)

    generic_text = """
    Our project management software helps teams work together. It includes
    task tracking, file sharing, and communication tools. Teams can see
    project progress and deadlines. The software works on all devices.
    """

    # Example: Playful, empowering brand voice (like Slack or Notion)
    config = Config(
        style_description="""
        Playful, empowering brand voice. Emphasize how users feel, not just
        features. Use "you" frequently. Show excitement. Make work sound fun.
        Confident but not arrogant. Casual professionalism.
        """,
        style_examples=[
            "Work doesn't have to feel like work",
            "You'll wonder how you ever managed without it",
            "Watch your team go from chaos to clarity",
            "Finally, a workspace as flexible as you are",
            "Get ready to actually enjoy Monday mornings",
        ],
        temperature=0.85,
    )

    print("📝 Generic description:")
    print(generic_text.strip())

    result = await improve(
        generic_text, critics=[CriticType.STYLE], max_iterations=3, config=config
    )

    print("\n✨ Brand voice version:")
    print(result.final_text.strip())


async def academic_to_blog() -> None:
    """Transform academic writing to engaging blog style."""
    print("\n\n📚 Academic → Blog Post")
    print("=" * 50)

    academic_text = """
    Recent studies have demonstrated that individuals who engage in regular
    physical exercise exhibit improved cognitive function across multiple
    domains. The neurobiological mechanisms underlying these improvements
    include increased neuroplasticity, enhanced neurotransmitter production,
    and improved cerebral blood flow.
    """

    config = Config(
        style_description="""
        Engaging blog style. Start with a hook or question. Use personal
        pronouns. Include relatable examples. Conversational but informative.
        Break up dense information. End with actionable advice.
        """,
        style_reference_text="""
        Ever wondered why you feel so sharp after a workout? Turns out,
        there's fascinating science behind that post-exercise mental clarity.
        Your brain literally changes when you move your body - and the
        benefits go way beyond just feeling good.
        """,
        temperature=0.9,
    )

    print("🎓 Academic text:")
    print(academic_text.strip())

    result = await improve(
        academic_text, critics=[CriticType.STYLE], max_iterations=3, config=config
    )

    print("\n📖 Blog version:")
    print(result.final_text.strip())


async def main() -> None:
    """Run all style transformation examples."""
    print("🎨 Sifaka Style Critic Examples")
    print("Demonstrating text transformation for different styles and audiences\n")

    await casual_transformation()
    await technical_to_executive()
    await match_brand_voice()
    await academic_to_blog()

    print("\n✅ All transformations complete!")
    print("\n💡 Style Critic Best Practices:")
    print("1. Provide clear style descriptions with specific guidance")
    print("2. Include 3-5 concrete examples of the target style")
    print("3. Use reference text when available")
    print("4. Higher temperature (0.8-0.9) for creative transformations")
    print("5. Run 2-3 iterations for best results")
    print("6. Combine with other critics (e.g., self_refine) for polish")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    else:
        asyncio.run(main())
