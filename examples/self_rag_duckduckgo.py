#!/usr/bin/env python
"""Example of Self-RAG critic using DuckDuckGo for fact-checking.

This example demonstrates:
1. How Self-RAG identifies factual claims that need verification
2. How DuckDuckGo search is used to fact-check those claims
3. How the corrected information is incorporated into the final text

Prerequisites:
- Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
- sifaka-tools is already installed in the sifaka venv
"""

import asyncio
import os
from sifaka import improve, Config
from sifaka.storage.file import FileStorage
from sifaka.tools import ToolRegistry


async def main() -> None:
    """Run Self-RAG with DuckDuckGo fact-checking."""

    # Check available tools (DuckDuckGo is auto-registered by sifaka-tools)
    print("🔧 Available tools:", ToolRegistry.list_available())
    print("✅ DuckDuckGo tool should be auto-registered as 'duckduckgo'")

    # Text with factual errors to demonstrate fact-checking
    text = """
    The Eiffel Tower, standing at 500 meters tall, is the tallest structure in Paris.
    Built in 1850 by Napoleon Bonaparte, it attracts over 100 million visitors annually.
    The tower is painted pink and requires repainting every 3 months to maintain its color.
    """

    print("\n📝 Original text (with factual errors):")
    print(text.strip())
    print()

    # Configure Sifaka with tools enabled
    config = Config(
        enable_tools=True,
        tool_timeout=15.0,  # Give DuckDuckGo time to search
        temperature=0.7,
        model="gpt-4o-mini",  # Or your preferred model
    )

    # Run Self-RAG improvement
    print("🚀 Running Self-RAG with DuckDuckGo fact-checking...")
    print("⏳ This may take 30-60 seconds as it searches for accurate information...")

    result = await improve(
        text,
        critics=["self_rag"],
        max_iterations=2,
        config=config,
        storage=FileStorage(),
    )

    print("\n✅ Fact-checked and corrected text:")
    print(result.final_text.strip())
    print("\n📊 Statistics:")
    print(f"   - Iterations: {result.iteration}")
    print(f"   - Processing time: {result.processing_time:.1f}s")
    print(f"   - Original length: {len(text.split())} words")
    print(f"   - Final length: {len(result.final_text.split())} words")

    # Show what Self-RAG identified and corrected
    print("\n🔍 Fact-checking summary:")
    for i, critique in enumerate(result.critiques):
        if critique.needs_improvement:
            print(f"\n   Iteration {i+1}:")
            # Extract key points from feedback
            feedback_lower = critique.feedback.lower()
            if "height" in feedback_lower or "500 meters" in feedback_lower:
                print("   • Height correction needed (actual: ~330 meters)")
            if "1850" in feedback_lower or "napoleon" in feedback_lower:
                print("   • Construction date/builder correction needed")
            if "100 million" in feedback_lower or "visitors" in feedback_lower:
                print("   • Visitor count correction needed")
            if "pink" in feedback_lower or "color" in feedback_lower:
                print("   • Color/painting schedule correction needed")

    # Note about tool usage
    print(
        "\n💡 Note: Self-RAG automatically uses available tools when it needs to verify facts."
    )


if __name__ == "__main__":
    print("🦆 Self-RAG with DuckDuckGo Fact-Checking Example")
    print("=" * 50)

    # Check for API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print(
            "❌ Error: Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable"
        )
        exit(1)

    # Run the example
    asyncio.run(main())
