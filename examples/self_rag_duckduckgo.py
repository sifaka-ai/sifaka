#!/usr/bin/env python
"""Example of Self-RAG critic using DuckDuckGo for fact-checking.

This example demonstrates:
1. How Self-RAG identifies factual claims that need verification
2. How DuckDuckGo search is used to fact-check those claims
3. How the corrected information is incorporated into the final text

Prerequisites:
- Set GOOGLE_API_KEY environment variable (uses Gemini)
- sifaka-tools is already installed in the sifaka venv
"""

import asyncio
import os
from sifaka import improve, Config
from sifaka.storage.file import FileStorage
from sifaka.tools import ToolRegistry

# Import sifaka_tools to trigger tool registration


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

    # Configure Sifaka with tools enabled using Google Gemini
    config = Config(
        enable_tools=True,
        tool_timeout=15.0,  # Give DuckDuckGo time to search
        temperature=0.7,
        provider="google",
        model="gemini-1.5-flash",  # Fast Google model
        critic_model="gemini-1.5-flash",  # Same model for critics
        api_key=os.getenv("GOOGLE_API_KEY"),
        critic_tool_settings={"self_rag": {"enable_tools": True}},
    )

    # Run Self-RAG improvement
    print("🚀 Running Self-RAG with DuckDuckGo fact-checking...")
    print("🌐 Using Google Gemini 1.5 Flash for fast, accurate fact-checking")
    print("⏳ This may take 30-60 seconds as it searches for accurate information...")

    result = await improve(
        text,
        critics=["self_rag"],
        max_iterations=3,
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
    print("\n🔍 Fact-checking details:")
    for i, critique in enumerate(result.critiques):
        print(f"\n   Iteration {i+1}:")
        print(f"   Needs improvement: {critique.needs_improvement}")
        print(f"   Confidence: {critique.confidence}")
        if critique.tools_used:
            print(f"   Tools used: {len(critique.tools_used)} searches")
        print("\n   Feedback:")
        # Print full feedback without truncation
        feedback_lines = critique.feedback.split("\n")
        for line in feedback_lines:
            # Don't truncate lines - show full content
            print(f"   {line}")

    # Note about tool usage
    print(
        "\n💡 Note: Self-RAG automatically uses available tools when it needs to verify facts."
    )


if __name__ == "__main__":
    print("🦆 Self-RAG with DuckDuckGo Fact-Checking Example")
    print("=" * 50)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: Please set GOOGLE_API_KEY environment variable")
        print("   This example uses Google Gemini for fact-checking")
        exit(1)

    # Run the example
    asyncio.run(main())
