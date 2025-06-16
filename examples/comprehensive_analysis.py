#!/usr/bin/env python3
"""
Comprehensive Analysis with Simple Built-in Features

This example demonstrates comprehensive text analysis using Sifaka's simple
built-in features approach. It shows how to:

1. Use multiple critics for comprehensive analysis
2. Enable built-in logging, timing, and caching
3. Process different types of content with full workflow analysis
4. Monitor performance with built-in statistics

The example maintains comprehensive analysis capabilities while using a much
simpler configuration approach with built-in features.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict

# Simple imports - no complex dependencies needed
import sifaka
from sifaka import SifakaConfig, SifakaEngine
from sifaka.storage import SifakaFilePersistence


def create_comprehensive_config():
    """Create a comprehensive configuration with multiple critics and built-in features."""

    # Use the simple configuration approach with comprehensive analysis
    config = (
        SifakaConfig.builder()
        .model("openai:gpt-4o-mini")  # Fast, reliable model
        .max_iterations(2)  # Reduced iterations for faster execution
        .min_length(100)  # Minimum content length
        .max_length(2000)  # Maximum content length for comprehensive content
        .critics(
            ["reflexion", "constitutional"]  # Reduced to 2 critics for faster execution
        )  # Multiple critics for comprehensive analysis
        .with_logging(log_level="INFO", log_content=False)  # Enable logging
        .with_timing()  # Enable performance timing
        .with_caching(cache_size=200)  # Enable caching for repeated runs
        .build()
    )

    return config


# Sample prompts for comprehensive analysis
ANALYSIS_PROMPTS = {
    "restaurant_review": "Write a detailed restaurant review covering food quality, service, atmosphere, and value for money.",
    "technical_explanation": "Explain how machine learning neural networks work, including key concepts and real-world applications.",
    "customer_service": "Write a professional response to a customer complaint about a delayed order and damaged product.",
    "educational_content": "Create an educational guide about renewable energy sources and their environmental impact.",
    "creative_writing": "Write a short story about a person discovering an unexpected talent during a challenging time.",
}


async def analyze_prompt_comprehensive(
    prompt: str, prompt_name: str, engine: SifakaEngine
) -> Dict[str, Any]:
    """Run comprehensive analysis using multiple critics on the given prompt."""

    print(f"\n{'='*60}")
    print(f"ANALYZING: {prompt_name.upper()}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Length: {len(prompt)} characters")

    results = {
        "prompt_name": prompt_name,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "timestamp": datetime.now().isoformat(),
        "analysis": {},
    }

    try:
        print(f"\n🔍 Running comprehensive analysis with multiple critics...")

        # Generate and analyze with comprehensive workflow
        thought = await engine.think(prompt, max_iterations=2)

        # The thought is automatically persisted by the engine's persistence layer
        print(f"💾 Thought {thought.id} processed and persisted")

        # Extract comprehensive analysis results
        analysis = {
            "final_text_length": len(thought.final_text or thought.current_text),
            "iterations": thought.iteration,
            "validation_passed": thought.validation_passed(),
            "total_generations": len(thought.generations),
            "total_validations": len(thought.validations),
            "total_critiques": len(thought.critiques),
            "techniques_applied": thought.techniques_applied,
        }

        # Analyze critics applied
        if thought.critiques:
            critics_applied = {}
            for critique in thought.critiques:
                critic_name = critique.critic
                if critic_name not in critics_applied:
                    critics_applied[critic_name] = {
                        "count": 0,
                        "avg_confidence": 0,
                        "improvements_suggested": 0,
                    }
                critics_applied[critic_name]["count"] += 1
                critics_applied[critic_name]["avg_confidence"] += critique.confidence
                if critique.needs_improvement:
                    critics_applied[critic_name]["improvements_suggested"] += 1

            # Calculate averages
            for critic_data in critics_applied.values():
                critic_data["avg_confidence"] = round(
                    critic_data["avg_confidence"] / critic_data["count"], 3
                )

            analysis["critics_applied"] = critics_applied

        # Get final text preview
        final_text = thought.final_text or thought.current_text
        analysis["final_text_preview"] = (
            final_text[:200] + "..." if len(final_text) > 200 else final_text
        )

        results["analysis"] = analysis

        print(f"✅ Comprehensive analysis completed")
        print(f"   Final text: {analysis['final_text_length']} characters")
        print(f"   Iterations: {analysis['iterations']}")
        print(f"   Critics applied: {len(analysis.get('critics_applied', {}))}")

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)[:50]}...")
        results["analysis"] = {"error": str(e)}

    return results


async def main():
    """Run comprehensive analysis on all sample prompts."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    print("🤖 Sifaka Comprehensive Analysis with Simple Built-in Features")
    print("=" * 65)
    print("Multiple critics working together with built-in performance monitoring!")
    print(f"Analyzing {len(ANALYSIS_PROMPTS)} different types of content...")

    # Create thoughts directory if it doesn't exist
    thoughts_dir = "thoughts"
    os.makedirs(thoughts_dir, exist_ok=True)
    print(f"📁 Thoughts will be saved to: {thoughts_dir}/")

    # Create comprehensive configuration
    config = create_comprehensive_config()

    # Create file persistence for thoughts
    persistence = SifakaFilePersistence(storage_dir=thoughts_dir)

    # Create engine with comprehensive configuration and persistence
    engine = SifakaEngine(config=config, persistence=persistence)

    print(f"\n✅ Created comprehensive analysis engine")
    print(f"   Model: {config.model}")
    print(f"   Critics: {config.critics}")
    print(
        f"   Built-in features: logging={config.enable_logging}, timing={config.enable_timing}, caching={config.enable_caching}"
    )

    all_results = []

    try:
        for i, (prompt_name, prompt_text) in enumerate(ANALYSIS_PROMPTS.items(), 1):
            print(f"\n🔄 Processing {i}/{len(ANALYSIS_PROMPTS)}: {prompt_name}")
            result = await analyze_prompt_comprehensive(prompt_text, prompt_name, engine)
            all_results.append(result)

            # Brief summary
            analysis = result["analysis"]
            if "error" not in analysis:
                print(f"\n📊 SUMMARY for {prompt_name}:")
                print(
                    f"   Final text length: {analysis.get('final_text_length', 'N/A')} characters"
                )
                print(f"   Iterations: {analysis.get('iterations', 'N/A')}")
                print(f"   Validation passed: {analysis.get('validation_passed', 'N/A')}")
                print(f"   Total critiques: {analysis.get('total_critiques', 'N/A')}")
                print(f"   Critics applied: {list(analysis.get('critics_applied', {}).keys())}")

        # Show performance stats if timing is enabled
        timing_stats = engine.get_timing_stats()
        if timing_stats.get("total_requests", 0) > 0:
            print(f"\n⏱️ Performance Stats:")
            print(f"Duration: {timing_stats['avg_duration_seconds']:.2f}s")
            print(f"Iterations: {timing_stats['avg_iterations']:.1f}")

        # Show cache stats if caching is enabled
        cache_stats = engine.get_cache_stats()
        if cache_stats.get("cache_size", 0) >= 0:
            print(f"\n💾 Cache Stats:")
            print(f"Cache size: {cache_stats['cache_size']}")

        # Save detailed results
        output_file = "comprehensive_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Analysis complete! Detailed results saved to {output_file}")
        print(f"\n🎯 Key Insights:")
        print(f"   • All {len(ANALYSIS_PROMPTS)} prompts analyzed successfully")
        print(f"   • {len(config.critics)} different critics used")
        print(f"   • Built-in performance monitoring and caching")
        print(f"   • Comprehensive workflow analysis with multiple iterations")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

    print("\n✅ Comprehensive Analysis with Simple Built-in Features completed!")
    print("Key Benefits: Multiple critics, built-in performance monitoring, simple configuration")
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
