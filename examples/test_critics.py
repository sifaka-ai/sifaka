#!/usr/bin/env python3
"""Test script for the new Sifaka critics implementation.

This script demonstrates the new research-based critics with rich metadata
and structured output. It tests each critic individually to verify they
work correctly with the enhanced SifakaThought model.
"""

import asyncio
import os
from datetime import datetime

# Set up environment variables for API keys
os.environ.setdefault("OPENAI_API_KEY", "your-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "your-anthropic-key")
os.environ.setdefault("GEMINI_API_KEY", "your-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "your-groq-key")

from sifaka.core.thought import SifakaThought
from sifaka.critics import (
    ConstitutionalCritic,
    MetaRewardingCritic,
    NCriticsCritic,
    PromptCritic,
    ReflexionCritic,
    SelfConsistencyCritic,
    SelfRAGCritic,
    SelfRefineCritic,
)


async def test_critic(critic, critic_name: str, thought: SifakaThought):
    """Test a single critic and display results."""
    print(f"\n{'='*60}")
    print(f"Testing {critic_name}")
    print(f"{'='*60}")
    
    # Get critic info
    info = critic.get_info()
    print(f"Model: {info['model']}")
    print(f"Paper: {info['paper_reference']}")
    print(f"Methodology: {info['methodology']}")
    print(f"Has Retrieval: {info['has_retrieval']}")
    
    try:
        # Run the critic
        print(f"\nRunning {critic_name} critique...")
        await critic.critique_async(thought)
        
        # Get the latest critique
        critiques = thought.get_current_iteration_critiques()
        if critiques:
            latest = critiques[-1]
            print(f"\nResults:")
            print(f"Needs Improvement: {latest.needs_improvement}")
            print(f"Confidence: {latest.confidence}")
            print(f"Feedback: {latest.feedback[:200]}{'...' if len(latest.feedback) > 200 else ''}")
            print(f"Suggestions: {len(latest.suggestions)} suggestions")
            for i, suggestion in enumerate(latest.suggestions[:2], 1):
                print(f"  {i}. {suggestion}")
            print(f"Processing Time: {latest.processing_time_ms:.1f}ms")
            print(f"Methodology: {latest.methodology}")
            
            # Show critic-specific metadata
            if latest.critic_metadata:
                print(f"Critic Metadata: {latest.critic_metadata}")
        else:
            print("No critique results found!")
            
    except Exception as e:
        print(f"Error testing {critic_name}: {e}")


async def main():
    """Main test function."""
    print("Testing Sifaka Critics Implementation")
    print("=" * 60)
    
    # Create a test thought with some sample text
    thought = SifakaThought(
        prompt="Write a brief explanation of artificial intelligence and its applications.",
        max_iterations=3,
    )
    
    # Add some sample text to critique
    sample_text = """Artificial intelligence (AI) is a technology that makes computers smart. 
    It can do things like recognize faces, translate languages, and play games. 
    AI is used in many places like phones, cars, and websites. Some people think 
    AI will take over the world, but others think it will help solve problems."""
    
    thought.add_generation(sample_text, "test_generator", 0.8)
    
    print(f"Sample text to critique:")
    print(f'"{sample_text}"')
    print()
    
    # Test each critic
    critics_to_test = [
        (ReflexionCritic(model_name="openai:gpt-4o-mini"), "ReflexionCritic"),
        (ConstitutionalCritic(model_name="anthropic:claude-3-5-haiku-20241022"), "ConstitutionalCritic"),
        (SelfRefineCritic(model_name="gemini-1.5-flash"), "SelfRefineCritic"),
        (NCriticsCritic(model_name="groq:llama-3.1-8b-instant"), "NCriticsCritic"),
        (SelfConsistencyCritic(model_name="openai:gpt-3.5-turbo"), "SelfConsistencyCritic"),
        (PromptCritic(model_name="anthropic:claude-3-haiku-20240307"), "PromptCritic"),
        (MetaRewardingCritic(model_name="gemini-1.5-flash"), "MetaRewardingCritic"),
        (SelfRAGCritic(model_name="groq:mixtral-8x7b-32768"), "SelfRAGCritic"),
    ]
    
    for critic, name in critics_to_test:
        await test_critic(critic, name, thought)
        
        # Move to next iteration for the next critic
        thought.iteration += 1
    
    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}")
    
    # Show final thought summary
    print(f"\nFinal Thought Summary:")
    print(f"Total Critiques: {len(thought.critiques)}")
    print(f"Total Suggestions: {sum(len(c.suggestions) for c in thought.critiques)}")
    print(f"Techniques Applied: {thought.techniques_applied}")
    
    # Show critique metadata summary
    print(f"\nCritique Metadata Summary:")
    for critique in thought.critiques:
        print(f"- {critique.critic}: {critique.confidence:.2f} confidence, "
              f"{critique.processing_time_ms:.1f}ms, "
              f"{len(critique.suggestions)} suggestions")


if __name__ == "__main__":
    asyncio.run(main())
