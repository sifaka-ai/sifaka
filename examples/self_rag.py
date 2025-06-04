#!/usr/bin/env python3
"""
Enhanced Self-RAG Critic Demo

This example demonstrates the enhanced SelfRAGCritic that's closer to the original
Self-RAG paper by Asai et al. 2023. The enhanced version includes:

1. Structured reflection simulating original Self-RAG special tokens
2. Automatic retrieval execution when tools are available
3. More faithful implementation of the Self-RAG methodology

Key improvements over the basic version:
- Simulates [Retrieve], [IsRel], [IsSup], [IsUse] token decisions
- Provides structured reflection with specific retrieval queries
- Can execute retrieval automatically when tools are configured
- Better alignment with the original paper's methodology
"""

import asyncio
import os
from dotenv import load_dotenv

from sifaka.core.thought import SifakaThought
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.tools import create_web_search_tools

# Load environment variables
load_dotenv()


async def demo_enhanced_self_rag():
    """Demonstrate the enhanced Self-RAG critic with retrieval capabilities."""
    
    print("🔍 Enhanced Self-RAG Critic Demo")
    print("=" * 50)
    
    # Create web search tools for retrieval augmentation
    print("\n📡 Setting up retrieval tools...")
    try:
        # Try to create web search tools (requires optional dependencies)
        web_tools = create_web_search_tools(providers=["duckduckgo"])
        print(f"✅ Created {len(web_tools)} web search tools")
        has_tools = len(web_tools) > 0
    except Exception as e:
        print(f"⚠️  Could not create web tools: {e}")
        print("💡 Install with: pip install 'pydantic-ai-slim[duckduckgo]'")
        web_tools = []
        has_tools = False
    
    # Create enhanced Self-RAG critic
    print("\n🤖 Creating Enhanced Self-RAG Critic...")
    critic = SelfRAGCritic(
        model_name="openai:gpt-4o-mini",  # Use a reliable model
        retrieval_tools=web_tools,
        retrieval_focus_areas=[
            "Factual Accuracy: Verify specific claims and statistics",
            "Currency: Check if information is current and up-to-date", 
            "Evidence Quality: Assess need for authoritative sources",
            "Completeness: Identify missing critical information",
        ]
    )
    
    print(f"✅ Created critic with {len(critic.retrieval_tools)} retrieval tools")
    print(f"🔧 Auto-execute retrieval: {critic.auto_execute_retrieval}")
    
    # Test cases with different retrieval needs
    test_cases = [
        {
            "name": "Factual Claims Needing Verification",
            "prompt": "Write about recent developments in AI safety research",
            "text": """Recent AI safety research has made significant breakthroughs. 
            OpenAI released GPT-5 in early 2024 with revolutionary safety features. 
            The model achieved 99.9% alignment accuracy in testing. Google's DeepMind 
            also announced their new safety framework called 'SafeAI' which has been 
            adopted by over 500 companies worldwide."""
        },
        {
            "name": "Well-Supported Content",
            "prompt": "Explain basic principles of machine learning",
            "text": """Machine learning is a subset of artificial intelligence that enables 
            computers to learn and improve from experience without being explicitly 
            programmed. The three main types are supervised learning (learning from 
            labeled data), unsupervised learning (finding patterns in unlabeled data), 
            and reinforcement learning (learning through trial and error with rewards)."""
        },
        {
            "name": "Time-Sensitive Information",
            "prompt": "Provide current stock market information",
            "text": """The stock market has been volatile recently. The S&P 500 closed at 4,200 
            points yesterday, showing a 2.5% increase from last week. Tesla stock 
            reached $800 per share, while Apple maintained its position at $150. 
            These figures reflect the current market sentiment."""
        }
    ]
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Create thought
        thought = SifakaThought(
            prompt=test_case["prompt"],
            current_text=test_case["text"]
        )
        
        print(f"\n📝 Original Text:")
        print(f"'{test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}'")
        
        # Perform critique
        print(f"\n🔍 Performing Enhanced Self-RAG Analysis...")
        try:
            await critic.critique_async(thought)
            
            # Get the latest critique
            if thought.critiques:
                latest_critique = thought.critiques[-1]
                
                print(f"\n📊 Self-RAG Assessment:")
                print(f"   Needs Improvement: {latest_critique.needs_improvement}")
                print(f"   Confidence: {latest_critique.confidence:.2f}")
                print(f"   Tools Used: {latest_critique.tools_used or 'None'}")
                
                print(f"\n💬 Feedback:")
                print(f"   {latest_critique.feedback}")
                
                if latest_critique.suggestions:
                    print(f"\n💡 Suggestions:")
                    for j, suggestion in enumerate(latest_critique.suggestions, 1):
                        print(f"   {j}. {suggestion}")
                
                print(f"\n🔧 Methodology:")
                print(f"   {latest_critique.methodology}")
                
                # Show enhanced metadata
                if hasattr(latest_critique, 'critic_metadata'):
                    metadata = latest_critique.critic_metadata
                    if metadata.get('self_rag_enhanced'):
                        print(f"\n🚀 Enhanced Self-RAG Features:")
                        print(f"   Auto-execute retrieval: {metadata.get('auto_execute_retrieval', False)}")
                        print(f"   Retrieval tools available: {metadata.get('retrieval_tools_available', False)}")
                        print(f"   Factual assessment: {metadata.get('factual_assessment', 'unknown')}")
                
        except Exception as e:
            print(f"❌ Critique failed: {e}")
            print(f"💡 Make sure you have a valid OpenAI API key in your environment")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 Demo Summary")
    print(f"{'='*60}")
    print(f"✅ Enhanced Self-RAG critic demonstrated")
    print(f"🔧 Retrieval tools available: {has_tools}")
    print(f"🎯 Closer to original Self-RAG methodology")
    print(f"📊 Structured reflection with token simulation")
    
    if not has_tools:
        print(f"\n💡 To enable full retrieval capabilities:")
        print(f"   pip install 'pydantic-ai-slim[duckduckgo]'")
        print(f"   pip install 'pydantic-ai-slim[tavily]'")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_self_rag())
