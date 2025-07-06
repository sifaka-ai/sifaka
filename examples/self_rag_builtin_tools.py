#!/usr/bin/env python
"""Self-RAG example using built-in tools for fact-checking.

This example demonstrates how Self-RAG can use any registered tool
for fact-checking and information retrieval, not just DuckDuckGo.

The built-in tools include:
- web_search: General web search
- wikipedia: Wikipedia articles
- arxiv: Research papers
"""

import asyncio
from sifaka import improve, Config
from sifaka.tools import ToolRegistry
from sifaka.storage import FileStorage


async def main():
    """Demonstrate Self-RAG with different built-in tools."""
    
    print("🤖 Self-RAG with Built-in Tools")
    print("=" * 50)
    
    # Set up file storage for thoughts
    storage = FileStorage(storage_dir="./thoughts")
    
    # Show available tools
    print("\nAvailable tools:")
    for tool_name in ToolRegistry.list_available():
        print(f"  ✓ {tool_name}")
    
    # Example 1: General fact-checking with web search
    print("\n\n📝 Example 1: Fact-Checking with Web Search")
    print("-" * 50)
    
    text1 = """
    The Great Wall of China is approximately 5,000 miles long and was built
    in the 3rd century BC. It's visible from space with the naked eye.
    Millions of workers constructed it over several dynasties.
    """
    
    print("Original text (with errors):")
    print(text1.strip())
    
    result1 = await improve(
        text1,
        critics=["self_rag"],
        storage=storage,
        config=Config(temperature=0.7)
    )
    
    print("\nFact-checked version:")
    print(result1.final_text)
    
    print("\nTool usage:")
    for tool in result1.tools_used:
        print(f"  - {tool.tool_name}: {tool.status} ({tool.result_count} results)")
    
    # Example 2: Academic content with arXiv preference
    print("\n\n📚 Example 2: Academic Content with Research Papers")
    print("-" * 50)
    
    text2 = """
    BERT (Bidirectional Encoder Representations from Transformers) was introduced
    in 2017 and uses masked language modeling. It has 110 million parameters
    in its base configuration. BERT revolutionized NLP tasks.
    """
    
    print("Original text:")
    print(text2.strip())
    
    # For academic content, the critic should prefer arXiv
    result2 = await improve(
        text2,
        critics=["self_rag"],
        storage=storage,
        config=Config(temperature=0.7)
    )
    
    print("\nEnhanced with research context:")
    print(result2.final_text)
    
    # Example 3: Historical facts with Wikipedia
    print("\n\n📖 Example 3: Historical Facts")
    print("-" * 50)
    
    text3 = """
    The Roman Empire fell in 476 AD when the last emperor was deposed.
    At its peak, it covered most of Europe, North Africa, and the Middle East.
    Latin was the official language throughout the empire.
    """
    
    print("Original text:")
    print(text3.strip())
    
    result3 = await improve(
        text3,
        critics=["self_rag"],
        storage=storage,
        config=Config(temperature=0.7)
    )
    
    print("\nHistorically verified version:")
    print(result3.final_text)
    
    # Show comprehensive tool usage
    print("\n\n📊 Tool Usage Summary")
    print("-" * 50)
    
    all_tools_used = {}
    for result in [result1, result2, result3]:
        for tool in result.tools_used:
            if tool.status == "success":
                all_tools_used[tool.tool_name] = all_tools_used.get(tool.tool_name, 0) + 1
    
    print("Tools successfully used:")
    for tool_name, count in all_tools_used.items():
        print(f"  - {tool_name}: {count} times")
    
    # Demonstrate tool flexibility
    print("\n\n🔧 Tool Selection Logic")
    print("-" * 50)
    print("Self-RAG automatically selects appropriate tools:")
    print("  • General facts → web_search")
    print("  • Academic/research → arxiv (if available)")
    print("  • Historical/encyclopedic → wikipedia")
    print("  • Falls back to any available tool if preferred not found")


if __name__ == "__main__":
    asyncio.run(main())