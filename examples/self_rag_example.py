"""Self-RAG critic example with thoughts output."""

import asyncio
import json
from pathlib import Path
from sifaka import improve


async def main():
    """Run Self-RAG critic and save thoughts."""
    text = "The Amazon rainforest produces oxygen and stores carbon dioxide."
    
    print("SELF-RAG CRITIC EXAMPLE")
    print("=" * 60)
    print(f"Original: {text}")
    print()
    
    result = await improve(
        text,
        critics=["self_rag"],
        max_iterations=2,
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\nFinal: {result.final_text}")
    
    # Save thoughts
    thoughts = {
        "critic": "self_rag",
        "original": text,
        "final": result.final_text,
        "iterations": result.iteration,
        "critiques": [
            {
                "iteration": i+1,
                "feedback": c.feedback,
                "suggestions": c.suggestions,
                "confidence": c.confidence
            }
            for i, c in enumerate(result.critiques)
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "self_rag_thoughts.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n✅ Thoughts saved to thoughts/self_rag_thoughts.json")


if __name__ == "__main__":
    asyncio.run(main())