"""Constitutional critic example with thoughts output."""

import asyncio
import json
from pathlib import Path
from sifaka import improve


async def main():
    """Run Constitutional critic and save thoughts."""
    text = "AI systems can analyze large datasets quickly and find patterns."
    
    print("CONSTITUTIONAL CRITIC EXAMPLE")
    print("=" * 60)
    print(f"Original: {text}")
    print()
    
    result = await improve(
        text,
        critics=["constitutional"],
        max_iterations=2,
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\nFinal: {result.final_text}")
    
    # Save thoughts
    thoughts = {
        "critic": "constitutional",
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
    
    with open(thoughts_dir / "constitutional_thoughts.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n✅ Thoughts saved to thoughts/constitutional_thoughts.json")


if __name__ == "__main__":
    asyncio.run(main())