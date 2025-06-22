"""Reflexion critic example with thoughts output."""

import asyncio
import json
from pathlib import Path
from sifaka import improve


async def main():
    """Run Reflexion critic and save thoughts."""
    text = "Python is used for web development and data science applications."
    
    print("REFLEXION CRITIC EXAMPLE")
    print("=" * 60)
    print(f"Original: {text}")
    print()
    
    result = await improve(
        text,
        critics=["reflexion"],
        max_iterations=2,
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\nFinal: {result.final_text}")
    
    # Save thoughts
    thoughts = {
        "critic": "reflexion",
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
    
    with open(thoughts_dir / "reflexion_thoughts.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n✅ Thoughts saved to thoughts/reflexion_thoughts.json")


if __name__ == "__main__":
    asyncio.run(main())