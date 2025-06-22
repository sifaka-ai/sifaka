"""Meta-Rewarding critic example with thoughts output."""

import asyncio
import json
from pathlib import Path
from sifaka import improve


async def main():
    """Run Meta-Rewarding critic and save thoughts."""
    text = "Climate change is causing global temperatures to rise."
    
    print("META-REWARDING CRITIC EXAMPLE")
    print("=" * 60)
    print(f"Original: {text}")
    print()
    
    result = await improve(
        text,
        critics=["meta_rewarding"],
        max_iterations=2,
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\nFinal: {result.final_text}")
    
    # Save thoughts
    thoughts = {
        "critic": "meta_rewarding",
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
    
    with open(thoughts_dir / "meta_rewarding_thoughts.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n✅ Thoughts saved to thoughts/meta_rewarding_thoughts.json")


if __name__ == "__main__":
    asyncio.run(main())