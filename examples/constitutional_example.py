"""Constitutional example: Generator creates, critic evaluates against principles."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show constitutional AI critique of generated content."""
    
    # Generation task
    prompt = "Explain the benefits of artificial intelligence in healthcare"
    
    print("CONSTITUTIONAL EXAMPLE - Principled Critique")
    print("=" * 60)
    print(f"Generation task: {prompt}")
    print("\nConstitutional principles being checked:")
    print("- Clarity and understanding")
    print("- Accuracy and evidence")
    print("- Completeness")
    print("- Objectivity and balance")
    print("- Engagement")
    print()
    
    validators = [
        LengthValidator(min_length=150, max_length=600)
    ]
    
    result = await improve(
        prompt,
        critics=["constitutional"],
        validators=validators,
        max_iterations=3,
        model="gpt-4o-mini",  # Can use smaller model with good critics
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show how principles guide improvement
    print("\n\n📋 CONSTITUTIONAL REVIEW CYCLE:")
    print("=" * 60)
    
    for i, critique in enumerate(result.critiques):
        print(f"\n--- Iteration {i+1} ---")
        print(f"🏛️ Constitutional assessment:")
        print(f"  Overall: {critique.feedback[:100]}...")
        print(f"  Confidence: {critique.confidence}")
        
        # Show which principles needed work
        print(f"\n  Principles needing improvement:")
        for j, suggestion in enumerate(critique.suggestions, 1):
            print(f"  {j}. {suggestion}")
        
        if i < len(result.generations):
            gen = result.generations[i]
            print(f"\n  Generator response: {len(gen.text)} chars in {gen.processing_time:.1f}s")
    
    # Save thoughts
    thoughts = {
        "workflow": "constitutional critique cycle",
        "prompt": prompt,
        "critic": "constitutional",
        "principles": [
            "Clarity", "Accuracy", "Completeness", 
            "Objectivity", "Engagement", "Structure", "Appropriateness"
        ],
        "iterations": result.iteration,
        "final_text": result.final_text,
        "principle_violations": [
            {
                "iteration": i+1,
                "feedback": crit.feedback,
                "suggestions": crit.suggestions,
                "confidence": crit.confidence
            }
            for i, crit in enumerate(result.critiques)
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "constitutional_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ Constitutional review saved to thoughts/constitutional_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())