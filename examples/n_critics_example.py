"""N-Critics example: Multiple perspectives evaluate generated content."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show how multiple critic perspectives improve generation."""
    
    prompt = "Explain how renewable energy can address climate change"
    
    print("N-CRITICS EXAMPLE - Multi-Perspective Evaluation")
    print("=" * 60)
    print(f"Task: {prompt}")
    print("\nCritic perspectives:")
    print("- Technical accuracy expert")
    print("- General readability critic")
    print("- Structure and flow editor")
    print("- Subject matter specialist")
    print()
    
    validators = [
        LengthValidator(min_length=200, max_length=800)
    ]
    
    result = await improve(
        prompt,
        critics=["n_critics"],
        validators=validators,
        max_iterations=3,
        model="gpt-4o-mini",
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show multi-perspective evaluation
    print("\n\n👥 MULTI-PERSPECTIVE EVALUATION:")
    print("=" * 60)
    
    for i, critique in enumerate(result.critiques):
        print(f"\n--- Iteration {i+1} ---")
        print(f"🎭 Ensemble feedback (confidence: {critique.confidence}):")
        print(f"  {critique.feedback[:200]}...")
        
        print(f"\n  Consensus suggestions ({len(critique.suggestions)}):")
        for j, suggestion in enumerate(critique.suggestions[:3], 1):  # Show first 3
            print(f"  {j}. {suggestion}")
        
        if i < len(result.generations):
            gen = result.generations[i]
            print(f"\n  Generator response:")
            print(f"  - Produced {len(gen.text)} chars")
            print(f"  - Addressed {gen.suggestion_implementation['implementation_count'] if gen.suggestion_implementation else 0} suggestions")
    
    # Save thoughts
    thoughts = {
        "workflow": "multi-perspective critique",
        "prompt": prompt,
        "critic": "n_critics",
        "perspectives": [
            "Technical accuracy",
            "General readability",
            "Structure and flow",
            "Subject expertise"
        ],
        "iterations": result.iteration,
        "final_text": result.final_text,
        "ensemble_evaluations": [
            {
                "iteration": i+1,
                "consensus_feedback": crit.feedback,
                "suggestions": crit.suggestions,
                "ensemble_confidence": crit.confidence,
                "needs_improvement": crit.needs_improvement
            }
            for i, crit in enumerate(result.critiques)
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "n_critics_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ Multi-perspective evaluation saved to thoughts/n_critics_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())