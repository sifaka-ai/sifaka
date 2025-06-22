"""Self-Consistency example: Multiple evaluations ensure consistent critique."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show how self-consistency uses multiple samples for reliable critique."""
    
    prompt = "Explain the concept of quantum computing to a general audience"
    
    print("SELF-CONSISTENCY EXAMPLE - Consensus Through Multiple Samples")
    print("=" * 60)
    print(f"Task: {prompt}")
    print("\nSelf-consistency approach:")
    print("- Generates multiple independent critiques")
    print("- Finds consensus among evaluations")
    print("- Provides high-confidence feedback")
    print()
    
    validators = [
        LengthValidator(min_length=150, max_length=500)
    ]
    
    result = await improve(
        prompt,
        critics=["self_consistency"],
        validators=validators,
        max_iterations=3,
        model="gpt-4o-mini",
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show consistency evaluation
    print("\n\n🎯 CONSISTENCY EVALUATION:")
    print("=" * 60)
    
    for i, critique in enumerate(result.critiques):
        print(f"\n--- Iteration {i+1} ---")
        
        # Self-consistency provides aggregated feedback
        print(f"📊 Consensus feedback (confidence: {critique.confidence}):")
        print(f"  {critique.feedback[:200]}...")
        
        # Show if it mentions consistency metrics
        if "consensus" in critique.feedback.lower() or "agreement" in critique.feedback.lower():
            print(f"\n  ✓ Multiple evaluations showed agreement")
        
        print(f"\n  Consensus suggestions ({len(critique.suggestions)}):")
        for j, suggestion in enumerate(critique.suggestions[:3], 1):
            print(f"  {j}. {suggestion}")
        
        if i < len(result.generations):
            gen = result.generations[i]
            print(f"\n  Generator response:")
            print(f"  - Text: {len(gen.text)} chars")
            print(f"  - Implemented: {gen.suggestion_implementation['implementation_count'] if gen.suggestion_implementation else 0} suggestions")
    
    # Save thoughts
    thoughts = {
        "workflow": "self-consistency evaluation",
        "prompt": prompt,
        "critic": "self_consistency",
        "approach": "multiple independent evaluations → consensus",
        "iterations": result.iteration,
        "final_text": result.final_text,
        "consensus_evaluations": [
            {
                "iteration": i+1,
                "consensus_feedback": crit.feedback,
                "consensus_suggestions": crit.suggestions,
                "consensus_confidence": crit.confidence,
                "evaluation_agreement": "high" if crit.confidence > 0.8 else "moderate"
            }
            for i, crit in enumerate(result.critiques)
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "self_consistency_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ Self-consistency evaluation saved to thoughts/self_consistency_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())