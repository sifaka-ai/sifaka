"""Self-Refine example: Iterative self-improvement with dimension analysis."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show self-refinement process across multiple dimensions."""
    
    prompt = "Write a paragraph about the importance of clean code"
    
    print("SELF-REFINE EXAMPLE - Multi-dimensional Improvement")
    print("=" * 60)
    print(f"Task: {prompt}")
    print("\nDimensions evaluated:")
    print("- Clarity and coherence")
    print("- Accuracy and correctness") 
    print("- Completeness and depth")
    print("- Engagement and readability")
    print("- Structure and organization")
    print()
    
    validators = [
        LengthValidator(min_length=100, max_length=400)
    ]
    
    result = await improve(
        prompt,
        critics=["self_refine"],
        validators=validators,
        max_iterations=3,
        model="gpt-4o-mini",
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final refined text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show refinement across dimensions
    print("\n\n🔄 SELF-REFINEMENT PROCESS:")
    print("=" * 60)
    
    for i, (critique, generation) in enumerate(zip(result.critiques, result.generations)):
        print(f"\n--- Refinement {i+1} ---")
        
        print(f"📊 Dimensional analysis:")
        print(f"  {critique.feedback[:150]}...")
        print(f"  Confidence: {critique.confidence}")
        
        print(f"\n  Areas for refinement:")
        for j, suggestion in enumerate(critique.suggestions, 1):
            print(f"  {j}. {suggestion}")
        
        print(f"\n  Refinement applied:")
        print(f"  - Text grew to {len(generation.text)} chars")
        print(f"  - Time: {generation.processing_time:.1f}s")
        
        if generation.suggestion_implementation:
            impl = generation.suggestion_implementation
            print(f"  - Addressed {impl['implementation_count']}/{len(impl['suggestions_given'])} suggestions")
    
    # Save thoughts
    thoughts = {
        "workflow": "self-refinement cycle",
        "prompt": prompt,
        "critic": "self_refine",
        "dimensions": [
            "clarity", "accuracy", "completeness",
            "engagement", "structure", "grammar", "relevance"
        ],
        "iterations": result.iteration,
        "final_text": result.final_text,
        "refinements": [
            {
                "iteration": i+1,
                "analysis": crit.feedback,
                "suggestions": crit.suggestions,
                "confidence": crit.confidence,
                "text_length": len(gen.text),
                "improvements_made": gen.suggestion_implementation['implementation_count'] 
                    if gen.suggestion_implementation else 0
            }
            for i, (crit, gen) in enumerate(zip(result.critiques, result.generations))
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "self_refine_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ Self-refinement process saved to thoughts/self_refine_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())