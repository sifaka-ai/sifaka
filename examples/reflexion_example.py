"""Reflexion example: Generator creates, critic evaluates, generator improves."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show the generate → critique → improve cycle with Reflexion."""
    
    # Start with a generation prompt rather than existing text
    prompt = "Write a brief introduction to machine learning"
    
    print("REFLEXION EXAMPLE - Generate → Critique → Improve")
    print("=" * 60)
    print(f"Generation prompt: {prompt}")
    print()
    
    # Use validators to ensure minimum quality
    validators = [
        LengthValidator(min_length=100, max_length=500)
    ]
    
    result = await improve(
        prompt,  # This will trigger initial generation
        critics=["reflexion"],
        validators=validators,
        max_iterations=3,
        model="gpt-4",  # Generator model
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show the generation → critique → improvement cycle
    print("\n\n🔄 GENERATION CYCLE:")
    print("=" * 60)
    
    for i, (critique, generation) in enumerate(zip(result.critiques, result.generations)):
        print(f"\n--- Iteration {i+1} ---")
        
        # Show validation results
        validations = [v for v in result.validations if v.validator == "LengthValidator"]
        if i < len(validations):
            val = validations[i]
            print(f"✓ Validation: {val.validator} - {'PASSED' if val.passed else 'FAILED'}")
            if not val.passed:
                print(f"  Details: {val.details}")
        
        # Show critique
        print(f"\n🔍 Critic feedback:")
        print(f"  {critique.feedback}")
        print(f"  Confidence: {critique.confidence}")
        print(f"  Suggestions: {len(critique.suggestions)}")
        
        # Show generation details
        print(f"\n✏️ Generator response:")
        print(f"  Model: {generation.model}")
        print(f"  Text length: {len(generation.text)} chars")
        print(f"  Processing time: {generation.processing_time:.2f}s")
        
        if generation.suggestion_implementation:
            impl = generation.suggestion_implementation
            print(f"  Implemented: {impl['implementation_count']}/{len(impl['suggestions_given'])} suggestions")
    
    # Save complete thoughts
    thoughts = {
        "workflow": "generate → critique → improve",
        "prompt": prompt,
        "critic": "reflexion",
        "generator_model": result.config.model if hasattr(result, 'config') else "gpt-4",
        "iterations": result.iteration,
        "final_text": result.final_text,
        "cycle": [
            {
                "iteration": i+1,
                "text_length": len(gen.text),
                "critique": {
                    "feedback": crit.feedback,
                    "confidence": crit.confidence,
                    "suggestions": crit.suggestions
                },
                "improvements_made": gen.suggestion_implementation['implementation_count'] if gen.suggestion_implementation else 0
            }
            for i, (crit, gen) in enumerate(zip(result.critiques, result.generations))
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "reflexion_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ Complete workflow saved to thoughts/reflexion_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())