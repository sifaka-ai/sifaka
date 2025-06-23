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
    print("\nModels:")
    print(f"  Generator: gpt-4 (powerful, for generation)")
    print(f"  Critic: gpt-4o-mini (smaller, faster, for evaluation)")
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
        model="gpt-4",  # Generator model (larger)
        critic_model="gpt-4o-mini",  # Critic model (smaller, faster)
        temperature=0.7,  # Generator temperature
        critic_temperature=0.3,  # Critic temperature (lower for consistency)
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show the generation → critique → improvement cycle
    print("\n\n🔄 GENERATION CYCLE:")
    print("=" * 60)
    
    # Capture all prompts for thoughts
    all_prompts = []
    
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
        print(f"\n🔍 Critic feedback (model: gpt-4o-mini):")
        print(f"  {critique.feedback}")
        print(f"  Confidence: {critique.confidence}")
        print(f"  Suggestions: {len(critique.suggestions)}")
        
        # Show generation details
        print(f"\n✏️ Generator response (model: gpt-4):")
        print(f"  Text length: {len(generation.text)} chars")
        print(f"  Processing time: {generation.processing_time:.2f}s")
        
        if generation.suggestion_implementation:
            impl = generation.suggestion_implementation
            print(f"  Implemented: {impl['implementation_count']}/{len(impl['suggestions_given'])} suggestions")
        
        # Capture the improvement prompt
        if generation.prompt:
            all_prompts.append({
                "iteration": i+1,
                "type": "improvement",
                "prompt": generation.prompt
            })
    
    # Save complete thoughts with ALL prompts
    thoughts = {
        "workflow": "generate → critique → improve",
        "initial_prompt": prompt,
        "models": {
            "generator": "gpt-4",
            "critic": "gpt-4o-mini"
        },
        "temperatures": {
            "generator": 0.7,
            "critic": 0.3
        },
        "critic_type": "reflexion",
        "iterations": result.iteration,
        "final_text": result.final_text,
        "all_prompts": all_prompts,  # Include all the actual prompts
        "cycle": [
            {
                "iteration": i+1,
                "text_length": len(gen.text),
                "critique": {
                    "model": "gpt-4o-mini",
                    "feedback": crit.feedback,
                    "confidence": crit.confidence,
                    "suggestions": crit.suggestions
                },
                "generation": {
                    "model": "gpt-4",
                    "prompt": gen.prompt,  # Include the actual generation prompt
                    "text": gen.text[:200] + "..." if len(gen.text) > 200 else gen.text,
                    "processing_time": gen.processing_time
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
    print(f"   - Initial prompt: {prompt}")
    print(f"   - All improvement prompts: {len(all_prompts)}")
    print(f"   - Full cycle details: {result.iteration} iterations")


if __name__ == "__main__":
    asyncio.run(main())