"""Meta-Rewarding example: Critic evaluates its own critique quality."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show meta-level evaluation of critique quality."""
    
    prompt = "Describe the future of work in the age of AI"
    
    print("META-REWARDING EXAMPLE - Self-Evaluating Critique")
    print("=" * 60)
    print(f"Task: {prompt}")
    print("\nMeta-rewarding evaluates:")
    print("- Quality of its own critique")
    print("- Reliability of feedback")
    print("- Confidence in suggestions")
    print()
    
    validators = [
        LengthValidator(min_length=200, max_length=700)
    ]
    
    result = await improve(
        prompt,
        critics=["meta_rewarding"],
        validators=validators,
        max_iterations=3,
        model="gpt-4o-mini",
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show meta-evaluation process
    print("\n\n🔄 META-EVALUATION PROCESS:")
    print("=" * 60)
    
    for i, critique in enumerate(result.critiques):
        print(f"\n--- Iteration {i+1} ---")
        
        # Extract meta-assessment from feedback
        feedback_parts = critique.feedback.split("Meta-assessment:")
        main_feedback = feedback_parts[0].strip()
        meta_assessment = feedback_parts[1].strip() if len(feedback_parts) > 1 else "Not provided"
        
        print(f"📝 Primary critique:")
        print(f"  {main_feedback[:150]}...")
        
        print(f"\n🔍 Meta-assessment of critique quality:")
        print(f"  {meta_assessment[:200]}...")
        
        print(f"\n  Confidence: {critique.confidence}")
        print(f"  Suggestions: {len(critique.suggestions)}")
        
        if i < len(result.generations):
            gen = result.generations[i]
            print(f"\n  Generator response: {len(gen.text)} chars")
    
    # Save thoughts
    thoughts = {
        "workflow": "meta-rewarding critique",
        "prompt": prompt,
        "critic": "meta_rewarding",
        "meta_evaluation": True,
        "iterations": result.iteration,
        "final_text": result.final_text,
        "meta_critiques": [
            {
                "iteration": i+1,
                "primary_feedback": crit.feedback.split("Meta-assessment:")[0].strip() 
                    if "Meta-assessment:" in crit.feedback else crit.feedback,
                "meta_assessment": crit.feedback.split("Meta-assessment:")[1].strip() 
                    if "Meta-assessment:" in crit.feedback else "Not provided",
                "suggestions": crit.suggestions,
                "self_assessed_confidence": crit.confidence
            }
            for i, crit in enumerate(result.critiques)
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "meta_rewarding_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ Meta-evaluation process saved to thoughts/meta_rewarding_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())