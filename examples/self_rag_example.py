"""Self-RAG example: Retrieval-augmented critique for factual content."""

import asyncio
import json
from pathlib import Path
from sifaka import improve
from sifaka.validators import LengthValidator


async def main():
    """Show how Self-RAG identifies when facts/data are needed."""
    
    prompt = "Write about the economic impact of electric vehicles"
    
    print("SELF-RAG EXAMPLE - Fact-Checking & Evidence")
    print("=" * 60)
    print(f"Task: {prompt}")
    print("\nSelf-RAG checks for:")
    print("- Factual claims needing evidence")
    print("- Statistics requiring sources")
    print("- Statements needing data support")
    print()
    
    validators = [
        LengthValidator(min_length=150, max_length=600)
    ]
    
    result = await improve(
        prompt,
        critics=["self_rag"],
        validators=validators,
        max_iterations=3,
        model="gpt-4o-mini",
        show_improvement_prompt=True,
        force_improvements=True
    )
    
    print(f"\n📝 Final text ({len(result.final_text)} chars):")
    print(result.final_text)
    
    # Show RAG critique process
    print("\n\n🔍 RETRIEVAL-AUGMENTED CRITIQUE:")
    print("=" * 60)
    
    for i, critique in enumerate(result.critiques):
        print(f"\n--- Iteration {i+1} ---")
        print(f"📚 Evidence assessment:")
        print(f"  {critique.feedback[:150]}...")
        print(f"  Confidence: {critique.confidence}")
        
        print(f"\n  Data/evidence needed:")
        for j, suggestion in enumerate(critique.suggestions, 1):
            if any(keyword in suggestion.lower() for keyword in ['data', 'statistic', 'source', 'evidence', 'fact']):
                print(f"  ⚠️  {suggestion}")
            else:
                print(f"  {j}. {suggestion}")
        
        if i < len(result.generations):
            gen = result.generations[i]
            print(f"\n  Generator added:")
            print(f"  - Text length: {len(gen.text)} chars")
            if gen.suggestion_implementation:
                impl = gen.suggestion_implementation
                print(f"  - Evidence added: {impl['implementation_count']}/{len(impl['suggestions_given'])} suggestions")
    
    # Save thoughts
    thoughts = {
        "workflow": "retrieval-augmented critique",
        "prompt": prompt,
        "critic": "self_rag",
        "focus": "factual accuracy and evidence",
        "iterations": result.iteration,
        "final_text": result.final_text,
        "evidence_requirements": [
            {
                "iteration": i+1,
                "assessment": crit.feedback,
                "evidence_needed": [s for s in crit.suggestions if any(
                    k in s.lower() for k in ['data', 'statistic', 'source', 'evidence']
                )],
                "all_suggestions": crit.suggestions,
                "confidence": crit.confidence
            }
            for i, crit in enumerate(result.critiques)
        ]
    }
    
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)
    
    with open(thoughts_dir / "self_rag_workflow.json", "w") as f:
        json.dump(thoughts, f, indent=2)
    
    print(f"\n\n✅ RAG critique process saved to thoughts/self_rag_workflow.json")


if __name__ == "__main__":
    asyncio.run(main())