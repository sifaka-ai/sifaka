"""Example of using N-Critics for multi-perspective ensemble evaluation.

N-Critics uses multiple critical perspectives to provide comprehensive feedback.
"""

import asyncio
from sifaka import improve

async def main():
    """Run N-Critics improvement example."""
    
    # Business proposal that needs multiple perspectives
    text = """
    We should invest all our money in cryptocurrency because it's the future. 
    Traditional investments are outdated. Bitcoin will definitely go to a million 
    dollars, so we can't lose. We should act fast before it's too late.
    """
    
    print("Original text:")
    print(text)
    print("\n" + "="*80 + "\n")
    
    try:
        # Run improvement with N-Critics for diverse perspectives
        result = await improve(
            text,
            critics=["n_critics"],
            max_iterations=3
        )
        
        print("Improved text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Show different perspectives from the ensemble
        print("\nEnsemble perspectives:")
        seen_feedback = set()
        for critique in result.critiques:
            if critique.critic == "n_critics" and critique.feedback not in seen_feedback:
                seen_feedback.add(critique.feedback)
                print(f"\n- Perspective: {critique.feedback[:100]}...")
                print(f"  Confidence: {critique.confidence:.2f}")
    
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())