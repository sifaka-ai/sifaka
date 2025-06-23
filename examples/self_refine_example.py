"""Example of using Self-Refine critic for iterative self-improvement.

Self-Refine focuses on quality improvements through iterative refinement.
"""

import asyncio
from sifaka import improve

async def main():
    """Run Self-Refine improvement example."""
    
    # Technical explanation that needs refinement
    text = """
    Quantum computers are different from regular computers. They use qubits instead 
    of bits. This makes them faster for some things. They might be useful for 
    breaking encryption and drug discovery.
    """
    
    print("Original text:")
    print(text)
    print("\n" + "="*80 + "\n")
    
    try:
        # Run improvement with Self-Refine critic
        result = await improve(
            text,
            critics=["self_refine"],
            max_iterations=3
        )
        
        print("Refined text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Show refinement progression
        print("\nRefinement process:")
        for i, generation in enumerate(result.generations):
            print(f"\nIteration {i + 1}:")
            print(f"  Length: {len(generation.text)} characters")
            if i > 0:
                prev_len = len(result.generations[i-1].text)
                change = len(generation.text) - prev_len
                print(f"  Change: {change:+d} characters")
    
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())