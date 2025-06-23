"""Example of using Self-Consistency critic for consensus-based evaluation.

Self-Consistency runs multiple independent evaluations and builds consensus.
"""

import asyncio
from sifaka import improve

async def main():
    """Run Self-Consistency improvement example."""
    
    # Complex reasoning that benefits from multiple evaluation paths
    text = """
    To solve climate change, we need to stop using all fossil fuels immediately. 
    This is the only solution that will work. Countries that don't comply should 
    face severe economic sanctions until they switch to renewable energy.
    """
    
    print("Original text:")
    print(text)
    print("\n" + "="*80 + "\n")
    
    try:
        # Run improvement with Self-Consistency critic
        result = await improve(
            text,
            critics=["self_consistency"],
            max_iterations=3
        )
        
        print("Consensus-improved text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Show consensus building process
        print("\nConsensus analysis:")
        consistency_critiques = [c for c in result.critiques if c.critic == "self_consistency"]
        
        if consistency_critiques:
            # Group by iteration
            by_iteration = {}
            for critique in consistency_critiques:
                # Estimate iteration from position
                iteration = len([c for c in consistency_critiques if c.timestamp <= critique.timestamp])
                if iteration not in by_iteration:
                    by_iteration[iteration] = []
                by_iteration[iteration].append(critique)
            
            for iteration, critiques in by_iteration.items():
                print(f"\nIteration {iteration}:")
                avg_confidence = sum(c.confidence for c in critiques) / len(critiques)
                print(f"  Average confidence: {avg_confidence:.2f}")
                print(f"  Consensus: {'Strong' if avg_confidence > 0.7 else 'Moderate' if avg_confidence > 0.5 else 'Weak'}")
    
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())