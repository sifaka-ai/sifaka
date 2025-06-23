"""Example of using Meta-Rewarding critic with validators.

Meta-Rewarding uses two-stage judgment with meta-evaluation of evaluation quality.
"""

import asyncio
from sifaka import improve
from sifaka.validators import LengthValidator

async def main():
    """Run Meta-Rewarding improvement example with validators."""
    
    # Academic abstract that needs improvement
    text = """
    This paper is about AI. We studied how AI works. We found that AI is good 
    at some things. Our results show AI can be useful. More research is needed.
    """
    
    print("Original text:")
    print(text)
    print("\n" + "="*80 + "\n")
    
    try:
        # Run improvement with Meta-Rewarding critic and validators
        validators = [
            LengthValidator(min_length=150, max_length=250)
        ]
        
        result = await improve(
            text,
            critics=["meta_rewarding"],
            max_iterations=3,
            validators=validators
        )
        
        print("Improved text:")
        print(result.final_text)
        print(f"\nIterations: {result.iteration}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Show validation results
        print("\nValidation results:")
        for validation in result.validations:
            status = "✓ Passed" if validation.passed else "✗ Failed"
            print(f"{status} - {validation.validator}: {validation.details}")
        
        # Show meta-evaluation process
        print("\nMeta-evaluation insights:")
        for critique in result.critiques:
            if critique.critic == "meta_rewarding" and critique.metadata:
                if "meta_evaluation" in critique.metadata:
                    print(f"\n- Meta-evaluation: {critique.metadata['meta_evaluation']}")
                print(f"  Final confidence: {critique.confidence:.2f}")
    
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())