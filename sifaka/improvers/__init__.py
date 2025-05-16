"""
Improvers for enhancing LLM outputs.
"""

# Import improver implementations
# These imports register the improvers with the registry
from sifaka.improvers import clarity_improver

# Re-export the clarity improver for convenience
from sifaka.improvers.clarity_improver import ClarityImprover, create_clarity_improver

# Convenience function
def clarity(model, temperature=0.7, system_prompt=None):
    """Create a clarity improver.
    
    Args:
        model: The model to use for improvement.
        temperature: The temperature to use for generation.
        system_prompt: The system prompt to use for the model.
        
    Returns:
        A clarity improver.
    """
    return create_clarity_improver(model, temperature, system_prompt)

__all__ = [
    "ClarityImprover",
    "create_clarity_improver",
    "clarity",
]
