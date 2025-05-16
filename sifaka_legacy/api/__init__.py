"""
Simplified API for Sifaka

This module provides a high-level, user-friendly API for working with the Sifaka
framework. It's designed to make common tasks simple while still allowing access
to the full power of Sifaka when needed.

Example usage:
```python
import sifaka

# Create a model
model = sifaka.model("openai", api_key="...")

# Create a chain
chain = (sifaka.Chain()
    .add_critic("prompt", instructions="Evaluate this text")
    .add_rule("length", max_length=500)
    .set_model(model))

# Run the chain
result = chain.run("Your input here")
```
"""

from .chain import Chain
from .factory import model, critic, rule, classifier, retriever
from .load import load_chain, load_model

__all__ = [
    "Chain",
    "model",
    "critic",
    "rule",
    "classifier",
    "retriever",
    "load_chain",
    "load_model",
]
