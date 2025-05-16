"""
Sifaka: A framework for building reliable LLM applications.

Sifaka provides a simplified API for building complex LLM-powered applications
with a focus on reliability, composability, and ease of use.

Basic Usage:
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

For more advanced usage, see the documentation.
"""

import logging
import os

# Initialize logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import core components from the simplified API
from sifaka.api import (
    Chain,
    model,
    critic,
    rule,
    classifier,
    retriever,
    load_chain,
    load_model,
)

# Re-export the API components as top-level imports
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

# Automatically initialize the registry if the environment variable is not set
if os.environ.get("SIFAKA_SKIP_REGISTRY_INIT") != "true":
    from sifaka.core import initialize_registry

    initialize_registry.initialize_registry()

# Package metadata
__version__ = "0.1.0"
