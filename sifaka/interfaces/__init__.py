"""
Interfaces module for Sifaka.

IMPORTANT: This module is deprecated. All interfaces have been moved to their component-specific locations.

## New Import Locations

1. **Core Interfaces**
   - Import from `sifaka.core.interfaces`
   - Example: `from sifaka.core.interfaces import Component, Configurable`

2. **Model Interfaces**
   - Import from `sifaka.models.interfaces`
   - Example: `from sifaka.models.interfaces import ModelProviderProtocol`

3. **Rule Interfaces**
   - Import from `sifaka.rules.interfaces`
   - Example: `from sifaka.rules.interfaces import Rule, RuleProtocol`

4. **Critic Interfaces**
   - Import from `sifaka.critics.interfaces`
   - Example: `from sifaka.critics.interfaces import Critic, TextValidator`

5. **Chain Interfaces**
   - Import from `sifaka.interfaces.chain`
   - Example: `from sifaka.interfaces.chain import Chain`

6. **Retrieval Interfaces**
   - Import from `sifaka.interfaces.retrieval`
   - Example: `from sifaka.interfaces.retrieval import Retriever`

## Usage Example

```python
# Old import (deprecated)
# from sifaka.interfaces import Component, ModelProvider

# New imports
from sifaka.core.interfaces import Component, Configurable
from sifaka.models.interfaces import ModelProviderProtocol
from sifaka.rules.interfaces import Rule
from sifaka.critics.interfaces import Critic

# Check if a class implements an interface
if isinstance(my_object, Component):
    print("Object implements Component interface")

# Create a class that implements an interface
class MyModelProvider:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    # Implement other required methods...

# Check implementation
provider = MyModelProvider("my-provider")
assert isinstance(provider, ModelProviderProtocol)
```
"""

# Raise an ImportError to force users to update their imports
raise ImportError(
    "The sifaka.interfaces module is deprecated. "
    "Please import interfaces from their component-specific locations:\n"
    "- Core interfaces: from sifaka.core.interfaces import ...\n"
    "- Model interfaces: from sifaka.models.interfaces import ...\n"
    "- Rule interfaces: from sifaka.rules.interfaces import ...\n"
    "- Critic interfaces: from sifaka.critics.interfaces import ...\n"
    "- Chain interfaces: from sifaka.interfaces.chain import ...\n"
    "- Retrieval interfaces: from sifaka.interfaces.retrieval import ..."
)
