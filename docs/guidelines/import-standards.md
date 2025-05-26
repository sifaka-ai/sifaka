# Sifaka Import Style Guidelines

This document establishes the official import style guidelines for the Sifaka codebase to ensure consistency, readability, and maintainability.

## 1. Import Hierarchy and Grouping

### 1.1 Import Order
Always organize imports in this order with blank lines between groups:

```python
# 1. Standard library imports
import asyncio
import logging
from typing import Any, Dict, List, Optional

# 2. Third-party imports
import pydantic
from pydantic import BaseModel

# 3. Sifaka imports (absolute paths only)
from sifaka import Chain, Thought, Model
from sifaka.core.interfaces import Validator, Critic
from sifaka.models import create_model
from sifaka.utils.logging import get_logger
```

### 1.2 Alphabetical Sorting
Within each group, sort imports alphabetically:

```python
# ✅ Good
import asyncio
import logging
from typing import Any, Dict

# ❌ Bad
import logging
import asyncio
from typing import Dict, Any
```

## 2. Sifaka-Specific Import Patterns

### 2.1 Public API Imports (Preferred)
Use package-level imports for core components:

```python
# ✅ Preferred - Clean and simple
from sifaka import Chain, Thought, Model, Validator, Critic, Retriever
```

### 2.2 Submodule Imports
Use submodule imports for specific implementations:

```python
# ✅ Good - Specific implementations
from sifaka.models import create_model
from sifaka.validators import LengthValidator, RegexValidator
from sifaka.critics import ReflexionCritic, SelfRefineCritic
from sifaka.storage import MemoryStorage, RedisStorage
```

### 2.3 Direct Module Imports (When Needed)
Use direct module imports for internal development or when you need specific classes:

```python
# ✅ Acceptable for internal development
from sifaka.core.chain.config import ChainConfig
from sifaka.core.thought.thought import Document, ValidationResult
```

## 3. Import Style Rules

### 3.1 Always Use Absolute Imports
Never use relative imports in the Sifaka codebase:

```python
# ✅ Good
from sifaka.core.interfaces import Model

# ❌ Bad
from ..interfaces import Model
from .interfaces import Model
```

### 3.2 Avoid Star Imports
Never use star imports:

```python
# ✅ Good
from sifaka.validators import LengthValidator, RegexValidator

# ❌ Bad
from sifaka.validators import *
```

### 3.3 Import What You Use
Only import what you actually use in the module:

```python
# ✅ Good - Only imports what's used
from sifaka import Chain
from sifaka.models import create_model

# ❌ Bad - Imports unused components
from sifaka import Chain, Thought, Model, Validator, Critic
from sifaka.models import create_model
```

## 4. Examples and Use Cases

### 4.1 Basic User Script
```python
#!/usr/bin/env python3
"""Example user script with proper imports."""

from sifaka import Chain
from sifaka.models import create_model
from sifaka.validators import LengthValidator

# Create and use components
model = create_model("openai:gpt-4")
validator = LengthValidator(min_length=50, max_length=500)
chain = Chain().with_model(model).validate_with(validator)
```

### 4.2 Internal Sifaka Module
```python
"""Internal module with comprehensive imports."""

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from sifaka.core.interfaces import Model, Validator
from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)
```

### 4.3 Example/Demo Script
```python
#!/usr/bin/env python3
"""Demo script showing clean imports."""

from sifaka import Chain
from sifaka.models import create_model
from sifaka.validators import LengthValidator, ContentValidator
from sifaka.critics import ReflexionCritic
from sifaka.storage import MemoryStorage
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
```

## 5. Import Guidelines by Module Type

### 5.1 Core Modules (`sifaka/core/`)
- Use absolute imports for all internal references
- Import interfaces from `sifaka.core.interfaces`
- Avoid importing from other core modules when possible

### 5.2 Implementation Modules (`sifaka/models/`, `sifaka/validators/`, etc.)
- Import interfaces from `sifaka.core.interfaces`
- Use factory patterns and avoid direct class imports in `__init__.py`
- Import utilities from `sifaka.utils`

### 5.3 Example Scripts (`examples/`)
- Use public API imports (`from sifaka import ...`)
- Demonstrate clean, user-friendly import patterns
- Group imports clearly for educational purposes

### 5.4 Test Files (`tests/`)
- Import what you're testing using the same patterns users would use
- Use public API imports when testing public functionality
- Use direct imports only when testing internal implementation details

## 6. Tools and Enforcement

### 6.1 Automated Formatting
Use these tools to enforce import style:

```bash
# Sort imports
isort --profile black --line-length 100 sifaka tests examples

# Format code
black --line-length 100 sifaka tests examples

# Lint imports
ruff check --select I sifaka tests examples
```

### 6.2 Pre-commit Configuration
Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black", "--line-length", "100"]

  - repo: https://github.com/psf/black
    rev: 24.2.0
    hooks:
      - id: black
        args: ["--line-length", "100"]
```

## 7. Migration Strategy

### 7.1 Updating Existing Code
When updating existing modules:

1. **Update imports first**: Change to the new style
2. **Test functionality**: Ensure everything still works
3. **Update documentation**: Reflect new import patterns
4. **Run formatting tools**: Apply automated formatting

### 7.2 Backward Compatibility
- All existing import patterns continue to work
- New code should use the preferred patterns
- Examples should be updated to demonstrate best practices

## 8. Common Patterns and Anti-Patterns

### 8.1 ✅ Good Patterns
```python
# Clean public API usage
from sifaka import Chain, Thought
from sifaka.models import create_model

# Specific implementations
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic

# Utilities
from sifaka.utils.logging import get_logger
```

### 8.2 ❌ Anti-Patterns
```python
# Don't mix import styles unnecessarily
from sifaka import Chain
from sifaka.core.chain import Chain  # Redundant

# Don't use overly specific imports for public API
from sifaka.core.chain.chain import Chain  # Too specific

# Don't import everything
from sifaka import *  # Never do this
```

---

These guidelines ensure that Sifaka maintains clean, consistent, and user-friendly import patterns while supporting both simple usage and advanced customization.
