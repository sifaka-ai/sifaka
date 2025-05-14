from typing import Any, List

"""
Base module for critics.

This module provides the foundational components for the Sifaka critic framework,
including base classes, protocols, and type definitions for text validation,
improvement, and critiquing.

## Overview
The module serves as the core foundation for the critic system, defining interfaces
and base implementations that enable text validation, improvement, and critiquing
functionality. Critics work alongside rules to provide a complete validation and
improvement system.

## Components
1. **Protocols**
   - TextValidator: Interface for text validation
   - TextImprover: Interface for text improvement
   - TextCritic: Interface for text critiquing

2. **Base Classes**
   - BaseCritic: Abstract base class for critics
   - Critic: Concrete implementation of BaseCritic

3. **Data Models**
   - CriticMetadata: Metadata for critic results
   - CriticOutput: Output from critic operations

4. **Factory Functions**
   - create_critic: Factory function for creating critic instances
   - create_basic_critic: Factory function for creating basic text critics

## Usage Examples
```python
from sifaka.critics.base.abstract import BaseCritic
from sifaka.critics.base.implementation import Critic
from sifaka.critics.base.metadata import CriticMetadata
from sifaka.critics.base.factories import create_critic

# Create a custom critic
class MyCritic(BaseCritic[str]):
    def __init__(self, name: str, description: str, config=None) -> None:
        super().__init__(name, description, config)

    def validate(self, text: str) -> bool:
        return len(text) > 0

    def def improve(self, text: str, feedback: Optional[str] = None) -> str:
        return text.upper() if text else ""

    def critique(self, text: str) -> BaseResult:
        return BaseResult(
            passed=True,
            message="Good text",
            score=0.8,
            issues=[],
            suggestions=[]
        )

# Create and use the critic
critic = create_critic(
    MyCritic,
    name="my_critic",
    description="A custom critic implementation"
)
text = "This is a test."
is_valid = critic.validate(text) if critic else ""
improved = critic.improve(text) if critic else ""
feedback = critic.critique(text) if critic else ""
```

## Error Handling
The module implements comprehensive error handling for:
1. Input Validation
   - Empty text checks
   - Type validation
   - Format verification
   - Content validation

2. Processing Errors
   - Validation failures
   - Improvement errors
   - Critique failures
   - Resource errors

3. Recovery Strategies
   - Default values
   - Fallback methods
   - State preservation
   - Error logging
"""
from .protocols import TextValidator, TextImprover, TextCritic
from .metadata import CriticMetadata, CriticOutput, CriticResultEnum
from .abstract import BaseCritic
from .implementation import Critic
from .factories import create_critic, create_basic_critic

__all__: List[Any] = [
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "CriticMetadata",
    "CriticOutput",
    "CriticResultEnum",
    "BaseCritic",
    "Critic",
    "create_critic",
    "create_basic_critic",
]
