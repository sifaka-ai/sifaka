# Sifaka Architecture Enhancement: Shared Retrieval and State Container

## Current Challenges

1. **Retrieval Isolation**: Currently, retrievers are only available to critics but not to models, limiting the potential for context-aware generation.
2. **State Management**: Information flow between components (models, validators, critics) lacks a unified structure.
3. **Persistence**: No standardized way to persist the state of interactions for analysis or resumption.

## Proposed Solutions

### 1. Universal Retrieval Access

Make retrievers available to both models and critics to enhance context-aware capabilities throughout the chain.

### 2. Unified State Container ("Thought")

Design a central state container that passes through the entire chain, accumulating context at each stage.

### 3. Flexible Persistence Mechanisms

Implement various persistence strategies to support different use cases (memory, disk, databases).

## Detailed Design

### "Thought" State Container

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

class Document(BaseModel):
    """A document retrieved from a retriever."""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None

class CriticFeedback(BaseModel):
    """Feedback from a critic."""
    critic_name: str
    issues: List[str]
    suggestions: List[str]
    metadata: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    """Result of a validation operation."""
    passed: bool
    message: str = ""
    score: Optional[float] = None
    issues: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None

class Thought(BaseModel):
    """Central state container for Sifaka."""
    # Version information
    version: str = "1.0.0"

    # Core content
    prompt: str
    text: Optional[str] = None
    system_prompt: Optional[str] = None

    # Retrieval context
    pre_generation_context: Optional[List[Document]] = None
    post_generation_context: Optional[List[Document]] = None

    # Validation information
    validation_results: Optional[Dict[str, ValidationResult]] = None

    # Critic feedback
    critique: Optional[Dict[str, Any]] = None
    critic_feedback: Optional[List[CriticFeedback]] = None

    # History tracking
    history: Optional[List["Thought"]] = None
    parent_id: Optional[str] = None

    # Metadata
    id: str = Field(default_factory=lambda: str(uuid4()))
    iteration: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    chain_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow unknown fields for forward compatibility
    }

    def next_iteration(self) -> "Thought":
        """Create a new Thought for the next iteration."""
        # Create a copy without history to avoid circular references
        current_without_history = self.model_copy(update={"history": None})

        # Create the new thought
        new_thought = self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "timestamp": datetime.now(),
                "parent_id": self.id,
                "history": [current_without_history],
            }
        )

        # If we already have history, extend it
        if self.history:
            new_thought.history.extend(self.history)

        return new_thought
```

## History Management Approaches

There are two main approaches to managing the history of thoughts:

### 1. Nested History (Shown Above)

In this approach, each Thought contains a reference to previous Thoughts in its history:

**Pros:**
- Complete history is available at any point
- Easy to trace the evolution of a thought
- Simple to implement with Pydantic's recursive models

**Cons:**
- Can lead to deep nesting and large objects
- Potential for circular references if not careful
- Memory usage grows with each iteration

### 2. Flat History with References

In this approach, we maintain a separate history collection and use IDs to reference:

```python
# Chain maintains the history
thoughts: List[Thought] = []

# Each thought only references its parent
class Thought(BaseModel):
    parent_id: Optional[str] = None
    # ... other fields

    def get_history(self, thought_collection: List["Thought"]) -> List["Thought"]:
        """Retrieve history from the collection."""
        if not self.parent_id:
            return []

        history = []
        current_id = self.parent_id

        while current_id:
            parent = next((t for t in thought_collection if t.id == current_id), None)
            if parent:
                history.append(parent)
                current_id = parent.parent_id
            else:
                break

        return history
```

**Pros:**
- More memory efficient
- Avoids circular references
- Easier to persist and serialize

**Cons:**
- Requires external collection management
- History retrieval is more complex
- Potential for orphaned thoughts

## Implementation Plan

1. Define the core "Thought" Pydantic model with versioning
2. Implement history management strategy (nested)
3. Refactor retrievers to be accessible to both models and critics
4. Modify chain execution to pass and update the Thought container
5. Implement persistence mechanisms (JSON, Redis, Milvus)
6. Update existing components to work with the new state container

## Benefits

- **Improved Context Awareness**: Both models and critics can leverage retrieved information
- **Transparent Process**: Clear visibility into the state at each stage of the chain
- **Enhanced Debugging**: Complete history of interactions for troubleshooting
- **Flexible Persistence**: Support for various storage mechanisms
- **Agent Memory**: Foundation for implementing different memory types

## Open Questions

- How should we handle large retrieval contexts in the state container?
  - Consider lazy loading or reference-based approaches for large documents
  - Implement pagination or truncation strategies for very large contexts
  - Use streaming for large document collections

- How can we optimize performance when passing potentially large state objects?
  - Use reference passing instead of deep copying when possible
  - Implement lazy evaluation for expensive operations
  - Consider using a database for storing large objects and keeping references

- Should we implement a caching mechanism for thoughts?
  - Memory cache for recent thoughts
  - Persistent cache for long-running sessions
  - Distributed cache for multi-process applications