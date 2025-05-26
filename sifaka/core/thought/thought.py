"""Thought state container for Sifaka."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel


class Document(BaseModel):
    """A document retrieved from a retriever."""

    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    passed: bool
    message: str = ""
    score: Optional[float] = None
    issues: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None


class CriticFeedback(BaseModel):
    """Feedback from a critic."""

    critic_name: str
    feedback: str = ""  # Main feedback text
    needs_improvement: bool = False
    confidence: float = 0.0
    violations: List[str] = []
    suggestions: List[str] = []
    metadata: Dict[str, Any] = {}
    processing_time_ms: Optional[float] = None


class ThoughtReference(BaseModel):
    """Reference to a thought in the chain history."""

    thought_id: str
    iteration: int
    timestamp: datetime
    summary: Optional[str] = None


class Thought(BaseModel):
    """Central state container for Sifaka."""

    prompt: str
    text: Optional[str] = None
    system_prompt: Optional[str] = None
    model_prompt: Optional[str] = None
    pre_generation_context: Optional[List[Document]] = None
    post_generation_context: Optional[List[Document]] = None
    validation_results: Optional[Dict[str, ValidationResult]] = None
    critic_feedback: Optional[List[CriticFeedback]] = None
    history: Optional[List[ThoughtReference]] = None
    parent_id: Optional[str] = None
    id: str = ""
    iteration: int = 0
    timestamp: Optional[datetime] = None
    chain_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid4())
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.now()
        super().__init__(**data)

    def next_iteration(self) -> "Thought":
        """Create a new Thought for the next iteration."""
        current_ref = ThoughtReference(
            thought_id=self.id,
            iteration=self.iteration,
            timestamp=self.timestamp or datetime.now(),
            summary=(
                f"Iteration {self.iteration}: {len(self.text or '')} chars"
                if self.text
                else f"Iteration {self.iteration}"
            ),
        )

        new_history = [current_ref]
        if self.history:
            new_history.extend(self.history)

        return self.model_copy(
            update={
                "id": str(uuid4()),
                "iteration": self.iteration + 1,
                "timestamp": datetime.now(),
                "parent_id": self.id,
                "history": new_history,
            }
        )

    def add_pre_generation_context(self, documents: List[Document]) -> "Thought":
        """Add pre-generation context to this thought."""
        current_context = self.pre_generation_context or []
        return self.model_copy(update={"pre_generation_context": current_context + documents})

    def add_post_generation_context(self, documents: List[Document]) -> "Thought":
        """Add post-generation context to this thought."""
        current_context = self.post_generation_context or []
        return self.model_copy(update={"post_generation_context": current_context + documents})

    def add_validation_result(self, name: str, result: ValidationResult) -> "Thought":
        """Add a validation result to this thought."""
        results = dict(self.validation_results or {})
        results[name] = result
        return self.model_copy(update={"validation_results": results})

    def add_critic_feedback(self, feedback: CriticFeedback) -> "Thought":
        """Add critic feedback to this thought."""
        feedback_list = list(self.critic_feedback or [])
        feedback_list.append(feedback)
        return self.model_copy(update={"critic_feedback": feedback_list})

    def set_text(self, text: str) -> "Thought":
        """Set the generated text for this thought."""
        return self.model_copy(update={"text": text})

    def set_model_prompt(self, model_prompt: str) -> "Thought":
        """Set the actual prompt sent to the model for this thought."""
        return self.model_copy(update={"model_prompt": model_prompt})

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Thought to a dictionary."""
        return self.model_dump(exclude={"history"})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        """Create a Thought from a dictionary."""
        version = data.get("version", "1.0.0")
        if version != "1.0.0":
            # Handle version differences if needed
            pass
        return cls.model_validate(data)
