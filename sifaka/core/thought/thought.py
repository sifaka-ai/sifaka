"""Thought state container for Sifaka."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from sifaka.core.thought.utils import create_thought_summary, parse_timestamp


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
    validator_name: Optional[str] = None  # Add validator_name for backward compatibility


class ToolCall(BaseModel):
    """Record of a tool call made during generation."""

    tool_name: str
    arguments: Dict[str, Any] = {}
    result: Optional[Any] = None
    timestamp: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


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
    model_name: Optional[str] = None  # Add model_name for backward compatibility
    pre_generation_context: Optional[List[Document]] = None
    post_generation_context: Optional[List[Document]] = None
    validation_results: Optional[Dict[str, ValidationResult]] = None
    critic_feedback: Optional[List[CriticFeedback]] = None
    tool_calls: Optional[List[ToolCall]] = None
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
            summary=create_thought_summary(self),
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
                # Preserve critic feedback from current iteration for next iteration's context
                "critic_feedback": self.critic_feedback,
                # Reset text and model_prompt for new iteration
                "text": None,
                "model_prompt": None,
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

    def add_validation_result(self, name_or_result, result=None) -> "Thought":
        """Add a validation result to this thought.

        Args:
            name_or_result: Either the name (str) or the ValidationResult object
            result: The ValidationResult object (if name_or_result is a string)

        Returns:
            A new Thought with the validation result added.
        """
        if isinstance(name_or_result, str) and result is not None:
            # Old signature: add_validation_result(name, result)
            name = name_or_result
            validation_result = result
        elif isinstance(name_or_result, ValidationResult) and result is None:
            # New signature: add_validation_result(result)
            validation_result = name_or_result
            name = validation_result.validator_name or "unknown_validator"
        else:
            raise ValueError("Invalid arguments to add_validation_result")

        results = dict(self.validation_results or {})
        results[name] = validation_result
        return self.model_copy(update={"validation_results": results})

    def add_critic_feedback(self, feedback: CriticFeedback) -> "Thought":
        """Add critic feedback to this thought."""
        feedback_list = list(self.critic_feedback or [])
        feedback_list.append(feedback)
        return self.model_copy(update={"critic_feedback": feedback_list})

    def add_tool_call(self, tool_call: ToolCall) -> "Thought":
        """Add a tool call record to this thought."""
        tool_calls_list = list(self.tool_calls or [])
        tool_calls_list.append(tool_call)
        return self.model_copy(update={"tool_calls": tool_calls_list})

    def set_text(self, text: str) -> "Thought":
        """Set the generated text for this thought."""
        return self.model_copy(update={"text": text})

    def set_model_prompt(self, model_prompt: str) -> "Thought":
        """Set the actual prompt sent to the model for this thought."""
        return self.model_copy(update={"model_prompt": model_prompt})

    def to_dict(self, exclude_prompt: bool = False) -> Dict[str, Any]:
        """Convert the Thought to a dictionary.

        Args:
            exclude_prompt: If True, exclude the original prompt field and only keep model_prompt.
        """
        data = self.model_dump()

        # Exclude original prompt if requested (keep only model_prompt)
        if exclude_prompt and "prompt" in data:
            del data["prompt"]

        # Convert datetime to string for JSON serialization
        if data.get("timestamp") and hasattr(data["timestamp"], "isoformat"):
            data["timestamp"] = data["timestamp"].isoformat()

        # Convert history timestamps to strings for JSON serialization
        if data.get("history"):
            for ref in data["history"]:
                if ref.get("timestamp") and hasattr(ref["timestamp"], "isoformat"):
                    ref["timestamp"] = ref["timestamp"].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        """Create a Thought from a dictionary."""
        version = data.get("version", "1.0.0")
        if version != "1.0.0":
            # Handle version differences if needed
            pass

        # Handle timestamp parsing
        if "timestamp" in data:
            data["timestamp"] = parse_timestamp(data["timestamp"])

        # Handle history timestamp parsing
        if "history" in data and data["history"]:
            for ref in data["history"]:
                if "timestamp" in ref:
                    ref["timestamp"] = parse_timestamp(ref["timestamp"])

        # Handle validation_results format conversion
        if "validation_results" in data and isinstance(data["validation_results"], list):
            # Convert list format to dict format
            validation_dict = {}
            for result_data in data["validation_results"]:
                if isinstance(result_data, dict):
                    validator_name = result_data.get("validator_name", "unknown_validator")
                    validation_dict[validator_name] = ValidationResult(**result_data)
            data["validation_results"] = validation_dict

        return cls.model_validate(data)
