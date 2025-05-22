"""
Thought state container for Sifaka.

This module defines the Thought class, which serves as a central state container
for passing information between models, validators, and critics in the Sifaka framework.
It encapsulates all the context needed at each stage of the chain, including the prompt,
retrieved context, validation results, and critic feedback.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document retrieved from a retriever.

    This class represents a document retrieved from a retriever, including
    the document text, metadata, and relevance score.

    Attributes:
        text: The text content of the document.
        metadata: Optional metadata about the document.
        score: Optional relevance score of the document.
    """

    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class ValidationResult(BaseModel):
    """Result of a validation operation.

    This class represents the outcome of validating text against specific criteria.
    It includes information about whether the validation passed, any issues found,
    and suggestions for improvement.

    Attributes:
        passed: Whether the validation passed.
        message: A message describing the validation result.
        score: Optional score for the validation (0.0 to 1.0).
        issues: Optional list of issues found during validation.
        suggestions: Optional list of suggestions for improvement.
    """

    passed: bool
    message: str = ""
    score: Optional[float] = None
    issues: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None


class CriticFeedback(BaseModel):
    """Feedback from a critic.

    This class represents feedback from a critic, including issues identified,
    suggestions for improvement, and any additional metadata.

    Attributes:
        critic_name: The name of the critic providing the feedback.
        issues: List of issues identified by the critic.
        suggestions: List of suggestions for improvement.
        metadata: Optional metadata about the feedback.
    """

    critic_name: str
    issues: List[str]
    suggestions: List[str]
    metadata: Optional[Dict[str, Any]] = None


class Thought(BaseModel):
    """Central state container for Sifaka.

    This class serves as a central state container for passing information between
    models, validators, and critics in the Sifaka framework. It encapsulates all
    the context needed at each stage of the chain, including the prompt, retrieved
    context, validation results, and critic feedback.

    The Thought container is designed to be immutable, with new instances created
    at each stage of the chain to represent the updated state.

    Attributes:
        version: The version of the Thought schema.
        
        prompt: The original prompt or task.
        text: The generated text (output from the model).
        system_prompt: Optional system prompt used for generation.
        
        pre_generation_context: Optional list of documents retrieved before generation.
        post_generation_context: Optional list of documents retrieved after generation.
        
        validation_results: Optional dictionary of validation results.
        critique: Optional dictionary containing critique information.
        critic_feedback: Optional list of feedback from critics.
        
        history: Optional list of previous Thought instances.
        parent_id: Optional ID of the parent Thought.
        
        id: Unique identifier for this Thought.
        iteration: The iteration number in the chain.
        timestamp: The timestamp when the thought was created.
        chain_id: Optional identifier for the chain.
        metadata: Optional dictionary for additional metadata.
    """

    # Version information
    version: str = "1.0.0"
    
    # Core content
    prompt: str
    text: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # Retrieval context
    pre_generation_context: Optional[List[Document]] = None
    post_generation_context: Optional[List[Document]] = None
    
    # Validation and critique information
    validation_results: Optional[Dict[str, ValidationResult]] = None
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
        """Create a new Thought for the next iteration.

        This method creates a new Thought instance with the same attributes as this one,
        but with the iteration number incremented and a new timestamp.

        Returns:
            A new Thought instance for the next iteration.
        """
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

    def add_pre_generation_context(self, documents: List[Document]) -> "Thought":
        """Add pre-generation context to this thought.

        Args:
            documents: List of documents to add as pre-generation context.

        Returns:
            A new Thought instance with updated pre-generation context.
        """
        current_context = self.pre_generation_context or []
        return self.model_copy(update={"pre_generation_context": current_context + documents})

    def add_post_generation_context(self, documents: List[Document]) -> "Thought":
        """Add post-generation context to this thought.

        Args:
            documents: List of documents to add as post-generation context.

        Returns:
            A new Thought instance with updated post-generation context.
        """
        current_context = self.post_generation_context or []
        return self.model_copy(update={"post_generation_context": current_context + documents})

    def add_validation_result(self, name: str, result: ValidationResult) -> "Thought":
        """Add a validation result to this thought.

        Args:
            name: The name of the validator.
            result: The validation result.

        Returns:
            A new Thought instance with the added validation result.
        """
        results = dict(self.validation_results or {})
        results[name] = result
        return self.model_copy(update={"validation_results": results})

    def add_critic_feedback(self, feedback: CriticFeedback) -> "Thought":
        """Add critic feedback to this thought.

        Args:
            feedback: The critic feedback to add.

        Returns:
            A new Thought instance with the added critic feedback.
        """
        feedback_list = list(self.critic_feedback or [])
        feedback_list.append(feedback)
        return self.model_copy(update={"critic_feedback": feedback_list})

    def set_text(self, text: str) -> "Thought":
        """Set the generated text for this thought.

        Args:
            text: The generated text.

        Returns:
            A new Thought instance with the updated text.
        """
        return self.model_copy(update={"text": text})

    def set_critique(self, critique: Dict[str, Any]) -> "Thought":
        """Set the critique for this thought.

        Args:
            critique: The critique information.

        Returns:
            A new Thought instance with the updated critique.
        """
        return self.model_copy(update={"critique": critique})

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Thought to a dictionary.

        Returns:
            A dictionary representation of the Thought.
        """
        return self.model_dump(exclude={"history"})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        """Create a Thought from a dictionary.

        Args:
            data: Dictionary representation of a Thought.

        Returns:
            A new Thought instance.
        """
        # Handle version differences if needed
        version = data.get("version", "1.0.0")
        if version != cls.model_fields["version"].default:
            # In the future, implement version migration logic here
            pass
            
        return cls.model_validate(data)
