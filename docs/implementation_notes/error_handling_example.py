"""
Example implementation demonstrating the standardized error handling system.

This example shows how to use the error handling utilities and exception hierarchy
in different components of the Sifaka framework.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, PrivateAttr

from sifaka.utils.errors import (
    SifakaError,
    ValidationError,
    ConfigurationError,
    ModelError,
    ClassifierError,
    CriticError,
    format_error_metadata,
    handle_errors,
    with_error_handling,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager, ComponentState

# Get a logger for this module
logger = get_logger(__name__)


# Example Model Component
class ExampleModelState(ComponentState):
    """State for the example model."""
    
    model: Optional[Any] = None
    api_key_valid: bool = False


class ExampleModel(BaseModel):
    """Example model demonstrating error handling patterns."""
    
    model_name: str
    api_key: str
    
    # State management
    _state_manager = PrivateAttr(default_factory=lambda: StateManager(ExampleModelState))
    
    def __init__(self, **data: Any) -> None:
        """Initialize the model with error handling for configuration."""
        super().__init__(**data)
        
        # Validate configuration
        if not self.model_name:
            raise ConfigurationError("Model name cannot be empty")
        
        if not self.api_key:
            raise ConfigurationError("API key cannot be empty")
    
    def warm_up(self) -> None:
        """Initialize the model with error handling."""
        state = self._state_manager.get_state()
        
        if state.initialized:
            return
        
        with with_error_handling("model initialization", logger=logger):
            # Simulate API key validation
            if self.api_key == "invalid":
                raise ValidationError("Invalid API key")
            
            # Simulate model loading
            state.model = {"name": self.model_name}
            state.api_key_valid = True
            state.initialized = True
    
    @handle_errors(reraise=True, log_errors=True)
    def generate(self, prompt: str) -> str:
        """Generate text with error handling."""
        state = self._state_manager.get_state()
        
        # Check initialization
        if not state.initialized:
            try:
                self.warm_up()
            except Exception as e:
                raise ModelError(f"Failed to initialize model: {e}", cause=e)
        
        # Validate input
        if not prompt:
            raise ValidationError("Prompt cannot be empty")
        
        try:
            # Simulate text generation
            if "error" in prompt.lower():
                raise Exception("Simulated API error")
            
            # Return generated text
            return f"Generated text for: {prompt}"
        except Exception as e:
            # Re-raise with context
            raise ModelError(f"Error generating text with {self.model_name}: {e}", cause=e)


# Example Classifier Component
class ClassificationResult(BaseModel):
    """Result of a classification operation."""
    
    label: str
    confidence: float
    metadata: Dict[str, Any] = {}


class ExampleClassifier(BaseModel):
    """Example classifier demonstrating error handling patterns."""
    
    name: str
    threshold: float = 0.5
    
    # State management
    _state_manager = PrivateAttr(default_factory=lambda: StateManager(ComponentState))
    
    def classify(self, text: str) -> ClassificationResult:
        """Classify text with error handling."""
        state = self._state_manager.get_state()
        
        try:
            # Handle empty text
            if not text:
                return ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={"reason": "empty_input"},
                )
            
            # Simulate classification
            if "error" in text.lower():
                raise Exception("Simulated classification error")
            
            # Return classification result
            return ClassificationResult(
                label="positive" if "good" in text.lower() else "negative",
                confidence=0.8,
                metadata={"processed_successfully": True},
            )
        except Exception as e:
            # Log the error
            logger.error(f"Classification error: {e}")
            state.error = f"Failed to classify text: {e}"
            
            # Return a fallback result
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata=format_error_metadata(e),
            )


# Example Critic Component
class CriticMetadata(BaseModel):
    """Metadata for a critique operation."""
    
    score: float
    feedback: str
    issues: List[str] = []
    suggestions: List[str] = []


class ExampleCritic(BaseModel):
    """Example critic demonstrating error handling patterns."""
    
    name: str
    
    # State management
    _state_manager = PrivateAttr(default_factory=lambda: StateManager(ComponentState))
    
    def critique(self, text: str) -> CriticMetadata:
        """Critique text with error handling."""
        state = self._state_manager.get_state()
        
        try:
            # Handle empty text
            if not text:
                return CriticMetadata(
                    score=0.0,
                    feedback="Empty input",
                    issues=["Text cannot be empty"],
                )
            
            # Simulate critique
            if "error" in text.lower():
                raise Exception("Simulated critique error")
            
            # Return critique result
            return CriticMetadata(
                score=0.8,
                feedback="Good quality text",
                issues=[],
                suggestions=["Consider adding more details"],
            )
        except Exception as e:
            # Log the error
            logger.error(f"Critique error: {e}")
            state.error = f"Failed to critique text: {e}"
            
            # Return a fallback result
            return CriticMetadata(
                score=0.0,
                feedback=f"Error during critique: {str(e)}",
                issues=["Critique process failed"],
                suggestions=[],
            )


# Example usage
def main() -> None:
    """Demonstrate error handling in different components."""
    
    # Model example
    try:
        model = ExampleModel(model_name="gpt-4", api_key="valid_key")
        model.warm_up()
        
        # Generate text
        text = model.generate("Hello, world!")
        print(f"Generated text: {text}")
        
        # Generate text with error
        try:
            error_text = model.generate("Trigger an error")
            print(f"Generated text: {error_text}")
        except ModelError as e:
            print(f"Caught model error: {e}")
    except SifakaError as e:
        print(f"Caught Sifaka error: {e}")
    
    # Classifier example
    classifier = ExampleClassifier(name="sentiment")
    
    # Classify text
    result = classifier.classify("This is good text")
    print(f"Classification result: {result.label} ({result.confidence})")
    
    # Classify text with error
    error_result = classifier.classify("Trigger an error")
    print(f"Error classification result: {error_result.label} ({error_result.confidence})")
    print(f"Error metadata: {error_result.metadata}")
    
    # Critic example
    critic = ExampleCritic(name="quality")
    
    # Critique text
    critique = critic.critique("This is good text")
    print(f"Critique result: {critique.score} - {critique.feedback}")
    
    # Critique text with error
    error_critique = critic.critique("Trigger an error")
    print(f"Error critique result: {error_critique.score} - {error_critique.feedback}")
    print(f"Error issues: {error_critique.issues}")


if __name__ == "__main__":
    main()
