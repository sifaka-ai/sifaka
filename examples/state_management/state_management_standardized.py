"""
Example demonstrating standardized state management in Sifaka.

This example shows how to use the standardized state management pattern
across different components in Sifaka.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

from sifaka.utils.state import (
    StateManager,
    ClassifierState,
    RuleState,
    CriticState,
    ModelState,
    create_classifier_state,
    create_rule_state,
    create_critic_state,
    create_model_state,
)


class ExampleClassifier(BaseModel):
    """Example classifier using standardized state management."""

    # Configuration (immutable)
    name: str
    description: str
    labels: List[str]

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management
    _state: StateManager[ClassifierState] = PrivateAttr(default_factory=create_classifier_state)

    def warm_up(self) -> None:
        """Initialize the classifier if needed."""
        if not self._state.is_initialized:
            # Initialize state
            state = self._state.initialize()
            
            # Load model
            state.model = self._load_model()
            state.vectorizer = self._create_vectorizer()
            state.initialized = True
            
            print(f"Initialized classifier: {self.name}")

    def _load_model(self) -> Dict[str, Any]:
        """Load the model."""
        print(f"Loading model for {self.name}...")
        # Simulate model loading
        return {"type": "example_model", "labels": self.labels}

    def _create_vectorizer(self) -> Dict[str, Any]:
        """Create the vectorizer."""
        print(f"Creating vectorizer for {self.name}...")
        # Simulate vectorizer creation
        return {"type": "example_vectorizer", "max_features": 1000}

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text."""
        # Ensure initialized
        self.warm_up()
        
        # Get state
        state = self._state.get_state()
        
        # Check cache
        if text in state.cache:
            print(f"Cache hit for {text[:10]}...")
            return state.cache[text]
        
        # Simulate classification
        print(f"Classifying text: {text[:10]}...")
        result = {
            "label": self.labels[0],
            "confidence": 0.9,
            "metadata": {
                "model_type": state.model["type"],
                "vectorizer_type": state.vectorizer["type"],
            }
        }
        
        # Update cache
        state.cache[text] = result
        
        return result

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the classifier's state."""
        if not self._state.is_initialized:
            return {"initialized": False}
        
        state = self._state.get_state()
        return {
            "initialized": state.initialized,
            "model_loaded": state.model is not None,
            "vectorizer_loaded": state.vectorizer is not None,
            "cache_size": len(state.cache),
        }

    def reset(self) -> None:
        """Reset the classifier's state."""
        self._state.reset()
        print(f"Reset classifier: {self.name}")


class ExampleRule(BaseModel):
    """Example rule using standardized state management."""

    # Configuration (immutable)
    name: str
    description: str
    threshold: float

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management
    _state: StateManager[RuleState] = PrivateAttr(default_factory=create_rule_state)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        if not self._state.is_initialized:
            # Initialize state
            state = self._state.initialize()
            
            # Create validator
            state.validator = self._create_validator()
            state.compiled_patterns = self._compile_patterns()
            state.initialized = True
            
            print(f"Initialized rule: {self.name}")

    def _create_validator(self) -> Dict[str, Any]:
        """Create the validator."""
        print(f"Creating validator for {self.name}...")
        # Simulate validator creation
        return {"type": "example_validator", "threshold": self.threshold}

    def _compile_patterns(self) -> Dict[str, Any]:
        """Compile patterns."""
        print(f"Compiling patterns for {self.name}...")
        # Simulate pattern compilation
        return {"pattern1": "compiled_pattern1", "pattern2": "compiled_pattern2"}

    def validate(self, text: str) -> Dict[str, Any]:
        """Validate text."""
        # Ensure initialized
        self.warm_up()
        
        # Get state
        state = self._state.get_state()
        
        # Check cache
        if text in state.cache:
            print(f"Cache hit for {text[:10]}...")
            return state.cache[text]
        
        # Simulate validation
        print(f"Validating text: {text[:10]}...")
        result = {
            "valid": True,
            "score": 0.9,
            "metadata": {
                "validator_type": state.validator["type"],
                "patterns_used": list(state.compiled_patterns.keys()),
            }
        }
        
        # Update cache
        state.cache[text] = result
        
        return result

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the rule's state."""
        if not self._state.is_initialized:
            return {"initialized": False}
        
        state = self._state.get_state()
        return {
            "initialized": state.initialized,
            "validator_loaded": state.validator is not None,
            "patterns_compiled": len(state.compiled_patterns) > 0,
            "cache_size": len(state.cache),
        }

    def reset(self) -> None:
        """Reset the rule's state."""
        self._state.reset()
        print(f"Reset rule: {self.name}")


def demonstrate_classifier_state() -> None:
    """Demonstrate standardized state management for classifiers."""
    print("\n=== Classifier State Management ===")

    # Create a classifier
    classifier = ExampleClassifier(
        name="example_classifier",
        description="An example classifier",
        labels=["positive", "negative", "neutral"],
    )

    # Check initial state
    print(f"Initial state: {classifier.get_state_summary()}")

    # Classify text (triggers initialization)
    result = classifier.classify("This is an example text")
    print(f"Classification result: {result}")

    # Check state after initialization
    print(f"State after initialization: {classifier.get_state_summary()}")

    # Classify again (uses cache)
    result = classifier.classify("This is an example text")
    print(f"Classification result (cached): {result}")

    # Reset state
    classifier.reset()
    print(f"State after reset: {classifier.get_state_summary()}")


def demonstrate_rule_state() -> None:
    """Demonstrate standardized state management for rules."""
    print("\n=== Rule State Management ===")

    # Create a rule
    rule = ExampleRule(
        name="example_rule",
        description="An example rule",
        threshold=0.5,
    )

    # Check initial state
    print(f"Initial state: {rule.get_state_summary()}")

    # Validate text (triggers initialization)
    result = rule.validate("This is an example text")
    print(f"Validation result: {result}")

    # Check state after initialization
    print(f"State after initialization: {rule.get_state_summary()}")

    # Validate again (uses cache)
    result = rule.validate("This is an example text")
    print(f"Validation result (cached): {result}")

    # Reset state
    rule.reset()
    print(f"State after reset: {rule.get_state_summary()}")


def main() -> None:
    """Run the example."""
    print("Standardized State Management Examples")
    print("=====================================")

    demonstrate_classifier_state()
    demonstrate_rule_state()


if __name__ == "__main__":
    main()
