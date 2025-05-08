# Composition Over Inheritance Implementation Plan for Critics

This document outlines the plan for implementing the Composition Over Inheritance pattern in the Sifaka critic system.

## Current Architecture

Currently, the critic system uses inheritance:
- `BaseCritic` is an abstract base class that provides common functionality
- Specific critics like `PromptCritic`, `ReflexionCritic`, `SelfRefineCritic`, etc. inherit from `BaseCritic`
- Critics also implement protocols like `TextValidator`, `TextImprover`, and `TextCritic`
- The system uses composition for internal components (PromptManager, ResponseParser, MemoryManager)

## Target Architecture

We'll refactor the critic system to use composition over inheritance:

1. Create a `CriticImplementation` protocol that defines the core critic logic
2. Create a `Critic` class that delegates to a `CriticImplementation`
3. Create specific implementations like `PromptCriticImplementation` that follow the protocol
4. Update factory functions to create critics with their implementations

## Implementation Status

| Critic | Implementation Status | Factory Function Updated | Notes |
|--------|----------------------|-------------------------|-------|
| PromptCritic | ✅ Completed | ✅ Completed | Basic prompt-based critic |
| ReflexionCritic | ✅ Completed | ✅ Completed | Critic with memory for reflections |
| SelfRefineCritic | ✅ Completed | ✅ Completed | Self-improving critic |
| SelfRAGCritic | ✅ Completed | ✅ Completed | Retrieval-augmented critic |
| ConstitutionalCritic | ❌ Pending | ❌ Pending | Principle-based critic |
| LACCritic | ❌ Pending | ❌ Pending | LLM-based actor-critic |
| FeedbackCritic | ❌ Pending | ❌ Pending | Component of LACCritic |
| ValueCritic | ❌ Pending | ❌ Pending | Component of LACCritic |

## Implementation Steps

### 1. CriticImplementation Protocol

The `CriticImplementation` protocol has been added to `protocols.py`:

```python
@runtime_checkable
class CriticImplementation(Protocol):
    """
    Protocol for critic implementations.

    This protocol defines the core critic logic that can be composed with
    the Critic class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.
    """

    def validate_impl(self, text: str) -> bool: ...
    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str: ...
    def critique_impl(self, text: str) -> Dict[str, Any]: ...
    def warm_up_impl(self) -> None: ...
```

### 2. Critic Class

The `CompositionCritic` class has been added to `base.py`:

```python
class CompositionCritic(BaseModel, TextValidator, TextImprover, TextCritic):
    """
    Critic that uses composition over inheritance.

    This class delegates critic operations to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    name: str = Field(description="Name of the critic", min_length=1)
    description: str = Field(description="Description of the critic", min_length=1)
    config: CriticConfig = Field(description="Configuration for the critic")
    _implementation: CriticImplementation = PrivateAttr()
    _state_manager: StateManager[CriticState] = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        config: CriticConfig,
        implementation: CriticImplementation,
        **kwargs: Any,
    ):
        """Initialize the critic."""
        super().__init__(name=name, description=description, config=config, **kwargs)
        self._implementation = implementation

    def validate(self, text: str) -> bool:
        """Validate text against quality standards."""
        return self._implementation.validate_impl(text)

    def improve(self, text: str, feedback: Any = None) -> str:
        """Improve text based on feedback."""
        return self._implementation.improve_impl(text, feedback)

    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback."""
        result = self._implementation.critique_impl(text)
        return CriticMetadata(
            score=result["score"],
            feedback=result["feedback"],
            issues=result["issues"],
            suggestions=result["suggestions"],
        )

    def warm_up(self) -> None:
        """Warm up the critic."""
        self._implementation.warm_up_impl()
```

### 3. Implementation Plan for Each Critic

For each critic, we need to:

1. Create an implementation class that follows the `CriticImplementation` protocol
2. Move the core logic from the critic to the implementation class
3. Update the factory function to create a `Critic` with the implementation

#### 3.1 PromptCritic Implementation

The `PromptCriticImplementation` class has been implemented in `implementations/prompt_implementation.py`:

```python
class PromptCriticImplementation:
    """
    Implementation of a prompt critic using language models.

    This class implements the CriticImplementation protocol for a prompt-based critic
    that uses language models to evaluate, validate, and improve text.
    """

    def __init__(
        self,
        config: Union[CriticConfig, PromptCriticConfig],
        llm_provider: Any,
        prompt_factory: Any = None,
    ) -> None:
        """
        Initialize the prompt critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider
            prompt_factory: Optional prompt factory
        """
        self.config = config
        self._state = CriticState()

        # Create components
        from ..managers.prompt_factories import PromptCriticPromptManager
        from ..managers.response import ResponseParser
        from ..services.critique import CritiqueService
        from ..managers.memory import MemoryManager

        # Store components in state
        self._state.model = llm_provider
        self._state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
        self._state.response_parser = ResponseParser()
        self._state.memory_manager = MemoryManager(buffer_size=10)

        # Create service and store in state cache
        self._state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._state.prompt_manager,
            response_parser=self._state.response_parser,
            memory_manager=self._state.memory_manager,
        )

        # Mark as initialized
        self._state.initialized = True

    def validate_impl(self, text: str) -> bool:
        """
        Validate text against quality standards.
        """
        if not self._state.initialized:
            raise RuntimeError("PromptCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            return False

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str:
        """
        Improve text based on feedback.
        """
        if not self._state.initialized:
            raise RuntimeError("PromptCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if feedback is None:
            feedback = "Please improve this text for clarity and effectiveness."

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        improved_text = critique_service.improve(text, feedback)

        # Track improvement in memory manager
        memory_item = json.dumps(
            {
                "original_text": text,
                "feedback": feedback,
                "improved_text": improved_text,
                "timestamp": __import__("time").time(),
            }
        )
        self._state.memory_manager.add_to_memory(memory_item)

        return improved_text

    def critique_impl(self, text: str) -> Dict[str, Any]:
        """
        Critique text and provide feedback.
        """
        if not self._state.initialized:
            raise RuntimeError("PromptCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        critique = critique_service.critique(text)

        # Convert to dictionary format
        return {
            "score": critique.score,
            "feedback": critique.feedback,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
        }

    def warm_up_impl(self) -> None:
        """
        Warm up the critic implementation.
        """
        # Already initialized in __init__
        pass
```

#### 3.2 ReflexionCritic Implementation

The `ReflexionCriticImplementation` class has been implemented in `implementations/reflexion_implementation.py`:

```python
class ReflexionCriticImplementation:
    """
    Implementation of a reflexion critic using language models with memory.

    This class implements the CriticImplementation protocol for a reflexion-based critic
    that uses language models and memory to evaluate, validate, and improve text.
    """

    def __init__(
        self,
        config: ReflexionCriticConfig,
        llm_provider: Any,
        prompt_factory: Any = None,
    ) -> None:
        """
        Initialize the reflexion critic implementation.

        Args:
            config: Configuration for the reflexion critic
            llm_provider: Language model provider
            prompt_factory: Optional custom prompt factory
        """
        self.config = config
        self._state = CriticState()

        # Create components
        from ..managers.prompt_factories import ReflexionCriticPromptManager
        from ..managers.response import ResponseParser
        from ..managers.memory import MemoryManager
        from ..services.critique import CritiqueService

        # Store components in state
        self._state.model = llm_provider
        self._state.prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
        self._state.response_parser = ResponseParser()
        self._state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)

        # Initialize critique service
        critique_service = CritiqueService(
            model=llm_provider,
            prompt_manager=self._state.prompt_manager,
            response_parser=self._state.response_parser,
            memory_manager=self._state.memory_manager,
            config=config,
        )

        # Store in cache
        self._state.cache = {
            "critique_service": critique_service,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reflection_depth": config.reflection_depth,
        }

        # Mark as initialized
        self._state.initialized = True

    def validate_impl(self, text: str) -> bool:
        """
        Validate text against quality standards.
        """
        self._check_input(text)

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str:
        """
        Improve text based on feedback and reflections.
        """
        self._check_input(text)
        feedback_str = self._format_feedback(feedback)

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.improve(text, feedback_str)

    def critique_impl(self, text: str) -> Dict[str, Any]:
        """
        Critique text and provide feedback.
        """
        self._check_input(text)

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        critique = critique_service.critique(text)

        # Convert to dictionary format
        return {
            "score": critique.score,
            "feedback": critique.feedback,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
        }

    def warm_up_impl(self) -> None:
        """
        Warm up the critic implementation.
        """
        if not self._state.initialized:
            # Initialize components if not already done
            # ...
            self._state.initialized = True
```

### 4. Update Factory Functions

The `create_prompt_critic` factory function has been updated to use the composition pattern:

```python
def create_prompt_critic(
    llm_provider: LanguageModel,
    name: str = "factory_critic",
    description: str = "Evaluates and improves text using language models",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    config: Optional[PromptCriticConfig] = None,
    **kwargs: Any,
) -> "CompositionCritic":
    """
    Create a prompt critic with the given parameters.

    This factory function creates a prompt critic using the composition over inheritance pattern.
    It creates a PromptCriticImplementation and composes it with a CompositionCritic.
    """
    # Import here to avoid circular imports
    from .base import CompositionCritic, create_composition_critic
    from .implementations import PromptCriticImplementation

    # Create configuration if not provided
    if config is None:
        config = PromptCriticConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            params=kwargs,
        )

    # Create implementation
    implementation = PromptCriticImplementation(
        config=config,
        llm_provider=llm_provider,
    )

    # Create and return critic
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )
```

## Files Modified

1. `sifaka/critics/protocols.py`
   - Added `CriticImplementation` protocol

2. `sifaka/critics/base.py`
   - Added `CompositionCritic` class that uses composition
   - Added `create_composition_critic` factory function
   - Updated imports and exports

3. `sifaka/critics/implementations/` (new directory)
   - Created `__init__.py` for the new directory
   - Created `prompt_implementation.py` with `PromptCriticImplementation`
   - Created `reflexion_implementation.py` with `ReflexionCriticImplementation`
   - Created `self_refine_implementation.py` with `SelfRefineCriticImplementation`
   - Created `self_rag_implementation.py` with `SelfRAGCriticImplementation`

4. `sifaka/critics/factories.py`
   - Updated `create_prompt_critic` factory function to use composition
   - Updated `create_reflexion_critic` factory function to use composition
   - Added `create_self_refine_critic` factory function to use composition
   - Added `create_self_rag_critic` factory function to use composition

5. `sifaka/critics/__init__.py`
   - Updated imports and exports to include new components

## Implementation Strategy

1. Start with the PromptCritic as it's a commonly used critic
2. Implement one critic at a time, following the pattern:
   - Create the implementation class
   - Update the factory function
   - Test the changes
3. Update the implementation status in this document as each critic is completed

## Benefits

- Reduced complexity by avoiding deep inheritance hierarchies
- Improved flexibility by allowing components to be combined in different ways
- Better testability by enabling testing of implementations in isolation
- Reduced coupling between components
- More consistent API across all critic types
- Follows the same pattern used in classifiers, chains, and retrieval

## Next Steps After Implementation

1. Update tests to work with the new pattern:
   - Update test fixtures to create critics using the new pattern
   - Update assertions to work with the new critic structure

2. Update documentation to reflect the new pattern:
   - Update docstrings to describe the new architecture
   - Update examples to use the new factory functions
   - Add implementation notes about the Composition Over Inheritance pattern
