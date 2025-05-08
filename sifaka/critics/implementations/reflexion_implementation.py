"""
Implementation of a reflexion critic using composition over inheritance.

This module provides a critic implementation that uses language models with memory
to evaluate, validate, and improve text outputs based on past reflections. It follows
the composition over inheritance pattern.

## Component Lifecycle

### Reflexion Critic Implementation Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Memory initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Critique generation
   - Text improvement with reflections
   - Feedback processing
   - Memory management

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

## Examples

```python
from sifaka.critics.implementations.reflexion_implementation import ReflexionCriticImplementation
from sifaka.critics.base import CompositionCritic, create_composition_critic
from sifaka.critics.models import ReflexionCriticConfig
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a reflexion critic configuration
config = ReflexionCriticConfig(
    name="my_reflexion_critic",
    description="A critic that learns from past feedback",
    system_prompt="You are an expert editor who learns from past feedback.",
    memory_buffer_size=5,
    reflection_depth=2
)

# Create a reflexion critic implementation
implementation = ReflexionCriticImplementation(config, provider)

# Create a critic with the implementation
critic = create_composition_critic(
    name="my_reflexion_critic",
    description="A critic that learns from past feedback",
    implementation=implementation
)

# Use the critic
text = "This is a sample technical document."
is_valid = critic.validate(text)
improved = critic.improve(text, "Add more detail and structure.")
feedback = critic.critique(text)
```
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from ..models import ReflexionCriticConfig
from ..utils.state import CriticState

# Configure logging
logger = logging.getLogger(__name__)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert editor and writing coach who helps improve text quality.
Your task is to evaluate, critique, and improve text based on clarity, coherence, grammar, and effectiveness.
Provide specific, actionable feedback and suggestions for improvement.
You maintain a memory of past improvements and use these reflections to guide future improvements."""


class ReflexionCriticImplementation:
    """
    Implementation of a reflexion critic using language models with memory.

    This class implements the CriticImplementation protocol for a reflexion-based critic
    that uses language models and memory to evaluate, validate, and improve text.

    ## Lifecycle Management

    The ReflexionCriticImplementation manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes memory buffer
       - Allocates resources

    2. **Operation**
       - Validates text input
       - Generates critiques
       - Improves text quality with reflections
       - Processes feedback
       - Manages memory of past reflections

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Architecture

    The ReflexionCriticImplementation uses a memory-based architecture:

    1. **Memory Manager**: Stores and retrieves past reflections and feedback
    2. **Prompt Manager**: Creates prompts that incorporate past reflections
    3. **Response Parser**: Parses responses from language models
    4. **Critique Service**: Coordinates the critique and improvement process

    ## Error Handling

    The ReflexionCriticImplementation handles errors through:
    - Input validation
    - State validation
    - Exception propagation
    - Graceful degradation
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

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If provider setup fails
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

    def _check_input(self, text: str) -> None:
        """
        Check if input text is valid.

        Args:
            text: Text to check

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If state is not initialized
        """
        if not self._state.initialized:
            raise RuntimeError("ReflexionCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _format_feedback(self, feedback: Any) -> str:
        """
        Format feedback to string.

        Args:
            feedback: Feedback to format

        Returns:
            Formatted feedback string
        """
        if feedback is None:
            return ""
        elif isinstance(feedback, dict) and "feedback" in feedback:
            return str(feedback["feedback"])
        else:
            return str(feedback)

    def validate_impl(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: Text to validate

        Returns:
            True if text is valid, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
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

        Args:
            text: Text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If improvement fails
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

        Args:
            text: Text to critique

        Returns:
            Dictionary with critique information

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
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

        This method initializes any resources needed by the critic implementation.
        """
        if not self._state.initialized:
            # Create components if not already initialized
            from ..managers.prompt_factories import ReflexionCriticPromptManager
            from ..managers.response import ResponseParser
            from ..managers.memory import MemoryManager
            from ..services.critique import CritiqueService
            
            # Initialize components if not already done
            if not hasattr(self._state, "prompt_manager") or self._state.prompt_manager is None:
                self._state.prompt_manager = ReflexionCriticPromptManager(self.config)
            
            if not hasattr(self._state, "response_parser") or self._state.response_parser is None:
                self._state.response_parser = ResponseParser()
            
            if not hasattr(self._state, "memory_manager") or self._state.memory_manager is None:
                self._state.memory_manager = MemoryManager(buffer_size=self.config.memory_buffer_size)
            
            # Initialize critique service if not already done
            if "critique_service" not in self._state.cache:
                critique_service = CritiqueService(
                    model=self._state.model,
                    prompt_manager=self._state.prompt_manager,
                    response_parser=self._state.response_parser,
                    memory_manager=self._state.memory_manager,
                    config=self.config,
                )
                self._state.cache["critique_service"] = critique_service
            
            # Mark as initialized
            self._state.initialized = True
