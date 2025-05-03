"""Core critic implementation for Sifaka.

This module provides the core implementation of the critic system, which serves as
the central component for text validation, improvement, and critiquing. The CriticCore
class delegates specific operations to specialized components while maintaining
a unified interface for text processing.

## Factory Function Pattern

This module follows the standard Sifaka factory function pattern:

1. The `create_core_critic()` factory function creates a CriticCore instance:
   - Takes configuration parameters directly
   - Creates a CriticConfig object
   - Instantiates and returns a CriticCore

2. Factory function handles parameter extraction and validation:
   - Extract parameters from function arguments
   - Create configuration objects as needed
   - Instantiate and return the component

3. Factory function provides a consistent interface:
   - Clear parameter names matching component attributes
   - Consistent error handling
   - Standardized return types

## Component Lifecycle

### CriticCore Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Manager initialization
   - Service setup
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Text improvement
   - Text critiquing
   - Feedback processing

3. **Cleanup Phase**
   - Resource release
   - State cleanup
   - Error recovery

### Component Interactions

1. **Language Model Provider**
   - Handles text generation
   - Manages model responses
   - Controls model parameters

2. **Prompt Manager**
   - Creates specialized prompts
   - Manages prompt templates
   - Validates prompt formats

3. **Response Parser**
   - Parses model responses
   - Validates response formats
   - Extracts structured data

4. **Memory Manager**
   - Stores past interactions
   - Retrieves relevant context
   - Manages memory buffer

## Error Handling

1. **Input Validation Errors**
   - Empty or invalid text
   - Invalid feedback format
   - Invalid violation format

2. **Model Interaction Errors**
   - Provider connection failures
   - Response parsing errors
   - Format validation failures

3. **Resource Management Errors**
   - Memory allocation failures
   - Service initialization errors
   - Manager setup failures

## Examples

```python
from sifaka.critics.core import create_core_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a core critic using the factory function
critic = create_core_critic(
    name="core_critic",
    description="A core critic implementation",
    system_prompt="You are an expert editor...",
    temperature=0.7,
    max_tokens=1000,
    llm_provider=provider
)

# Validate text
text = "This is a sample technical document."
is_valid = critic.validate(text)
print(f"Text is valid: {is_valid}")

# Critique text
critique = critic.critique(text)
print(f"Score: {critique.score}")
print(f"Feedback: {critique.feedback}")

# Improve text
improved_text = critic.improve(text, "The text needs more detail.")
print(f"Improved text: {improved_text}")
```
"""

from typing import Any, Dict, List, Optional

from .base import BaseCritic
from .models import CriticConfig, CriticMetadata
from .managers.memory import MemoryManager
from .managers.prompt import DefaultPromptManager, PromptManager
from .managers.response import ResponseParser
from .services.critique import CritiqueService
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CriticCore(BaseCritic):
    """Core critic implementation that delegates to specialized components.

    This class serves as the central implementation of the critic system,
    providing a unified interface for text processing while delegating
    specific operations to specialized components.

    ## Lifecycle Management

    The CriticCore manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up managers
       - Creates services
       - Allocates resources

    2. **Operation**
       - Processes text input
       - Delegates to services
       - Manages responses
       - Handles errors

    3. **Cleanup**
       - Releases resources
       - Cleans up state
       - Logs final status

    ## Error Handling

    The CriticCore implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies violation format

    2. **Service Management**
       - Handles service errors
       - Manages resource allocation
       - Controls state transitions

    3. **Model Interaction**
       - Handles provider errors
       - Manages response parsing
       - Validates output formats

    ## Examples

    ```python
    from sifaka.critics.core import create_core_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a core critic using the factory function
    critic = create_core_critic(
        name="core_critic",
        description="A core critic implementation",
        system_prompt="You are an expert editor...",
        temperature=0.7,
        max_tokens=1000,
        llm_provider=OpenAIProvider(api_key="your-api-key")
    )

    # Use the critic
    text = "This is a sample technical document."
    is_valid = critic.validate(text)
    critique = critic.critique(text)
    improved_text = critic.improve(text, "The text needs more detail.")
    ```
    """

    def __init__(
        self,
        config: CriticConfig,
        llm_provider: Any,
        prompt_manager: Optional[PromptManager] = None,
        response_parser: Optional[ResponseParser] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """Initialize a CriticCore instance.

        This method sets up the core critic with its configuration and components.
        It creates or uses provided managers and services for text processing.

        ## Lifecycle Steps
        1. Configuration validation
        2. Manager initialization
        3. Service creation
        4. Resource allocation

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider
            prompt_manager: Optional prompt manager
            response_parser: Optional response parser
            memory_manager: Optional memory manager

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        super().__init__(config)

        # Create managers
        self._prompt_manager = prompt_manager or self._create_prompt_manager()
        self._response_parser = response_parser or ResponseParser()
        self._memory_manager = memory_manager

        # Create services
        self._critique_service = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._prompt_manager,
            response_parser=self._response_parser,
            memory_manager=self._memory_manager,
        )

        # Store the language model provider
        self._model = llm_provider

    def validate(self, text: str) -> bool:
        """Validate text against quality standards.

        This method checks if the given text meets the quality standards
        defined by the critic's configuration.

        ## Lifecycle Steps
        1. Input validation
        2. Service delegation
        3. Result processing

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        return self._critique_service.validate(text)

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on violations.

        This method enhances the given text by addressing the specified
        violations and applying improvements.

        ## Lifecycle Steps
        1. Input validation
        2. Violation processing
        3. Service delegation
        4. Result formatting

        Args:
            text: The text to improve
            violations: List of violations to address

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or violations are invalid
            RuntimeError: If improvement fails
        """
        return self._critique_service.improve(text, violations)

    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback.

        This method analyzes the given text and provides detailed feedback
        about its quality and potential improvements.

        ## Lifecycle Steps
        1. Input validation
        2. Service delegation
        3. Result processing
        4. Metadata creation

        Args:
            text: The text to critique

        Returns:
            CriticMetadata: Structured feedback about the text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        return self._critique_service.critique(text)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on feedback.

        This method enhances the given text using the provided feedback
        to guide the improvement process.

        ## Lifecycle Steps
        1. Input validation
        2. Feedback processing
        3. Service delegation
        4. Result formatting

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or feedback is invalid
            RuntimeError: If improvement fails
        """
        return self._critique_service.improve(text, feedback)

    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text against quality standards.

        This method performs asynchronous validation of text against
        the quality standards defined by the critic's configuration.

        ## Lifecycle Steps
        1. Input validation
        2. Service delegation
        3. Result processing

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        return await self._critique_service.avalidate(text)

    async def acritique(self, text: str) -> CriticMetadata:
        """Asynchronously critique text and provide feedback.

        This method performs asynchronous analysis of the given text
        and provides detailed feedback about its quality.

        ## Lifecycle Steps
        1. Input validation
        2. Service delegation
        3. Result processing
        4. Metadata creation

        Args:
            text: The text to critique

        Returns:
            CriticMetadata: Structured feedback about the text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        return await self._critique_service.acritique(text)

    async def aimprove(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Asynchronously improve text based on violations.

        This method performs asynchronous improvement of the given text
        by addressing the specified violations.

        ## Lifecycle Steps
        1. Input validation
        2. Violation processing
        3. Service delegation
        4. Result formatting

        Args:
            text: The text to improve
            violations: List of violations to address

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or violations are invalid
            RuntimeError: If improvement fails
        """
        return await self._critique_service.aimprove(text, violations)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on feedback.

        This method performs asynchronous improvement of the given text
        using the provided feedback to guide the process.

        ## Lifecycle Steps
        1. Input validation
        2. Feedback processing
        3. Service delegation
        4. Result formatting

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or feedback is invalid
            RuntimeError: If improvement fails
        """
        return await self._critique_service.aimprove(text, feedback)

    def _create_prompt_manager(self) -> PromptManager:
        """Create a default prompt manager.

        This method creates a default prompt manager instance using
        the critic's configuration.

        ## Lifecycle Steps
        1. Configuration validation
        2. Manager creation
        3. Resource allocation

        Returns:
            PromptManager: A configured prompt manager instance

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If manager creation fails
        """
        return DefaultPromptManager(self.config)


def create_core_critic(
    name: str,
    description: str,
    llm_provider: Any,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    prompt_manager: Optional[PromptManager] = None,
    response_parser: Optional[ResponseParser] = None,
    memory_manager: Optional[MemoryManager] = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a core critic.

    This factory function creates a configured CriticCore instance.
    It provides a standardized way to create critics with various configurations.

    ## Lifecycle Steps
    1. Parameter extraction
    2. Configuration creation
    3. Component initialization
    4. Critic instantiation

    Args:
        name: Name of the critic
        description: Description of what this critic does
        llm_provider: Language model provider to use
        system_prompt: System prompt for the language model
        temperature: Temperature for the language model
        max_tokens: Maximum tokens for the language model
        min_confidence: Minimum confidence threshold for valid critiques
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the critique result cache (0 to disable)
        priority: Priority of the critic (higher values = higher priority)
        cost: Computational cost of using this critic
        prompt_manager: Optional prompt manager to use
        response_parser: Optional response parser to use
        memory_manager: Optional memory manager to use
        **kwargs: Additional keyword arguments for the config

    Returns:
        CriticCore: The created critic
    """
    # Extract config parameters
    config_params = {}
    if system_prompt is not None:
        config_params["system_prompt"] = system_prompt
    if temperature is not None:
        config_params["temperature"] = temperature
    if max_tokens is not None:
        config_params["max_tokens"] = max_tokens
    if min_confidence is not None:
        config_params["min_confidence"] = min_confidence
    if max_attempts is not None:
        config_params["max_attempts"] = max_attempts
    if cache_size is not None:
        config_params["cache_size"] = cache_size
    if priority is not None:
        config_params["priority"] = priority
    if cost is not None:
        config_params["cost"] = cost

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = CriticConfig(
        name=name,
        description=description,
        **config_params,
    )

    # Create critic
    return CriticCore(
        config=config,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        response_parser=response_parser,
        memory_manager=memory_manager,
    )
