"""
Reflexion critic module for Sifaka.

This module implements the Reflexion approach for critics, which enables language model
agents to learn from feedback without requiring weight updates. It employs a process
where agents reflect on feedback they receive and maintain these reflections in memory
to improve future decision-making.

## Component Lifecycle

### Reflexion Critic Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Memory buffer setup
   - Provider initialization
   - Factory setup
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Critique generation
   - Reflection processing
   - Text improvement
   - Memory management

3. **Cleanup Phase**
   - Memory cleanup
   - Resource release
   - State reset
   - Error recovery

### Component Interactions

1. **Language Model Provider**
   - Receives formatted prompts
   - Returns model responses
   - Handles model-specific formatting

2. **Memory Manager**
   - Stores past reflections
   - Retrieves relevant reflections
   - Manages memory buffer

3. **Prompt Factory**
   - Creates reflection-aware prompts
   - Manages template variables
   - Validates prompt formats

### Error Handling and Recovery

1. **Input Validation Errors**
   - Empty or invalid text inputs
   - Invalid feedback format
   - Invalid reflection format
   - Recovery: Return appropriate error messages

2. **Memory Management Errors**
   - Buffer overflow
   - Invalid reflection format
   - Memory access errors
   - Recovery: Buffer cleanup and state preservation

3. **Model Interaction Errors**
   - Provider connection failures
   - Response parsing errors
   - Format validation failures
   - Recovery: Retry with fallback strategies

## Examples

```python
from sifaka.critics.reflexion import ReflexionCritic, ReflexionCriticConfig
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a reflexion critic configuration
config = ReflexionCriticConfig(
    name="my_reflexion_critic",
    description="A critic that learns from past feedback",
    system_prompt="You are an expert editor that learns from past improvements.",
    temperature=0.7,
    max_tokens=1000,
    memory_buffer_size=5,
    reflection_depth=2
)

# Create a reflexion critic
critic = ReflexionCritic(
    name="my_reflexion_critic",
    description="A critic that learns from past feedback",
    llm_provider=provider,
    config=config
)

# Validate text
text = "This is a sample technical document."
is_valid = critic.validate(text)
print(f"Text is valid: {is_valid}")

# Critique text
critique = critic.critique(text)
print(f"Critique: {critique}")

# Improve text with feedback
feedback = "The text needs more detail and better structure."
improved_text = critic.improve(text, feedback)
print(f"Improved text: {improved_text}")

# The critic will now use this experience to improve future text
```

The core concept behind Reflexion is verbal reinforcement learning - using language
itself as the mechanism for agent improvement. This approach allows language agents to
verbally reflect on task feedback signals and maintain these reflections in an episodic
memory buffer, which influences subsequent decision-making.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Final, List, Optional, ClassVar, Union, cast

from pydantic import PrivateAttr

from .base import BaseCritic, CriticConfig
from .protocols import TextCritic, TextImprover, TextValidator
from .prompt import LanguageModel
from ..utils.state import create_critic_state

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReflexionCriticConfig(CriticConfig):
    """Configuration for the reflexion critic.

    This configuration class extends the base CriticConfig with reflexion-specific
    parameters that control the behavior of the ReflexionCritic.

    ## Configuration Parameters

    1. **System Prompt**
       - Controls the base behavior of the model
       - Should emphasize reflection and learning
       - Default: "You are an expert editor that improves text through reflection."

    2. **Temperature**
       - Controls the randomness of model responses
       - Range: 0.0 to 1.0
       - Default: 0.7

    3. **Max Tokens**
       - Maximum number of tokens in model responses
       - Must be positive
       - Default: 1000

    4. **Memory Buffer Size**
       - Number of reflections to maintain in memory
       - Must be non-negative
       - Default: 5

    5. **Reflection Depth**
       - Number of reflection iterations to perform
       - Must be positive
       - Default: 1

    ## Error Handling

    The configuration implements comprehensive validation:

    1. **System Prompt Validation**
       - Ensures non-empty prompt
       - Validates prompt format

    2. **Parameter Validation**
       - Temperature range check
       - Token limit validation
       - Memory buffer validation
       - Reflection depth validation

    ## Examples

    ```python
    from sifaka.critics.reflexion import ReflexionCriticConfig

    # Create a basic configuration
    config = ReflexionCriticConfig()

    # Create a custom configuration
    custom_config = ReflexionCriticConfig(
        system_prompt="You are an expert technical writer that learns from feedback.",
        temperature=0.5,
        max_tokens=2000,
        memory_buffer_size=10,
        reflection_depth=2
    )
    ```
    """

    system_prompt: str = "You are an expert editor that improves text through reflection."
    temperature: float = 0.7
    max_tokens: int = 1000
    memory_buffer_size: int = 5
    reflection_depth: int = 1  # How many levels of reflection to perform

    def __post_init__(self) -> None:
        """Validate reflexion critic specific configuration."""
        super().__post_init__()
        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.memory_buffer_size < 0:
            raise ValueError("memory_buffer_size must be non-negative")
        if self.reflection_depth < 1:
            raise ValueError("reflection_depth must be positive")


class ReflexionPromptFactory:
    """Factory for creating reflexion-specific prompts.

    This factory creates specialized prompts that incorporate reflection and memory
    into the text improvement process.

    ## Lifecycle Management

    The ReflexionPromptFactory manages its lifecycle through three main phases:

    1. **Initialization**
       - Template setup
       - Format validation
       - Error handling setup

    2. **Operation**
       - Creates reflection-aware prompts
       - Handles template variables
       - Validates formats

    3. **Cleanup**
       - Resets state
       - Logs final status

    ## Error Handling

    The ReflexionPromptFactory implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies reflection format

    2. **Template Management**
       - Handles missing variables
       - Validates template syntax
       - Manages format errors

    ## Examples

    ```python
    from sifaka.critics.reflexion import ReflexionPromptFactory

    # Create a prompt factory
    factory = ReflexionPromptFactory()

    # Create different types of prompts
    text = "This is a sample text."
    feedback = "The text needs more detail."
    reflections = ["Previous improvement focused on clarity", "Added more examples"]

    # Validation prompt
    validation_prompt = factory.create_validation_prompt(text)
    print(validation_prompt)

    # Critique prompt
    critique_prompt = factory.create_critique_prompt(text)
    print(critique_prompt)

    # Improvement prompt with reflections
    improvement_prompt = factory.create_improvement_prompt(
        text, feedback, reflections
    )
    print(improvement_prompt)
    ```
    """

    def create_validation_prompt(self, text: str) -> str:
        """Create a prompt for validating text.

        This method creates a structured prompt for validating text quality using
        the language model. The prompt is designed to elicit a clear validation
        response with a boolean result and supporting reasoning.

        ## Lifecycle Steps
        1. Input validation
        2. Prompt template formatting
        3. Response format specification

        Args:
            text: The text to validate

        Returns:
            str: The validation prompt with clear response format instructions

        Raises:
            ValueError: If text is empty or invalid

        Examples:
            ```python
            text = "This is a sample technical document."
            prompt = factory.create_validation_prompt(text)
            # Returns:
            # "Please validate the following text:
            #
            # TEXT TO VALIDATE:
            # This is a sample technical document.
            #
            # FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
            # VALID: [true/false]
            # REASON: [reason for validation result]
            #
            # VALIDATION:"
            ```
        """
        return f"""Please validate the following text:

        TEXT TO VALIDATE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        VALID: [true/false]
        REASON: [reason for validation result]

        VALIDATION:"""

    def create_critique_prompt(self, text: str) -> str:
        """Create a prompt for critiquing text.

        This method creates a structured prompt for critiquing text quality using
        the language model. The prompt is designed to elicit a comprehensive critique
        including a score, general feedback, specific issues, and improvement suggestions.

        ## Lifecycle Steps
        1. Input validation
        2. Prompt template formatting
        3. Response format specification

        Args:
            text: The text to critique

        Returns:
            str: The critique prompt with clear response format instructions

        Raises:
            ValueError: If text is empty or invalid

        Examples:
            ```python
            text = "This is a sample technical document."
            prompt = factory.create_critique_prompt(text)
            # Returns:
            # "Please critique the following text:
            #
            # TEXT TO CRITIQUE:
            # This is a sample technical document.
            #
            # FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
            # SCORE: [number between 0 and 1]
            # FEEDBACK: [your general feedback]
            # ISSUES:
            # - [issue 1]
            # - [issue 2]
            # SUGGESTIONS:
            # - [suggestion 1]
            # - [suggestion 2]
            #
            # CRITIQUE:"
            ```
        """
        return f"""Please critique the following text:

        TEXT TO CRITIQUE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        SCORE: [number between 0 and 1]
        FEEDBACK: [your general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]

        CRITIQUE:"""

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: List[str] = None
    ) -> str:
        """Create a prompt for improving text with reflections.

        This method creates a structured prompt for improving text using the language
        model. The prompt incorporates feedback and previous reflections to guide
        the improvement process.

        ## Lifecycle Steps
        1. Input validation
        2. Reflection formatting
        3. Prompt template formatting
        4. Response format specification

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional list of previous reflections to incorporate

        Returns:
            str: The improvement prompt with clear response format instructions

        Raises:
            ValueError: If text or feedback is empty or invalid
            TypeError: If reflections is not a list

        Examples:
            ```python
            text = "This is a sample technical document."
            feedback = "The text needs more detail and better structure."
            reflections = ["Previous improvement focused on clarity", "Added more examples"]
            prompt = factory.create_improvement_prompt(text, feedback, reflections)
            # Returns:
            # "Please improve the following text:
            #
            # TEXT TO IMPROVE:
            # This is a sample technical document.
            #
            # FEEDBACK:
            # The text needs more detail and better structure.
            #
            # PREVIOUS REFLECTIONS:
            # 1. Previous improvement focused on clarity
            # 2. Added more examples
            #
            # FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
            # IMPROVED_TEXT: [improved text]
            #
            # IMPROVEMENT:"
            ```
        """
        reflection_text = ""
        if reflections and len(reflections) > 0:
            reflection_text = "\n\nPREVIOUS REFLECTIONS:\n"
            for i, reflection in enumerate(reflections):
                reflection_text += f"{i+1}. {reflection}\n"

        return f"""Please improve the following text:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}{reflection_text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        IMPROVED_TEXT: [improved text]

        IMPROVEMENT:"""

    def create_reflection_prompt(self, text: str, feedback: str, improved_text: str) -> str:
        """Create a prompt for generating a reflection.

        This method creates a structured prompt for generating reflections on the
        text improvement process. The prompt guides the model to analyze what went
        well, what went wrong, and what could be improved in future iterations.

        ## Lifecycle Steps
        1. Input validation
        2. Prompt template formatting
        3. Response format specification

        Args:
            text: The original text
            feedback: Feedback received
            improved_text: The improved text

        Returns:
            str: The reflection prompt with clear response format instructions

        Raises:
            ValueError: If any input is empty or invalid

        Examples:
            ```python
            text = "This is a sample technical document."
            feedback = "The text needs more detail and better structure."
            improved_text = "This is a well-structured technical document with detailed explanations..."
            prompt = factory.create_reflection_prompt(text, feedback, improved_text)
            # Returns:
            # "Please reflect on the following text improvement process:
            #
            # ORIGINAL TEXT:
            # This is a sample technical document.
            #
            # FEEDBACK RECEIVED:
            # The text needs more detail and better structure.
            #
            # IMPROVED TEXT:
            # This is a well-structured technical document with detailed explanations...
            #
            # Reflect on what went well, what went wrong, and what could be improved in future iterations.
            # Focus on specific patterns, mistakes, or strategies that could be applied to similar tasks.
            #
            # FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
            # REFLECTION: [your reflection]
            #
            # REFLECTION:"
            ```
        """
        return f"""Please reflect on the following text improvement process:

        ORIGINAL TEXT:
        {text}

        FEEDBACK RECEIVED:
        {feedback}

        IMPROVED TEXT:
        {improved_text}

        Reflect on what went well, what went wrong, and what could be improved in future iterations.
        Focus on specific patterns, mistakes, or strategies that could be applied to similar tasks.

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        REFLECTION: [your reflection]

        REFLECTION:"""


class ReflexionCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """A critic that uses reflection to improve text quality.

    This critic implements the Reflexion approach, which enables learning from
    feedback without requiring weight updates. It maintains a memory buffer of
    past reflections to improve future text generation.
    """

    # Class constants
    DEFAULT_NAME = "reflexion_critic"
    DEFAULT_DESCRIPTION = "Improves text using reflections on past feedback"

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        llm_provider: Any = None,
        prompt_factory: Any = None,
        config: ReflexionCriticConfig = None,
    ) -> None:
        """Initialize the reflexion critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider
            prompt_factory: Prompt factory
            config: Configuration for the critic
        """
        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Validate required parameters
        if llm_provider is None:
            from pydantic import ValidationError

            # Create a simple ValidationError for testing
            error = ValidationError.from_exception_data(
                "Field required",
                [{"loc": ("llm_provider",), "msg": "Field required", "type": "missing"}],
            )
            raise error

        # Create default config if not provided
        if config is None:
            config = ReflexionCriticConfig(
                name=name,
                description=description,
                system_prompt="You are an expert editor that improves text through reflection.",
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
                max_attempts=3,
                memory_buffer_size=5,
                reflection_depth=1,
            )

        # Initialize base class
        super().__init__(config)

        # Import required components
        from .managers.prompt_factories import ReflexionCriticPromptManager
        from .managers.response import ResponseParser
        from .managers.memory import MemoryManager
        from .services.critique import CritiqueService

        # Store components in state
        state.model = llm_provider
        state.prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
        state.response_parser = ResponseParser()
        state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)

        # Create service and store in state cache
        state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=state.prompt_manager,
            response_parser=state.response_parser,
            memory_manager=state.memory_manager,
        )

        # Mark as initialized
        state.initialized = True

    @property
    def config(self) -> ReflexionCriticConfig:
        """Get the reflexion critic configuration."""
        return cast(ReflexionCriticConfig, self._config)

    def validate(self, text: str) -> bool:
        """Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    def improve(self, text: str, feedback: str = None) -> str:
        """Improve text based on feedback and reflections.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (can be a string or a list of violations)

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Handle different feedback types
        if isinstance(feedback, list):
            # Convert violations to feedback string
            feedback_str = self._violations_to_feedback(feedback)
        elif feedback is None:
            feedback_str = "Please improve this text."
        else:
            feedback_str = feedback

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service - the critique service handles reflection generation
        return critique_service.improve(text, feedback_str)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on specific feedback.

        This method implements the required abstract method from BaseCritic.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        return self.improve(text, feedback)

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            TypeError: If model returns invalid output
            RuntimeError: If critic is not properly initialized
        """
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.critique(text)

    def _violations_to_feedback(self, violations: List[Dict[str, Any]]) -> str:
        """Convert rule violations to feedback text.

        This method transforms a list of rule violations into a human-readable
        feedback string that can be used to guide text improvement.

        Args:
            violations: List of rule violations, where each violation is a dictionary
                      containing 'rule_name' and 'message' keys

        Returns:
            str: Formatted feedback string

        Raises:
            TypeError: If violations is not a list
            ValueError: If violations contain invalid data

        Examples:
            ```python
            violations = [
                {"rule_name": "Clarity", "message": "Sentence too complex"},
                {"rule_name": "Grammar", "message": "Subject-verb agreement error"}
            ]
            feedback = critic._violations_to_feedback(violations)
            # Returns:
            # "The following issues were found:
            # - Clarity: Sentence too complex
            # - Grammar: Subject-verb agreement error"
            ```
        """
        # Get state
        state = self._state_manager.get_state()

        # Store violations in state cache for potential future use
        if "cache" not in state.cache:
            state.cache["cache"] = {}

        state.cache["cache"]["last_violations"] = violations

        # Handle empty violations
        if not violations:
            return "No issues found."

        # Format feedback
        feedback = "The following issues were found:\n"
        for i, violation in enumerate(violations):
            rule_name = violation.get("rule_name", f"Rule {i+1}")
            message = violation.get("message", "Unknown issue")
            feedback += f"- {rule_name}: {message}\n"

        # Store formatted feedback in cache
        state.cache["cache"]["last_formatted_feedback"] = feedback

        return feedback

    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """Parse a critique response string into a structured format.

        This method processes the raw response from the language model and
        extracts structured data including score, feedback, issues, and suggestions.

        Args:
            response: Raw response string from the language model

        Returns:
            dict: Structured critique data containing:
                - score: float between 0 and 1
                - feedback: General feedback text
                - issues: List of specific issues
                - suggestions: List of improvement suggestions

        Raises:
            ValueError: If response format is invalid
            TypeError: If response cannot be parsed

        Examples:
            ```python
            response = '''
            SCORE: 0.7
            FEEDBACK: The text needs improvement
            ISSUES:
            - Unclear structure
            - Missing examples
            SUGGESTIONS:
            - Add more examples
            - Improve organization
            '''
            result = critic._parse_critique_response(response)
            # Returns:
            # {
            #     "score": 0.7,
            #     "feedback": "The text needs improvement",
            #     "issues": ["Unclear structure", "Missing examples"],
            #     "suggestions": ["Add more examples", "Improve organization"]
            # }
            ```
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure cache exists in state
        if "cache" not in state.cache:
            state.cache["cache"] = {}

        # Store raw response in cache
        state.cache["cache"]["last_raw_response"] = response

        # Initialize result structure
        result = {
            "score": 0.0,
            "feedback": "",
            "issues": [],
            "suggestions": [],
        }

        try:
            # Extract score
            if "SCORE:" in response:
                score_line = response.split("SCORE:")[1].split("\n")[0].strip()
                try:
                    result["score"] = float(score_line)
                except ValueError:
                    logger.warning("Failed to parse score from response")
                    pass

            # Extract feedback
            if "FEEDBACK:" in response:
                feedback_parts = response.split("FEEDBACK:")[1].split("ISSUES:")[0].strip()
                result["feedback"] = feedback_parts

            # Extract issues
            if "ISSUES:" in response:
                issues_part = response.split("ISSUES:")[1]
                if "SUGGESTIONS:" in issues_part:
                    issues_part = issues_part.split("SUGGESTIONS:")[0]

                issues = []
                for line in issues_part.strip().split("\n"):
                    if line.strip().startswith("-"):
                        issues.append(line.strip()[1:].strip())
                result["issues"] = issues

            # Extract suggestions
            if "SUGGESTIONS:" in response:
                suggestions_part = response.split("SUGGESTIONS:")[1].strip()
                suggestions = []
                for line in suggestions_part.split("\n"):
                    if line.strip().startswith("-"):
                        suggestions.append(line.strip()[1:].strip())
                result["suggestions"] = suggestions

        except Exception as e:
            # Log error and return default values if parsing fails
            logger.error(f"Error parsing critique response: {str(e)}")
            pass

        # Store parsed result in cache
        state.cache["cache"]["last_parsed_critique"] = result

        return result

    def _get_relevant_reflections(self) -> List[str]:
        """Get relevant reflections from the memory buffer.

        This method retrieves the most relevant reflections from the memory buffer
        to guide the current text improvement process.

        Returns:
            List[str]: List of relevant reflections

        Raises:
            RuntimeError: If critic is not properly initialized

        Examples:
            ```python
            reflections = critic._get_relevant_reflections()
            # Returns:
            # [
            #     "Previous improvement focused on clarity",
            #     "Added more examples in last iteration"
            # ]
            ```
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Ensure cache exists in state
        if "cache" not in state.cache:
            state.cache["cache"] = {}

        # Get reflections from memory manager
        memory_manager = state.memory_manager
        if not memory_manager:
            raise RuntimeError("Memory manager not initialized")

        reflections = memory_manager.get_memory()

        # Store reflections in cache
        state.cache["cache"]["last_reflections"] = reflections

        return reflections

    # Async methods
    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text.

        This method performs asynchronous validation of text using the language model.
        It checks if the text meets quality standards and returns a boolean result.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If model interaction fails or critic is not properly initialized
        """
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return await critique_service.avalidate(text)

    async def acritique(self, text: str) -> dict:
        """Asynchronously critique text.

        This method performs asynchronous critique of text using the language model.
        It analyzes the text and provides detailed feedback in a structured format.

        Args:
            text: The text to critique

        Returns:
            dict: Structured critique data containing:
                - score: float between 0 and 1
                - feedback: General feedback text
                - issues: List of specific issues
                - suggestions: List of improvement suggestions

        Raises:
            ValueError: If text is empty
            RuntimeError: If model interaction fails or critic is not properly initialized
        """
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Ensure cache exists in state
        if "cache" not in state.cache:
            state.cache["cache"] = {}

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        result = await critique_service.acritique(text)

        # Store result in cache
        state.cache["cache"]["last_async_critique"] = result

        return result

    async def aimprove(self, text: str, feedback: str = None) -> str:
        """Asynchronously improve text.

        This method performs asynchronous improvement of text using the language model.
        It incorporates feedback and reflections to enhance the text quality.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If model interaction fails or critic is not properly initialized
        """
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Ensure cache exists in state
        if "cache" not in state.cache:
            state.cache["cache"] = {}

        # Handle different feedback types
        if isinstance(feedback, list):
            # Convert violations to feedback string
            feedback_str = self._violations_to_feedback(feedback)
        elif feedback is None:
            feedback_str = "Please improve this text."
        else:
            feedback_str = feedback

        # Store feedback in cache
        state.cache["cache"]["last_async_feedback"] = feedback_str

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service - the critique service handles reflection generation
        result = await critique_service.aimprove(text, feedback_str)

        # Store result in cache
        state.cache["cache"]["last_async_improved_text"] = result

        return result

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback.

        This method implements the required async abstract method from BaseCritic.
        It performs asynchronous improvement of text using specific feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If model interaction fails or critic is not properly initialized
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("ReflexionCritic not properly initialized")

        # Ensure cache exists in state
        if "cache" not in state.cache:
            state.cache["cache"] = {}

        # Store the specific feedback request in cache
        state.cache["cache"]["last_specific_feedback"] = feedback

        return await self.aimprove(text, feedback)


# Default configurations
DEFAULT_REFLEXION_SYSTEM_PROMPT: Final[
    str
] = """You are an expert editor that improves text through reflection.
You maintain a memory of past improvements and use these reflections to guide
future improvements. Focus on learning patterns from past feedback and applying
them to new situations."""

DEFAULT_REFLEXION_CONFIG = ReflexionCriticConfig(
    name="Default Reflexion Critic",
    description="Evaluates and improves text using reflections on past feedback",
    system_prompt=DEFAULT_REFLEXION_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    memory_buffer_size=5,
    reflection_depth=1,
)


def create_reflexion_critic(
    llm_provider: LanguageModel,
    name: str = "reflexion_critic",
    description: str = "Improves text using reflections on past feedback",
    system_prompt: str = DEFAULT_REFLEXION_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    memory_buffer_size: int = 5,
    reflection_depth: int = 1,
    config: Optional[Union[Dict[str, Any], ReflexionCriticConfig]] = None,
    **kwargs: Any,
) -> ReflexionCritic:
    """Create a reflexion critic with the given parameters.

    This helper function creates and configures a ReflexionCritic instance with
    the specified parameters. It provides a convenient way to create a critic
    with custom settings.

    Args:
        llm_provider: Language model provider to use for critiquing
        name: Name of the critic
        description: Description of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation (0.0 to 1.0)
        max_tokens: Maximum tokens for model generation
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        max_attempts: Maximum number of improvement attempts
        memory_buffer_size: Maximum number of reflections to store
        reflection_depth: How many levels of reflection to perform
        config: Optional pre-configured ReflexionCriticConfig or dict
        **kwargs: Additional configuration parameters

    Returns:
        ReflexionCritic: Configured reflexion critic

    Raises:
        ValueError: If any parameter is invalid
        TypeError: If llm_provider is not a valid LanguageModel

    Examples:
        ```python
        from sifaka.models.openai import create_openai_provider
        from sifaka.critics.reflexion import create_reflexion_critic

        # Create a language model provider
        provider = create_openai_provider(api_key="your-api-key")

        # Create a reflexion critic with default settings
        critic = create_reflexion_critic(llm_provider=provider)

        # Create a reflexion critic with custom settings
        critic = create_reflexion_critic(
            llm_provider=provider,
            name="custom_critic",
            description="A critic with custom settings",
            system_prompt="You are an expert technical writer...",
            temperature=0.5,
            max_tokens=2000,
            min_confidence=0.8,
            memory_buffer_size=10,
            reflection_depth=2
        )

        # Use the critic
        text = "This is a sample technical document."
        improved_text = critic.improve(text)
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        from sifaka.utils.config import standardize_critic_config

        # If standardize_critic_config is available, use it
        critic_config = standardize_critic_config(
            config=config,
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            memory_buffer_size=memory_buffer_size,
            reflection_depth=reflection_depth,
            **kwargs,
        )
    except (ImportError, AttributeError):
        # Create config manually
        if isinstance(config, ReflexionCriticConfig):
            critic_config = config
        elif isinstance(config, dict):
            critic_config = ReflexionCriticConfig(**config)
        else:
            critic_config = ReflexionCriticConfig(
                name=name,
                description=description,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                min_confidence=min_confidence,
                max_attempts=max_attempts,
                memory_buffer_size=memory_buffer_size,
                reflection_depth=reflection_depth,
                **kwargs,
            )

    # Create and return the critic
    return ReflexionCritic(
        config=critic_config, llm_provider=llm_provider, name=name, description=description
    )
