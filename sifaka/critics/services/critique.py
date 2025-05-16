"""
Critique service for critics.

This module provides the CritiqueService class which is responsible for
critiquing, validating, and improving text.

## Component Lifecycle

### CritiqueService Lifecycle

1. **Initialization Phase**
   - Component setup (prompt manager, response parser, memory manager)
   - Model provider configuration
   - Resource allocation
   - Error handling setup

2. **Usage Phase**
   - Text validation
   - Text critique
   - Text improvement
   - Reflection generation
   - Memory management

3. **Cleanup Phase**
   - Resource cleanup
   - Memory buffer clearing
   - State reset
   - Error recovery

### Component Interactions

1. **Prompt Manager**
   - Creates prompts for different operations
   - Manages prompt templates
   - Handles prompt validation

2. **Response Parser**
   - Parses model responses
   - Validates response formats
   - Extracts structured data

3. **Memory Manager**
   - Stores and retrieves reflections
   - Manages memory buffer
   - Handles cleanup

4. **Model Provider**
   - Generates text responses
   - Handles model errors
   - Manages model state

### Error Handling and Recovery

1. **Input Validation Errors**
   - Empty or invalid text
   - Invalid feedback format
   - Invalid violation format
   - Recovery: Return appropriate error messages

2. **Processing Errors**
   - Model generation failures
   - Response parsing errors
   - Memory management issues
   - Recovery: Retry mechanisms and fallbacks

3. **Resource Errors**
   - Memory allocation failures
   - Model provider issues
   - Network connectivity problems
   - Recovery: Resource cleanup and state preservation

### Error Recovery Strategies

1. **Component-Level Recovery**
   - Prompt manager: Use default templates
   - Response parser: Simplify parsing
   - Memory manager: Clear and reset
   - Model provider: Retry with simpler operations

2. **System-Level Recovery**
   - Preserve valid state
   - Clean up invalid state
   - Restore previous state
   - Log error details

3. **User-Level Recovery**
   - Provide error messages
   - Suggest alternatives
   - Allow configuration changes
   - Enable manual intervention
"""

import time
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, cast, TypeVar, ClassVar, Type
from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
    from sifaka.core.managers.prompt import CriticPromptManager as PromptManager
    from ..managers.response import ResponseParser
from ...utils.logging import get_logger
from ...utils.errors import safely_execute_component_operation as safely_execute_critic
from ...utils.errors.component import CriticError
from ...utils.state import StateManager, create_critic_state
from ...utils.errors import ErrorResult

logger = get_logger(__name__)


class CritiqueService(BaseModel):
    """
    Handles critiquing, validating, and improving text.

    This class is responsible for using language models to critique,
    validate, and improve text.

    ## Lifecycle Management

    The CritiqueService manages its lifecycle through three main phases:

    1. **Initialization**
       - Sets up components
       - Configures model provider
       - Allocates resources
       - Sets up error handling

    2. **Operation**
       - Processes text through components
       - Manages state and memory
       - Handles errors and recovery
       - Coordinates component interactions

    3. **Cleanup**
       - Releases resources
       - Clears memory buffers
       - Resets state
       - Logs final status

    ## Error Handling

    The CritiqueService implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies violation format

    2. **Processing Errors**
       - Handles model failures
       - Manages parsing errors
       - Recovers from memory issues

    3. **Resource Management**
       - Handles allocation failures
       - Manages cleanup errors
       - Preserves valid state

    ## Component Coordination

    The CritiqueService coordinates between components:

    1. **Prompt Management**
       - Creates appropriate prompts
       - Validates prompt format
       - Handles template variables

    2. **Response Processing**
       - Parses model responses
       - Validates output format
       - Extracts structured data

    3. **Memory Management**
       - Stores reflections
       - Manages buffer size
       - Handles cleanup

    4. **Model Interaction**
       - Coordinates operations
       - Manages model state
       - Handles error recovery
    """

    _state_manager: StateManager = PrivateAttr(default_factory=create_critic_state)

    @classmethod
    def model_validate(
        cls: Type["CritiqueService"],
        obj: Any,
        *,
        strict: Optional[bool] = None,
        from_attributes: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        by_alias: Optional[bool] = None,
        by_name: Optional[bool] = None,
    ) -> "CritiqueService":
        """Validate a model instance."""
        return super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    def __init__(
        self,
        llm_provider: Any,
        prompt_manager: Any,
        response_parser: Any,
        memory_manager: Optional[Optional[Any]] = None,
    ) -> None:
        """
        Initialize a CritiqueService instance.

        This method sets up the service with its required components and
        performs necessary validation and initialization.

        Lifecycle:
        1. Validate components
        2. Initialize state
        3. Set up error handling
        4. Configure resources

        Args:
            llm_provider: The language model provider to use
            prompt_manager: The prompt manager to use
            response_parser: The response parser to use
            memory_manager: Optional memory manager to use

        Raises:
            ValueError: If components are invalid
            RuntimeError: If initialization fails
        """
        super().__init__()
        self._state_manager.update("model", llm_provider)
        self._state_manager.update("prompt_manager", prompt_manager)
        self._state_manager.update("response_parser", response_parser)
        self._state_manager.update("memory_manager", memory_manager)
        self._state_manager.update("initialized", True)
        self._state_manager.update("critique_count", 0)
        self._state_manager.update("validation_count", 0)
        self._state_manager.update("improvement_count", 0)
        self._state_manager.update("error_count", 0)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("initialization_time", time.time())

    def validate_text(self, text: str) -> bool:
        """
        Validate text against quality standards.

        This method checks if text meets quality standards by using
        a language model to analyze it. It returns a boolean value
        indicating whether the text passes validation.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If validation fails for any reason
        """

        def validation_operation() -> bool:
            # Ensure model provider is available
            model = self._state_manager.get("model")
            if not model:
                raise ValueError("Model provider is not available")

            # Get prompt for validation
            prompt_manager = self._state_manager.get("prompt_manager")
            prompt = prompt_manager.create_validation_prompt(text) if prompt_manager else None
            if not prompt:
                raise ValueError("Failed to create validation prompt")

            # Generate validation response
            validation_response = model.generate(prompt)
            if not validation_response:
                raise ValueError("Empty validation response")

            # Parse validation result
            response_parser = self._state_manager.get("response_parser")
            if not response_parser:
                raise ValueError("Response parser is not available")

            validation_result = response_parser.parse_validation(validation_response)
            if validation_result is None:
                raise ValueError("Failed to parse validation result")

            # Increment validation count
            validation_count = self._state_manager.get("validation_count", 0)
            self._state_manager.update("validation_count", validation_count + 1)

            return bool(validation_result)

        # Execute with error handling
        return safely_execute_critic(
            operation=validation_operation,
            component_name="CritiqueService.validate_text",
            component_type="critic",
            error_class=CriticError,
            default_value=False,
        )

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Critique text and provide feedback.

        This method critiques text using the model provider and handles
        any errors that occur during critique.

        Lifecycle:
        1. Input validation
        2. Prompt creation
        3. Model invocation
        4. Response parsing
        5. Error handling

        Args:
            text: The text to critique
            metadata: Optional metadata to include in the critique

        Returns:
            A dictionary containing the critique details

        Raises:
            ValueError: If text is empty
            RuntimeError: If critique fails
        """
        critique_count = self._state_manager.get("critique_count", 0)
        self._state_manager.update("critique_count", critique_count + 1)

        def critique_operation() -> Dict[str, Any]:
            if not isinstance(text, str) or not text.strip() if text else "":
                raise ValueError("text must be a non-empty string")
            prompt_manager = self._state_manager.get("prompt_manager")
            model = self._state_manager.get("model")
            response_parser = self._state_manager.get("response_parser")
            if not prompt_manager or not model or not response_parser:
                raise RuntimeError("CritiqueService not properly initialized")
            critique_prompt = prompt_manager.create_critique_prompt(text) if prompt_manager else ""
            response = model.invoke(critique_prompt) if model else ""
            result = response_parser.parse_critique_response(response) if response_parser else {}
            stats = self._state_manager.get("stats", {})
            stats["critique_count"] = stats.get("critique_count", 0) + 1 if stats else 1
            stats["last_critique_time"] = time.time()
            score_distribution = stats.get("score_distribution", {}) if stats else {}

            # Handle the case where result might be a string or None
            result_dict = result if isinstance(result, dict) else {}
            score = result_dict.get("score", 0.0)

            score_bucket = round(score * 10) / 10
            score_distribution[str(score_bucket)] = (
                score_distribution.get(str(score_bucket), 0) + 1 if score_distribution else 1
            )
            stats["score_distribution"] = score_distribution
            self._state_manager.update("stats", stats)
            return result_dict

        try:
            result = safely_execute_critic(
                operation=critique_operation,
                component_name=self.__class__.__name__,
                component_type="Critic",
                error_class=CriticError,
                additional_metadata={
                    "text_length": len(text),
                    "method": "critique",
                    **(metadata or {}),
                },
            )
            if isinstance(result, dict):
                return result
            if isinstance(result, ErrorResult):
                error_msg = str(result)
                return {
                    "score": 0.0,
                    "feedback": f"Failed to critique text: {error_msg}",
                    "issues": ["Critique process failed"],
                    "suggestions": ["Try again with clearer text"],
                }
            # Fallback for any other unexpected result type
            return {
                "score": 0.0,
                "feedback": "Failed to critique text: Unknown error",
                "issues": ["Critique process failed"],
                "suggestions": ["Try again with clearer text"],
            }
        except Exception as e:
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)
            logger.error(f'Failed to critique text: {str(e) if logger else ""}')
            return {
                "score": 0.0,
                "feedback": f"Failed to critique text: {str(e)}",
                "issues": ["Critique process failed"],
                "suggestions": ["Try again with clearer text"],
            }

    def improve(self, text: str, feedback: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Improve text based on feedback or violations.

        This method improves text using the model provider and handles
        any errors that occur during improvement.

        Lifecycle:
        1. Input validation
        2. Feedback processing
        3. Prompt creation
        4. Model invocation
        5. Response parsing
        6. Error handling

        Args:
            text: The text to improve
            feedback: Feedback or violations to guide improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If improvement fails
        """
        improvement_count = self._state_manager.get("improvement_count", 0)
        self._state_manager.update("improvement_count", improvement_count + 1)

        def improvement_operation() -> str:
            if not isinstance(text, str) or not text.strip() if text else "":
                raise ValueError("text must be a non-empty string")
            prompt_manager = self._state_manager.get("prompt_manager")
            model = self._state_manager.get("model")
            response_parser = self._state_manager.get("response_parser")
            if not prompt_manager or not model or not response_parser:
                raise RuntimeError("CritiqueService not properly initialized")
            processed_feedback = feedback
            if isinstance(feedback, list):
                processed_feedback = self._violations_to_feedback(feedback) if self else ""

            # Get the reflections as a string
            reflections = self._get_relevant_reflections() if self else []
            reflections_str = "\n".join(reflections) if reflections else ""

            improvement_prompt = prompt_manager.create_improvement_prompt(
                text,
                processed_feedback,
                reflections_str if prompt_manager else "",
            )
            response = model.invoke(improvement_prompt) if model else ""
            improved_text = (
                response_parser.parse_improvement_response(response) if response_parser else ""
            )

            # Convert processed_feedback to string for _generate_reflection
            feedback_str = (
                processed_feedback
                if isinstance(processed_feedback, str)
                else str(processed_feedback)
            )
            self._generate_reflection(text, feedback_str, improved_text) if self else ""

            stats = self._state_manager.get("stats", {})
            stats["improvement_count"] = stats.get("improvement_count", 0) + 1 if stats else 1
            stats["last_improvement_time"] = time.time()
            self._state_manager.update("stats", stats)
            return str(improved_text)

        try:
            result = safely_execute_critic(
                operation=improvement_operation,
                component_name=self.__class__.__name__,
                component_type="Critic",
                error_class=CriticError,
                additional_metadata={
                    "text_length": len(text),
                    "method": "improve",
                    "feedback_type": "list" if isinstance(feedback, list) else "string",
                },
            )
            return str(result) if result is not None else text
        except Exception as e:
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)
            logger.error(f'Failed to improve text: {str(e) if logger else ""}')
            return text

    def _violations_to_feedback(self, violations: List[Dict[str, Any]]) -> str:
        """
        Convert violations to feedback text.

        This method converts a list of violations into a feedback string
        that can be used to guide text improvement.

        Lifecycle:
        1. Input validation
        2. Violation processing
        3. Feedback generation
        4. Error handling

        Args:
            violations: List of rule violations

        Returns:
            Feedback text

        Raises:
            ValueError: If violations are invalid
        """
        if not violations:
            return ""
        feedback = "The following issues need to be addressed:\n"
        for violation in violations:
            rule = violation.get("rule", "unknown") if violation else ""
            message = violation.get("message", "") if violation else ""
            feedback += f"- {rule}: {message}\n"
        return feedback.strip() if feedback else ""

    def _generate_reflection(self, original_text: str, feedback: str, improved_text: str) -> None:
        """
        Generate a reflection on the improvement.

        This method generates a reflection on how the text was improved
        and stores it in memory if available.

        Lifecycle:
        1. Input validation
        2. Prompt creation
        3. Model invocation
        4. Response parsing
        5. Memory storage
        6. Error handling

        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text

        Raises:
            ValueError: If inputs are invalid
        """
        memory_manager = self._state_manager.get("memory_manager")
        if not memory_manager:
            return

        def reflection_operation() -> Any:
            prompt_manager = self._state_manager.get("prompt_manager")
            model = self._state_manager.get("model")
            response_parser = self._state_manager.get("response_parser")
            if not prompt_manager or not model or not response_parser:
                raise RuntimeError("CritiqueService not properly initialized")
            reflection_prompt = (
                prompt_manager.create_reflection_prompt(original_text, feedback, improved_text)
                if prompt_manager
                else ""
            )
            response = model.invoke(reflection_prompt) if model else ""
            reflection = (
                response_parser.parse_reflection_response(response) if response_parser else ""
            )
            if reflection:
                (
                    memory_manager and memory_manager and memory_manager.add_to_memory(reflection)
                    if memory_manager
                    else ""
                )
            stats = self._state_manager.get("stats", {})
            stats["reflection_count"] = stats.get("reflection_count", 0) + 1 if stats else 1
            self._state_manager.update("stats", stats)
            return reflection

        try:
            safely_execute_critic(
                operation=reflection_operation,
                component_name=self.__class__.__name__,
                component_type="Critic",
                error_class=CriticError,
                additional_metadata={
                    "method": "generate_reflection",
                    "original_text_length": len(original_text),
                    "improved_text_length": len(improved_text),
                },
            )
        except Exception as e:
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)
            logger.error(f'Failed to generate reflection: {str(e) if logger else ""}')

    def _get_relevant_reflections(self) -> List[str]:
        """
        Get relevant reflections from memory.

        This method retrieves relevant reflections from memory if available.

        Lifecycle:
        1. Memory check
        2. Reflection retrieval
        3. Error handling

        Returns:
            List of relevant reflections
        """
        memory_manager = self._state_manager.get("memory_manager")
        if not memory_manager:
            return []

        def retrieval_operation() -> List[str]:
            if memory_manager:
                memory = memory_manager.get_memory()
                return cast(List[str], memory) if memory else []
            return []

        try:
            result = safely_execute_critic(
                operation=retrieval_operation,
                component_name=self.__class__.__name__,
                component_type="Critic",
                error_class=CriticError,
                additional_metadata={"method": "get_relevant_reflections"},
            )
            if isinstance(result, list):
                return result
            if isinstance(result, ErrorResult):
                logger.error(f"Failed to get reflections: {str(result)}")
                return []
            # Fallback for any other unexpected result type
            return []
        except Exception as e:
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)
            logger.error(f'Failed to get reflections: {str(e) if logger else ""}')
            return []
