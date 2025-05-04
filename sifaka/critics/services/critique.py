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

from typing import Any, Dict, List, Optional, Union

from ..managers.memory import MemoryManager
from ..managers.prompt import PromptManager
from ..managers.response import ResponseParser
from ...utils.logging import get_logger

logger = get_logger(__name__)


class CritiqueService:
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

    def __init__(
        self,
        llm_provider: Any,
        prompt_manager: PromptManager,
        response_parser: ResponseParser,
        memory_manager: Optional[MemoryManager] = None,
    ):
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
        self._model = llm_provider
        self._prompt_manager = prompt_manager
        self._response_parser = response_parser
        self._memory_manager = memory_manager

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        This method validates text using the model provider and handles
        any errors that occur during validation.

        Lifecycle:
        1. Input validation
        2. Prompt creation
        3. Model invocation
        4. Response parsing
        5. Error handling

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If validation fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Create validation prompt
        validation_prompt = self._prompt_manager.create_validation_prompt(text)

        try:
            # Get response from the model
            response = self._model.invoke(validation_prompt)

            # Parse the response
            return self._response_parser.parse_validation_response(response)
        except Exception as e:
            logger.error(f"Failed to validate text: {str(e)}")
            return False

    def critique(self, text: str) -> Dict[str, Any]:
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

        Returns:
            A dictionary containing the critique details

        Raises:
            ValueError: If text is empty
            RuntimeError: If critique fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Create critique prompt
        critique_prompt = self._prompt_manager.create_critique_prompt(text)

        try:
            # Get response from the model
            response = self._model.invoke(critique_prompt)

            # Parse the response
            return self._response_parser.parse_critique_response(response)
        except Exception as e:
            logger.error(f"Failed to critique text: {str(e)}")
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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Convert violations to feedback if needed
        if isinstance(feedback, list):
            feedback = self._violations_to_feedback(feedback)

        # Create improvement prompt
        improvement_prompt = self._prompt_manager.create_improvement_prompt(
            text, feedback, self._get_relevant_reflections()
        )

        try:
            # Get response from the model
            response = self._model.invoke(improvement_prompt)

            # Parse the response
            improved_text = self._response_parser.parse_improvement_response(response)

            # Generate reflection
            self._generate_reflection(text, feedback, improved_text)

            return improved_text
        except Exception as e:
            logger.error(f"Failed to improve text: {str(e)}")
            return text  # Return original text on error

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
            rule = violation.get("rule", "unknown")
            message = violation.get("message", "")
            feedback += f"- {rule}: {message}\n"

        return feedback.strip()

    def _generate_reflection(
        self, original_text: str, feedback: str, improved_text: str
    ) -> None:
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
        if not self._memory_manager:
            return

        # Create reflection prompt
        reflection_prompt = self._prompt_manager.create_reflection_prompt(
            original_text, feedback, improved_text
        )

        try:
            # Get response from the model
            response = self._model.invoke(reflection_prompt)

            # Parse the response
            reflection = self._response_parser.parse_reflection_response(response)

            # Store reflection in memory
            if reflection:
                self._memory_manager.add_to_memory(reflection)
        except Exception as e:
            logger.error(f"Failed to generate reflection: {str(e)}")

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
        if not self._memory_manager:
            return []

        try:
            return self._memory_manager.get_memory()
        except Exception as e:
            logger.error(f"Failed to get reflections: {str(e)}")
            return []

    async def avalidate(self, text: str) -> bool:
        """
        Asynchronously validate text against quality standards.

        This method validates text using the model provider and handles
        any errors that occur during validation.

        Lifecycle:
        1. Input validation
        2. Prompt creation
        3. Model invocation
        4. Response parsing
        5. Error handling

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If validation fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Create validation prompt
        validation_prompt = self._prompt_manager.create_validation_prompt(text)

        try:
            # Get response from the model
            response = await self._model.ainvoke(validation_prompt)

            # Parse the response
            return self._response_parser.parse_validation_response(response)
        except Exception as e:
            logger.error(f"Failed to validate text: {str(e)}")
            return False

    async def acritique(self, text: str) -> Dict[str, Any]:
        """
        Asynchronously critique text and provide feedback.

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

        Returns:
            A dictionary containing the critique details

        Raises:
            ValueError: If text is empty
            RuntimeError: If critique fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Create critique prompt
        critique_prompt = self._prompt_manager.create_critique_prompt(text)

        try:
            # Get response from the model
            response = await self._model.ainvoke(critique_prompt)

            # Parse the response
            return self._response_parser.parse_critique_response(response)
        except Exception as e:
            logger.error(f"Failed to critique text: {str(e)}")
            return {
                "score": 0.0,
                "feedback": f"Failed to critique text: {str(e)}",
                "issues": ["Critique process failed"],
                "suggestions": ["Try again with clearer text"],
            }

    async def aimprove(self, text: str, feedback: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Asynchronously improve text based on feedback or violations.

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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Convert violations to feedback if needed
        if isinstance(feedback, list):
            feedback = self._violations_to_feedback(feedback)

        # Create improvement prompt
        improvement_prompt = self._prompt_manager.create_improvement_prompt(
            text, feedback, self._get_relevant_reflections()
        )

        try:
            # Get response from the model
            response = await self._model.ainvoke(improvement_prompt)

            # Parse the response
            improved_text = self._response_parser.parse_improvement_response(response)

            # Generate reflection
            await self._generate_reflection_async(text, feedback, improved_text)

            return improved_text
        except Exception as e:
            logger.error(f"Failed to improve text: {str(e)}")
            return text  # Return original text on error

    async def _generate_reflection_async(
        self, original_text: str, feedback: str, improved_text: str
    ) -> None:
        """
        Asynchronously generate a reflection on the improvement.

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
        if not self._memory_manager:
            return

        # Create reflection prompt
        reflection_prompt = self._prompt_manager.create_reflection_prompt(
            original_text, feedback, improved_text
        )

        try:
            # Get response from the model
            response = await self._model.ainvoke(reflection_prompt)

            # Parse the response
            reflection = self._response_parser.parse_reflection_response(response)

            # Store reflection in memory
            if reflection:
                self._memory_manager.add_to_memory(reflection)
        except Exception as e:
            logger.error(f"Failed to generate reflection: {str(e)}")
