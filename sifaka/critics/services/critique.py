"""
Critique service for critics.

This module provides the CritiqueService class which is responsible for
critiquing, validating, and improving text.
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
        
        Args:
            llm_provider: The language model provider to use
            prompt_manager: The prompt manager to use
            response_parser: The response parser to use
            memory_manager: Optional memory manager to use
        """
        self._model = llm_provider
        self._prompt_manager = prompt_manager
        self._response_parser = response_parser
        self._memory_manager = memory_manager
        
    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.
        
        Args:
            text: The text to validate
            
        Returns:
            True if the text meets quality standards, False otherwise
            
        Raises:
            ValueError: If text is empty
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
        
        Args:
            text: The text to critique
            
        Returns:
            A dictionary containing the critique details
            
        Raises:
            ValueError: If text is empty
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
                "issues": ["Failed to parse model response"],
                "suggestions": ["Try again with clearer text"],
            }
            
    def improve(self, text: str, feedback: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Improve text based on feedback.
        
        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            
        Returns:
            The improved text
            
        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
            
        # Handle different feedback types
        if isinstance(feedback, list):
            # Convert violations to feedback string
            feedback_str = self._violations_to_feedback(feedback)
        else:
            feedback_str = feedback
            
        # Get reflections from memory if available
        reflections = None
        if self._memory_manager:
            reflections = self._memory_manager.get_memory()
            
        # Create improvement prompt
        improve_prompt = self._prompt_manager.create_improvement_prompt(
            text, feedback_str, reflections
        )
        
        try:
            # Get response from the model
            response = self._model.invoke(improve_prompt)
            
            # Parse the response
            improved_text = self._response_parser.parse_improvement_response(response)
            
            # Generate reflection if memory manager is available
            if self._memory_manager:
                self._generate_reflection(text, feedback_str, improved_text)
                
            return improved_text
        except Exception as e:
            logger.error(f"Failed to improve text: {str(e)}")
            raise ValueError(f"Failed to improve text: {str(e)}") from e
            
    def _violations_to_feedback(self, violations: List[Dict[str, Any]]) -> str:
        """
        Convert rule violations to feedback text.
        
        Args:
            violations: List of rule violations
            
        Returns:
            Formatted feedback
        """
        if not violations:
            return "No issues found."
            
        feedback = "The following issues were found:\n"
        for i, violation in enumerate(violations):
            rule_name = violation.get("rule_name", f"Rule {i+1}")
            message = violation.get("message", "Unknown issue")
            feedback += f"- {rule_name}: {message}\n"
            
        return feedback
        
    def _generate_reflection(
        self, original_text: str, feedback: str, improved_text: str
    ) -> None:
        """
        Generate a reflection on the improvement process.
        
        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text
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
            
            # Add reflection to memory
            if reflection:
                self._memory_manager.add_to_memory(reflection)
        except Exception as e:
            logger.error(f"Failed to generate reflection: {str(e)}")
            # Silently fail if reflection generation fails
            pass
            
    async def avalidate(self, text: str) -> bool:
        """
        Asynchronously validate text against quality standards.
        
        Args:
            text: The text to validate
            
        Returns:
            True if the text meets quality standards, False otherwise
            
        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
            
        # Create validation prompt
        validation_prompt = self._prompt_manager.create_validation_prompt(text)
        
        try:
            # Get response from the model
            if hasattr(self._model, "ainvoke"):
                response = await self._model.ainvoke(validation_prompt)
            else:
                # Fall back to synchronous invoke if ainvoke is not available
                response = self._model.invoke(validation_prompt)
                
            # Parse the response
            return self._response_parser.parse_validation_response(response)
        except Exception as e:
            logger.error(f"Failed to validate text: {str(e)}")
            return False
            
    async def acritique(self, text: str) -> Dict[str, Any]:
        """
        Asynchronously critique text and provide feedback.
        
        Args:
            text: The text to critique
            
        Returns:
            A dictionary containing the critique details
            
        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
            
        # Create critique prompt
        critique_prompt = self._prompt_manager.create_critique_prompt(text)
        
        try:
            # Get response from the model
            if hasattr(self._model, "ainvoke"):
                response = await self._model.ainvoke(critique_prompt)
            else:
                # Fall back to synchronous invoke if ainvoke is not available
                response = self._model.invoke(critique_prompt)
                
            # Parse the response
            return self._response_parser.parse_critique_response(response)
        except Exception as e:
            logger.error(f"Failed to critique text: {str(e)}")
            return {
                "score": 0.0,
                "feedback": f"Failed to critique text: {str(e)}",
                "issues": ["Failed to parse model response"],
                "suggestions": ["Try again with clearer text"],
            }
            
    async def aimprove(self, text: str, feedback: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Asynchronously improve text based on feedback.
        
        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            
        Returns:
            The improved text
            
        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
            
        # Handle different feedback types
        if isinstance(feedback, list):
            # Convert violations to feedback string
            feedback_str = self._violations_to_feedback(feedback)
        else:
            feedback_str = feedback
            
        # Get reflections from memory if available
        reflections = None
        if self._memory_manager:
            reflections = self._memory_manager.get_memory()
            
        # Create improvement prompt
        improve_prompt = self._prompt_manager.create_improvement_prompt(
            text, feedback_str, reflections
        )
        
        try:
            # Get response from the model
            if hasattr(self._model, "ainvoke"):
                response = await self._model.ainvoke(improve_prompt)
            else:
                # Fall back to synchronous invoke if ainvoke is not available
                response = self._model.invoke(improve_prompt)
                
            # Parse the response
            improved_text = self._response_parser.parse_improvement_response(response)
            
            # Generate reflection if memory manager is available
            if self._memory_manager:
                await self._generate_reflection_async(text, feedback_str, improved_text)
                
            return improved_text
        except Exception as e:
            logger.error(f"Failed to improve text: {str(e)}")
            raise ValueError(f"Failed to improve text: {str(e)}") from e
            
    async def _generate_reflection_async(
        self, original_text: str, feedback: str, improved_text: str
    ) -> None:
        """
        Asynchronously generate a reflection on the improvement process.
        
        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text
        """
        if not self._memory_manager:
            return
            
        # Create reflection prompt
        reflection_prompt = self._prompt_manager.create_reflection_prompt(
            original_text, feedback, improved_text
        )
        
        try:
            # Get response from the model
            if hasattr(self._model, "ainvoke"):
                response = await self._model.ainvoke(reflection_prompt)
            else:
                # Fall back to synchronous invoke if ainvoke is not available
                response = self._model.invoke(reflection_prompt)
                
            # Parse the response
            reflection = self._response_parser.parse_reflection_response(response)
            
            # Add reflection to memory
            if reflection:
                self._memory_manager.add_to_memory(reflection)
        except Exception as e:
            logger.error(f"Failed to generate reflection: {str(e)}")
            # Silently fail if reflection generation fails
            pass
