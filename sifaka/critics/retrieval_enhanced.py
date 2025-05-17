"""
Retrieval-enhanced critics for Sifaka.

This module provides base classes and utilities for enhancing critics with retrieval capabilities.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Type, TypeVar

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError, RetrieverError
from sifaka.retrievers.base import Retriever
from sifaka.retrievers.augmenter import RetrievalAugmenter

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Critic)


class RetrievalEnhancedCritic(Critic):
    """Base class for retrieval-enhanced critics.
    
    This class provides a foundation for enhancing any critic with retrieval capabilities.
    It wraps an existing critic and adds retrieval functionality to its critique and improve methods.
    
    Attributes:
        base_critic: The base critic to enhance with retrieval.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
    """
    
    def __init__(
        self,
        base_critic: Critic,
        retrieval_augmenter: RetrievalAugmenter,
        include_passages_in_critique: bool = True,
        include_passages_in_improve: bool = True,
        max_passages: int = 5,
        **options: Any,
    ):
        """Initialize the retrieval-enhanced critic.
        
        Args:
            base_critic: The base critic to enhance with retrieval.
            retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
            include_passages_in_critique: Whether to include retrieved passages in the critique.
            include_passages_in_improve: Whether to include retrieved passages in the improve method.
            max_passages: Maximum number of passages to retrieve.
            **options: Additional options to pass to the base critic.
            
        Raises:
            ImproverError: If the base critic or retrieval augmenter is not provided.
        """
        if not base_critic:
            raise ImproverError("Base critic not provided")
        
        if not retrieval_augmenter:
            raise ImproverError("Retrieval augmenter not provided")
        
        # Initialize with the base critic's model and system prompt
        super().__init__(
            model=base_critic.model,
            system_prompt=base_critic.system_prompt,
            temperature=base_critic.temperature,
            **options,
        )
        
        self.base_critic = base_critic
        self.retrieval_augmenter = retrieval_augmenter
        self.include_passages_in_critique = include_passages_in_critique
        self.include_passages_in_improve = include_passages_in_improve
        self.max_passages = max_passages
        
        # Store retrieval context for reuse between critique and improve
        self._retrieval_context: Optional[Dict[str, Any]] = None
    
    def validate(self, text: str) -> bool:
        """Validate text using the base critic.
        
        Args:
            text: The text to validate.
            
        Returns:
            True if the text is valid, False otherwise.
        """
        return self.base_critic.validate(text)
    
    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the base critic enhanced with retrieval.
        
        Args:
            text: The text to critique.
            
        Returns:
            A dictionary with critique information.
            
        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        try:
            # Get retrieval context
            self._retrieval_context = self.retrieval_augmenter.get_retrieval_context(text)
            
            # Get base critique from the base critic
            base_critique = self.base_critic._critique(text)
            
            # Add retrieval context to the critique if enabled
            if self.include_passages_in_critique and self._retrieval_context["passage_count"] > 0:
                base_critique["retrieved_passages"] = self._retrieval_context["passages"]
                base_critique["formatted_passages"] = self._retrieval_context["formatted_passages"]
                
                # Add a suggestion to incorporate retrieved information if not already present
                if "suggestions" in base_critique and isinstance(base_critique["suggestions"], list):
                    base_critique["suggestions"].append("Incorporate information from retrieved passages")
                
                # Add an issue related to retrieved information if not already present
                if "issues" in base_critique and isinstance(base_critique["issues"], list):
                    base_critique["issues"].append("Could be enhanced with additional information from retrieved sources")
            
            return base_critique
        
        except Exception as e:
            logger.error(f"Error critiquing text with retrieval enhancement: {str(e)}")
            raise ImproverError(f"Error critiquing text with retrieval enhancement: {str(e)}")
    
    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the base critic enhanced with retrieval.
        
        Args:
            text: The text to improve.
            critique: The critique information.
            
        Returns:
            The improved text.
            
        Raises:
            ImproverError: If the text cannot be improved.
        """
        try:
            # If we don't have retrieval context or it's not enabled for improve, use base improve
            if not self.include_passages_in_improve or not self._retrieval_context or self._retrieval_context["passage_count"] == 0:
                return self.base_critic._improve(text, critique)
            
            # Create a custom improve prompt that includes retrieved information
            prompt = self._create_improve_prompt(text, critique)
            
            # Generate improved text
            response = self._generate(prompt)
            
            # Extract improved text from response
            improved_text = response.strip()
            
            # Remove any markdown code block markers
            if improved_text.startswith("```") and improved_text.endswith("```"):
                improved_text = improved_text[3:-3].strip()
            
            return improved_text
        
        except Exception as e:
            logger.error(f"Error improving text with retrieval enhancement: {str(e)}")
            raise ImproverError(f"Error improving text with retrieval enhancement: {str(e)}")
    
    def _create_improve_prompt(self, text: str, critique: Dict[str, Any]) -> str:
        """Create a prompt for improving text with retrieval enhancement.
        
        Args:
            text: The text to improve.
            critique: The critique information.
            
        Returns:
            A prompt for improving the text.
        """
        # Format issues and suggestions
        issues_str = ""
        if "issues" in critique and isinstance(critique["issues"], list) and critique["issues"]:
            issues_str = "Issues:\n" + "\n".join(f"- {issue}" for issue in critique["issues"])
        
        suggestions_str = ""
        if "suggestions" in critique and isinstance(critique["suggestions"], list) and critique["suggestions"]:
            suggestions_str = "Suggestions:\n" + "\n".join(f"- {suggestion}" for suggestion in critique["suggestions"])
        
        # Create the prompt
        prompt = f"""
        Please improve the following text based on the critique and retrieved information:
        
        Original text:
        ```
        {text}
        ```
        
        {issues_str}
        
        {suggestions_str}
        
        Retrieved information:
        {self._retrieval_context["formatted_passages"]}
        
        Instructions:
        1. Address all issues mentioned in the critique
        2. Incorporate relevant information from the retrieved passages
        3. Maintain the original style and tone
        4. Ensure factual accuracy and logical coherence
        5. Cite the passage number when incorporating information (e.g., [Passage 1])
        
        Improved text:
        """
        
        return prompt


def enhance_critic_with_retrieval(
    critic: T,
    retrieval_augmenter: RetrievalAugmenter,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> RetrievalEnhancedCritic:
    """Enhance a critic with retrieval capabilities.
    
    Args:
        critic: The critic to enhance with retrieval.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the enhanced critic.
        
    Returns:
        A retrieval-enhanced critic.
        
    Raises:
        ImproverError: If the critic or retrieval augmenter is not provided.
    """
    return RetrievalEnhancedCritic(
        base_critic=critic,
        retrieval_augmenter=retrieval_augmenter,
        include_passages_in_critique=include_passages_in_critique,
        include_passages_in_improve=include_passages_in_improve,
        max_passages=max_passages,
        **options,
    )
