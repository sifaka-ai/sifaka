"""
Implementation of a Self-RAG critic using composition over inheritance.

This module provides a critic implementation that uses Self-Reflective Retrieval-Augmented
Generation to evaluate, validate, and improve text outputs. It follows the
composition over inheritance pattern.

Based on Self-RAG: https://arxiv.org/abs/2310.11511

## Component Lifecycle

### Self-RAG Critic Implementation Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Retriever setup
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Retrieval decision
   - Information retrieval
   - Response generation
   - Reflection generation
   - Text improvement

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

## Examples

```python
from sifaka.critics.implementations.self_rag_implementation import SelfRAGCriticImplementation
from sifaka.critics.base import create_composition_critic
from sifaka.critics.models import SelfRAGCriticConfig
from sifaka.models.providers import OpenAIProvider
from sifaka.retrieval import create_simple_retriever

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a retriever
documents = {
    "health_insurance": "To file a claim for health reimbursement, follow these steps: "
                       "1. Complete the claim form with your personal and policy information. "
                       "2. Attach all original receipts and medical documentation. "
                       "3. Make copies of all documents for your records.",
    "travel_insurance": "For travel insurance claims, you need to provide: "
                       "1. Proof of travel (boarding passes, itinerary). "
                       "2. Incident report or documentation of the event. "
                       "3. Original receipts for expenses being claimed."
}
retriever = create_simple_retriever(documents=documents)

# Create a configuration
config = SelfRAGCriticConfig(
    name="self_rag_critic",
    description="A critic that uses self-reflective retrieval-augmented generation",
    system_prompt="You are an expert at deciding when to retrieve information.",
    temperature=0.7,
    max_tokens=1000,
    retrieval_threshold=0.5
)

# Create a self-rag critic implementation
implementation = SelfRAGCriticImplementation(config, provider, retriever)

# Create a critic with the implementation
critic = create_composition_critic(
    name="my_self_rag_critic",
    description="A critic that uses retrieval to improve responses",
    implementation=implementation
)

# Use the critic
text = "What are the steps to file a health insurance claim?"
is_valid = critic.validate(text)
improved = critic.improve(text)
feedback = critic.critique(text)
```
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from ..models import SelfRAGCriticConfig
from ..utils.state import CriticState
from ...retrieval import Retriever

# Configure logging
logger = logging.getLogger(__name__)


class SelfRAGCriticImplementation:
    """
    Implementation of a Self-RAG critic using language models with retrieval.

    This class implements the CriticImplementation protocol for a Self-RAG critic
    that uses language models and retrieval to evaluate, validate, and improve text.

    ## Lifecycle Management

    The SelfRAGCriticImplementation manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Sets up retriever
       - Initializes state

    2. **Operation**
       - Decides whether to retrieve
       - Retrieves relevant information
       - Generates responses
       - Reflects on responses

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    1. **Input Validation**
       - Empty text checks
       - Type validation
       - Format verification

    2. **Retrieval Errors**
       - Query processing errors
       - Source unavailable
       - No relevant results

    3. **Generation Errors**
       - Model unavailable
       - Token limit exceeded
       - Invalid responses
    """

    def __init__(
        self,
        config: SelfRAGCriticConfig,
        llm_provider: Any,
        retriever: Retriever,
        prompt_factory: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Self-RAG critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use
            retriever: Retriever to use for information retrieval
            prompt_factory: Optional prompt factory for creating prompts

        Raises:
            ValueError: If config is invalid
            TypeError: If llm_provider or retriever is invalid
        """
        # Initialize state
        self._state = CriticState()
        self._state.initialized = False

        # Validate inputs
        if not config:
            raise ValueError("Config must be provided")
        if not llm_provider:
            raise ValueError("Language model provider must be provided")
        if not retriever:
            raise ValueError("Retriever must be provided")

        # Store components in state
        self._state.model = llm_provider
        self._state.config = config
        self._state.initialized = True

        # Store configuration and retriever in state cache
        self._state.cache = {
            "retriever": retriever,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "retrieval_threshold": config.retrieval_threshold,
            "retrieval_prompt_template": config.retrieval_prompt_template,
            "generation_prompt_template": config.generation_prompt_template,
            "reflection_prompt_template": config.reflection_prompt_template,
        }

        # Store prompt factory if provided
        if prompt_factory:
            self._state.cache["prompt_factory"] = prompt_factory

    def _check_input(self, text: str) -> None:
        """
        Check if input is valid.

        Args:
            text: The text to check

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critic is not properly initialized
        """
        if not self._state.initialized:
            raise RuntimeError("SelfRAGCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _get_task_from_feedback(self, feedback: Optional[Any] = None) -> str:
        """
        Get task from feedback.

        Args:
            feedback: Optional feedback containing the task

        Returns:
            The task as a string
        """
        if feedback is None:
            return "Answer the following question or complete the following task."

        if isinstance(feedback, dict) and "task" in feedback:
            task = feedback.get("task", "")
            if task:
                return task

        if isinstance(feedback, str) and feedback.strip():
            return feedback

        return "Answer the following question or complete the following task."

    def validate_impl(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        self._check_input(text)

        # For SelfRAG, validation is always True as it focuses on improvement
        return True

    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str:
        """
        Improve text through self-reflective retrieval-augmented generation.

        Args:
            text: Text to improve
            feedback: Optional feedback to guide improvement, can be a dict with 'task' key

        Returns:
            str: Improved text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If improvement fails
        """
        self._check_input(text)

        # Get task from feedback
        task = self._get_task_from_feedback(feedback)

        # Run the full Self-RAG process
        result = self._run_self_rag(task, text)

        # Return the improved response
        return result.get("response", text)

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

        # Use a generic task for critique
        task = "Evaluate the quality and accuracy of this text."

        # Run the full Self-RAG process
        result = self._run_self_rag(task, text)

        # Extract reflection as feedback
        reflection = result.get("reflection", "")

        # Parse reflection for issues and suggestions
        issues = []
        suggestions = []

        # Extract issues and suggestions from reflection
        for line in reflection.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                if (
                    "should" in line.lower()
                    or "could" in line.lower()
                    or "recommend" in line.lower()
                ):
                    suggestions.append(line[2:])
                else:
                    issues.append(line[2:])

        # Calculate score based on issues
        score = 1.0 if not issues else max(0.0, 1.0 - (len(issues) * 0.1))

        return {
            "score": score,
            "feedback": reflection,
            "issues": issues,
            "suggestions": suggestions,
        }

    def warm_up_impl(self) -> None:
        """
        Warm up the critic implementation.

        This method initializes any resources needed by the critic implementation.
        """
        # Already initialized in __init__
        pass

    def _run_self_rag(self, task: str, response: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full Self-RAG process.

        Args:
            task: The task or question to process
            response: Optional initial response to improve

        Returns:
            Dictionary containing response, retrieval_query, retrieved_context, and reflection

        Raises:
            ValueError: If task is empty
            RuntimeError: If critic is not properly initialized
        """
        if not self._state.initialized:
            raise RuntimeError("SelfRAGCriticImplementation not properly initialized")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        # Step 1: Ask the model if it needs retrieval
        retrieval_template = self._state.cache.get("retrieval_prompt_template")
        if not retrieval_template:
            retrieval_template = (
                "Do you need external knowledge to answer this question? If so, what would you search for?\n\n"
                "Task:\n{task}\n\n"
                "If you need external knowledge, respond with a search query.\n"
                "If you don't need external knowledge, respond with 'No external knowledge needed.'"
            )
        retrieval_prompt = retrieval_template.format(task=task)

        retrieval_query = self._state.model.generate(
            retrieval_prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        # Check if retrieval is needed
        no_retrieval_phrases = [
            "no external knowledge",
            "no additional knowledge",
            "no retrieval",
            "no search",
            "not needed",
            "unnecessary",
        ]

        if any(phrase in retrieval_query.lower() for phrase in no_retrieval_phrases):
            context = ""
        else:
            # Step 2: Retrieve information
            retriever = self._state.cache.get("retriever")
            if not retriever:
                raise RuntimeError("Retriever not initialized")
            context = retriever.retrieve(retrieval_query)

        # Step 3: Generate response with (or without) retrieved context
        generation_template = self._state.cache.get("generation_prompt_template")
        if not generation_template:
            generation_template = (
                "Please answer the following task using the provided context (if available).\n\n"
                "Context:\n{context}\n\n"
                "Task:\n{task}\n\n"
                "Answer:"
            )
        generation_prompt = generation_template.format(context=context, task=task)

        if response is None or not response.strip():
            # Generate new response
            response = self._state.model.generate(
                generation_prompt,
                system_prompt=self._state.cache.get("system_prompt", ""),
                temperature=self._state.cache.get("temperature", 0.7),
                max_tokens=self._state.cache.get("max_tokens", 1000),
            ).strip()

        # Step 4: Ask model to reflect on whether the answer is good and the retrieval helped
        reflection_template = self._state.cache.get("reflection_prompt_template")
        if not reflection_template:
            reflection_template = (
                "Reflect on whether your answer used relevant information and addressed the task accurately.\n\n"
                "Task:\n{task}\n\n"
                "Retrieved Context:\n{context}\n\n"
                "Your Response:\n{response}\n\n"
                "Reflection:"
            )
        reflection_prompt = reflection_template.format(
            task=task, context=context, response=response
        )

        reflection = self._state.model.generate(
            reflection_prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        return {
            "response": response,
            "retrieval_query": retrieval_query,
            "retrieved_context": context,
            "reflection": reflection,
        }
