"""
Self-RAG critic module for Sifaka.

This module implements the Self-Reflective Retrieval-Augmented Generation approach for critics,
which enables language models to decide when and what to retrieve, and reflect on the
relevance and utility of the retrieved information.

## Overview
The SelfRAGCritic is a specialized implementation of the critic interface
that combines retrieval-augmented generation with self-reflection. It enables
language models to decide when and what information to retrieve, and then
reflect on the relevance and utility of the retrieved information for improving
text quality. This approach enhances the model's ability to provide accurate,
well-informed critiques and improvements.

## Components
- **SelfRAGCritic**: Main class implementing TextValidator, TextImprover, and TextCritic
- **create_self_rag_critic**: Factory function for creating SelfRAGCritic instances
- **Retriever**: Component that retrieves relevant information from external sources
- **PromptManager**: Creates prompts for different stages of the Self-RAG process

## Architecture
The SelfRAGCritic follows a retrieval-augmented architecture:
- Uses standardized state management with _state_manager
- Implements a multi-stage process (retrieval decision, retrieval, generation, reflection)
- Provides automatic retrieval decisions based on task requirements
- Implements both sync and async interfaces
- Provides comprehensive error handling and recovery
- Tracks performance and retrieval statistics

## Usage Examples
```python
from sifaka.critics.implementations.self_rag import create_self_rag_critic
from sifaka.models.providers import OpenAIProvider
from sifaka.retrieval import SimpleRetriever

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a retriever with some documents
documents = [
    "Health insurance claims must be filed within 90 days of service.",
    "To file a claim, you need to submit the claim form and receipts.",
    "Claims can be submitted online or by mail."
]
retriever = SimpleRetriever(documents=documents)

# Create a Self-RAG critic
critic = create_self_rag_critic(
    llm_provider=provider,
    retriever=retriever,
    system_prompt="You are an expert assistant that provides accurate information.",
    temperature=0.7,
    retrieval_threshold=0.6
)

# Use the critic to improve text
task = "What are the steps to file a claim for health reimbursement?"
result = critic.run(task, response=None) if critic else ""
print(f"Response: {result['response']}")
print(f"Reflection: {result['reflection']}")

# Critique existing text
response = "To file a claim, just send an email."
critique = critic.critique(response, {"task": task}) if critic else ""
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")

# Improve existing text
improved_text = critic.improve(response, {"task": task}) if critic else ""
print(f"Improved text: {improved_text}")
```

## Error Handling
The module implements comprehensive error handling for:
- Input validation (empty text, invalid types)
- Initialization errors (missing provider, invalid config)
- Processing errors (model failures, timeout issues)
- Resource management (cleanup, state preservation)

## Component Lifecycle
1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Retriever setup
   - Factory initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Retrieval decision
   - Information retrieval
   - Response generation
   - Reflection generation
   - Critique generation
   - Text improvement

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

## References
Based on Self-RAG: https://arxiv.org/abs/2310.11511
"""

import json
import time
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import ConfigDict, Field, PrivateAttr

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...core.base import BaseResult as CriticResult
from sifaka.utils.config.critics import SelfRAGCriticConfig
from sifaka.interfaces import TextCritic, TextImprover, TextValidator
from ...retrieval import Retriever


class SelfRAGCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the Self-Reflective Retrieval-Augmented Generation approach.

    This critic enables language models to decide when and what to retrieve,
    and reflect on the relevance and utility of the retrieved information. It implements
    the TextValidator, TextImprover, and TextCritic interfaces to provide a comprehensive
    set of text analysis capabilities with retrieval augmentation.

    Based on Self-RAG: https://arxiv.org/abs/2310.11511

    ## Architecture
    The SelfRAGCritic follows a retrieval-augmented architecture:
    - Uses standardized state management with _state_manager
    - Implements a multi-stage process (retrieval decision, retrieval, generation, reflection)
    - Provides automatic retrieval decisions based on task requirements
    - Implements both sync and async interfaces
    - Provides comprehensive error handling and recovery
    - Tracks performance and retrieval statistics

    ## Lifecycle
    1. **Initialization**: Set up with configuration and dependencies
       - Create/validate config
       - Initialize language model provider
       - Set up retriever component
       - Initialize memory manager
       - Set up state tracking

    2. **Operation**: Process text through various methods
       - validate(): Check if text meets quality standards
       - critique(): Analyze text and provide detailed feedback
       - improve(): Enhance text through retrieval-augmented generation
       - run(): Execute the full Self-RAG process (retrieval decision, retrieval, generation, reflection)

    3. **Cleanup**: Manage resources and finalize state
       - Release resources
       - Reset state
       - Log final status
       - Track performance metrics

    ## Examples
    ```python
    from sifaka.critics.implementations.self_rag import create_self_rag_critic
    from sifaka.models.providers import OpenAIProvider
    from sifaka.retrieval import SimpleRetriever

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a retriever with some documents
    documents = [
        "Health insurance claims must be filed within 90 days of service.",
        "To file a claim, you need to submit the claim form and receipts.",
        "Claims can be submitted online or by mail."
    ]
    retriever = SimpleRetriever(documents=documents)

    # Create a Self-RAG critic
    critic = create_self_rag_critic(
        llm_provider=provider,
        retriever=retriever
    )

    # Use the critic to improve text
    task = "What are the steps to file a claim for health reimbursement?"
    result = critic.run(task, response=None) if critic else ""
    print(f"Response: {result['response']}")
    print(f"Reflection: {result['reflection']}")
    ```

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model: Language model provider
      - retriever: Retrieval component
      - initialized: Initialization status
      - cache: Temporary data storage
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: SelfRAGCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        retriever: Retriever,
        config: Optional[Optional[SelfRAGCriticConfig]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Self-RAG critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: Language model provider to use
            retriever: Retriever to use for information retrieval
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider or retriever is not a valid provider
        """
        # Validate required parameters
        if llm_provider is None:
            raise ValueError("llm_provider cannot be None")
        if retriever is None:
            raise ValueError("retriever cannot be None")

        # Create config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_SELF_RAG_CRITIC_CONFIG

            config = DEFAULT_SELF_RAG_CRITIC_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store components in state
            self._state_manager.update("model", llm_provider)
            self._state_manager.update("retriever", retriever)

            # Store configuration in cache
            cache = {
                "retrieval_threshold": config and config.retrieval_threshold,
                "retrieval_prompt_template": config and config.retrieval_prompt_template,
                "generation_prompt_template": config and config.generation_prompt_template,
                "reflection_prompt_template": config and config.reflection_prompt_template,
                "system_prompt": config and config.system_prompt,
                "temperature": config and config.temperature,
                "max_tokens": config and config.max_tokens,
                "reflection_enabled": config and config.reflection_enabled,
            }
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time() if time else 0)
        except Exception as e:
            self.record_error(e)
            raise ValueError(f"Failed to initialize SelfRAGCritic: {str(e)}") from e

    def _check_input(self, text: str) -> None:
        """
        Validate input text and initialization state.

        Args:
            text: The text to validate

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not self._state_manager.get("initialized", False):
            raise RuntimeError("SelfRAGCritic not properly initialized")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Task string

        Raises:
            ValueError: If metadata is None or missing task key
        """
        if metadata is None or "task" not in metadata:
            raise ValueError("metadata must contain a 'task' key")
        return metadata["task"]

    def process(self, input: str) -> CriticResult:
        """
        Process the input text and return a result.

        This is the main method required by BaseComponent.

        Args:
            input: The text to process

        Returns:
            CriticResult: The result of processing the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time() if time else 0

        try:
            # Validate input
            if not isinstance(input, str) or not input.strip():
                raise ValueError("Input must be a non-empty string")

            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRAGCritic not properly initialized")

            # Create a default task if none provided
            task = "Provide information about the following text"

            # Run the Self-RAG process
            result = self.run(task, input) if self else ""

            # Extract reflection as feedback
            reflection = result.get("reflection", "") if result else ""

            # Parse reflection for issues and suggestions
            issues = []
            suggestions = []

            # Extract issues and suggestions from reflection
            for line in reflection.split("\n") if reflection else "":
                line = line.strip() if line else ""
                if line.startswith("- ") if line else "" or line.startswith("* ") if line else "":
                    if (
                        "should" in line.lower()
                        if line
                        else (
                            "" or "could" in line.lower()
                            if line
                            else "" or "recommend" in line.lower() if line else ""
                        )
                    ):
                        suggestions.append(line[2:]) if suggestions else ""
                    else:
                        issues.append(line[2:]) if issues else ""

            # Calculate score based on issues
            score = 1.0 if not issues else 0.5

            # Create result
            critic_result = CriticResult(
                passed=True,  # Self-RAG critics always pass
                message=result.get("response", "") if result else "",
                metadata={
                    "operation": "process",
                    "retrieval_query": result.get("retrieval_query", "") if result else "",
                    "retrieved_context": result.get("retrieved_context", "") if result else "",
                    "reflection": reflection,
                },
                score=score,
                issues=issues,
                suggestions=suggestions,
                processing_time_ms=(time.time() - start_time) * 1000 if time else 0,
            )

            # Update statistics
            self.update_statistics(critic_result) if self else ""

            return critic_result

        except Exception as e:
            self.record_error(e)
            processing_time = (time.time() - start_time) * 1000 if time else 0
            return CriticResult(
                passed=False,
                message=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
                score=0.0,
                issues=[f"Processing error: {str(e)}"],
                suggestions=["Retry with different input"],
                processing_time_ms=processing_time,
            )

    @property
    def config(self) -> SelfRAGCriticConfig:
        """Get the Self-RAG critic configuration."""
        return cast(SelfRAGCriticConfig, self._config)

    def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate
            metadata: Optional metadata containing the task

        Returns:
            True if text is valid, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text) if self else ""

        # For SelfRAG, validation is always True as it focuses on improvement
        return True

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text) if self else ""

        # Get task from metadata
        task = self._get_task_from_metadata(metadata) if self else ""

        # Run the full Self-RAG process
        result = self.run(task, text, metadata) if self else ""

        # Extract reflection as feedback
        reflection = result.get("reflection", "") if result else ""

        # Parse reflection for issues and suggestions
        issues = []
        suggestions = []

        # Extract issues and suggestions from reflection
        for line in reflection.split("\n") if reflection else "":
            line = line.strip() if line else ""
            if line.startswith("- ") if line else "" or line.startswith("* ") if line else "":
                if (
                    "should" in line.lower()
                    if line
                    else (
                        "" or "could" in line.lower()
                        if line
                        else "" or "recommend" in line.lower() if line else ""
                    )
                ):
                    suggestions.append(line[2:]) if suggestions else ""
                else:
                    issues.append(line[2:]) if issues else ""

        # Calculate score based on issues
        score = 1.0 if not issues else 0.5

        return {
            "score": score,
            "feedback": reflection,
            "issues": issues,
            "suggestions": suggestions,
        }

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text through self-reflective retrieval-augmented generation.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text) if self else ""

        # Get task from metadata
        task = self._get_task_from_metadata(metadata) if self else ""

        # Run the full Self-RAG process
        result = self.run(task, text, metadata) if self else ""

        # Return the improved response
        return result.get("response", text) if result else ""

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Use the feedback as context for generation
        cache = self._state_manager.get("cache", {})
        generation_template = cache.get("generation_prompt_template")
        if not generation_template:
            generation_template = (
                "Please answer the following task using the provided context (if available).\n\n"
                "Context:\n{context}\n\n"
                "Task:\n{task}\n\n"
                "Answer:"
            )
        generation_prompt = generation_template.format(
            context=f"Feedback: {feedback}",
            task=f"Improve the following text based on the feedback:\n{text}",
        )

        # Generate improved response
        model = self._state_manager.get("model")
        improved_text = model.generate(
            generation_prompt,
            system_prompt=cache.get("system_prompt", ""),
            temperature=cache.get("temperature", 0.7),
            max_tokens=cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    def run(
        self,
        task: str,
        response: Optional[Optional[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full Self-RAG process.

        Args:
            task: The task or question to process
            response: Optional initial response to improve
            metadata: Optional metadata

        Returns:
            Dictionary containing response, retrieval_query, retrieved_context, and reflection

        Raises:
            ValueError: If task is empty
            RuntimeError: If critic is not properly initialized
        """
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("SelfRAGCritic not properly initialized")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        # Step 1: Ask model to decide whether to retrieve and what to retrieve
        cache = self._state_manager.get("cache", {})
        retrieval_template = cache.get("retrieval_prompt_template")
        if not retrieval_template:
            retrieval_template = (
                "For the following task, decide whether you need to retrieve information and what to retrieve.\n\n"
                "Task:\n{task}\n\n"
                "Current Response (if any):\n{response}\n\n"
                "Do you need to retrieve information? If yes, what would you like to retrieve? "
                "If no, explain why retrieval is not necessary."
            )
        retrieval_prompt = retrieval_template.format(
            task=task, response=response or "No response yet."
        )

        model = self._state_manager.get("model")
        retrieval_decision = model.generate(
            retrieval_prompt,
            system_prompt=cache.get("system_prompt", ""),
            temperature=cache.get("temperature", 0.7),
            max_tokens=cache.get("max_tokens", 1000),
        ).strip()

        # Step 2: Extract retrieval query and decide whether to retrieve
        retrieval_threshold = cache.get("retrieval_threshold", 0.5)
        should_retrieve = False
        retrieval_query = ""

        if "yes" in retrieval_decision.lower() or "retrieve" in retrieval_decision.lower():
            should_retrieve = True
            # Extract query from decision
            retrieval_query = task  # Default to using the task as the query
            for line in retrieval_decision.split("\n"):
                if "query:" in line.lower() or "retrieve:" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        retrieval_query = parts[1].strip()
                        break

        # Step 3: Retrieve information and generate response
        context = ""
        if should_retrieve and retrieval_query:
            # Retrieve information
            retriever = self._state_manager.get("retriever")
            results = retriever.retrieve(retrieval_query, top_k=3)
            if results:
                context = "\n\n".join([result.content for result in results])

        # Generate response if not provided
        if response is None:
            generation_template = cache.get("generation_prompt_template")
            if not generation_template:
                generation_template = (
                    "Please answer the following task using the provided context (if available).\n\n"
                    "Context:\n{context}\n\n"
                    "Task:\n{task}\n\n"
                    "Answer:"
                )
            generation_prompt = generation_template.format(
                context=context or "No relevant information found.", task=task
            )

            response = model.generate(
                generation_prompt,
                system_prompt=cache.get("system_prompt", ""),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            ).strip()

        # Step 4: Ask model to reflect on whether the answer is good and the retrieval helped
        reflection = ""
        if cache.get("reflection_enabled", True):
            reflection_template = cache.get("reflection_prompt_template")
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

            reflection = model.generate(
                reflection_prompt,
                system_prompt=cache.get("system_prompt", ""),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            ).strip()

        return {
            "response": response,
            "retrieval_query": retrieval_query,
            "retrieved_context": context,
            "reflection": reflection,
        }

    # Async methods are implemented similarly to the synchronous ones
    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Asynchronously validate text."""
        self._check_input(text)
        return True

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously analyze text and provide detailed feedback."""
        self._check_input(text)
        task = self._get_task_from_metadata(metadata)
        result = await self.arun(task, text, metadata)

        # Extract reflection as feedback
        reflection = result.get("reflection", "")
        issues = []
        suggestions = []

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

        score = 1.0 if not issues else 0.5

        return {
            "score": score,
            "feedback": reflection,
            "issues": issues,
            "suggestions": suggestions,
        }

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronously improve text through self-reflective retrieval-augmented generation."""
        self._check_input(text)
        task = self._get_task_from_metadata(metadata)
        result = await self.arun(task, text, metadata)
        return result.get("response", text)

    async def arun(
        self, task: str, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously run the full Self-RAG process."""
        # For simplicity, we'll use the synchronous implementation for now
        return self.run(task, response, metadata)


def create_self_rag_critic(
    llm_provider: Any,
    retriever: Retriever,
    name: str = "self_rag_critic",
    description: str = "Improves text through self-reflective retrieval-augmented generation",
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retrieval_threshold: Optional[float] = None,
    retrieval_prompt_template: Optional[Optional[str]] = None,
    generation_prompt_template: Optional[Optional[str]] = None,
    reflection_prompt_template: Optional[Optional[str]] = None,
    config: Optional[Union[Dict[str, Any], SelfRAGCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRAGCritic:
    """
    Create a Self-RAG critic with the given parameters.

    This factory function creates a configured Self-RAG critic instance
    that enables language models to decide when and what to retrieve,
    and reflect on the relevance and utility of the retrieved information.
    It provides a convenient way to create a self-reflective retrieval-augmented
    critic with customized settings.

    ## Architecture
    The factory function follows the Factory Method pattern to:
    - Create standardized configuration objects
    - Instantiate critic classes with consistent parameters
    - Support optional parameter overrides
    - Provide type safety through return types
    - Handle error cases gracefully

    ## Lifecycle
    1. **Configuration**: Create and validate configuration
       - Use default configuration as base
       - Apply provided parameter overrides
       - Validate configuration values
       - Handle configuration errors

    2. **Instantiation**: Create and initialize critic
       - Create SelfRAGCritic instance
       - Initialize with resolved dependencies
       - Apply configuration
       - Handle initialization errors

    ## Examples
    ```python
    from sifaka.critics.implementations.self_rag import create_self_rag_critic
    from sifaka.models.providers import OpenAIProvider
    from sifaka.retrieval import SimpleRetriever

    # Create with basic parameters
    provider = OpenAIProvider(api_key="your-api-key")
    documents = [
        "Health insurance claims must be filed within 90 days of service.",
        "To file a claim, you need to submit the claim form and receipts.",
        "Claims can be submitted online or by mail."
    ]
    retriever = SimpleRetriever(documents=documents)
    critic = create_self_rag_critic(
        llm_provider=provider,
        retriever=retriever,
        system_prompt="You are an expert assistant that provides accurate information.",
        temperature=0.7,
        retrieval_threshold=0.6
    )

    # Create with custom configuration
    from sifaka.utils.config.critics import SelfRAGCriticConfig
    config = SelfRAGCriticConfig(
        name="custom_self_rag_critic",
        description="A custom Self-RAG critic",
        system_prompt="You are an expert assistant that provides accurate information.",
        temperature=0.5,
        max_tokens=2000,
        retrieval_threshold=0.7
    )
    critic = create_self_rag_critic(
        llm_provider=provider,
        retriever=retriever,
        config=config
    )
    ```

    Args:
        llm_provider: The language model provider to use
        retriever: The retriever to use for information retrieval
        name: The name of the critic
        description: A description of the critic
        min_confidence: The minimum confidence threshold
        max_attempts: The maximum number of attempts
        cache_size: The size of the cache
        priority: The priority of the critic
        cost: The cost of the critic
        system_prompt: The system prompt to use
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate
        retrieval_threshold: The threshold for deciding whether to retrieve
        retrieval_prompt_template: The template for retrieval prompts
        generation_prompt_template: The template for generation prompts
        reflection_prompt_template: The template for reflection prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional configuration parameters

    Returns:
        A configured SelfRAGCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider or retriever is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_SELF_RAG_CRITIC_CONFIG

            config = DEFAULT_SELF_RAG_CRITIC_CONFIG.model_copy()

            # Update config with provided values
            updates = {}
            if name is not None:
                updates["name"] = name
            if description is not None:
                updates["description"] = description
            if system_prompt is not None:
                updates["system_prompt"] = system_prompt
            if temperature is not None:
                updates["temperature"] = temperature
            if max_tokens is not None:
                updates["max_tokens"] = max_tokens
            if min_confidence is not None:
                updates["min_confidence"] = min_confidence
            if max_attempts is not None:
                updates["max_attempts"] = max_attempts
            if cache_size is not None:
                updates["cache_size"] = cache_size
            if priority is not None:
                updates["priority"] = priority
            if cost is not None:
                updates["cost"] = cost
            if retrieval_threshold is not None:
                updates["retrieval_threshold"] = retrieval_threshold
            if retrieval_prompt_template is not None:
                updates["retrieval_prompt_template"] = retrieval_prompt_template
            if generation_prompt_template is not None:
                updates["generation_prompt_template"] = generation_prompt_template
            if reflection_prompt_template is not None:
                updates["reflection_prompt_template"] = reflection_prompt_template

            # Add any additional kwargs
            updates.update(kwargs)

            config = config.model_copy(update=updates)
        elif isinstance(config, dict):
            from sifaka.utils.config.critics import SelfRAGCriticConfig

            config = SelfRAGCriticConfig(**config)

        # Create and return the critic
        return SelfRAGCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            retriever=retriever,
            config=config,
        )
    except Exception as e:
        raise ValueError(f"Failed to create Self-RAG critic: {str(e)}") from e
