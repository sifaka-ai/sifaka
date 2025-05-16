"""
Self-RAG Critic Implementation

This module implements the Self-RAG (Retrieval-Augmented Generation) critic,
which enhances text evaluation and improvement through dynamic retrieval.
It enables language models to decide when to retrieve, what to retrieve,
and how to use retrieved information through self-reflection.

Based on Self-RAG: https://arxiv.org/abs/2310.11511
"""

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, cast, TypeVar

from pydantic import ConfigDict, PrivateAttr
from sifaka.core.base import BaseComponent
from sifaka.core.config import BaseConfig
from sifaka.core.results import BaseResult, CriticResult, create_critic_result
from sifaka.interfaces.critic import TextCritic, TextImprover, TextValidator, CritiqueResult
from sifaka.retrieval.core import RetrievedDocument
from sifaka.utils.state import create_critic_state

# Import RetrieverProtocol with type parameter
from sifaka.interfaces.retrieval import RetrieverProtocol
from sifaka.utils.errors import CriticError
from sifaka.utils.logging import get_logger

# Type variable for retriever document type
T = TypeVar("T")

# Constants - defined at module level to avoid recreation
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can retrieve relevant information "
    "to provide accurate and informative responses. When necessary, use "
    "the retrieved information to enhance your response. Always indicate "
    "when you're using retrieved information by citing the source."
)

DEFAULT_RETRIEVAL_PROMPT_TEMPLATE = (
    "Given the following task, decide whether retrieval would be helpful. "
    "Task: {task}\n\n"
    "Should I retrieve information? Respond with YES or NO and a brief explanation."
)

DEFAULT_GENERATION_PROMPT_TEMPLATE = (
    "Task: {task}\n\n"
    "Retrieved Information:\n{retrieved_info}\n\n"
    "Using the retrieved information above, respond to the task."
)

DEFAULT_REFLECTION_PROMPT_TEMPLATE = (
    "Task: {task}\n\n"
    "Retrieved Information:\n{retrieved_info}\n\n"
    "Your Response: {response}\n\n"
    "Reflect on your response. Did you use the retrieved information appropriately? "
    "Is the response accurate and helpful? What could be improved?"
)

logger = get_logger(__name__)


class SelfRAGCriticConfig(BaseConfig):
    """
    Configuration for the Self-RAG critic.

    This class defines the configuration parameters specific to the Self-RAG critic,
    extending the base configuration with parameters for retrieval, generation,
    and reflection.

    Attributes:
        retrieval_threshold: Threshold for retrieval relevance
        system_prompt: System prompt for the language model
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        retrieval_prompt_template: Template for retrieval decision prompts
        generation_prompt_template: Template for generation prompts
        reflection_prompt_template: Template for reflection prompts
        reflection_enabled: Whether to enable reflection
    """

    retrieval_threshold: float = 0.7
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 1000
    retrieval_prompt_template: str = DEFAULT_RETRIEVAL_PROMPT_TEMPLATE
    generation_prompt_template: str = DEFAULT_GENERATION_PROMPT_TEMPLATE
    reflection_prompt_template: str = DEFAULT_REFLECTION_PROMPT_TEMPLATE
    reflection_enabled: bool = True


class SelfRAGCritic(BaseComponent[str, BaseResult], TextValidator, TextImprover, TextCritic):
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

    # Private attribute to store the actual SelfRAGCriticConfig
    _self_rag_config: SelfRAGCriticConfig = PrivateAttr()

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        retriever: RetrieverProtocol[Any, Any, Any],
        config: Optional[BaseConfig] = None,
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
            # Import only needed here, the class was already imported at the module level
            from ...utils.config.critics import DEFAULT_SELF_RAG_CRITIC_CONFIG

            # Create a copy of the default config and update it
            config_dict = deepcopy(DEFAULT_SELF_RAG_CRITIC_CONFIG)
            config_dict.update({"name": name, "description": description, **kwargs})

            # Create the critic config using the already imported class
            critic_config = SelfRAGCriticConfig.model_validate(config_dict)

            # Store it for later access
            self._self_rag_config = critic_config

            # Create a BaseConfig wrapper for BaseComponent
            base_config = BaseConfig(
                name=name,
                description=description,
                params={
                    "config": critic_config,
                    "llm_provider": llm_provider,
                    "retriever": retriever,
                },
            )

            # Initialize base component with the base config
            super().__init__(name=name, description=description, config=base_config)
        else:
            # Check if it's a SelfRAGCriticConfig directly or inside a BaseConfig
            if isinstance(config, SelfRAGCriticConfig):
                self._self_rag_config = config

                # Create a BaseConfig wrapper
                base_config = BaseConfig(
                    name=name, description=description, params={"config": config}
                )

                # Initialize with the wrapped config
                super().__init__(name=name, description=description, config=base_config)
            else:
                # For BaseConfig, try to extract SelfRAGCriticConfig from params
                params = getattr(config, "params", {})
                config_obj = params.get("config")

                if isinstance(config_obj, SelfRAGCriticConfig):
                    self._self_rag_config = config_obj
                else:
                    # Create a new SelfRAGCriticConfig using the already imported class
                    self._self_rag_config = SelfRAGCriticConfig(
                        name=name, description=description, **kwargs
                    )

                # Initialize with the original config
                super().__init__(name=name, description=description, config=config)

        try:
            # Store components in state
            self._state_manager.update("model", llm_provider)
            self._state_manager.update("retriever", retriever)

            # Define default values for missing attributes - using constants defined at module level
            # Get attributes with fallbacks for missing ones
            retrieval_threshold = getattr(self._self_rag_config, "retrieval_threshold", 0.7)
            retrieval_prompt_template = getattr(
                self._self_rag_config,
                "retrieval_prompt_template",
                DEFAULT_RETRIEVAL_PROMPT_TEMPLATE,
            )
            generation_prompt_template = getattr(
                self._self_rag_config,
                "generation_prompt_template",
                DEFAULT_GENERATION_PROMPT_TEMPLATE,
            )
            reflection_prompt_template = getattr(
                self._self_rag_config,
                "reflection_prompt_template",
                DEFAULT_REFLECTION_PROMPT_TEMPLATE,
            )
            reflection_enabled = getattr(self._self_rag_config, "reflection_enabled", True)
            system_prompt = getattr(self._self_rag_config, "system_prompt", DEFAULT_SYSTEM_PROMPT)
            temperature = getattr(self._self_rag_config, "temperature", 0.7)
            max_tokens = getattr(self._self_rag_config, "max_tokens", 1000)

            cache = {
                "retrieval_threshold": retrieval_threshold,
                "retrieval_prompt_template": retrieval_prompt_template,
                "generation_prompt_template": generation_prompt_template,
                "reflection_prompt_template": reflection_prompt_template,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reflection_enabled": reflection_enabled,
            }
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
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
        # Explicitly cast to str to satisfy mypy
        return str(metadata["task"])

    def process(self, input: str) -> BaseResult:
        """
        Process the input text and return a result.

        This is the main method required by BaseComponent.

        Args:
            input: The text to process

        Returns:
            BaseResult: The result of processing the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

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
            result = self.run(task, input)

            # Extract reflection as feedback
            reflection = result.get("reflection", "") if isinstance(result, dict) else ""

            # Parse reflection for issues and suggestions
            issues: List[str] = []
            suggestions: List[str] = []

            # Extract issues and suggestions from reflection
            if reflection:
                for line in reflection.split("\n"):
                    line = line.strip()
                    if line and (line.startswith("- ") or line.startswith("* ")):
                        if (
                            "should" in line.lower()
                            or "could" in line.lower()
                            or "recommend" in line.lower()
                        ):
                            suggestions.append(line[2:])
                        else:
                            issues.append(line[2:])

            # Calculate score based on issues
            score = 1.0 if not issues else 0.5

            # Create result
            response_text = result.get("response", "") if isinstance(result, dict) else ""
            retrieval_query = result.get("retrieval_query", "") if isinstance(result, dict) else ""
            retrieved_context = (
                result.get("retrieved_context", "") if isinstance(result, dict) else ""
            )

            critic_result: BaseResult = BaseResult(
                passed=True,  # Self-RAG critics always pass
                message=response_text,
                metadata={
                    "operation": "process",
                    "retrieval_query": retrieval_query,
                    "retrieved_context": retrieved_context,
                    "reflection": reflection,
                },
                score=score,
                issues=issues,
                suggestions=suggestions,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            self.update_statistics(critic_result)

            return critic_result

        except Exception as e:
            self.record_error(e)
            processing_time = (time.time() - start_time) * 1000
            return BaseResult(
                passed=False,
                message=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
                score=0.0,
                issues=[f"Processing error: {str(e)}"],
                suggestions=["Retry with different input"],
                processing_time_ms=processing_time,
            )

    def validate(self, text: str) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            True if text is valid, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # For SelfRAG, validation is always True as it focuses on improvement
        return True

    def critique(self, text: str) -> CritiqueResult:
        """
        Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Use a default task since the interface doesn't allow for metadata
        metadata = {"task": "Analyze the text"}
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Run the full Self-RAG process
        result = self.run(task, text, metadata)

        # Extract reflection as feedback
        reflection = result.get("reflection", "") if isinstance(result, dict) else ""

        # Parse reflection for issues and suggestions
        issues: List[str] = []
        suggestions: List[str] = []

        # Extract issues and suggestions from reflection
        if reflection:
            for line in reflection.split("\n"):
                line = line.strip()
                if line and (line.startswith("- ") or line.startswith("* ")):
                    if (
                        "should" in line.lower()
                        or "could" in line.lower()
                        or "recommend" in line.lower()
                    ):
                        suggestions.append(line[2:])
                    else:
                        issues.append(line[2:])

        # Calculate score based on issues
        score = 1.0 if not issues else 0.5

        # Create critique result
        result_dict: CritiqueResult = {
            "score": score,
            "feedback": reflection,
            "issues": issues,
            "suggestions": suggestions,
        }

        return result_dict

    def improve(self, text: str, feedback: str = "") -> str:
        """
        Improve text through self-reflective retrieval-augmented generation.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (not used directly, metadata is used instead)

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # This implementation uses metadata instead of direct feedback parameter
        metadata = {"task": "Improve the text"} if not feedback else {"task": feedback}
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Run the full Self-RAG process
        result = self.run(task, text, metadata)

        # Return the improved response
        if isinstance(result, dict) and "response" in result:
            response = result.get("response", "")
            return str(response)
        return str(text)

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
        result = model.generate(
            generation_prompt,
            system_prompt=cache.get("system_prompt", ""),
            temperature=cache.get("temperature", 0.7),
            max_tokens=cache.get("max_tokens", 1000),
        )

        # Ensure we return a string
        if isinstance(result, str):
            return result.strip()
        elif hasattr(result, "output"):
            return str(result.output).strip()
        else:
            return str(result).strip()

    def run(
        self,
        task: str,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,  # Kept for compatibility
    ) -> Dict[str, Any]:
        """
        Run the full Self-RAG process.

        Args:
            task: The task or question to process
            response: Optional initial response to improve
            metadata: Optional metadata (not used directly in this implementation)

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
        # Note: retrieval_threshold is not currently used but kept for future implementation
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


def create_self_rag_critic(
    llm_provider: Any,
    retriever: RetrieverProtocol[Any, Any, Any],
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
    retrieval_prompt_template: Optional[str] = None,
    generation_prompt_template: Optional[str] = None,
    reflection_prompt_template: Optional[str] = None,
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
            # Create a default config with provided values
            # Build config dict with only non-None values
            config_dict: Dict[str, Any] = {
                "name": name,
                "description": description,
            }

            # Add optional parameters only if they're not None
            if min_confidence is not None:
                config_dict["min_confidence"] = min_confidence
            if max_attempts is not None:
                config_dict["max_attempts"] = max_attempts
            if cache_size is not None:
                config_dict["cache_size"] = cache_size
            if priority is not None:
                config_dict["priority"] = priority
            if cost is not None:
                config_dict["cost"] = cost
            if system_prompt is not None:
                config_dict["system_prompt"] = system_prompt
            if temperature is not None:
                config_dict["temperature"] = temperature
            if max_tokens is not None:
                config_dict["max_tokens"] = max_tokens
            if retrieval_threshold is not None:
                config_dict["retrieval_threshold"] = retrieval_threshold

            # Initialize params as a dictionary if any custom templates are provided
            params: Dict[str, Any] = {}

            # Add custom parameters to the params dictionary
            if retrieval_prompt_template is not None:
                params["retrieval_prompt_template"] = retrieval_prompt_template
            if generation_prompt_template is not None:
                params["generation_prompt_template"] = generation_prompt_template
            if reflection_prompt_template is not None:
                params["reflection_prompt_template"] = reflection_prompt_template

            # Only add params to config_dict if it's not empty
            if params:
                config_dict["params"] = params

            # Add any additional kwargs
            config_dict.update(kwargs)

            # Create the config using config_dict
            config = SelfRAGCriticConfig(**config_dict)
        elif isinstance(config, dict):
            # Convert dict to config
            critic_config = SelfRAGCriticConfig.model_validate(config)
            config = critic_config
        # Otherwise, use the provided config

        # Create the critic with wrapped BaseConfig
        base_config = BaseConfig(
            name=name,
            description=description,
            params={"config": config, "llm_provider": llm_provider, "retriever": retriever},
        )

        return SelfRAGCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            retriever=retriever,
            config=base_config,
        )
    except Exception as e:
        raise ValueError(f"Failed to create Self-RAG critic: {str(e)}") from e
