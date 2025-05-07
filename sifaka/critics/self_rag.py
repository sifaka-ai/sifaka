"""
Self-RAG critic module for Sifaka.

This module implements the Self-Reflective Retrieval-Augmented Generation approach for critics,
which enables language models to decide when and what to retrieve, and reflect on the
relevance and utility of the retrieved information.

Based on Self-RAG: https://arxiv.org/abs/2310.11511

Example:
    ```python
    from sifaka.critics.self_rag import create_self_rag_critic
    from sifaka.models.providers import OpenAIProvider
    from sifaka.retrieval import SimpleRetriever

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a retriever
    documents = {
        "health insurance": "To file a claim for health reimbursement, follow these steps: 1) Complete the claim form...",
        "travel insurance": "For travel insurance claims, you need to provide: 1) Proof of travel 2) Incident report..."
    }
    retriever = SimpleRetriever(documents=documents)

    # Create a self-rag critic
    critic = create_self_rag_critic(
        llm_provider=provider,
        retriever=retriever
    )

    # Use the critic to improve text
    task = "What are the steps to file a claim for health reimbursement?"
    result = critic.run(task, response=None)
    print(f"Response: {result['response']}")
    print(f"Reflection: {result['reflection']}")
    ```
"""

from typing import Any, Dict, List, Optional, Union, cast

from pydantic import ConfigDict, Field, PrivateAttr

from .base import BaseCritic, TextCritic, TextImprover, TextValidator, create_critic_state
from .models import CriticConfig, PromptCriticConfig
from ..models.base import ModelProvider
from ..retrieval import Retriever


class SelfRAGCriticConfig(PromptCriticConfig):
    """
    Configuration for Self-RAG critics.

    This model extends PromptCriticConfig with Self-RAG-specific settings
    for critics that decide when and what to retrieve, and reflect on
    the relevance and utility of the retrieved information.

    ## Lifecycle Management

    1. **Initialization**
       - Set base configuration
       - Configure Self-RAG settings
       - Validate field values
       - Create immutable instance

    2. **Validation**
       - Check field types
       - Verify value ranges
       - Ensure required fields
       - Validate custom rules

    3. **Usage**
       - Access configuration values
       - Create modified instances
       - Serialize to/from JSON
       - Validate against schema

    Examples:
        ```python
        from sifaka.critics.self_rag import SelfRAGCriticConfig

        # Create a Self-RAG critic config
        config = SelfRAGCriticConfig(
            name="self_rag_critic",
            description="A Self-RAG critic",
            system_prompt="You are an expert at deciding when to retrieve information.",
            temperature=0.7,
            max_tokens=1000,
            retrieval_threshold=0.5
        )

        # Access configuration values
        print(f"System prompt: {config.system_prompt}")
        print(f"Retrieval threshold: {config.retrieval_threshold}")

        # Create modified config
        new_config = config.model_copy(
            update={"retrieval_threshold": 0.7}
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(
        default="You are an expert at deciding when to retrieve information and reflecting on its relevance.",
        description="System prompt for the model",
    )
    retrieval_threshold: float = Field(
        default=0.5, description="Threshold for retrieval confidence", ge=0.0, le=1.0
    )
    retrieval_prompt_template: Optional[str] = Field(
        default=(
            "Do you need external knowledge to answer this question? If so, what would you search for?\n\n"
            "Task:\n{task}\n\n"
            "If you need external knowledge, respond with a search query.\n"
            "If you don't need external knowledge, respond with 'No external knowledge needed.'"
        ),
        description="Template for retrieval prompts",
    )
    generation_prompt_template: Optional[str] = Field(
        default=(
            "Please answer the following task using the provided context (if available).\n\n"
            "Context:\n{context}\n\n"
            "Task:\n{task}\n\n"
            "Answer:"
        ),
        description="Template for generation prompts",
    )
    reflection_prompt_template: Optional[str] = Field(
        default=(
            "Reflect on whether your answer used relevant information and addressed the task accurately.\n\n"
            "Task:\n{task}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Your Response:\n{response}\n\n"
            "Reflection:"
        ),
        description="Template for reflection prompts",
    )


class SelfRAGCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the Self-Reflective Retrieval-Augmented Generation approach.

    This critic enables language models to decide when and what to retrieve,
    and reflect on the relevance and utility of the retrieved information.

    Based on Self-RAG: https://arxiv.org/abs/2310.11511

    ## Lifecycle Management

    The SelfRAGCritic manages its lifecycle through three main phases:

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

    ## Examples

    Basic usage with factory function:

    ```python
    from sifaka.critics import create_self_rag_critic
    from sifaka.models.openai import create_openai_provider
    from sifaka.retrieval import SimpleRetriever

    # Create a language model provider
    provider = create_openai_provider(
        model_name="gpt-4",
        api_key="your-api-key"
    )

    # Create a retriever with a document collection
    documents = {
        "health_insurance": """
        To file a claim for health reimbursement, follow these steps:
        1. Complete the claim form with your personal and policy information
        2. Attach all original receipts and medical documentation
        3. Make copies of all documents for your records
        4. Submit the claim through the online portal, mobile app, or by mail
        5. Track your claim status using the provided claim number
        6. Expect processing within 14-30 days depending on complexity
        """,
        "travel_insurance": """
        For travel insurance claims, you need to provide:
        1. Proof of travel (boarding passes, itinerary)
        2. Incident report or documentation of the event
        3. Original receipts for expenses being claimed
        4. Completed claim form with policy details
        5. Submit within the timeframe specified in your policy
        """
    }
    retriever = SimpleRetriever(documents=documents)

    # Create a self-rag critic
    critic = create_self_rag_critic(
        llm_provider=provider,
        retriever=retriever,
        name="insurance_rag_critic",
        description="A critic for insurance-related queries",
        system_prompt="You are an expert at retrieving and using insurance information.",
        temperature=0.7,
        max_tokens=1000,
    )

    # Process a task
    task = "What are the steps to file a claim for health reimbursement?"
    result = critic.run(task)

    # Print results
    print(f"Retrieval Query: {result['retrieval_query']}")
    print(f"Retrieved Context: {result['retrieved_context']}")
    print(f"Response: {result['response']}")
    print(f"Reflection: {result['reflection']}")
    ```

    Advanced usage with custom configuration:

    ```python
    from sifaka.critics.self_rag import SelfRAGCritic, SelfRAGCriticConfig
    from sifaka.models.anthropic import create_anthropic_provider
    from sifaka.retrieval.vector import VectorRetriever
    from sifaka.retrieval.embeddings import OpenAIEmbeddings

    # Create a language model provider
    provider = create_anthropic_provider(
        model_name="claude-3-opus-20240229",
        api_key="your-api-key"
    )

    # Create an embeddings model
    embeddings = OpenAIEmbeddings(api_key="your-openai-api-key")

    # Create a vector retriever (more advanced than SimpleRetriever)
    documents = [
        {"id": "doc1", "text": "Python is a high-level programming language known for its readability and versatility."},
        {"id": "doc2", "text": "JavaScript is primarily used for web development and runs in browsers."},
        {"id": "doc3", "text": "Rust offers memory safety without garbage collection and is used for systems programming."}
    ]
    retriever = VectorRetriever(documents=documents, embeddings=embeddings)

    # Create a custom configuration with specialized prompts
    config = SelfRAGCriticConfig(
        name="programming_rag_critic",
        description="A critic for programming language questions",
        system_prompt="You are an expert programmer who knows when to retrieve information about programming languages.",
        temperature=0.5,
        max_tokens=2000,
        retrieval_threshold=0.7,
        retrieval_prompt_template=(
            "You are answering a question about programming languages.\n\n"
            "Question: {task}\n\n"
            "Do you need to retrieve specific information about programming languages to answer this question?\n"
            "If yes, what specific information would you search for? Be precise.\n"
            "If no, respond with 'No retrieval needed.'"
        ),
        generation_prompt_template=(
            "You are explaining programming concepts.\n\n"
            "Retrieved Information:\n{context}\n\n"
            "Question: {task}\n\n"
            "Provide a clear, accurate explanation:"
        ),
        reflection_prompt_template=(
            "Reflect on your answer to the programming question:\n\n"
            "Question: {task}\n\n"
            "Retrieved Information:\n{context}\n\n"
            "Your Answer:\n{response}\n\n"
            "Evaluate your answer for accuracy, completeness, and relevance to the question.\n"
            "Did you use the retrieved information effectively?\n"
            "What could be improved?"
        )
    )

    # Create a self-rag critic with custom configuration
    critic = SelfRAGCritic(
        config=config,
        llm_provider=provider,
        retriever=retriever
    )

    # Use the critic with different methods
    task = "What are the key differences between Python and Rust?"

    # Run the full process
    result = critic.run(task)
    print(f"Response: {result['response']}")

    # Improve an existing response
    initial_response = "Python and Rust are programming languages."
    improved_response = critic.improve(initial_response, {"task": task})
    print(f"Improved response: {improved_response}")

    # Get a critique
    critique = critic.critique(initial_response, {"task": task})
    print(f"Score: {critique['score']}")
    print(f"Issues: {critique['issues']}")

    # Use asynchronous methods
    import asyncio

    async def process_task():
        task = "Explain the benefits of JavaScript for web development."
        result = await critic.arun(task)
        return result

    async_result = asyncio.run(process_task())
    ```
    """

    # Class constants
    DEFAULT_NAME = "self_rag_critic"
    DEFAULT_DESCRIPTION = "Improves text through self-reflective retrieval-augmented generation"

    # State management using state manager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: SelfRAGCriticConfig,
        llm_provider: Any,
        retriever: Retriever,
    ) -> None:
        """
        Initialize the Self-RAG critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use
            retriever: Retriever to use for information retrieval

        Raises:
            ValueError: If config is invalid
            TypeError: If llm_provider or retriever is invalid
        """
        # Initialize base class
        super().__init__(config)

        # Get state from state manager
        state = self._state_manager.get_state()

        # Store components in state
        state.model = llm_provider
        state.initialized = True

        # Store configuration and retriever in state cache
        state.cache = {
            "retriever": retriever,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "retrieval_threshold": config.retrieval_threshold,
            "retrieval_prompt_template": config.retrieval_prompt_template,
            "generation_prompt_template": config.generation_prompt_template,
            "reflection_prompt_template": config.reflection_prompt_template,
        }

    def _check_input(self, text: str) -> None:
        """
        Check if input is valid.

        Args:
            text: The text to check

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critic is not properly initialized
        """
        # Get state from state manager
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("SelfRAGCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Get task from metadata.

        Args:
            metadata: Optional metadata containing the task

        Returns:
            The task as a string

        Raises:
            ValueError: If metadata is missing required keys
        """
        if metadata is None:
            return "Answer the following question or complete the following task."

        task = metadata.get("task", "")
        if not task:
            return "Answer the following question or complete the following task."

        return task

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
        self._check_input(text)

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
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Run the full Self-RAG process
        result = self.run(task, text, metadata)

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
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Run the full Self-RAG process
        result = self.run(task, text, metadata)

        # Return the improved response
        return result.get("response", text)

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

        # Get state from state manager
        state = self._state_manager.get_state()

        # Use the feedback as context for generation
        generation_template = state.cache.get("generation_prompt_template")
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
        improved_text = state.model.generate(
            generation_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    def run(
        self, task: str, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
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
        # Get state from state manager
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("SelfRAGCritic not properly initialized")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        # Step 1: Ask the model if it needs retrieval
        retrieval_template = state.cache.get("retrieval_prompt_template")
        if not retrieval_template:
            retrieval_template = (
                "Do you need external knowledge to answer this question? If so, what would you search for?\n\n"
                "Task:\n{task}\n\n"
                "If you need external knowledge, respond with a search query.\n"
                "If you don't need external knowledge, respond with 'No external knowledge needed.'"
            )
        retrieval_prompt = retrieval_template.format(task=task)

        retrieval_query = state.model.generate(
            retrieval_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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
            retriever = state.cache.get("retriever")
            if not retriever:
                raise RuntimeError("Retriever not initialized")
            context = retriever.retrieve(retrieval_query)

        # Step 3: Generate response with (or without) retrieved context
        generation_template = state.cache.get("generation_prompt_template")
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
            response = state.model.generate(
                generation_prompt,
                system_prompt=state.cache.get("system_prompt", ""),
                temperature=state.cache.get("temperature", 0.7),
                max_tokens=state.cache.get("max_tokens", 1000),
            ).strip()

        # Step 4: Ask model to reflect on whether the answer is good and the retrieval helped
        reflection_template = state.cache.get("reflection_prompt_template")
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

        reflection = state.model.generate(
            reflection_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return {
            "response": response,
            "retrieval_query": retrieval_query,
            "retrieved_context": context,
            "reflection": reflection,
        }

    # Async methods
    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Asynchronously validate text.

        Args:
            text: The text to validate
            metadata: Optional metadata containing the task

        Returns:
            True if text is valid, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # For SelfRAG, validation is always True as it focuses on improvement
        return True

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Run the full Self-RAG process
        result = await self.arun(task, text, metadata)

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

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Asynchronously improve text through self-reflective retrieval-augmented generation.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Run the full Self-RAG process
        result = await self.arun(task, text, metadata)

        # Return the improved response
        return result.get("response", text)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """
        Asynchronously improve text based on specific feedback.

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

        # Get state from state manager
        state = self._state_manager.get_state()

        # Use the feedback as context for generation
        generation_template = state.cache.get("generation_prompt_template")
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
        improved_text = await state.model.agenerate(
            generation_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        )
        improved_text = improved_text.strip()

        return improved_text

    async def arun(
        self, task: str, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously run the full Self-RAG process.

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
        # Get state from state manager
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("SelfRAGCritic not properly initialized")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        # Step 1: Ask the model if it needs retrieval
        retrieval_template = state.cache.get("retrieval_prompt_template")
        if not retrieval_template:
            retrieval_template = (
                "Do you need external knowledge to answer this question? If so, what would you search for?\n\n"
                "Task:\n{task}\n\n"
                "If you need external knowledge, respond with a search query.\n"
                "If you don't need external knowledge, respond with 'No external knowledge needed.'"
            )
        retrieval_prompt = retrieval_template.format(task=task)

        retrieval_query = await state.model.agenerate(
            retrieval_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        )
        retrieval_query = retrieval_query.strip()

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
            # Note: We're using sync retrieval here as there's no async interface defined
            # In a real implementation, you might want to add async support to the Retriever interface
            import asyncio

            retriever = state.cache.get("retriever")
            if not retriever:
                raise RuntimeError("Retriever not initialized")
            context = await asyncio.to_thread(retriever.retrieve, retrieval_query)

        # Step 3: Generate response with (or without) retrieved context
        generation_template = state.cache.get("generation_prompt_template")
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
            response = await state.model.agenerate(
                generation_prompt,
                system_prompt=state.cache.get("system_prompt", ""),
                temperature=state.cache.get("temperature", 0.7),
                max_tokens=state.cache.get("max_tokens", 1000),
            )
            response = response.strip()

        # Step 4: Ask model to reflect on whether the answer is good and the retrieval helped
        reflection_template = state.cache.get("reflection_prompt_template")
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

        reflection = await state.model.agenerate(
            reflection_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        )
        reflection = reflection.strip()

        return {
            "response": response,
            "retrieval_query": retrieval_query,
            "retrieved_context": context,
            "reflection": reflection,
        }


def create_self_rag_critic(
    llm_provider: Any,
    retriever: Retriever,
    name: str = "self_rag_critic",
    description: str = "Improves text through self-reflective retrieval-augmented generation",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at deciding when to retrieve information and reflecting on its relevance.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    retrieval_threshold: float = 0.5,
    retrieval_prompt_template: Optional[str] = None,
    generation_prompt_template: Optional[str] = None,
    reflection_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRAGCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRAGCritic:
    """
    Create a Self-RAG critic with the given parameters.

    This factory function creates a configured SelfRAGCritic instance.
    It provides a standardized way to create critics with various configurations.

    Args:
        llm_provider: Language model provider to use
        retriever: Retriever to use for information retrieval
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        retrieval_threshold: Threshold for retrieval confidence
        retrieval_prompt_template: Optional custom template for retrieval prompts
        generation_prompt_template: Optional custom template for generation prompts
        reflection_prompt_template: Optional custom template for reflection prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        SelfRAGCritic: The created critic

    Examples:
        ```python
        from sifaka.critics.self_rag import create_self_rag_critic
        from sifaka.models.providers import OpenAIProvider
        from sifaka.retrieval import SimpleRetriever

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a retriever
        documents = {
            "health insurance": "To file a claim for health reimbursement, follow these steps: 1) Complete the claim form...",
            "travel insurance": "For travel insurance claims, you need to provide: 1) Proof of travel 2) Incident report..."
        }
        retriever = SimpleRetriever(documents=documents)

        # Create a Self-RAG critic with default settings
        critic = create_self_rag_critic(
            llm_provider=provider,
            retriever=retriever
        )

        # Create a Self-RAG critic with custom settings
        critic = create_self_rag_critic(
            llm_provider=provider,
            retriever=retriever,
            name="custom_rag_critic",
            description="Custom RAG critic",
            system_prompt="You are an expert at retrieving and using information.",
            temperature=0.8,
            max_tokens=1500,
            retrieval_threshold=0.7
        )

        # Use the critic
        task = "What are the steps to file a claim for health reimbursement?"
        result = critic.run(task, response=None)
        print(f"Response: {result['response']}")
        print(f"Reflection: {result['reflection']}")
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        from ..utils.config import standardize_critic_config

        # If standardize_critic_config is available, use it
        critic_config = standardize_critic_config(
            config_class=SelfRAGCriticConfig,
            config=config,
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            retrieval_threshold=retrieval_threshold,
            retrieval_prompt_template=retrieval_prompt_template,
            generation_prompt_template=generation_prompt_template,
            reflection_prompt_template=reflection_prompt_template,
            **kwargs,
        )
    except ImportError:
        # If standardize_critic_config is not available, create config manually
        if config is None:
            # Create new config
            config_params = {
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "min_confidence": min_confidence,
                "max_attempts": max_attempts,
                "cache_size": cache_size,
                "priority": priority,
                "cost": cost,
                "retrieval_threshold": retrieval_threshold,
                "params": kwargs,
            }

            # Add optional parameters if provided
            if retrieval_prompt_template is not None:
                config_params["retrieval_prompt_template"] = retrieval_prompt_template
            if generation_prompt_template is not None:
                config_params["generation_prompt_template"] = generation_prompt_template
            if reflection_prompt_template is not None:
                config_params["reflection_prompt_template"] = reflection_prompt_template

            critic_config = SelfRAGCriticConfig(**config_params)
        elif isinstance(config, dict):
            # Convert dict to config
            critic_config = SelfRAGCriticConfig(**config)
        else:
            # Use provided config
            critic_config = config

    # Create and return the critic
    return SelfRAGCritic(
        config=critic_config,
        llm_provider=llm_provider,
        retriever=retriever,
    )
