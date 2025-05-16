"""
Self-RAG (Retrieval-Augmented Generation) critic for improving text.

This module provides a critic that uses retrieval to improve text generation
by augmenting the generation process with relevant information.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import json
from ..models import ModelProvider
from ..types import ValidationResult
from ..di import inject


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrievers that find relevant information."""

    def retrieve(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information for a query.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional parameters for retrieval

        Returns:
            A list of retrieved documents/information
        """
        ...


class SelfRAGCritic:
    """
    Critic that uses Self-Reflective Retrieval-Augmented Generation.

    This critic enables language models to decide when to retrieve information,
    what to retrieve, and how to use the retrieved information through self-reflection.
    """

    @inject(model_provider="model.openai")
    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        retriever: Optional[RetrieverProtocol] = None,
        system_prompt: str = "You are a helpful assistant that uses retrieval to provide accurate information.",
        retrieval_threshold: float = 0.7,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """
        Initialize the Self-RAG critic.

        Args:
            model_provider: Model provider to use (injected if not provided)
            retriever: Retriever to use for information retrieval
            system_prompt: System prompt for the model
            retrieval_threshold: Threshold for when to use retrieval
            temperature: Temperature for generation
            **kwargs: Additional arguments to pass to the model provider
        """
        if not model_provider:
            from ..di import resolve

            model_provider = resolve("model.openai")

        self.model_provider = model_provider
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.retrieval_threshold = retrieval_threshold
        self.temperature = temperature
        self.kwargs = kwargs

        # Templates for different stages
        self.retrieval_prompt_template = (
            "Given the following task, decide whether retrieval would be helpful. "
            "Task: {task}\n\n"
            "Should I retrieve information? Respond with YES or NO and a brief explanation."
        )
        self.generation_prompt_template = (
            "Task: {task}\n\n"
            "Retrieved Information:\n{retrieved_info}\n\n"
            "Using the retrieved information above, respond to the task."
        )
        self.reflection_prompt_template = (
            "Task: {task}\n\n"
            "Retrieved Information:\n{retrieved_info}\n\n"
            "Your Response: {response}\n\n"
            "Reflect on your response. Did you use the retrieved information appropriately? "
            "Is the response accurate and helpful? What could be improved?"
        )

    def validate(self, text: str, task: str = "") -> ValidationResult:
        """
        Validate text using retrieval augmentation.

        Args:
            text: The text to validate
            task: The task context for validation

        Returns:
            A ValidationResult indicating whether the text passes quality criteria
        """
        # If no retriever is available, validate without retrieval
        if not self.retriever:
            return self._validate_without_retrieval(text)

        # Run the full Self-RAG process
        result = self.run(task, text)

        # Extract critique information from the reflection
        reflection = result.get("reflection", "")

        # Determine quality based on reflection
        score = 0.7  # Default score
        passed = True
        issues = []
        suggestions = []

        # Parse reflection to extract issues and suggestions
        if "could be improved" in reflection.lower():
            passed = False
            score = 0.5

            # Extract issues and suggestions from reflection
            reflection_lines = reflection.split("\n")
            for line in reflection_lines:
                if "issue:" in line.lower() or "problem:" in line.lower():
                    issues.append(line.split(":", 1)[1].strip())
                elif "suggestion:" in line.lower() or "improvement:" in line.lower():
                    suggestions.append(line.split(":", 1)[1].strip())

        # Create validation result
        return ValidationResult(
            passed=passed,
            score=score,
            message=reflection,
            issues=issues,
            suggestions=suggestions,
        )

    def _validate_without_retrieval(self, text: str) -> ValidationResult:
        """
        Validate text without using retrieval.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult indicating whether the text passes quality criteria
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Create validation prompt
        prompt = f"""
        {self.system_prompt}

        Evaluate the quality of the following text:

        ---
        {text}
        ---

        Please provide your evaluation in the following JSON format:
        {{
            "score": <a number between 0.0 and 1.0>,
            "feedback": "<overall feedback>",
            "issues": ["<issue 1>", "<issue 2>", ...],
            "suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
        }}

        Only respond with valid JSON.
        """

        response = self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

        # Parse response
        try:
            result = json.loads(response)
            # Ensure all required fields exist
            result.setdefault("score", 0.5)
            result.setdefault("feedback", "")
            result.setdefault("issues", [])
            result.setdefault("suggestions", [])

            # Create validation result
            return ValidationResult(
                passed=result["score"] >= 0.7,
                score=result["score"],
                message=result["feedback"],
                issues=result["issues"],
                suggestions=result["suggestions"],
            )
        except json.JSONDecodeError:
            # Fallback if the model doesn't generate valid JSON
            return ValidationResult(
                passed=True,
                score=0.5,
                message=response,
                issues=[],
                suggestions=[],
            )

    def critique(self, text: str, task: str = "") -> dict:
        """
        Evaluate text with retrieval augmentation.

        Args:
            text: The text to evaluate
            task: The task context for evaluation

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        # If no retriever is available, critique without retrieval
        if not self.retriever:
            return self._critique_without_retrieval(text)

        # Run the full Self-RAG process
        result = self.run(task, text)

        # Extract critique information from the reflection
        reflection = result.get("reflection", "")

        # Structure critique based on reflection
        critique = {
            "score": 0.7,  # Default score
            "feedback": reflection,
            "issues": [],
            "suggestions": [],
            "retrieved_info": result.get("retrieved_info", ""),
        }

        # Parse reflection to extract issues and suggestions
        if "could be improved" in reflection.lower():
            critique["score"] = 0.5

            # Extract issues and suggestions from reflection
            reflection_lines = reflection.split("\n")
            for line in reflection_lines:
                if "issue:" in line.lower() or "problem:" in line.lower():
                    critique["issues"].append(line.split(":", 1)[1].strip())
                elif "suggestion:" in line.lower() or "improvement:" in line.lower():
                    critique["suggestions"].append(line.split(":", 1)[1].strip())

        return critique

    def _critique_without_retrieval(self, text: str) -> dict:
        """
        Critique text without using retrieval.

        Args:
            text: The text to critique

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Create critique prompt
        prompt = f"""
        {self.system_prompt}

        Evaluate the quality of the following text:

        ---
        {text}
        ---

        Please provide your evaluation in the following JSON format:
        {{
            "score": <a number between 0.0 and 1.0>,
            "feedback": "<overall feedback>",
            "issues": ["<issue 1>", "<issue 2>", ...],
            "suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
        }}

        Only respond with valid JSON.
        """

        response = self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

        # Parse response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if the model doesn't generate valid JSON
            return {
                "score": 0.5,
                "feedback": response,
                "issues": [],
                "suggestions": [],
            }

    def improve(self, text: str, issues: Optional[List[str]] = None, task: str = "") -> str:
        """
        Improve text using retrieval augmentation.

        Args:
            text: The text to improve
            issues: Optional list of issues to address
            task: The task context for improvement

        Returns:
            Improved text
        """
        # If no task provided, use the text as the task
        if not task:
            task = f"Improve this text: {text}"

        # If no retriever is available or issues are specified, improve without retrieval
        if not self.retriever or (issues and len(issues) > 0):
            return self._improve_without_retrieval(text, issues)

        # Run the full Self-RAG process
        result = self.run(task, text)

        # Return the improved response
        return result.get("response", text)

    def _improve_without_retrieval(self, text: str, issues: Optional[List[str]] = None) -> str:
        """
        Improve text without using retrieval.

        Args:
            text: The text to improve
            issues: Optional list of issues to address

        Returns:
            Improved text
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Format issues if provided
        issues_text = ""
        if issues and len(issues) > 0:
            issues_text = "Address the following issues:\n" + "\n".join(
                [f"- {issue}" for issue in issues]
            )

        # Create improvement prompt
        prompt = f"""
        {self.system_prompt}

        Improve the following text:

        ---
        {text}
        ---

        {issues_text}

        Please provide only the improved text without any explanations or additional commentary.
        """

        return self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

    def run(self, task: str, response: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full Self-RAG process.

        This method implements the complete Self-RAG process including retrieval decision,
        retrieval, generation, and reflection.

        Args:
            task: The task to run Self-RAG for
            response: Optional existing response to evaluate

        Returns:
            A dictionary with the results of the Self-RAG process
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        result = {
            "task": task,
            "should_retrieve": False,
            "retrieved_info": "",
            "response": response or "",
            "reflection": "",
        }

        # Step 1: Decide whether to retrieve information
        if not response:  # Only decide for new generations
            should_retrieve = self._should_retrieve(task)
            result["should_retrieve"] = should_retrieve

            # Step 2: Retrieve information if needed
            if should_retrieve and self.retriever:
                retrieved_docs = self.retriever.retrieve(task)
                retrieved_info = self._format_retrieved_docs(retrieved_docs)
                result["retrieved_info"] = retrieved_info

                # Step 3: Generate response with retrieved information
                result["response"] = self._generate_with_retrieval(task, retrieved_info)
            else:
                # Generate without retrieval
                result["response"] = self._generate_without_retrieval(task)

        # Step 4: Generate reflection (for both new and existing responses)
        if self.retriever:
            # Retrieve information for reflection if not already retrieved
            if not result["retrieved_info"]:
                retrieved_docs = self.retriever.retrieve(task)
                result["retrieved_info"] = self._format_retrieved_docs(retrieved_docs)

            # Generate reflection
            result["reflection"] = self._reflect(task, result["retrieved_info"], result["response"])

        return result

    def _should_retrieve(self, task: str) -> bool:
        """
        Decide whether to retrieve information for a task.

        Args:
            task: The task to decide retrieval for

        Returns:
            True if retrieval should be performed, False otherwise
        """
        # Format the prompt using the template
        prompt = self.retrieval_prompt_template.format(task=task)

        # Add the system prompt
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        # Generate a response
        response = self.model_provider.generate(
            full_prompt,
            temperature=0.3,  # Lower temperature for more deterministic decisions
            **self.kwargs,
        )

        # Check if the response indicates retrieval
        return "yes" in response.lower()

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for inclusion in prompts.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted string of retrieved information
        """
        if not docs:
            return "No relevant information found."

        formatted = ""
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            formatted += f"[{source}]: {content}\n\n"

        return formatted

    def _generate_with_retrieval(self, task: str, retrieved_info: str) -> str:
        """
        Generate a response using retrieved information.

        Args:
            task: The task to generate a response for
            retrieved_info: The retrieved information to use

        Returns:
            Generated response
        """
        # Format the prompt using the template
        prompt = self.generation_prompt_template.format(
            task=task,
            retrieved_info=retrieved_info,
        )

        # Add the system prompt
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        # Generate a response
        return self.model_provider.generate(
            full_prompt, temperature=self.temperature, **self.kwargs
        )

    def _generate_without_retrieval(self, task: str) -> str:
        """
        Generate a response without using retrieval.

        Args:
            task: The task to generate a response for

        Returns:
            Generated response
        """
        # Create a simple prompt for the task
        prompt = f"{self.system_prompt}\n\n{task}"

        # Generate a response
        return self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

    def _reflect(self, task: str, retrieved_info: str, response: str) -> str:
        """
        Generate a reflection on the response.

        Args:
            task: The original task
            retrieved_info: The retrieved information
            response: The response to reflect on

        Returns:
            Generated reflection
        """
        # Format the prompt using the template
        prompt = self.reflection_prompt_template.format(
            task=task,
            retrieved_info=retrieved_info,
            response=response,
        )

        # Add the system prompt
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        # Generate a reflection
        return self.model_provider.generate(
            full_prompt, temperature=self.temperature, **self.kwargs
        )
