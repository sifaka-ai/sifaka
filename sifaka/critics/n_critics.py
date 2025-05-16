"""
N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics.

This module provides a critic that uses an ensemble of critics for self-refinement.

Based on the paper:
"N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics"
Shahriar Mousavi, Roxana Leontie Rios Gutierrez, Deepak Rengarajan, Vishal Gundecha, 
Anand Raju Babu, Avisek Naug, Srinivas Chappidi
arXiv:2310.18679 [cs.CL]
https://arxiv.org/abs/2310.18679
"""

import json
from typing import Dict, Any, Optional, List, Tuple

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver


class NCriticsCritic(Critic):
    """Critic that uses an ensemble of critics for self-refinement.

    This critic implements the N-Critics technique from the paper
    "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics" (Mousavi et al., 2023).

    N-Critics leverages an ensemble of specialized critics, each focusing on different aspects
    of the text, to provide comprehensive feedback and guide the refinement process.

    Attributes:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        num_critics: Number of specialized critics to use.
        max_refinement_iterations: Maximum number of refinement iterations.
    """

    def __init__(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        num_critics: int = 3,
        max_refinement_iterations: int = 2,
        **options: Any,
    ):
        """Initialize the N-Critics critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            num_critics: Number of specialized critics to use.
            max_refinement_iterations: Maximum number of refinement iterations.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert language model that uses an ensemble of specialized critics "
                "to provide comprehensive feedback and guide the refinement process. "
                "You follow the N-Critics approach to provide structured guidance."
            )

        super().__init__(model, system_prompt, temperature, **options)

        self.num_critics = max(1, min(5, num_critics))  # Clamp between 1 and 5
        self.max_refinement_iterations = max(1, max_refinement_iterations)
        
        # Define the critic roles
        self.critic_roles = [
            "Factual Accuracy Critic: Focus on identifying factual errors and inaccuracies.",
            "Coherence and Clarity Critic: Focus on improving the logical flow and clarity of the text.",
            "Completeness Critic: Focus on identifying missing information or incomplete explanations.",
            "Style and Tone Critic: Focus on improving the writing style, tone, and language usage.",
            "Relevance Critic: Focus on ensuring the content is relevant to the intended purpose."
        ][:self.num_critics]  # Use only the specified number of critics

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the N-Critics technique.

        This method implements the ensemble critic approach from the N-Critics paper:
        1. Generate critiques from multiple specialized critics
        2. Aggregate the critiques into a comprehensive assessment

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        try:
            # Generate critiques from each specialized critic
            critic_critiques = []
            for role in self.critic_roles:
                critique = self._generate_critic_critique(text, role)
                critic_critiques.append(critique)

            # Aggregate critiques
            aggregated_critique = self._aggregate_critiques(critic_critiques)
            
            # Determine if improvement is needed
            needs_improvement = any(critique.get("needs_improvement", True) for critique in critic_critiques)
            
            # Prepare the final critique
            final_critique = {
                "needs_improvement": needs_improvement,
                "message": aggregated_critique["summary"],
                "critic_critiques": critic_critiques,
                "aggregated_critique": aggregated_critique,
                # For compatibility with base Critic
                "issues": aggregated_critique["issues"],
                "suggestions": aggregated_critique["suggestions"],
            }

            return final_critique
        except Exception as e:
            raise ImproverError(f"Error critiquing text with N-Critics: {str(e)}")

    def _generate_critic_critique(self, text: str, role: str) -> Dict[str, Any]:
        """Generate a critique from a specialized critic.

        Args:
            text: The text to critique.
            role: The role of the specialized critic.

        Returns:
            A dictionary with the critique from the specialized critic.

        Raises:
            ImproverError: If the critique cannot be generated.
        """
        prompt = f"""
        You are a specialized critic with the following role:
        {role}

        Your task is to critique the following text based on your specialized role:

        ```
        {text}
        ```

        Please provide a detailed critique that:
        1. Identifies specific issues related to your specialized role
        2. Explains why these issues are problematic
        3. Suggests concrete improvements
        4. Rates the text on a scale of 1-10 for your specific area of focus

        Format your response as JSON with the following fields:
        - "role": your specialized role
        - "needs_improvement": boolean indicating whether the text needs improvement in your area
        - "score": your rating of the text on a scale of 1-10
        - "issues": a list of specific issues you identified
        - "suggestions": a list of specific suggestions for improvement
        - "explanation": a brief explanation of your overall assessment

        JSON response:
        """

        try:
            response = self._generate(prompt)

            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                # No JSON found, create a default response
                return {
                    "role": role,
                    "needs_improvement": True,
                    "score": 5,
                    "issues": ["Unable to parse critique response"],
                    "suggestions": ["General improvement needed"],
                    "explanation": "Unable to parse critique response, but proceeding with improvement",
                }

            json_str = response[json_start:json_end]
            critique = json.loads(json_str)

            # Ensure all required fields are present
            critique.setdefault("role", role)
            critique.setdefault("needs_improvement", True)
            critique.setdefault("score", 5)
            critique.setdefault("issues", ["General improvement needed"])
            critique.setdefault("suggestions", ["Improve based on the feedback provided"])
            critique.setdefault("explanation", "Text needs improvement")

            return critique
        except json.JSONDecodeError:
            # Failed to parse JSON, create a default response
            return {
                "role": role,
                "needs_improvement": True,
                "score": 5,
                "issues": ["Unable to parse critique response"],
                "suggestions": ["General improvement needed"],
                "explanation": "Unable to parse critique response, but proceeding with improvement",
            }
        except Exception as e:
            raise ImproverError(f"Error generating critique from specialized critic: {str(e)}")

    def _aggregate_critiques(self, critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate critiques from multiple specialized critics.

        Args:
            critiques: List of critiques from specialized critics.

        Returns:
            A dictionary with the aggregated critique.
        """
        # Extract all issues and suggestions
        all_issues = []
        all_suggestions = []
        average_score = 0.0

        for critique in critiques:
            all_issues.extend(critique.get("issues", []))
            all_suggestions.extend(critique.get("suggestions", []))
            average_score += critique.get("score", 5)

        # Calculate average score
        if critiques:
            average_score /= len(critiques)

        # Generate a summary of the critiques
        critiques_summary = "\n\n".join([
            f"Critic: {critique.get('role', 'Unknown')}\n"
            f"Score: {critique.get('score', 5)}/10\n"
            f"Explanation: {critique.get('explanation', 'No explanation provided')}"
            for critique in critiques
        ])

        prompt = f"""
        You are an expert at aggregating feedback from multiple critics. Please synthesize the following critiques
        into a coherent summary that captures the key issues and suggestions:

        {critiques_summary}

        Your summary should:
        1. Identify the most important issues across all critiques
        2. Highlight the most valuable suggestions
        3. Provide a balanced assessment of the text's strengths and weaknesses
        4. Be concise but comprehensive

        Summary:
        """

        try:
            summary = self._generate(prompt)
        except Exception:
            # If generation fails, create a simple summary
            summary = "Multiple issues were identified by the critics. The text needs improvement in several areas."

        return {
            "summary": summary,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "average_score": average_score,
        }

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the N-Critics technique.

        This method implements the iterative refinement process from the N-Critics paper,
        using feedback from multiple critics to guide the improvement.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        current_text = text
        current_critique = critique
        
        # Iterative refinement process
        for iteration in range(self.max_refinement_iterations):
            try:
                # Generate improved text based on aggregated critique
                improved_text = self._generate_improved_text(
                    current_text, 
                    current_critique["aggregated_critique"]["summary"],
                    current_critique["critic_critiques"]
                )
                
                # Re-evaluate the improved text
                improved_critique = self._critique(improved_text)
                
                # Check if the text has improved
                current_score = current_critique["aggregated_critique"]["average_score"]
                improved_score = improved_critique["aggregated_critique"]["average_score"]
                
                if improved_score > current_score:
                    current_text = improved_text
                    current_critique = improved_critique
                else:
                    # No improvement, stop iterating
                    break
                
                # If the score is high enough, stop iterating
                if improved_score >= 9.0:
                    break
            except Exception as e:
                # If an error occurs during improvement, return the current text
                break
        
        return current_text

    def _generate_improved_text(self, text: str, summary: str, critiques: List[Dict[str, Any]]) -> str:
        """Generate improved text based on critiques.

        Args:
            text: The text to improve.
            summary: The summary of the critiques.
            critiques: The critiques from specialized critics.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        # Format critiques for the prompt
        critiques_text = "\n\n".join([
            f"Critic: {critique.get('role', 'Unknown')}\n"
            f"Issues: {', '.join(critique.get('issues', ['No issues identified']))}\n"
            f"Suggestions: {', '.join(critique.get('suggestions', ['No suggestions provided']))}"
            for critique in critiques
        ])
        
        prompt = f"""
        You are a language refinement agent as described in the paper "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics" (Mousavi et al., 2023).

        Your task is to improve the following text based on the critiques provided by an ensemble of specialized critics:

        Original text:
        ```
        {text}
        ```

        Summary of critiques:
        {summary}

        Detailed critiques:
        {critiques_text}

        Please rewrite the text to address the issues identified by the critics. Maintain the
        original meaning and intent, but improve the quality based on the feedback.

        Improved text:
        """

        try:
            response = self._generate(prompt)

            # Extract improved text from response
            improved_text = response.strip()

            # Remove any markdown code block markers
            if improved_text.startswith("```") and improved_text.endswith("```"):
                improved_text = improved_text[3:-3].strip()

            return improved_text
        except Exception as e:
            raise ImproverError(f"Error generating improved text: {str(e)}")


@register_improver("n_critics")
def create_n_critics_critic(
    model: Model,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    num_critics: int = 3,
    max_refinement_iterations: int = 2,
    **options: Any,
) -> NCriticsCritic:
    """Create an N-Critics critic.

    This factory function creates an NCriticsCritic based on the paper
    "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics" (Mousavi et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        num_critics: Number of specialized critics to use.
        max_refinement_iterations: Maximum number of refinement iterations.
        **options: Additional options to pass to the NCriticsCritic.

    Returns:
        An NCriticsCritic instance.
    """
    return NCriticsCritic(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        num_critics=num_critics,
        max_refinement_iterations=max_refinement_iterations,
        **options,
    )
