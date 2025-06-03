"""Prompt-based customizable critic for Sifaka.

This module implements a flexible, prompt-based critic that can be customized
for specific evaluation criteria, domains, or use cases through configurable
prompts and evaluation guidelines.

This critic is not based on a specific research paper but provides a practical
framework for implementing custom evaluation criteria using prompt engineering
and structured output formatting.

The PromptCritic enables:
1. Custom evaluation criteria through configurable prompts
2. Domain-specific critique through specialized guidelines
3. Task-specific evaluation through tailored instructions
4. Flexible adaptation to different use cases and requirements

IMPLEMENTATION APPROACH:

This critic uses prompt engineering to create customizable evaluation criteria
without requiring separate model training or fine-tuning. It leverages the
underlying language model's ability to follow detailed instructions and
evaluation guidelines provided through carefully crafted prompts.

The approach is inspired by prompt engineering best practices and instruction
following research, allowing users to specify custom evaluation criteria,
rubrics, and guidelines that the critic will apply consistently.

DESIGN PRINCIPLES:
1. Flexibility: Support for arbitrary evaluation criteria and guidelines
2. Consistency: Structured output format regardless of custom criteria
3. Adaptability: Easy modification for different domains and use cases
4. Transparency: Clear specification of evaluation criteria and reasoning
5. Integration: Seamless integration with Sifaka's validation and critique framework

CUSTOMIZATION OPTIONS:
- Custom evaluation criteria and rubrics
- Domain-specific guidelines and best practices
- Task-specific evaluation instructions
- Specialized terminology and concepts
- Custom scoring and assessment methods

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance custom
evaluation by providing external guidelines, examples, domain knowledge,
or reference materials relevant to the specific evaluation criteria.
"""

from typing import Any, Dict, List, Optional

from pydantic_ai import Agent

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic, CritiqueFeedback


class PromptCritic(BaseCritic):
    """Customizable prompt-based critic for flexible evaluation criteria.

    This critic allows users to specify custom evaluation criteria, guidelines,
    and instructions through configurable prompts, enabling domain-specific
    and task-specific critique without requiring separate model training.

    Enhanced with validation context awareness to ensure custom evaluation
    criteria work harmoniously with validation requirements.
    """

    def __init__(
        self,
        model_name: str = "anthropic:claude-3-haiku-20240307",
        custom_criteria: Optional[str] = None,
        evaluation_guidelines: Optional[str] = None,
        domain_context: Optional[str] = None,
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Prompt-based critic.

        Args:
            model_name: The model name for the PydanticAI agent
            custom_criteria: Custom evaluation criteria and instructions
            evaluation_guidelines: Specific guidelines for evaluation process
            domain_context: Domain-specific context and terminology
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.custom_criteria = custom_criteria or self._get_default_criteria()
        self.evaluation_guidelines = evaluation_guidelines or self._get_default_guidelines()
        self.domain_context = domain_context or "General text evaluation"

        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Custom prompt-based critic implementation. "
            "Not based on a specific research paper but follows prompt engineering best practices "
            "for instruction following and structured evaluation."
        )
        methodology = (
            "Prompt-based methodology: Customizable evaluation through configurable criteria and guidelines. "
            "Uses prompt engineering for flexible, domain-specific critique without model training. "
            "Adaptable to various use cases and evaluation requirements."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            **agent_kwargs,
        )

    def _get_default_criteria(self) -> str:
        """Get default evaluation criteria."""
        return """Evaluate the text based on the following criteria:
1. Clarity: Is the text clear, well-structured, and easy to understand?
2. Accuracy: Is the information accurate and factually correct?
3. Completeness: Does the text adequately address the given task or topic?
4. Relevance: Is the content relevant and focused on the main topic?
5. Quality: Is the overall quality appropriate for the intended purpose?"""

    def _get_default_guidelines(self) -> str:
        """Get default evaluation guidelines."""
        return """Follow these guidelines when evaluating:
- Be specific and constructive in your feedback
- Provide actionable suggestions for improvement
- Consider the intended audience and purpose
- Balance criticism with recognition of strengths
- Focus on the most important issues first"""

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Prompt-based critic."""
        return f"""You are a customizable Prompt-based critic for text evaluation.

Your role is to evaluate text according to specific custom criteria and guidelines
provided by the user, offering flexible and adaptable critique for various
domains, tasks, and use cases.

DOMAIN CONTEXT:
{self.domain_context}

CUSTOM EVALUATION CRITERIA:
{self.custom_criteria}

EVALUATION GUIDELINES:
{self.evaluation_guidelines}

RESPONSE FORMAT:
- needs_improvement: boolean indicating if text needs improvement based on custom criteria
- message: detailed analysis according to the specified evaluation criteria
- suggestions: 1-3 specific suggestions based on the custom guidelines
- confidence: float 0.0-1.0 based on assessment certainty using the given criteria
- reasoning: explanation of evaluation process and application of custom criteria

EVALUATION PROCESS:
1. Apply the custom evaluation criteria systematically
2. Follow the specified evaluation guidelines
3. Consider the domain context and specialized requirements
4. Provide feedback that aligns with the custom criteria
5. Generate actionable suggestions based on the guidelines

Be consistent in applying the custom criteria while maintaining flexibility
to adapt to the specific requirements and context provided."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for custom prompt-based methodology."""
        if not thought.current_text:
            return "No text available for custom prompt-based critique."

        prompt_parts = [
            "CUSTOM PROMPT-BASED CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            f"Domain Context: {self.domain_context}",
            "",
            "TEXT TO EVALUATE:",
            thought.current_text,
            "",
            "CUSTOM EVALUATION CRITERIA:",
            "=" * 35,
            self.custom_criteria,
            "",
            "EVALUATION GUIDELINES:",
            "=" * 25,
            self.evaluation_guidelines,
            "",
        ]

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend(
                [
                    "VALIDATION REQUIREMENTS:",
                    "=" * 25,
                    validation_context,
                    "",
                    "NOTE: Custom evaluation should consider validation requirements",
                    "alongside the specified custom criteria and guidelines.",
                    "",
                ]
            )

        # Add previous custom critiques
        if thought.iteration > 0:
            prev_prompt_critiques = [
                c
                for c in thought.critiques
                if c.iteration == thought.iteration - 1 and c.critic == "PromptCritic"
            ]
            if prev_prompt_critiques:
                prompt_parts.extend(
                    [
                        "PREVIOUS CUSTOM EVALUATION:",
                        "=" * 30,
                    ]
                )
                for critique in prev_prompt_critiques[-1:]:  # Last prompt critique
                    prompt_parts.extend(
                        [
                            f"Previous Assessment: {critique.feedback[:150]}{'...' if len(critique.feedback) > 150 else ''}",
                            f"Previous Suggestions: {', '.join(critique.suggestions)}",
                            "",
                        ]
                    )

        # Add custom evaluation instructions
        prompt_parts.extend(
            [
                "CUSTOM EVALUATION INSTRUCTIONS:",
                "=" * 40,
                "1. Apply the custom evaluation criteria systematically to the text",
                "2. Follow the specified evaluation guidelines throughout the process",
                "3. Consider the domain context and any specialized requirements",
                "4. Provide specific feedback that aligns with the custom criteria",
                "5. Generate actionable suggestions based on the evaluation guidelines",
                "6. Maintain consistency with the specified evaluation approach",
                "",
                "EVALUATION FOCUS:",
                "- Apply each criterion from the custom evaluation criteria",
                "- Follow the evaluation guidelines for feedback style and approach",
                "- Consider domain-specific requirements and context",
                "- Provide constructive, actionable feedback",
                "- Maintain consistency with previous evaluations using the same criteria",
                "",
                "Evaluate the text thoroughly according to the custom criteria and guidelines provided.",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Prompt-based critic-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)

        # Add Prompt-based critic-specific metadata
        prompt_metadata = {
            "methodology": "custom_prompt_based",
            "domain_context": self.domain_context,
            "has_custom_criteria": bool(self.custom_criteria),
            "has_custom_guidelines": bool(self.evaluation_guidelines),
            "customizable": True,
            "criteria_length": len(self.custom_criteria) if self.custom_criteria else 0,
            "guidelines_length": (
                len(self.evaluation_guidelines) if self.evaluation_guidelines else 0
            ),
        }

        base_metadata.update(prompt_metadata)
        return base_metadata

    def update_criteria(self, new_criteria: str) -> None:
        """Update the custom evaluation criteria.

        Args:
            new_criteria: New evaluation criteria to use
        """
        self.custom_criteria = new_criteria
        # Update system prompt with new criteria
        self.system_prompt = self._create_system_prompt()
        # Recreate agent with updated prompt
        self.agent = Agent(
            model=self.model_name,
            output_type=CritiqueFeedback,
            system_prompt=self.system_prompt,
            tools=self.retrieval_tools,
        )

    def update_guidelines(self, new_guidelines: str) -> None:
        """Update the evaluation guidelines.

        Args:
            new_guidelines: New evaluation guidelines to use
        """
        self.evaluation_guidelines = new_guidelines
        # Update system prompt with new guidelines
        self.system_prompt = self._create_system_prompt()
        # Recreate agent with updated prompt
        self.agent = Agent(
            model=self.model_name,
            output_type=CritiqueFeedback,
            system_prompt=self.system_prompt,
            tools=self.retrieval_tools,
        )

    def update_domain_context(self, new_context: str) -> None:
        """Update the domain context.

        Args:
            new_context: New domain context to use
        """
        self.domain_context = new_context
        # Update system prompt with new context
        self.system_prompt = self._create_system_prompt()
        # Recreate agent with updated prompt
        self.agent = Agent(
            model=self.model_name,
            output_type=CritiqueFeedback,
            system_prompt=self.system_prompt,
            tools=self.retrieval_tools,
        )
