"""Example custom critic plugin for Sifaka.

This example shows how to create a domain-specific critic
that can be used to improve text according to custom criteria.
"""

from typing import Any, Dict, Optional

from sifaka import improve_sync
from sifaka.core.config import ModelConfig
from sifaka.critics.core.base import BaseCritic


class AcademicWritingCritic(BaseCritic):
    """A critic specialized for improving academic writing.

    This critic focuses on:
    - Clarity and precision of language
    - Proper citation format
    - Academic tone and objectivity
    - Logical argument structure
    """

    name = "academic_writing"

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        citation_style: str = "APA",
        formality_level: str = "high",
        **kwargs,
    ):
        """Initialize the academic writing critic.

        Args:
            model_config: LLM configuration
            citation_style: Citation format (APA, MLA, Chicago, etc.)
            formality_level: Desired formality (high, medium)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(model_config, **kwargs)
        self.citation_style = citation_style
        self.formality_level = formality_level

    def _build_critique_prompt(
        self,
        original_text: str,
        current_text: str,
        attempt: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the critique prompt for academic writing."""

        prompt = f"""You are an expert academic writing critic. Review the following text and provide specific feedback for improvement.

Focus on these aspects:
1. **Clarity and Precision**: Is the language clear and unambiguous? Are technical terms properly defined?
2. **Academic Tone**: Is the writing objective and formal? Avoid colloquialisms and personal opinions.
3. **Argument Structure**: Is the argument logical and well-supported? Are claims backed by evidence?
4. **Citation Format**: Are sources properly cited in {self.citation_style} format?
5. **Paragraph Structure**: Does each paragraph have a clear topic sentence and supporting details?

Original Text:
{original_text}

Current Version (Attempt {attempt}):
{current_text}

Provide a detailed critique identifying specific areas for improvement. Be constructive and specific about what needs to be changed and why."""

        return prompt

    def _build_improvement_prompt(
        self,
        original_text: str,
        current_text: str,
        critique: str,
        attempt: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the improvement prompt based on critique."""

        prompt = f"""You are an expert academic writer. Improve the following text based on the critique provided.

Ensure the improved version:
- Maintains {self.formality_level} formality
- Uses {self.citation_style} citation format
- Addresses all points raised in the critique
- Preserves the original meaning while enhancing clarity
- Follows academic writing conventions

Current Text:
{current_text}

Critique:
{critique}

Provide only the improved text, without any explanations or meta-commentary."""

        return prompt

    def _parse_critique_response(self, response: str) -> str:
        """Parse the critique from the model response."""
        # For academic writing, we want to preserve detailed feedback
        return response.strip()

    def _parse_improvement_response(self, response: str) -> str:
        """Parse the improved text from the model response."""
        # Remove any potential meta-commentary
        lines = response.strip().split("\n")

        # Filter out lines that look like headers or explanations
        filtered_lines = []
        for line in lines:
            if not (
                line.startswith("Improved")
                or line.startswith("Here")
                or line.startswith("Version")
                or line.strip() == ""
            ):
                filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()


class TechnicalDocumentationCritic(BaseCritic):
    """A critic specialized for technical documentation.

    This critic focuses on:
    - Code examples and their correctness
    - API documentation completeness
    - Step-by-step instructions clarity
    - Technical accuracy
    """

    name = "technical_docs"

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        programming_language: Optional[str] = None,
        doc_style: str = "sphinx",  # sphinx, doxygen, markdown
        **kwargs,
    ):
        """Initialize the technical documentation critic."""
        super().__init__(model_config, **kwargs)
        self.programming_language = programming_language
        self.doc_style = doc_style

    def _build_critique_prompt(
        self,
        original_text: str,
        current_text: str,
        attempt: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build critique prompt for technical documentation."""

        lang_context = (
            f" for {self.programming_language}" if self.programming_language else ""
        )

        prompt = f"""You are an expert technical writer{lang_context}. Review this documentation and provide specific feedback.

Evaluate these aspects:
1. **Completeness**: Are all parameters, return values, and exceptions documented?
2. **Code Examples**: Are examples correct, runnable, and illustrative?
3. **Clarity**: Can a developer understand how to use this without ambiguity?
4. **Technical Accuracy**: Are technical details correct and up-to-date?
5. **Structure**: Is the documentation well-organized with clear sections?

Documentation Style: {self.doc_style}

Original Documentation:
{original_text}

Current Version (Attempt {attempt}):
{current_text}

Provide specific feedback on what needs improvement and why."""

        return prompt

    def _build_improvement_prompt(
        self,
        original_text: str,
        current_text: str,
        critique: str,
        attempt: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build improvement prompt for technical documentation."""

        prompt = f"""You are an expert technical writer. Improve this documentation based on the critique.

Requirements:
- Use {self.doc_style} documentation format
- Include clear, working code examples
- Ensure all technical details are accurate
- Make it easy for developers to understand and use
- Address all points from the critique

Current Documentation:
{current_text}

Critique:
{critique}

Provide only the improved documentation text."""

        return prompt

    def _parse_critique_response(self, response: str) -> str:
        """Parse critique response."""
        return response.strip()

    def _parse_improvement_response(self, response: str) -> str:
        """Parse improvement response."""
        return response.strip()


# To register these critics:
# 1. Manual registration:
# from sifaka.critics import register_critic
# register_critic("academic_writing", AcademicWritingCritic)
# register_critic("technical_docs", TechnicalDocumentationCritic)

# 2. Or via entry points in setup.py:
#    entry_points={
#        "sifaka.critics": [
#            "academic_writing = my_plugin:AcademicWritingCritic",
#            "technical_docs = my_plugin:TechnicalDocumentationCritic",
#        ],
#    }


if __name__ == "__main__":
    # Example usage
    from sifaka.critics import register_critic

    # Register custom critics
    register_critic("academic_writing", AcademicWritingCritic)
    register_critic("technical_docs", TechnicalDocumentationCritic)

    # Example 1: Improve academic writing
    academic_text = """
    Machine learning is really cool and lots of people use it.
    It can do amazing things like recognize pictures and translate languages.
    According to some researchers, it's getting better every day.
    """

    print("Academic Writing Example:")
    print("Original:", academic_text)

    # This would make actual API calls in practice
    result = improve_sync(
        academic_text,
        critics=["academic_writing"],  # Custom critics use strings after registration
        max_iterations=2,
        critic_kwargs={"citation_style": "APA", "formality_level": "high"},
    )
    print("Improved:", result.final_text)

    # Example 2: Improve technical documentation
    tech_docs = """
    This function does some stuff with data.
    It takes some parameters and returns something.
    You should probably check for errors.
    """

    print("\nTechnical Documentation Example:")
    print("Original:", tech_docs)

    result = improve_sync(
        tech_docs,
        critics=["technical_docs"],  # Custom critics use strings after registration
        max_iterations=2,
        critic_kwargs={"programming_language": "Python", "doc_style": "sphinx"},
    )
    print("Improved:", result.final_text)
