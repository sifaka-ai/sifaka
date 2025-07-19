"""Prompt-engineered critic for flexible, domain-specific text evaluation.

This module provides a highly adaptable critic implementation that uses custom
prompts to define evaluation criteria, enabling rapid deployment of specialized
critics without writing new code. Perfect for domain experts who understand
their content requirements but don't want to implement custom critic classes.

## Core Philosophy:

The PromptCritic democratizes critic creation by allowing natural language
specification of evaluation criteria. This approach recognizes that domain
experts often know exactly what makes content good in their field but may
not have the programming expertise to implement custom validators.

## Key Capabilities:

- **Custom Criteria**: Define any evaluation criteria in natural language
- **Domain Flexibility**: Adapt to any content type or industry requirement
- **Rapid Prototyping**: Test new evaluation approaches without coding
- **Expert Knowledge**: Encode domain expertise directly in prompts
- **Iterative Refinement**: Easily adjust criteria based on results

## Common Use Cases:

### Academic Writing
- Citation format compliance
- Argument structure validation
- Academic tone consistency
- Literature review completeness

### Technical Documentation
- API documentation standards
- Code example accuracy
- Prerequisite clarity
- Step-by-step completeness

### Legal Documents
- Clause completeness
- Defined terms usage
- Risk disclosure adequacy
- Regulatory compliance language

### Marketing Content
- Brand voice adherence
- Feature-benefit alignment
- Call-to-action effectiveness
- SEO keyword integration

## Usage Patterns:

    >>> # Custom evaluation criteria
    >>> critic = PromptCritic(
    ...     custom_prompt=\"\"\"Evaluate this text for:
    ...     1. Technical accuracy
    ...     2. Implementation completeness
    ...     3. Security considerations
    ...     4. Performance implications
    ...     Rate each dimension and provide specific improvements.\"\"\"
    ... )
    >>>
    >>> # Domain-specific critic
    >>> medical_critic = PromptCritic(
    ...     custom_prompt=\"\"\"As a medical reviewer, evaluate for:
    ...     - Clinical accuracy
    ...     - Patient safety considerations
    ...     - Regulatory compliance (FDA, HIPAA)
    ...     - Clarity for patient understanding\"\"\"
    ... )
    >>>
    >>> # Multi-stakeholder evaluation
    >>> stakeholder_critic = PromptCritic(
    ...     custom_prompt='''Evaluate from three perspectives:
    ...     1. Engineering: Technical feasibility
    ...     2. Product: User value and experience
    ...     3. Business: ROI and market fit'''
    ... )

## Prompt Engineering Best Practices:

1. **Be Specific**: Detailed criteria yield better evaluations
2. **Use Structure**: Numbered lists and categories improve analysis
3. **Include Examples**: Show what good looks like in your domain
4. **Define Priorities**: Indicate which criteria matter most
5. **Request Evidence**: Ask for specific examples in feedback

## Pre-built Critics:

This module includes factory functions for common use cases:
- `create_academic_critic()`: Academic paper evaluation
- `create_business_critic()`: Business document assessment
- `create_creative_critic()`: Creative writing analysis

These serve as both ready-to-use critics and templates for customization.

## Integration Notes:

- Works seamlessly with all Sifaka validators and other critics
- Can be combined in multi-critic pipelines for comprehensive evaluation
- Supports the same configuration options as all BaseCritic implementations
- Provides structured responses despite flexible prompt input
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import SifakaResult
from .core.base import BaseCritic


class PromptResponse(BaseModel):
    """Structured response format for prompt-based evaluation results.

    Provides consistent output structure regardless of the custom evaluation
    criteria, ensuring seamless integration with the Sifaka improvement pipeline.
    The response captures both high-level assessment and detailed feedback.

    Attributes:
        feedback: Comprehensive evaluation based on the custom criteria defined
            in the prompt. Should directly address each evaluation dimension
            specified by the user.
        suggestions: Specific, actionable improvements derived from the custom
            criteria. Each suggestion should map to identified issues.
        needs_improvement: Binary assessment of whether the text meets the
            custom criteria. True if any significant issues found.
        confidence: Model's confidence in its evaluation (0.0-1.0). Affected by:
            - Clarity of custom criteria
            - Text complexity
            - Domain familiarity
        metadata: Extensible field for custom evaluation data such as:
            - Per-criterion scores
            - Domain-specific metrics
            - Detailed analysis breakdowns
    """

    feedback: str = Field(..., description="Overall feedback based on custom criteria")
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom evaluation data"
    )


class PromptCritic(BaseCritic):
    """Flexible critic using custom prompts for domain-specific evaluation.

    Enables rapid deployment of specialized critics through natural language
    prompt engineering rather than code implementation. Ideal for domain experts
    who understand their content requirements but want to avoid custom development.

    ## When to Use This Critic:

    ✅ **Ideal for:**
    - Domain-specific evaluation criteria not covered by standard critics
    - Rapid prototyping of new evaluation approaches
    - Industry-specific compliance requirements
    - Multi-dimensional evaluation with custom rubrics
    - Experimental or evolving evaluation criteria
    - Non-technical users who understand content requirements

    ❌ **Avoid when:**
    - Standard critics (style, constitutional, etc.) meet your needs
    - You need guaranteed consistency across evaluations
    - Evaluation criteria are vague or undefined
    - Performance is critical (custom code is more efficient)

    🎯 **Perfect applications:**
    - **Regulatory Compliance**: "Check for GDPR compliance language"
    - **Brand Guidelines**: "Ensure alignment with 2024 brand voice guide"
    - **Technical Standards**: "Validate against API documentation standards"
    - **Academic Requirements**: "Check thesis statement and citation format"
    - **Accessibility**: "Evaluate for WCAG 2.1 AA compliance"
    - **Industry Jargon**: "Ensure appropriate medical terminology usage"

    ## Prompt Engineering Tips:

    1. **Structure**: Use numbered lists or categories for clarity
    2. **Specificity**: Define exact criteria, not general quality
    3. **Examples**: Include examples of good/bad in your prompt
    4. **Priorities**: Indicate which criteria are most important
    5. **Evidence**: Request specific examples in the feedback

    ## Advanced Usage:

        >>> # Multi-perspective evaluation
        >>> critic = PromptCritic(
        ...     custom_prompt='''Evaluate from multiple stakeholder perspectives:
        ...
        ...     CUSTOMER: Is the value proposition clear? Benefits obvious?
        ...     LEGAL: Any claims that need substantiation? Risk exposure?
        ...     BRAND: Does this align with our premium positioning?
        ...
        ...     Provide perspective-specific feedback and overall recommendations.'''
        ... )
        >>>
        >>> # Rubric-based evaluation
        >>> critic = PromptCritic(
        ...     custom_prompt='''Score each criterion (1-5) and explain:
        ...
        ...     1. Clarity (weight: 30%): Is the main point immediately clear?
        ...     2. Evidence (weight: 40%): Are claims supported with data?
        ...     3. Action (weight: 30%): Is the next step obvious?
        ...
        ...     Calculate weighted score and identify top improvement priority.'''
        ... )
    """

    def __init__(
        self,
        custom_prompt: str = "Evaluate this text for quality and suggest improvements.",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """Initialize a custom prompt-based critic.

        Creates a flexible critic that evaluates text based on user-defined
        criteria expressed in natural language. The prompt defines what aspects
        of the text to evaluate and how to assess quality.

        Args:
            custom_prompt: Natural language prompt defining evaluation criteria.
                Should clearly specify what to evaluate and how to judge quality.
                Can include multiple criteria, scoring rubrics, or perspectives.
                Default provides general quality evaluation.
            model: LLM model for evaluation. GPT-4o-mini recommended for
                cost-effective evaluation. Use GPT-4 for complex criteria.
            temperature: Generation temperature (0.0-1.0). Lower values (0.3-0.5)
                provide more consistent evaluation, higher values (0.7-0.9)
                offer more creative feedback.
            provider: LLM provider (OpenAI, Anthropic, etc.)
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object

        Example:
            >>> # Technical documentation critic
            >>> critic = PromptCritic(
            ...     custom_prompt='''Evaluate this technical documentation for:
            ...     1. Completeness: Are all parameters documented?
            ...     2. Clarity: Can a developer understand without context?
            ...     3. Examples: Are there practical usage examples?
            ...     4. Errors: How are error cases handled?
            ...     Score each dimension and provide specific improvements.'''
            ... )
            >>>
            >>> # Inclusive language critic
            >>> critic = PromptCritic(
            ...     custom_prompt='''Review for inclusive language:
            ...     - Avoid gendered assumptions
            ...     - Use person-first language for disabilities
            ...     - Check for cultural sensitivity
            ...     - Ensure accessibility considerations
            ...     Highlight specific issues and suggest alternatives.'''
            ... )

        Prompt design tips:
            - Start with clear evaluation objectives
            - Break down complex criteria into specific dimensions
            - Include scoring or rating instructions if needed
            - Request specific examples in the feedback
            - Consider the target audience for the content
        """
        # Initialize with custom config
        if config is None:
            config = Config()

        super().__init__(model, temperature, config, provider, api_key)
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        """Return the identifier for this critic type.

        Returns:
            "prompt" - Used in configuration and logging to identify
            custom prompt-based evaluation.
        """
        return "prompt"

    def _get_response_type(self) -> type[BaseModel]:
        """Specify the structured response format for prompt-based evaluation.

        Returns:
            PromptResponse class that provides consistent structure
            regardless of custom evaluation criteria, ensuring smooth
            integration with the Sifaka improvement pipeline.
        """
        return PromptResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create evaluation messages using the custom prompt.

        Constructs the evaluation request by combining the user's custom
        prompt with the text to evaluate and any relevant context from
        previous iterations.

        Args:
            text: Text to evaluate against custom criteria
            result: SifakaResult containing iteration history and context

        Returns:
            List of message dictionaries with:
            - System message: Establishes the critic as a flexible evaluator
            - User message: Combines custom prompt, text, and context

        Message structure:
            1. Custom evaluation criteria (from user's prompt)
            2. Text to evaluate
            3. Previous iteration context (if applicable)
            4. Request for specific, actionable feedback

        The prompt is designed to elicit:
            - Specific examples from the text
            - Clear rationale for assessments
            - Actionable improvement suggestions
            - Confidence-appropriate responses
        """
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""{self.custom_prompt}

Text to evaluate:
{text}
{previous_context}

Please provide specific, actionable feedback based on the evaluation criteria."""

        return [
            {
                "role": "system",
                "content": "You are a customizable text critic that evaluates based on user-defined criteria.",
            },
            {"role": "user", "content": user_prompt},
        ]


def create_academic_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs: Any
) -> PromptCritic:
    """Create a pre-configured critic for academic writing evaluation.

    Input parameters are validated at runtime.

    Provides comprehensive evaluation of academic texts across key scholarly
    dimensions. Suitable for research papers, dissertations, academic articles,
    and scholarly communications.

    Args:
        model: LLM model for evaluation. Default gpt-4o-mini balances
            quality and cost. Consider gpt-4 for nuanced academic evaluation.
        temperature: Generation temperature (0.0-1.0). Default 0.7 provides
            balanced evaluation. Lower for more consistent assessment.
        **kwargs: Additional arguments passed to PromptCritic.
            See CriticFactoryParams in core.type_defs for expected fields:
            - provider: LLM provider override
            - api_key: API key override
            - config: Full configuration object

    Returns:
        PromptCritic configured for academic writing evaluation with
        focus on scholarly standards and conventions

    Evaluation criteria:
        1. **Clarity and Precision**: Technical accuracy, unambiguous language
        2. **Logical Flow**: Argument structure, transitions, coherence
        3. **Evidence Usage**: Citation quality, source credibility, support
        4. **Academic Tone**: Formal register, objective voice, scholarly style
        5. **Field Contribution**: Originality, significance, research gap

    Example:
        >>> # Standard academic evaluation
        >>> critic = create_academic_critic()
        >>>
        >>> # High-precision evaluation for journal submission
        >>> critic = create_academic_critic(
        ...     model="gpt-4",
        ...     temperature=0.5
        ... )
        >>>
        >>> # Evaluate a research abstract
        >>> result = await improve(
        ...     abstract_text,
        ...     critics=[critic],
        ...     max_iterations=3
        ... )

    Best suited for:
        - Research paper drafts
        - Dissertation chapters
        - Grant proposals
        - Academic abstracts
        - Literature reviews
        - Conference papers
    """
    prompt = """Evaluate this text as an academic paper excerpt. Consider:
1. Clarity and precision of language
2. Logical flow and argumentation
3. Use of evidence and citations
4. Academic tone and style
5. Contribution to the field"""

    # Create PromptCritic with parameters
    return PromptCritic(
        custom_prompt=prompt,
        model=model,
        temperature=temperature,
        **kwargs,
    )


def create_business_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs: Any
) -> PromptCritic:
    """Create a pre-configured critic for business document evaluation.

    Optimized for professional business communications including proposals,
    reports, presentations, and strategic documents. Focuses on clarity,
    professionalism, and business impact.

    Args:
        model: LLM model for evaluation. Default gpt-4o-mini provides
            good business judgment. Use gpt-4 for executive-level content.
        temperature: Generation temperature (0.0-1.0). Default 0.7 balances
            consistency with insightful feedback.
        **kwargs: Additional arguments passed to PromptCritic.
            See CriticFactoryParams in core.type_defs for expected fields:
            - provider: LLM provider override
            - api_key: API key override
            - config: Full configuration object

    Returns:
        PromptCritic configured for business document evaluation with
        focus on professional communication standards

    Evaluation criteria:
        1. **Message Clarity**: Clear value proposition, obvious next steps
        2. **Professional Tone**: Appropriate formality, credibility, polish
        3. **Value Focus**: Benefits over features, ROI, business impact
        4. **Structure**: Logical flow, scannable format, executive summary
        5. **Persuasiveness**: Compelling arguments, evidence, call-to-action

    Example:
        >>> # Standard business document review
        >>> critic = create_business_critic()
        >>>
        >>> # Executive presentation review
        >>> critic = create_business_critic(
        ...     model="gpt-4",
        ...     temperature=0.6  # More consistent for formal docs
        ... )
        >>>
        >>> # Evaluate a business proposal
        >>> result = await improve(
        ...     proposal_text,
        ...     critics=[critic],
        ...     validators=[LengthValidator(max_length=5000)]
        ... )

    Best suited for:
        - Business proposals and RFPs
        - Executive summaries
        - Strategic plans
        - Marketing materials
        - Investor presentations
        - Project reports
        - Professional emails
    """
    prompt = """Evaluate this business document. Consider:
1. Clarity of message and call-to-action
2. Professional tone and language
3. Value proposition and benefits
4. Structure and organization
5. Persuasiveness and impact"""

    return PromptCritic(
        custom_prompt=prompt, model=model, temperature=temperature, **kwargs
    )


def create_creative_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs: Any
) -> PromptCritic:
    """Create a pre-configured critic for creative writing evaluation.

    Tailored for fiction, creative non-fiction, poetry, and other artistic
    writing. Emphasizes narrative craft, emotional resonance, and artistic merit
    over technical correctness.

    Args:
        model: LLM model for evaluation. Default gpt-4o-mini offers
            good creative insights. Consider gpt-4 for literary analysis.
        temperature: Generation temperature (0.0-1.0). Default 0.7 allows
            for creative interpretation. Higher values (0.8-0.9) provide
            more diverse creative feedback.
        **kwargs: Additional arguments passed to PromptCritic.
            See CriticFactoryParams in core.type_defs for expected fields:
            - provider: LLM provider override
            - api_key: API key override
            - config: Full configuration object

    Returns:
        PromptCritic configured for creative writing evaluation with
        focus on artistic elements and reader engagement

    Evaluation criteria:
        1. **Narrative Flow**: Pacing, tension, story arc, scene transitions
        2. **Character Voice**: Authenticity, consistency, development, dialogue
        3. **Imagery**: Sensory details, metaphors, descriptive power, showing vs telling
        4. **Emotional Impact**: Reader engagement, empathy, memorable moments
        5. **Originality**: Fresh perspectives, unique voice, creative choices

    Example:
        >>> # Standard creative writing review
        >>> critic = create_creative_critic()
        >>>
        >>> # Literary fiction evaluation
        >>> critic = create_creative_critic(
        ...     model="gpt-4",
        ...     temperature=0.8  # More creative feedback
        ... )
        >>>
        >>> # Evaluate a short story excerpt
        >>> result = await improve(
        ...     story_excerpt,
        ...     critics=[critic],
        ...     max_iterations=5  # Allow more creative iterations
        ... )

    Best suited for:
        - Short stories and novels
        - Creative non-fiction
        - Poetry and verse
        - Screenplay and dialogue
        - Personal narratives
        - Descriptive passages
        - Character sketches

    Note:
        This critic prioritizes artistic merit over technical rules.
        Combine with other critics (e.g., style, grammar) for comprehensive
        creative writing improvement.
    """
    prompt = """Evaluate this creative writing. Consider:
1. Narrative flow and pacing
2. Character development and voice
3. Descriptive language and imagery
4. Emotional impact and engagement
5. Originality and creativity"""

    return PromptCritic(
        custom_prompt=prompt, model=model, temperature=temperature, **kwargs
    )
