"""Research-backed critics for sophisticated text analysis and improvement.

This package provides a comprehensive collection of critic implementations based
on cutting-edge academic research in natural language processing, AI alignment,
and computational linguistics. Each critic embodies a unique approach to text
analysis, offering diverse perspectives for comprehensive content improvement.

## Research-Based Critics:

### 🧠 **ReflexionCritic**
📖 *"Reflexion: Language Agents with Verbal Reinforcement Learning"* (Shinn et al., 2023)

**Core Innovation**: Self-reflection on previous iterations for refined feedback
- **Best for**: Complex writing tasks requiring multiple improvement cycles
- **Strengths**: Learning from iteration history, sophisticated reasoning
- **Use cases**: Research papers, creative writing, complex technical documentation

### 🔄 **SelfRefineCritic**
📖 *"Self-Refine: Iterative Refinement with Self-Feedback"* (Madaan et al., 2023)

**Core Innovation**: Multi-dimensional quality assessment with iterative refinement
- **Best for**: General-purpose text improvement across quality dimensions
- **Strengths**: Balanced evaluation, clear improvement guidance
- **Use cases**: Blog posts, articles, general content polishing

### ⚖️ **ConstitutionalCritic**
📖 *"Constitutional AI: Harmlessness from AI Feedback"* (Bai et al., 2022)

**Core Innovation**: Principle-based evaluation with customizable ethics framework
- **Best for**: Content requiring adherence to specific values or guidelines
- **Strengths**: Consistent ethical evaluation, customizable principles
- **Use cases**: Corporate communications, public content, policy documents

### 🎯 **MetaRewardingCritic**
📖 *"Meta-Rewarding Language Models"* (Wu et al., 2024)

**Core Innovation**: Three-stage self-improving critique (initial → meta-eval → refined)
- **Best for**: High-stakes content requiring premium critique quality
- **Strengths**: Self-correcting evaluation, highest quality feedback
- **Use cases**: Executive communications, legal documents, critical publications

### 🤝 **SelfConsistencyCritic**
📖 *"Self-Consistency Improves Chain of Thought Reasoning"* (Wang et al., 2022)

**Core Innovation**: Multiple evaluation paths with consensus building
- **Best for**: Reducing evaluation bias and improving reliability
- **Strengths**: Robust consensus, reduced evaluation variance
- **Use cases**: Important decisions, baseline establishment, A/B testing

### 👥 **NCriticsCritic**
📖 *"N-Critics: Self-Refinement with Ensemble of Critics"* (Tian et al., 2023)

**Core Innovation**: Multiple expert perspectives with ensemble evaluation
- **Best for**: Comprehensive evaluation requiring diverse viewpoints
- **Strengths**: Multi-perspective coverage, catches blind spots
- **Use cases**: Complex documents, multi-stakeholder content, comprehensive reviews

### 🔍 **SelfRAGCritic**
📖 *"Self-RAG: Learning to Retrieve, Generate, and Critique"* (Asai et al., 2023)

**Core Innovation**: Tool-enabled fact-checking with retrieval-augmented critique
- **Best for**: Content requiring factual accuracy and external verification
- **Strengths**: Real-time fact-checking, evidence-based evaluation
- **Use cases**: News articles, research content, factual documentation

### 🎓 **SelfTaughtEvaluatorCritic**
📖 *"Self-Taught Evaluator: The Path to GPT-4 Level Performance Without Human Annotations"* (Wang et al., 2024)

**Core Innovation**: Contrasting outputs with reasoning traces for self-improving evaluation
- **Best for**: High-quality evaluation without human-labeled data
- **Strengths**: Generates contrasting examples, transparent reasoning, self-improvement
- **Use cases**: Complex evaluation tasks, comparative analysis, evaluation system development

### 🎭 **Agent4DebateCritic**
📖 *"Agent4Debate: Multiagent Competitive Debate Dynamics"* (Chen et al., 2024)

**Core Innovation**: Multi-agent competitive debate for text improvement through argumentation
- **Best for**: Weighing complex trade-offs between competing improvements
- **Strengths**: Adversarial testing, structured reasoning, transparent decisions
- **Use cases**: High-stakes content, controversial topics, strategic documents

## Specialized Critics:

### 🎨 **StyleCritic**
**Focus**: Writing style transformation and brand voice consistency
- **Best for**: Style adaptation, brand voice alignment, audience targeting
- **Features**: Custom style profiles, tone analysis, voice transformation
- **Use cases**: Marketing content, brand communications, audience-specific writing

### 🛠️ **PromptCritic**
**Focus**: Custom prompt-engineered critics for specific use cases
- **Best for**: Domain-specific evaluation without custom code development
- **Features**: Flexible prompt templates, rapid prototyping, specialized evaluation
- **Use cases**: Academic writing, technical documentation, industry-specific content

## Usage Patterns:

    >>> from sifaka import improve
    >>>
    >>> # Single critic for focused improvement
    >>> result = await improve(text, critics=["reflexion"])
    >>>
    >>> # Multi-critic ensemble for comprehensive analysis
    >>> result = await improve(
    ...     text,
    ...     critics=["self_refine", "constitutional", "n_critics"],
    ...     max_iterations=5
    ... )
    >>>
    >>> # Domain-specific configuration
    >>> academic_critics = ["self_consistency", "self_rag", "constitutional"]
    >>> business_critics = ["style", "constitutional", "meta_rewarding"]
    >>> creative_critics = ["reflexion", "self_refine", "style"]
    >>>
    >>> # Custom critic creation
    >>> from sifaka.critics import create_critic
    >>> custom_critic = create_critic(
    ...     "constitutional",
    ...     principles=["Be concise", "Use active voice", "Avoid jargon"]
    ... )

## Advanced Usage:

### Custom Critic Development

    >>> from sifaka.critics import BaseCritic
    >>> from sifaka.core.models import CritiqueResult
    >>>
    >>> class DomainExpertCritic(BaseCritic):
    ...     @property
    ...     def name(self) -> str:
    ...         return "domain_expert"
    ...
    ...     async def _create_messages(self, text, result):
    ...         return [{
    ...             "role": "system",
    ...             "content": "You are a domain expert providing specialized feedback"
    ...         }, {
    ...             "role": "user",
    ...             "content": f"Evaluate this text: {text}"
    ...         }]

### Critic Combination Strategies

**Sequential Processing**: Critics run in order, each building on previous feedback
- Best for: Iterative refinement with increasing sophistication
- Example: `["self_refine", "constitutional", "meta_rewarding"]`

**Parallel Processing**: Critics evaluate independently for diverse perspectives
- Best for: Comprehensive analysis with multiple viewpoints
- Example: `["n_critics", "self_consistency", "self_rag"]`

**Hierarchical Processing**: General → Specific → Quality assurance
- Best for: Systematic improvement with quality gates
- Example: `["self_refine", "style", "constitutional"]`

## Performance Considerations:

- **Resource Usage**: Meta-rewarding (3x tokens) > Self-consistency (3x calls) > Standard critics
- **Speed**: Self-refine, Style (fast) > Constitutional, N-critics (medium) > Meta-rewarding, Self-consistency (slow)
- **Quality**: Meta-rewarding (highest) > Self-consistency > N-critics > Standard critics
- **Specialization**: Self-RAG (factual), Style (voice), Constitutional (values)

Choose critics based on your quality requirements, performance constraints, and specific content needs.
"""

from .agent4debate import Agent4DebateCritic
from .constitutional import ConstitutionalCritic
from .core.base import BaseCritic, CriticResponse
from .core.factory import (
    create_critic,
    create_critics,
    list_available_critics,
    register_critic,
)
from .core.registry import CriticRegistry
from .meta_rewarding import MetaRewardingCritic
from .n_critics import NCriticsCritic
from .prompt import PromptCritic, create_academic_critic
from .reflexion import ReflexionCritic
from .self_consistency import SelfConsistencyCritic
from .self_rag import SelfRAGCritic
from .self_refine import SelfRefineCritic
from .self_taught_evaluator import SelfTaughtEvaluatorCritic
from .style import StyleCritic, style_critic_from_file

# Public API exports
__all__ = [
    # Base classes for custom critic development
    "BaseCritic",
    "CriticResponse",
    # Research-based critics
    "ReflexionCritic",  # Self-reflective learning
    "SelfRefineCritic",  # Multi-dimensional refinement
    "ConstitutionalCritic",  # Principle-based evaluation
    "MetaRewardingCritic",  # Self-improving critique quality
    "SelfConsistencyCritic",  # Consensus-based evaluation
    "NCriticsCritic",  # Multi-perspective ensemble
    "SelfRAGCritic",  # Tool-enabled fact-checking
    "SelfTaughtEvaluatorCritic",  # Contrasting outputs with reasoning traces
    "Agent4DebateCritic",  # Multi-agent competitive debate
    # Specialized critics
    "StyleCritic",  # Style and voice transformation
    "PromptCritic",  # Custom prompt-engineered critics
    # Factory and management
    "create_critic",  # Simple critic instantiation
    "create_critics",  # Batch critic creation
    "list_available_critics",  # Discovery and introspection
    "register_critic",  # Custom critic registration
    "CriticRegistry",  # Critic management system
    # Convenience factory functions
    "create_academic_critic",  # Pre-configured academic writing critic
    "style_critic_from_file",  # Style critic from file-based configuration
]
