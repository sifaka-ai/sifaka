# Critics: A Visual Guide

Critics in Sifaka provide automated feedback and improvement suggestions for generated text. This guide presents each critic type with visual explanations, progressing from simple to complex approaches.

## Overview: The Critic Landscape

### Complexity Levels

| Level | Critics | Key Features |
|-------|---------|--------------|
| ⭐ **Beginner** | PromptCritic | Custom prompts, simple rules |
| ⭐⭐ **Intermediate** | SelfRefineCritic | Iterative improvement loops |
| ⭐⭐⭐ **Intermediate-Advanced** | ReflexionCritic<br/>ConstitutionalCritic | Self-reflection & learning<br/>Principle-based evaluation |
| ⭐⭐⭐⭐ **Advanced** | SelfRAGCritic<br/>SelfConsistencyCritic<br/>NCriticsCritic | External knowledge integration<br/>Consensus from multiple critiques<br/>Multi-role perspective analysis |
| ⭐⭐⭐⭐⭐ **Expert** | MetaRewardingCritic | Two-stage meta-evaluation |

### Quick Picker

<details>
<summary><strong>🎯 I want simple, custom feedback</strong></summary>

**→ PromptCritic** - Define your own criteria with custom prompts
</details>

<details>
<summary><strong>🔄 I want iterative improvement</strong></summary>

**→ SelfRefineCritic** - Multiple rounds of self-critique and revision
</details>

<details>
<summary><strong>🪞 I want learning from experience</strong></summary>

**→ ReflexionCritic** - Self-reflection with episodic memory
</details>

<details>
<summary><strong>⚖️ I want ethical/principle compliance</strong></summary>

**→ ConstitutionalCritic** - Evaluation against constitutional principles
</details>

<details>
<summary><strong>🔍 I want fact-checking with external knowledge</strong></summary>

**→ SelfRAGCritic** - Retrieval-augmented critique with external sources
</details>

<details>
<summary><strong>🎯 I want reliable, consensus-based feedback</strong></summary>

**→ SelfConsistencyCritic** - Multiple critiques with majority agreement
</details>

<details>
<summary><strong>🎭 I want comprehensive multi-perspective analysis</strong></summary>

**→ NCriticsCritic** - Multiple specialized roles (content, style, structure, etc.)
</details>

<details>
<summary><strong>🏆 I want maximum quality assurance</strong></summary>

**→ MetaRewardingCritic** - Two-stage judgment with meta-evaluation
</details>

### Interactive Critic Flow Diagram

```mermaid
graph TD
    A[Input Text] --> B{Choose Critic Type}

    B -->|Simple & Custom| C[PromptCritic]
    B -->|Iterative| D[SelfRefineCritic]
    B -->|Learning| E[ReflexionCritic]
    B -->|Ethical| F[ConstitutionalCritic]
    B -->|Knowledge-Based| G[SelfRAGCritic]
    B -->|Consensus| H[SelfConsistencyCritic]
    B -->|Meta-Evaluation| I[MetaRewardingCritic]
    B -->|Multi-Role| J[NCriticsCritic]

    C --> K[Custom Prompt Analysis]
    D --> L[Multiple Refinement Rounds]
    E --> M[Self-Reflection Process]
    F --> N[Principle Checking]
    G --> O[Retrieval + Critique]
    H --> P[Multiple Critiques → Consensus]
    I --> Q[Base Critique → Meta-Judge]
    J --> R[Multiple Roles → Aggregate]

    K --> S[Improved Text]
    L --> S
    M --> S
    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S

    style A fill:#e1f5fe
    style S fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#fce4ec
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#ffebee
    style J fill:#ffebee
```

**All critics share the same interface:**
- `critique(thought)` → Analyze and provide feedback
- `improve(thought)` → Generate improved version

---

## 1. PromptCritic 🎯
**Complexity: ⭐ (Beginner)**

The simplest critic - you define exactly what to look for using custom prompts.

### Visual Flow
```
Input Text ──→ [Your Custom Prompt] ──→ Critique ──→ [Your Improvement Prompt] ──→ Better Text
```

### Key Concept
Think of this as having a human reviewer with specific instructions. You tell the critic exactly what criteria to use and how to improve text.

### When to Use
- You have specific, well-defined criteria
- Domain-specific requirements (legal, medical, technical)
- Simple, focused improvements
- Getting started with critics

### Example
```python
from sifaka.critics.prompt import PromptCritic

critic = PromptCritic(
    model=model,
    critique_prompt="Check if this text is professional and clear. List any issues.",
    improvement_prompt="Rewrite this text to be more professional: {text}"
)
```

---

## 2. SelfRefineCritic 🔄
**Complexity: ⭐⭐ (Intermediate)**

Iteratively improves text through multiple rounds of self-critique and revision.

### Visual Flow
```
Input Text ──→ [Critique] ──→ [Improve] ──→ [Critique Again] ──→ [Improve Again] ──→ Final Text
     │              │              │              │                    │
     │              │              │              │                    │
     └─── Round 1 ──┴─── Round 2 ──┴─── Round 3 ──┴─── Continue ──────┘
                                                    Until Satisfied
```

### Key Concept
Like a writer who reviews their work multiple times, making incremental improvements with each pass. Each iteration builds on the previous critique to create progressively better text.

### When to Use
- General text improvement through iteration
- When you want multiple refinement passes
- Tasks that benefit from incremental improvement
- When single-pass critique isn't sufficient

### Example
```python
from sifaka.critics.self_refine import SelfRefineCritic

critic = SelfRefineCritic(
    model=model,
    max_iterations=3,  # Up to 3 rounds of improvement
    improvement_criteria=["clarity", "accuracy", "engagement"]
)
```

---

## 3. ReflexionCritic 🪞
**Complexity: ⭐⭐⭐ (Intermediate-Advanced)**

Uses self-reflection to understand failures and successes, learning from experience.

### Visual Flow
```
Input Text ──→ [Initial Critique] ──→ [Self-Reflection] ──→ [Improved Text]
     │               │                       │                    │
     │               │                       │                    │
     └─── Task ──────┴─── "Why did ─────────┴─── Apply ─────────┘
       Feedback           this fail?"           Insights
```

### Key Concept
Like a thoughtful student who not only identifies mistakes but reflects on *why* they happened within the current task. Performs deeper analysis by reflecting on its own critique process.

### When to Use
- When you want deeper, more thoughtful critique
- Complex reasoning tasks requiring self-awareness
- When you have external feedback signals to incorporate
- Tasks where understanding the "why" matters

### Example
```python
from sifaka.critics.reflexion import ReflexionCritic

critic = ReflexionCritic(
    model=model,
    reflection_prompt_template="Reflect on why this critique was effective or ineffective",
    improve_prompt_template="Based on reflection, improve this text"
)
```

---

## 4. ConstitutionalCritic ⚖️
**Complexity: ⭐⭐⭐ (Intermediate-Advanced)**

Evaluates text against a set of constitutional principles to ensure compliance and ethical alignment.

### Visual Flow
```
Input Text ──→ [Check Against Principles] ──→ [Identify Violations] ──→ [Generate Feedback] ──→ Constitutional Critique
     │                    │                         │                        │
     │                    │                         │                        │
     └─── Principle 1 ────┼─── Principle 2 ────────┼─── Principle N ────────┘
          Principle 2      │    Principle 3         │
          Principle 3      │    ...                 │
          ...              │                        │
```

### Key Concept
Like a constitutional court that evaluates decisions against established principles. Systematically checks text against each constitutional principle and provides specific feedback on violations and improvements needed.

### When to Use
- Ensuring ethical AI outputs
- Compliance with organizational guidelines
- Safety-critical applications
- Content moderation and policy enforcement

### Example
```python
from sifaka.critics.constitutional import ConstitutionalCritic

principles = [
    "Be helpful and informative",
    "Avoid harmful or offensive content",
    "Provide accurate information",
    "Respect user privacy"
]

critic = ConstitutionalCritic(model=model, principles=principles)
```

---

## 5. SelfRAGCritic 🔍
**Complexity: ⭐⭐⭐⭐ (Advanced)**

Combines retrieval-augmented generation with self-reflection tokens to provide structured quality assessment.

### Visual Flow
```
Input Text ──→ [Retrieval Decision] ──→ [Retrieve Docs] ──→ [Self-RAG Critique] ──→ [Utility Assessment] ──→ Improved Text
     │              │                      │                    │                      │
     │              │                      │                    │                      │
     └─── Analyze ──┴─── If Needed ───────┴─── [Retrieve] ─────┴─── [Relevant] ──────┘
        Task Type        Get Context         [Relevant]         [Utility]
                                            [Support]
```

### Key Concept
Like a researcher who systematically decides when to look up information and then provides structured assessment using reflection tokens ([Retrieve], [Relevant], [Support], [Utility]) to evaluate text quality and factual support.

### When to Use
- Fact-checking and accuracy verification
- Domain-specific knowledge requirements
- When external context matters for evaluation
- Research and analysis tasks requiring structured assessment

### Example
```python
from sifaka.critics.self_rag import SelfRAGCritic

critic = SelfRAGCritic(
    model=model,
    retriever=your_retriever,  # External knowledge source
    use_reflection_tokens=True,
    max_retrieved_docs=5
)
```

---

## 6. SelfConsistencyCritic 🎯
**Complexity: ⭐⭐⭐⭐ (Advanced)**

Generates multiple independent critiques and uses consensus to determine the most reliable feedback.

### Visual Flow
```
Input Text ──→ [Critique #1] ──→ [Critique #2] ──→ [Critique #3] ──→ [Majority Vote] ──→ Consensus Feedback
     │              │              │              │                    │
     │              │              │              │                    │
     └─── Same ─────┼─── Same ─────┼─── Same ─────┼─── Aggregate ─────┘
        Model        │    Model     │    Model     │   Results
                     │              │              │
                 Different      Different      Different
                 Reasoning      Reasoning      Reasoning
```

### Key Concept
Like a panel of reviewers who independently evaluate the same text and then use majority voting to determine the most reliable consensus. Reduces the impact of inconsistent or low-quality individual critiques through multiple reasoning paths.

### When to Use
- High-stakes decisions requiring reliability
- Reducing critique noise and errors
- When single critiques might be inconsistent
- Quality assurance scenarios requiring consensus

### Example
```python
from sifaka.critics.self_consistency import SelfConsistencyCritic

critic = SelfConsistencyCritic(
    model=model,
    num_iterations=5,  # Generate 5 different critiques
    consensus_threshold=0.6,  # 60% agreement required
    aggregation_method="majority_vote"
)
```

---

## 7. MetaRewardingCritic 🏆
**Complexity: ⭐⭐⭐⭐⭐ (Expert)**

Two-stage judgment process: first critique, then meta-evaluate the quality of that critique to provide the most reliable assessment.

### Visual Flow
```
Input Text ──→ [Initial Judgment] ──→ [Meta-Judgment] ──→ [Combine Assessments] ──→ Final Critique
     │              │                      │                      │
     │              │                      │                      │
     └─── Stage 1 ──┴─── Stage 2 ─────────┴─── Integration ──────┘
        (Judge Text)   (Judge the Judge)     (Best Assessment)
```

### Key Concept
Like having a senior reviewer who not only evaluates the text but also evaluates the quality of their own evaluation. The meta-judge asks "How good was my initial judgment?" and uses both perspectives to provide the most reliable final assessment.

### When to Use
- Maximum quality assurance scenarios
- High-stakes content evaluation
- When single-stage critique might miss important issues
- Research and development requiring thorough evaluation

### Example
```python
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.critics.constitutional import ConstitutionalCritic

base_critic = ConstitutionalCritic(model=model, principles=principles)

critic = MetaRewardingCritic(
    model=model,
    base_critic=base_critic,
    meta_judge_model_name="openai:gpt-4",  # Separate model for meta-judgment
    use_scoring=True,
    score_range=(1, 10)
)
```

---

## 8. NCriticsCritic 🎭
**Complexity: ⭐⭐⭐⭐ (Advanced)**

Uses multiple specialized roles with the same model to provide comprehensive, multi-perspective feedback.

### Visual Flow
```
Input Text ──→ [Content Expert] ────┐
           ──→ [Style Editor] ───────┤
           ──→ [Structure Analyst] ──┼──→ [Aggregate Scores] ──→ [Combined Feedback]
           ──→ [Audience Specialist] ─┤      & Feedback
           ──→ [Quality Assurance] ──┘
                    │
              Same Model, Different Roles
```

### Key Concept
Like having one expert reviewer wear different "hats" - evaluating the same text from multiple specialized perspectives (content, style, structure, audience, quality) and then combining all the feedback.

### When to Use
- Comprehensive evaluation from multiple perspectives
- When you want thorough coverage but only have one model
- Structured, role-based feedback
- Quality assurance with detailed breakdowns

### Example
```python
from sifaka.critics.n_critics import NCriticsCritic

critic = NCriticsCritic(
    model=model,
    num_critics=5,  # Number of specialized roles
    critic_roles=[
        "Content Expert: Focus on factual accuracy and completeness",
        "Style Editor: Focus on writing style, tone, and clarity",
        "Structure Analyst: Focus on organization and logical flow",
        "Audience Specialist: Focus on target audience appropriateness",
        "Quality Assurance: Focus on overall quality and consistency"
    ]
)
```

---

## Choosing the Right Critic

### Decision Tree
```
Start Here
    │
    ├─ Simple, specific criteria? ──→ PromptCritic
    │
    ├─ Need iterative improvement? ──→ SelfRefineCritic
    │
    ├─ Learning from experience? ──→ ReflexionCritic
    │
    ├─ Ethical/guideline compliance? ──→ ConstitutionalCritic
    │
    ├─ Need external knowledge? ──→ SelfRAGCritic
    │
    ├─ Need reliable consensus? ──→ SelfConsistencyCritic
    │
    ├─ Need meta-evaluation? ──→ MetaRewardingCritic
    │
    └─ Need multi-role analysis? ──→ NCriticsCritic
```

### Complexity vs. Comprehensiveness
```
Comprehensiveness ↑
                  │
                  │    MetaRewarding ●
                  │
                  │  SelfConsistency ●    NCritics ●
                  │
                  │    SelfRAG ●
                  │
                  │  Constitutional ●  Reflexion ●
                  │
                  │    SelfRefine ●
                  │
                  │  PromptCritic ●
                  │
                  └────────────────────────────→ Complexity

Note: Higher comprehensiveness means broader coverage of potential issues,
not necessarily better results. Choose based on your specific needs.
```

---

## Best Practices

### 🚀 **Getting Started**
1. Start with `PromptCritic` to understand the basics
2. Move to `SelfRefineCritic` for general improvement
3. Add specialized critics as needed

### ⚡ **Performance**
- Use simpler critics for real-time applications
- Reserve complex critics for batch processing
- Consider caching for repeated critiques

### 🎯 **Effectiveness**
- Combine multiple critics for comprehensive coverage
- Use domain-specific prompts in `PromptCritic`
- Provide external feedback to `ReflexionCritic`

### 🔧 **Integration**
```python
# Chain multiple critics
chain = Chain(model=model, prompt="Your prompt")
chain.improve_with(PromptCritic(model=model, ...))
chain.improve_with(ConstitutionalCritic(model=model, ...))

# Or use multi-role approach
multi_role_critic = NCriticsCritic(
    model=model,
    num_critics=5,
    critic_roles=[
        "Content Expert: Focus on factual accuracy",
        "Style Editor: Focus on writing clarity",
        "Structure Analyst: Focus on organization",
        "Audience Specialist: Focus on target audience",
        "Quality Assurance: Focus on overall quality"
    ]
)
chain.improve_with(multi_role_critic)
```

---

## Research Papers & Implementation Notes

Each critic is based on research papers with some adaptations for practical use:

- **PromptCritic**: Custom implementation for domain-specific critique criteria
- **SelfRefineCritic**: Based on "Self-Refine: Iterative Refinement with Self-Feedback"
- **ReflexionCritic**: Simplified from full Actor/Evaluator/Self-Reflection architecture
- **ConstitutionalCritic**: Adapts core principle-based evaluation from "Constitutional AI: Harmlessness from AI Feedback" for text critique (original focuses on training methodology)
- **SelfRAGCritic**: Combines retrieval with self-reflection tokens from "Self-RAG"
- **SelfConsistencyCritic**: Adapted from reasoning to text critique domain
- **NCriticsCritic**: Based on "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics"
- **MetaRewardingCritic**: Adapts core two-stage judgment from "Meta-Rewarding Language Models" for text critique (original focuses on training methodology)

For full research context, see the [API Reference](api/api-reference.md#critics).

---

*This guide provides visual explanations to help you understand and choose the right critic for your needs. Each critic serves different purposes and complexity levels - start simple and add sophistication as needed.*
