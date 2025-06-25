# Sifaka Critic Selection Guide

This guide helps you choose the right critics for your text improvement needs.

## Quick Selection Matrix

| Task Type | Primary Critics | Secondary Critics | Why |
|-----------|----------------|-------------------|-----|
| **Academic Writing** | `self_consistency`, `constitutional` | `self_rag`, `meta_rewarding` | Ensures rigor, accuracy, and principle adherence |
| **Blog Posts** | `self_refine`, `n_critics` | `reflexion` | Focuses on engagement and multiple reader perspectives |
| **Technical Documentation** | `self_rag`, `constitutional` | `self_consistency` | Prioritizes accuracy and clarity |
| **Marketing Copy** | `n_critics`, `self_refine` | `constitutional` | Multiple perspectives and polished output |
| **Code Comments** | `constitutional`, `self_refine` | `reflexion` | Clear, helpful, and constructive |
| **Customer Communication** | `constitutional`, `n_critics` | `self_refine` | Respectful tone with stakeholder perspectives |
| **Creative Writing** | `reflexion`, `self_refine` | `meta_rewarding` | Iterative improvement with quality focus |
| **Fact-Heavy Content** | `self_rag`, `self_consistency` | `constitutional` | Verification and consensus on facts |
| **Legal/Compliance** | `constitutional`, `self_consistency` | `meta_rewarding` | Principle adherence with verification |
| **Educational Content** | `n_critics`, `self_refine` | `reflexion` | Multiple learning perspectives |

## Detailed Critic Profiles

### 1. **ReflexionCritic** 🔄
**Best for:** Iterative improvement tasks where learning from previous attempts matters
- ✅ **Use when:** Content needs progressive refinement
- ✅ **Use when:** You have multiple iterations available
- ✅ **Use when:** Context from earlier attempts is valuable
- ❌ **Avoid when:** You need single-shot improvement
- ❌ **Avoid when:** Historical context isn't relevant
- **Speed:** Medium | **Cost:** Medium | **Iterations:** 3-5 optimal

### 2. **ConstitutionalCritic** 📋
**Best for:** Ensuring content adheres to principles and guidelines
- ✅ **Use when:** Safety and accuracy are paramount
- ✅ **Use when:** You have specific principles to follow
- ✅ **Use when:** Ethical considerations matter
- ❌ **Avoid when:** Creative freedom is priority
- ❌ **Avoid when:** Principles might constrain innovation
- **Speed:** Fast | **Cost:** Low | **Iterations:** 1-2 usually sufficient

### 3. **SelfRefineCritic** ✨
**Best for:** General quality improvement across multiple dimensions
- ✅ **Use when:** You want well-rounded improvements
- ✅ **Use when:** No specific constraints exist
- ✅ **Use when:** Polish and flow matter
- ❌ **Avoid when:** You need specialized expertise
- ❌ **Avoid when:** Fact-checking is critical
- **Speed:** Fast | **Cost:** Low | **Iterations:** 2-3 optimal

### 4. **NCriticsCritic** 👥
**Best for:** Getting diverse perspectives on content
- ✅ **Use when:** Multiple stakeholder views matter
- ✅ **Use when:** You're unsure what to improve
- ✅ **Use when:** Comprehensive feedback needed
- ❌ **Avoid when:** You need focused, specific feedback
- ❌ **Avoid when:** Speed is critical
- **Speed:** Slower | **Cost:** Higher | **Iterations:** 1-2 comprehensive

### 5. **SelfRAGCritic** 🔍
**Best for:** Content requiring factual accuracy and verification
- ✅ **Use when:** Facts and claims need verification
- ✅ **Use when:** Citations would improve credibility
- ✅ **Use when:** Technical accuracy matters
- ❌ **Avoid when:** Content is opinion-based
- ❌ **Avoid when:** Creative expression is key
- **Speed:** Medium | **Cost:** Medium (Higher with retrieval) | **Iterations:** 1-2

### 6. **MetaRewardingCritic** 🎯
**Best for:** High-stakes content needing quality assurance
- ✅ **Use when:** Critique quality itself matters
- ✅ **Use when:** You need confidence in feedback
- ✅ **Use when:** Self-evaluation is valuable
- ❌ **Avoid when:** Quick feedback needed
- ❌ **Avoid when:** Simple improvements suffice
- **Speed:** Slower | **Cost:** Higher | **Iterations:** 1-2 deep

### 7. **SelfConsistencyCritic** 🎲
**Best for:** Achieving consensus through multiple evaluations
- ✅ **Use when:** Consistency matters more than speed
- ✅ **Use when:** You want robust, reliable feedback
- ✅ **Use when:** Variance in critique is concerning
- ❌ **Avoid when:** Deterministic feedback needed
- ❌ **Avoid when:** Budget is tight
- **Speed:** Slowest | **Cost:** Highest (3x normal) | **Iterations:** 1 (but 3 internal)

### 8. **PromptCritic** 🛠️
**Best for:** Custom evaluation criteria and experiments
- ✅ **Use when:** You have specific requirements
- ✅ **Use when:** Existing critics don't fit
- ✅ **Use when:** Experimenting with new approaches
- ❌ **Avoid when:** Standard critics suffice
- ❌ **Avoid when:** You want proven approaches
- **Speed:** Variable | **Cost:** Low | **Iterations:** Variable

## Effective Critic Combinations

### For Maximum Quality
```python
critics = ["self_consistency", "constitutional", "meta_rewarding"]
```
**Why:** Consensus + principles + quality verification

### For Fast Improvement
```python
critics = ["self_refine", "constitutional"]
```
**Why:** Quick general improvements with safety checks

### For Technical Content
```python
critics = ["self_rag", "self_consistency", "n_critics"]
```
**Why:** Fact-checking + consensus + expert perspectives

### For Creative Content
```python
critics = ["reflexion", "self_refine", "prompt:creative_writing"]
```
**Why:** Iterative learning + polish + custom creative criteria

### For User-Facing Content
```python
critics = ["constitutional", "n_critics", "self_refine"]
```
**Why:** Safe + multiple perspectives + polished

## Performance Considerations

### Speed Ranking (Fastest to Slowest)
1. **constitutional** - Single evaluation, principle-based
2. **self_refine** - Single evaluation, structured
3. **prompt** - Single evaluation, custom
4. **reflexion** - Depends on history depth
5. **self_rag** - Fact identification overhead
6. **meta_rewarding** - Two-stage process
7. **n_critics** - Multiple perspectives
8. **self_consistency** - Multiple evaluations (3x)

### Cost Efficiency
- **Most Efficient:** `constitutional`, `self_refine`
- **Moderate:** `reflexion`, `self_rag`, `prompt`
- **Higher Cost:** `meta_rewarding`, `n_critics`
- **Highest Cost:** `self_consistency` (3x tokens)

### Iteration Patterns
- **Single iteration often enough:** `constitutional`, `self_consistency`
- **2-3 iterations optimal:** `self_refine`, `self_rag`
- **3-5 iterations beneficial:** `reflexion`
- **1-2 deep iterations:** `meta_rewarding`, `n_critics`

## Decision Tree

```
Start: What's your primary goal?
│
├─> Accuracy/Facts Critical?
│   ├─> Yes: Use self_rag
│   │   └─> Add self_consistency for consensus
│   └─> No: Continue
│
├─> Multiple Stakeholders?
│   ├─> Yes: Use n_critics
│   │   └─> Add constitutional for safety
│   └─> No: Continue
│
├─> Iterative Improvement Needed?
│   ├─> Yes: Use reflexion
│   │   └─> Add self_refine for polish
│   └─> No: Continue
│
├─> Specific Principles/Rules?
│   ├─> Yes: Use constitutional
│   │   └─> Add meta_rewarding for verification
│   └─> No: Use self_refine
│
└─> Custom Requirements?
    └─> Yes: Use prompt with custom criteria
```

## Common Patterns by Industry

### Academia/Research
```python
critics = ["self_rag", "self_consistency", "constitutional"]
config = Config(temperature=0.3, max_iterations=3)
```

### Marketing/Sales
```python
critics = ["n_critics", "self_refine", "constitutional"]
config = Config(temperature=0.7, max_iterations=2)
```

### Technical Writing
```python
critics = ["self_rag", "constitutional", "self_refine"]
config = Config(temperature=0.4, max_iterations=2)
```

### Creative Writing
```python
critics = ["reflexion", "self_refine", "meta_rewarding"]
config = Config(temperature=0.8, max_iterations=4)
```

### Customer Service
```python
critics = ["constitutional", "n_critics"]
config = Config(temperature=0.5, max_iterations=2)
```

## Tips for Best Results

1. **Start Simple**: Begin with 1-2 critics and add more if needed
2. **Order Matters**: Place fact-checking critics early, polish critics later
3. **Monitor Costs**: Self-consistency is 3x more expensive
4. **Use Validators**: Combine critics with validators for constraints
5. **Configure Temperature**: Lower for formal, higher for creative
6. **Set Iteration Limits**: Most improvements happen in first 2-3 iterations

## Future Considerations

As you use Sifaka, consider:
- Tracking which critics work best for your use cases
- Creating custom prompt critics for repeated patterns
- Building retrieval backends for SelfRAG if accuracy is critical
- Monitoring performance metrics to optimize selection

Remember: The best critic combination depends on your specific context, constraints, and quality requirements. Start with the recommendations above and adjust based on results.
