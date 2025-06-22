# Sifaka API Reference

## Core Function

### `improve(text, **kwargs) -> SifakaResult`

The main function for improving text through iterative critique.

**Parameters:**

- `text` (str): The text to improve
- `max_iterations` (int, default=3): Maximum number of improvement iterations
- `model` (str, default="gpt-4o-mini"): The language model to use
- `critics` (List[str], optional): List of critics to use. Available options:
  - `"reflexion"` - Self-reflection and learning from mistakes
  - `"constitutional"` - Principle-based ethical evaluation
  - `"self_refine"` - Iterative self-improvement
  - `"n_critics"` - Multi-perspective ensemble critique
  - `"self_rag"` - Retrieval-augmented critique
  - `"meta_rewarding"` - Meta-evaluation of judgments
  - `"self_consistency"` - Consensus-based evaluation
  - `"prompt"` - Custom prompt-based critique
- `validators` (List[Validator], optional): Validation constraints
- `temperature` (float, default=0.7): Model temperature for generation
- `timeout_seconds` (int, default=300): Timeout in seconds
- `storage` (StorageBackend, optional): Storage backend for results

**Returns:**
- `SifakaResult`: Complete result with audit trail

**Example:**
```python
result = await improve(
    "Write about renewable energy",
    critics=["reflexion", "constitutional"],
    max_iterations=3
)
```

## Critics

### ReflexionCritic
Implements self-reflection and learning from mistakes.

**Research:** [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

**Usage:**
```python
await improve(text, critics=["reflexion"])
```

**Features:**
- Context-aware reflection
- Dynamic confidence calculation
- Learning from previous iterations

### ConstitutionalCritic
Principle-based evaluation following Constitutional AI approach.

**Research:** [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

**Usage:**
```python
await improve(text, critics=["constitutional"])
```

**Features:**
- Structured principle evaluation
- Violation detection and scoring
- Ethical compliance assessment

### SelfRefineCritic
Iterative self-improvement through self-feedback.

**Research:** [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

**Usage:**
```python
await improve(text, critics=["self_refine"])
```

**Features:**
- Quality-based assessment
- Iterative refinement suggestions
- Context from previous iterations

### NCriticsCritic
Multi-perspective ensemble critique.

**Research:** [N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics](https://arxiv.org/abs/2310.18679)

**Usage:**
```python
await improve(text, critics=["n_critics"])
```

**Features:**
- Multiple critical perspectives
- Consensus-based scoring
- Configurable perspectives

**Custom Perspectives:**
```python
from sifaka.critics.n_critics import NCriticsCritic

custom_critic = NCriticsCritic(perspectives=[
    "Technical accuracy: Focus on correctness",
    "Clarity: Focus on readability",
    "Completeness: Focus on thoroughness"
])
```

### SelfRAGCritic
Retrieval-augmented critique for factual accuracy.

**Research:** [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

**Usage:**
```python
await improve(text, critics=["self_rag"])
```

**Features:**
- Retrieval need assessment
- Factual accuracy evaluation
- Evidence quality analysis

### MetaRewardingCritic
Two-stage judgment with meta-evaluation.

**Research:** [Meta-Rewarding: Learning to Judge Judges with Self-Generated Meta-Judgments](https://arxiv.org/abs/2407.19594)

**Usage:**
```python
await improve(text, critics=["meta_rewarding"])
```

**Features:**
- Initial judgment stage
- Meta-judgment of evaluation quality
- Reliability scoring

### SelfConsistencyCritic
Consensus-based evaluation through multiple independent assessments.

**Research:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

**Usage:**
```python
await improve(text, critics=["self_consistency"])
```

**Features:**
- Multiple independent evaluations
- Consistency analysis
- Majority consensus building

### PromptCritic
Custom prompt-based critique system.

**Usage:**
```python
await improve(text, critics=["prompt"])
```

**Features:**
- Customizable evaluation criteria
- Flexible prompt templates
- Domain-specific assessment

## Validators

### LengthValidator
Enforces text length constraints.

**Usage:**
```python
from sifaka.validators import LengthValidator

validator = LengthValidator(min_length=100, max_length=1000)
await improve(text, validators=[validator])
```

**Parameters:**
- `min_length` (int): Minimum required length
- `max_length` (int): Maximum allowed length

### ContentValidator
Validates presence of required terms and concepts.

**Usage:**
```python
from sifaka.validators import ContentValidator

validator = ContentValidator(
    required_terms=["methodology", "results", "conclusion"],
    forbidden_terms=["maybe", "perhaps"]
)
await improve(text, validators=[validator])
```

**Parameters:**
- `required_terms` (List[str]): Terms that must be present
- `forbidden_terms` (List[str]): Terms that must not be present

## Storage Backends

### MemoryStorage
In-memory storage (default, non-persistent).

**Usage:**
```python
from sifaka.storage import MemoryStorage

storage = MemoryStorage()
result = await improve(text, storage=storage)
```

### FileStorage
File-based persistent storage.

**Usage:**
```python
from sifaka.storage import FileStorage

storage = FileStorage(storage_dir="./sifaka_results")
result = await improve(text, storage=storage)

# Load result later
loaded = await storage.load(result.id)
```

**Parameters:**
- `storage_dir` (str): Directory to store results
- `max_files` (int, default=1000): Maximum files to keep

## Models

### SifakaResult
Complete result with audit trail.

**Attributes:**
- `final_text` (str): The final improved text
- `original_text` (str): The original input text
- `iteration` (int): Number of iterations completed
- `generations` (List[Generation]): All text generations
- `critiques` (List[CritiqueResult]): All critique results
- `validations` (List[ValidationResult]): All validation results
- `id` (str): Unique result identifier
- `created_at` (datetime): Creation timestamp
- `confidence` (float): Overall confidence score

**Properties:**
- `current_text` (str): Most recent generation or original
- `all_passed` (bool): Whether all validations passed
- `needs_improvement` (bool): Whether critics suggest improvement

### CritiqueResult
Individual critic feedback.

**Attributes:**
- `critic` (str): Name of the critic
- `feedback` (str): Detailed feedback text
- `suggestions` (List[str]): Specific improvement suggestions
- `needs_improvement` (bool): Whether improvement is needed
- `confidence` (float): Confidence in the assessment
- `timestamp` (datetime): When critique was created

### ValidationResult
Individual validation result.

**Attributes:**
- `validator` (str): Name of the validator
- `passed` (bool): Whether validation passed
- `score` (float): Validation score (0.0-1.0)
- `details` (str): Detailed validation information
- `timestamp` (datetime): When validation was performed

## Configuration

### Model Configuration
Configure the underlying language model.

**Supported Models:**
- OpenAI: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- Gemini: `gemini-1.5-pro`, `gemini-1.5-flash`

**Example:**
```python
result = await improve(
    text,
    model="gpt-4o",
    temperature=0.3  # More deterministic
)
```

### Timeout Management
Control execution time with timeouts.

**Example:**
```python
result = await improve(
    text,
    timeout_seconds=60,  # 1 minute timeout
    critics=["reflexion"]
)
```

## Error Handling

### SifakaError
Base exception for all Sifaka errors.

### ConfigurationError
Raised for invalid configuration parameters.

### ModelProviderError
Raised for model provider API issues.

### TimeoutError
Raised when operation times out.

### ValidationError
Raised when validation fails.

### CriticError
Raised when critics fail to evaluate.

**Example Error Handling:**
```python
from sifaka.core.exceptions import TimeoutError

try:
    result = await improve(text)
except TimeoutError:
    print("Operation timed out")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage

### Custom Critics
Create custom critics by implementing the Critic interface.

```python
from sifaka.core.interfaces import Critic
from sifaka.core.models import CritiqueResult

class CustomCritic(Critic):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Your custom critique logic
        return CritiqueResult(
            critic="custom",
            feedback="Custom feedback",
            suggestions=["Custom suggestion"],
            needs_improvement=True,
            confidence=0.8
        )
    
    @property
    def name(self) -> str:
        return "custom"
```

### Plugin System
Register custom storage backends.

```python
from sifaka.plugins import register_storage_backend
from sifaka.storage.base import StorageBackend

class CustomStorage(StorageBackend):
    # Implement abstract methods
    pass

register_storage_backend("custom", CustomStorage)

# Use it
from sifaka.plugins import create_storage_backend
storage = create_storage_backend("custom")
```

### Batch Processing
Process multiple texts efficiently.

```python
texts = ["Text 1", "Text 2", "Text 3"]
results = []

for text in texts:
    result = await improve(
        text,
        critics=["reflexion"],
        max_iterations=2
    )
    results.append(result)

# Analyze batch results
avg_confidence = sum(r.confidence for r in results) / len(results)
```

## Best Practices

1. **Choose Critics Wisely**: Different critics serve different purposes
   - Use `reflexion` for general improvement
   - Use `constitutional` for ethical content
   - Use `n_critics` for comprehensive analysis
   - Use `self_rag` for factual content

2. **Set Appropriate Limits**: Balance quality and iterations
   - Use `max_iterations=3` for most cases
   - Increase iterations for important content

3. **Use Validators**: Enforce hard constraints
   - Always validate length requirements
   - Check for required terminology
   - Prevent forbidden content

4. **Monitor Performance**: Track quality
   - Monitor confidence scores
   - Analyze improvement patterns

5. **Handle Errors Gracefully**: Plan for failures
   - Handle timeout scenarios
   - Provide fallback strategies

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models  
- `GOOGLE_API_KEY`: Google API key for Gemini models

## Rate Limits

Sifaka respects provider rate limits and includes built-in retry logic. For high-volume usage:

1. Use cheaper models like `gpt-4o-mini`
2. Implement delays between requests
3. Consider batching strategies
4. Monitor API quotas

## Troubleshooting

### Common Issues

1. **Performance Issues**: Use lower iterations
2. **Slow Performance**: Reduce max_iterations, use faster models
3. **Low Quality**: Use better critics, increase iterations
4. **API Errors**: Check API keys and quotas

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logs for debugging
result = await improve(text, critics=["reflexion"])
```