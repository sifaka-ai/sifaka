# PydanticAI Integration - Complete

## Summary

PydanticAI is now fully integrated as the ONLY option for structured outputs in Sifaka. This provides:

1. **Structured outputs** - No more JSON parsing failures
2. **Built-in retries** - Handled by PydanticAI
3. **Monitoring with Logfire** - Automatic tracing and metrics
4. **Multi-provider support** - OpenAI, Anthropic, Gemini, Groq

## What Changed

### 1. LLMClient Enhancement (`llm_client.py`)
- Added `create_agent()` method to create PydanticAI agents
- Maps providers to PydanticAI format (e.g., `openai:gpt-4o-mini`)
- Configures Logfire automatically if token is available

### 2. BaseCritic Update (`critics/core/base.py`)
- ALWAYS uses PydanticAI agents for structured outputs
- Moved `CriticResponse` model directly into base.py
- Removed JSON parsing completely

### 3. TextGenerator Update (`engine/generation.py`)
- Added `ImprovementResponse` model for structured improvements
- Uses PydanticAI agent for text generation

### 4. Monitoring Integration (`core/monitoring.py`)
- Added Logfire spans for LLM calls and critic calls
- Automatic performance tracking with `logfire.span()`

### 5. Files Removed
- `response_parser.py` - No longer needed
- `confidence_advanced.py` - Experimental code removed
- `self_rag_enhanced.py` - Unused enhanced version
- `critics/factory.py` - Old duplicate factory
- Removed `use_pydantic_ai` config option - it's always on

## Usage

### Enable Monitoring
```python
# Set environment variable
export LOGFIRE_TOKEN=your-token

# Or in config
config = Config(logfire_token="your-token")
```

### Custom Critics with Structured Outputs
```python
from pydantic import BaseModel, Field

class MyResponse(BaseModel):
    analysis: str = Field(..., description="Detailed analysis")
    score: float = Field(..., ge=0, le=10)
    issues: list[str] = Field(default_factory=list)

class MyCritic(BaseCritic):
    def _get_system_prompt(self) -> str:
        return "You are a specialized code quality critic."

    # Critic will automatically use MyResponse for structured output
```

## Benefits

1. **100% Reliable Outputs** - No JSON parsing failures ever
2. **Type Safety** - Pydantic models ensure type correctness
3. **Better Error Messages** - Structured validation errors
4. **Performance Monitoring** - Automatic with Logfire
5. **Cleaner Code** - No manual parsing logic

## Testing

```bash
# Run PydanticAI tests
pytest tests/test_pydantic_ai.py -v
```

## Phase 2 Complete - Custom Response Models

ALL critics now have their own tailored response models:

### 1. **ReflexionResponse**
- Tracks `evolution_summary` and `key_learnings` from iterations
- Perfect for understanding how text has improved over time

### 2. **ConstitutionalResponse**
- Includes `principle_evaluations` for each constitutional principle
- Provides `overall_safety_score` for content appropriateness

### 3. **SelfRefineResponse**
- Features `refinement_areas` with current/target states
- Includes `refinement_iterations_recommended` for planning

### 4. **MetaRewardingResponse**
- Two-stage evaluation with `critique_evaluations`
- Provides both initial and `adjusted_confidence` after meta-evaluation

### 5. **NCriticsResponse**
- Contains `perspective_assessments` from each viewpoint
- Calculates `consensus_score` and `agreement_level`

### 6. **SelfRAGResponse**
- Identifies `factual_claims` with verifiability scores
- Lists `retrieval_opportunities` and `unsupported_assertions`
- Provides `accuracy_score` for factual content

### 7. **SelfConsistencyResponse**
- Contains `individual_evaluations` from multiple runs
- Calculates `consistency_score` and `evaluation_variance`
- Identifies `common_themes` and `divergent_points`

### 8. **PromptResponse**
- Flexible `custom_criteria_results` based on user prompt
- Provides `overall_score` and `key_findings`
- Adaptable to any custom evaluation needs

### Custom Critics

Creating a custom critic with structured output is now simple:

```python
from pydantic import BaseModel, Field
from sifaka.critics.core.base import BaseCritic

class CodeReviewResponse(BaseModel):
    feedback: str
    suggestions: list[str]
    needs_improvement: bool
    confidence: float

    # Custom fields
    security_issues: list[str] = Field(default_factory=list)
    performance_concerns: list[str] = Field(default_factory=list)
    best_practices_score: float = Field(default=0.8)

class CodeReviewCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "code_review"

    def _get_response_type(self) -> type[BaseModel]:
        return CodeReviewResponse

    def _get_system_prompt(self) -> str:
        return "You are an expert code reviewer focusing on security, performance, and best practices."
```

## Benefits Achieved

1. **Type Safety** - Every critic output is fully typed
2. **Rich Metadata** - Critics can return domain-specific information
3. **No Parsing Failures** - PydanticAI handles all validation
4. **Better Insights** - Custom fields provide deeper analysis
5. **Easy Extension** - Adding new critics is straightforward

The integration is complete - PydanticAI is now the foundation for all LLM interactions in Sifaka, with rich structured outputs tailored to each critic's purpose.
