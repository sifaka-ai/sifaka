# Feedback Summarizer

> **🚧 Coming Soon**: Enhanced feedback summarization using both local and API-based models is currently in development and will be available in an upcoming release. The current implementation is being improved for better performance and reliability.

The `FeedbackSummarizer` is a customizable utility that can summarize validation results and critic feedback using various local or API-based models. It supports multiple summarization models including T5, BART, Pegasus, and others, with T5 as the default.

## Features

- **Multiple model support**: T5, BART, Pegasus, and custom models
- **API-based summarization**: OpenAI, Anthropic, and other API providers
- **Configurable parameters**: Custom prompts, length limits, and model settings
- **Fallback mechanisms**: Graceful degradation when models are unavailable
- **Caching**: Improved performance with summary caching
- **Easy integration**: Works seamlessly with existing critics

## Quick Start

### Basic Usage

```python
from sifaka.critics import FeedbackSummarizer

# Create summarizer with T5 (default)
summarizer = FeedbackSummarizer()

# Summarize feedback from a thought
summary = summarizer.summarize_thought_feedback(thought)
print(f"Summary: {summary}")
```

### Using Different Models

```python
# T5-based summarization
t5_summarizer = FeedbackSummarizer(
    model_name="t5-small",
    model_type="t5",
    max_length=100
)

# BART-based summarization
bart_summarizer = FeedbackSummarizer(
    model_name="facebook/bart-base",
    model_type="bart",
    max_length=120,
    custom_prompt="Create a concise summary:"
)

# API-based summarization
api_summarizer = FeedbackSummarizer(
    model_type="api",
    api_model="openai:gpt-3.5-turbo",
    max_length=80
)
```

### Integration with Critics

```python
from sifaka.critics import SelfRefineCritic, FeedbackSummarizer

class SummarizingSelfRefineCritic(SelfRefineCritic):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.feedback_summarizer = FeedbackSummarizer(
            model_name="t5-small",
            max_length=120
        )

    def improve(self, thought):
        # Get summarized feedback
        summary = self.feedback_summarizer.summarize_thought_feedback(thought)

        # Use summary in improvement prompt
        prompt = f"Improve based on: {summary}"
        return self.model.generate(prompt)

# Use the enhanced critic
critic = SummarizingSelfRefineCritic(model)
chain = Chain(model=model, prompt="...").improve_with(critic)
```

## Configuration Options

### Model Types

- **`t5`**: T5-based models (e.g., "t5-small", "t5-base")
- **`bart`**: BART-based models (e.g., "facebook/bart-base")
- **`pegasus`**: Pegasus models for summarization
- **`auto`**: Automatic model type detection
- **`api`**: API-based models (OpenAI, Anthropic, etc.)

### Parameters

```python
FeedbackSummarizer(
    model_name="t5-small",           # HuggingFace model name
    model_type="auto",               # Model type
    api_model=None,                  # API model for "api" type
    max_length=150,                  # Maximum summary length
    min_length=30,                   # Minimum summary length
    custom_prompt=None,              # Custom summarization prompt
    cache_summaries=True,            # Enable caching
    fallback_to_truncation=True,     # Fallback when models fail
    **model_kwargs                   # Additional model arguments
)
```

## Advanced Usage

### Custom Prompts

```python
summarizer = FeedbackSummarizer(
    custom_prompt="Provide a brief, actionable summary of this feedback:"
)
```

### Selective Summarization

```python
# Summarize only validation results
validation_summary = summarizer.summarize_validation_results(
    thought.validation_results
)

# Summarize only critic feedback
critic_summary = summarizer.summarize_critic_feedback(
    thought.critic_feedback
)

# Summarize both with custom options
full_summary = summarizer.summarize_thought_feedback(
    thought,
    include_validation=True,
    include_critic_feedback=True,
    custom_prompt="Key points:"
)
```

### Performance Optimization

```python
# Enable caching for repeated inputs
summarizer = FeedbackSummarizer(cache_summaries=True)

# Clear cache when needed
summarizer.clear_cache()

# Disable fallback for strict requirements
summarizer = FeedbackSummarizer(fallback_to_truncation=False)
```

## Model Recommendations

### Local Models

- **T5-small**: Fast, good quality, low memory usage
- **T5-base**: Better quality, higher memory usage
- **BART-base**: Good for general summarization
- **Pegasus**: Specialized for summarization tasks

### API Models

- **OpenAI GPT-3.5-turbo**: Fast, cost-effective
- **OpenAI GPT-4**: Higher quality, more expensive
- **Anthropic Claude**: Good balance of speed and quality

## Examples

See the following examples for complete demonstrations:

- `examples/critics/feedback_summarizer_demo.py` - Comprehensive demo with all model types
- `examples/anthropic/self_refine_with_summarizer.py` - Integration with Self-Refine critic

## Dependencies

For local models:
```bash
pip install transformers torch
```

For API models, ensure you have the appropriate API keys:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models

## Error Handling

The FeedbackSummarizer includes robust error handling:

1. **Model loading failures**: Falls back to truncation or returns error message
2. **API failures**: Graceful degradation to local models or fallback
3. **Invalid input**: Returns appropriate default messages
4. **Memory issues**: Automatic input truncation for large texts

## Best Practices

1. **Choose appropriate models**: Use smaller models for speed, larger for quality
2. **Set reasonable limits**: Configure `max_length` based on your use case
3. **Use caching**: Enable caching for repeated summarization tasks
4. **Test fallbacks**: Ensure fallback mechanisms work in your environment
5. **Monitor performance**: Track summarization time and quality
