# Performance Monitoring

Sifaka includes comprehensive performance monitoring and observability features to help you understand and optimize your text improvement operations.

## Logfire Integration

Sifaka integrates with [Logfire](https://logfire.pydantic.dev/) for production monitoring and distributed tracing.

### Setup

1. Get your Logfire token from https://logfire.pydantic.dev/
2. Set the environment variable:
   ```bash
   export LOGFIRE_TOKEN='your-token-here'
   ```
   Or add to your `.env` file:
   ```
   LOGFIRE_TOKEN=your-token-here
   ```

### What's Tracked

When Logfire is enabled, Sifaka automatically tracks:

- **Overall Performance**
  - Total duration and iterations
  - Token usage and generation rate
  - Confidence score progression
  - Success/failure status

- **LLM Calls**
  - Individual call timing
  - Token usage per call
  - Model and provider information
  - Call type (critic vs generation)

- **Critic Evaluations**
  - Which critics ran
  - Confidence scores
  - Improvement suggestions
  - Processing time per critic

- **Nested Spans**
  - `sifaka_improve` - Top level operation
  - `critic_call` - Individual critic evaluations
  - `llm_call` - LLM API calls
  - `text_generation_llm_call` - Text generation calls

### Viewing Metrics

After running operations with Logfire enabled, you'll see output like:
```
Logfire project URL: https://logfire-us.pydantic.dev/your-org/your-project
```

Visit this URL to see detailed traces, performance metrics, and insights.

## Built-in Monitoring

Even without Logfire, Sifaka tracks performance metrics locally:

```python
from sifaka import improve_sync

result = improve_sync("Your text here")

# Access performance data
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Iterations: {result.iteration}")
print(f"Total tokens: {sum(g.tokens_used for g in result.generations)}")
```

## Performance Tips

1. **Monitor token usage** - Track costs and optimize prompts
2. **Watch iteration counts** - High iterations may indicate unclear requirements
3. **Check confidence progression** - Ensure critics are converging
4. **Identify slow critics** - Some critics may need optimization
5. **Track error rates** - Monitor and fix recurring issues
