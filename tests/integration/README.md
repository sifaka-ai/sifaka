# Integration Tests

This directory contains integration tests that interact with real LLM APIs. These tests verify end-to-end functionality but are not run by default in CI.

## Running Integration Tests

### Prerequisites

1. Set up API keys for the providers you want to test:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_API_KEY="your-google-key"
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev,all]"
   ```

### Running Tests

Run all integration tests:
```bash
pytest tests/integration/ --integration
```

Run with a specific provider:
```bash
pytest tests/integration/ --integration --llm-provider=anthropic
```

Run with custom timeout:
```bash
INTEGRATION_TEST_TIMEOUT=60 pytest tests/integration/ --integration
```

Run specific test file:
```bash
pytest tests/integration/test_critics_integration.py --integration
```

### Test Categories

- **test_critics_integration.py**: Tests all critic implementations with real LLM calls
- **test_memory_integration.py**: Tests memory-bounded operations and cleanup
- **test_validators_integration.py**: Tests validation logic with real improvements

### Writing New Integration Tests

1. Mark tests with `@pytest.mark.integration`
2. Use the provided fixtures:
   - `api_key`: Gets API key for the selected provider
   - `llm_provider`: Gets the provider name
   - `integration_timeout`: Gets timeout value

Example:
```python
@pytest.mark.integration
def test_my_integration(api_key, llm_provider, integration_timeout):
    result = improve(
        "Test text",
        llm_provider=llm_provider,
        api_key=api_key,
        timeout=integration_timeout,
    )
    assert result.final_text != "Test text"
```

### CI Configuration

Integration tests are automatically run in CI with mock responses to avoid API costs and ensure reliability.

#### Using Mocks in CI

When running in CI or with mock responses:
```bash
# Automatically enabled in CI
CI=true pytest tests/integration/ --integration

# Or manually enable mocks
USE_MOCK_LLM=true pytest tests/integration/ --integration
```

Mock responses provide:
- Realistic LLM behavior for all critics
- Consistent results for reliable testing
- No API costs or rate limits
- Fast execution times

#### Running with Real APIs

To run with real API calls (not recommended for CI):
```bash
# Disable mocks and provide real API keys
CI=false USE_MOCK_LLM=false pytest tests/integration/ --integration
```

### Writing Tests with Mock Support

Integration tests automatically use mocks when `CI=true` or `USE_MOCK_LLM=true`. The mock system:

1. Provides realistic responses for each critic type
2. Simulates multiple iteration improvements
3. Handles token counting and latency
4. Supports all three providers (OpenAI, Anthropic, Google)

Example test that works with both mocks and real APIs:
```python
@pytest.mark.integration
def test_critic_with_mock_support(api_key, llm_provider, use_mocks):
    # Test automatically uses mocks in CI
    result = improve(
        "Test text",
        llm_provider=llm_provider,
        api_key=api_key,
        critics=["reflexion"],
    )

    # Assertions work for both mock and real responses
    assert result.final_text != "Test text"
    assert result.iteration >= 1
```

### Cost Considerations

Integration tests make real API calls which incur costs. To minimize expenses:

- CI uses mock responses by default (no cost)
- Use small texts and few iterations for real API tests
- Set reasonable timeouts
- Run full integration suite with real APIs only for releases
- Monitor API usage and costs regularly
