# Testing Guide

This guide explains the testing framework and practices used in Sifaka.

## Test Categories

### 1. Edge Cases
Edge case tests ensure the system handles boundary conditions and unexpected inputs correctly. See [Edge Cases](edge_cases.md) for details.

Key areas tested:
- Empty/null inputs
- Very long inputs
- Special characters
- Unicode characters
- Concurrent requests
- Failure recovery
- Memory usage
- Timeouts

### 2. Performance Tests
Performance tests verify the system meets performance requirements. See [Performance Testing](performance.md) for details.

Key metrics tested:
- Single request performance
- Batch request performance
- Large input handling
- Rule validation speed
- Critic processing speed
- Memory usage
- Concurrent request handling
- Component-specific benchmarks

### 3. Security Tests
Security tests verify the system's security measures. See [Security Testing](security.md) for details.

Key areas tested:
- API key handling
- Input sanitization
- Data privacy
- Rate limiting
- Error message security
- File path security
- Configuration validation
- Output sanitization
- Authentication
- Session management

### 4. Integration Tests
Integration tests verify component interactions. See [Integration Testing](integration.md) for details.

Key scenarios tested:
- Model-Critic integration
- Model-Domain integration
- Critic-Domain integration
- Full system workflow
- Concurrent operations
- Error handling
- Performance under load

## Running Tests

### Running All Tests
```bash
python -m pytest
```

### Running Specific Test Categories
```bash
# Run edge case tests
python -m pytest tests/edge_cases/

# Run performance tests
python -m pytest tests/performance/

# Run security tests
python -m pytest tests/security/

# Run integration tests
python -m pytest tests/integration/
```

### Running with Coverage
```bash
python -m pytest --cov=sifaka tests/
```

## Writing Tests

### Test Structure
Each test module should:
1. Import necessary modules
2. Define a test class inheriting from `unittest.TestCase`
3. Implement `setUp` method for test fixtures
4. Include docstrings explaining test purpose
5. Use clear test method names
6. Include assertions to verify behavior

Example:
```python
import unittest
from sifaka.models.mock import MockProvider

class TestExample(unittest.TestCase):
    """Tests for example functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MockProvider(
            config={
                "name": "mock_model",
                "description": "Mock model for testing",
                "params": {"delay": 0.1}
            }
        )

    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        result = self.mock_model.generate("test")
        self.assertIsNotNone(result)
        self.assertIn("text", result)
```

### Best Practices
1. **Isolation**: Each test should be independent
2. **Clarity**: Use descriptive test names and docstrings
3. **Coverage**: Test both success and failure cases
4. **Mocking**: Use mocks for external dependencies
5. **Performance**: Keep tests fast and efficient
6. **Maintenance**: Keep tests up to date with code changes

### Test Templates
See [Test Module Template](../templates/test_module_template.md) for a detailed template.

## Continuous Integration
Tests are run automatically on:
- Pull requests
- Merges to main branch
- Release creation

See [Contributing](../contributing.md) for more details on the development workflow.