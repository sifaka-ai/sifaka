# Sifaka Test Suite

This directory contains the comprehensive test suite for the Sifaka framework. The tests cover various components, functionalities, and integration points of the framework.

## Test Structure

The test suite is organized as follows:

```
tests/
├── conftest.py           # Common fixtures and utilities
├── rules/                # Tests for rule components
│   ├── test_base.py      # Tests for base rule classes
│   ├── test_length.py    # Tests for LengthRule
│   ├── test_format.py    # Tests for FormatRule
│   ├── test_prohibited_content.py # Tests for ProhibitedContentRule
│   └── test_safety.py    # Tests for safety rules
├── test_chain.py         # Tests for Chain class
└── test_integration.py   # Integration tests for multiple components
```

## Running Tests

To run the tests, you can use pytest from the project root:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chain.py

# Run specific test class
pytest tests/rules/test_base.py::TestRuleResult

# Run specific test method
pytest tests/rules/test_base.py::TestRuleResult::test_rule_result_initialization
```

## Test Coverage

To generate test coverage reports, run:

```bash
# Generate coverage report
pytest --cov=sifaka

# Generate HTML coverage report
pytest --cov=sifaka --cov-report=html
```

This will create a `htmlcov` directory with an HTML report that you can open in your browser.

## Adding New Tests

When adding new tests:

1. Keep tests focused on a single unit of functionality
2. Use descriptive test names that reflect what's being tested
3. Follow the existing test structure and naming conventions
4. Make use of fixtures in `conftest.py` for common test scenarios
5. Use mock objects for external dependencies

## Continuous Integration

These tests are automatically run as part of the CI pipeline for the project. All tests must pass before a pull request can be merged.

## Test Dependencies

The test suite requires the following dependencies:

- pytest
- pytest-cov
- pytest-mock

These can be installed via:

```bash
pip install -r requirements-dev.txt
```

## Troubleshooting

### Import Errors

If you encounter import errors when running the tests, it may be due to one of the following issues:

1. **Module Path Issues**: Ensure that the Sifaka package is properly installed or available in your PYTHONPATH. You can install it in development mode with:

   ```bash
   pip install -e .
   ```

2. **Import Structure Mismatch**: If the import structure in the tests doesn't match the actual package structure, you may need to update the imports in the test files. Check the actual import paths in the Sifaka package and adjust the tests accordingly.

3. **Missing Dependencies**: Make sure all required dependencies are installed. This includes both the production dependencies in `requirements.txt` and the development dependencies in `requirements-dev.txt`.

### Test Failures

If tests are failing, check the following:

1. **Mock Configuration**: Ensure that mocks are properly configured for the expected behavior.
2. **API Changes**: The tests might need to be updated if the Sifaka API has changed.
3. **Environment Differences**: Some tests may behave differently in different environments. Check environment variables or configuration that might affect the tests.

## Known Compatibility Issues

There are several compatibility issues between the current test suite and the actual implementation of Sifaka:

1. **Module Restructuring**: Some modules have been moved or renamed:
   - Safety rules are now in `sifaka.rules.content.safety` instead of `sifaka.rules.safety`
   - Formatting rules have been reorganized into the `sifaka.rules.formatting` subpackage

2. **Configuration Parameters**: Many configuration classes have different parameter names than what the tests expect:
   - `ProhibitedContentConfig` doesn't accept `prohibited_terms` as a parameter
   - `ClassifierAdapter` doesn't accept `positive_class` as a parameter
   - Format-related configs don't accept `required_format` parameter

3. **Implementation Differences**: Some implementations differ from what the tests expect:
   - The Chain implementation's error handling doesn't match test expectations
   - The critique mechanism in Chain has changed (dict vs object with __dict__)
   - PromptCriticConfig requires a description parameter that the tests don't provide

### Fix Plan

To make the tests compatible with the current implementation:

1. **Update Import Paths**: Update all import statements in test files to use the correct current paths.

2. **Review API Documentation**: Check the current API documentation to understand the correct parameter names and usage.

3. **Update Test Parameters**: Update the tests to use the correct parameter names for configuration classes.

4. **Update Mock Behaviors**: Ensure mocks are configured to match the actual behavior of the current implementation.

5. **Fix Expected Values**: Update assertions to match the expected values from the current implementation.

6. **Check Deprecated Features**: Be aware of deprecated features and consider testing against both legacy and new APIs where necessary.

This may require a significant refactoring of the test suite, but it will ensure proper test coverage for the current version of Sifaka.