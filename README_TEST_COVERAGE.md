# Test Coverage Documentation

This document provides an overview of test coverage for the Sifaka project and explains some of the special test cases.

## Recent Test Coverage Improvements

The team has been focused on improving test coverage, particularly for modules that previously had 0% or low coverage. Here are the key improvements:

| Module | Before | After | Notes |
|--------|--------|-------|-------|
| utils/validation.py | 20% | 100% | Comprehensive testing of all validation utility functions |
| monitoring.py | 34% | 98% | Added detailed tests covering monitoring functionality |
| critics/style.py | 9% | 90%* | Comprehensive tests via isolation techniques |
| models/mock.py | 0% | 90%* | Comprehensive tests via isolation techniques |
| critics/managers/response.py | 13% | 80% | Added extensive tests for edge cases |
| critics/services/critique.py | 13% | 100% | Comprehensive tests including async methods |

\* Note: These modules may show lower percentages in the combined coverage report due to isolation testing.

## Special Testing Techniques

### Isolation Testing

Some modules require special testing techniques due to compatibility issues with Pydantic v2 and LangChain. For these modules, we've implemented isolated test runners that:

1. Patch the import system to avoid problematic imports
2. Run the tests in a clean environment
3. Generate coverage data separately

The following modules are tested using isolation techniques:
- sifaka/critics/protocols.py
- sifaka/critics/reflexion.py
- sifaka/critics/style.py
- sifaka/models/mock.py
- sifaka/rules/formatting/whitespace.py

These modules may show 0% coverage in the combined report even though they have tests.

### Async Testing

For modules that include asynchronous functions, we use pytest-asyncio to properly test the async code paths. Examples include:
- critics/services/critique.py - Test coverage includes both synchronous and asynchronous methods

## Running Tests with Coverage

To run the tests with coverage reporting:

```bash
# Run all tests with coverage
python run_all_tests_with_coverage.py

# Run specific tests with coverage
python -m pytest tests/path/to/test_file.py -v --cov=sifaka.path.to.module --cov-report=term
```

## Current Overall Coverage

The current overall test coverage for the project is 49% (up from 38% before the recent improvements). We continue to work on improving coverage for the remaining modules, with the following modules identified as the next targets:

1. critics/prompt.py (41%)
2. critics/reflexion.py (0%*)
3. classifiers/* (various low coverage modules)

\* Note: This module has tests but they're run in isolation due to compatibility issues.