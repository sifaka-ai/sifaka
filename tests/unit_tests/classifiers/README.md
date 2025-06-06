# Sifaka Classifier Tests

This directory contains comprehensive unit tests for all Sifaka classifier implementations. The tests cover functionality, error handling, edge cases, and performance characteristics of each classifier.

## Test Structure

### Base Classes
- **`test_base.py`** - Tests for `BaseClassifier`, `CachedClassifier`, `ClassificationResult`, and `TimingMixin`

### Classifier Implementations
- **`test_bias.py`** - Tests for `BiasClassifier` and `CachedBiasClassifier`
- **`test_emotion.py`** - Tests for `EmotionClassifier` and `CachedEmotionClassifier`
- **`test_intent.py`** - Tests for `IntentClassifier` and `CachedIntentClassifier`
- **`test_language.py`** - Tests for `LanguageClassifier` and `CachedLanguageClassifier`
- **`test_readability.py`** - Tests for `ReadabilityClassifier` and `CachedReadabilityClassifier`
- **`test_sentiment.py`** - Tests for `SentimentClassifier` and `CachedSentimentClassifier`
- **`test_spam.py`** - Tests for `SpamClassifier` and `CachedSpamClassifier`
- **`test_toxicity.py`** - Tests for `ToxicityClassifier` and `CachedToxicityClassifier`

## Test Coverage

Each classifier test file includes comprehensive coverage of:

### Core Functionality
- ✅ Initialization with default and custom parameters
- ✅ Async and sync classification methods
- ✅ Primary classification algorithms (ML models, external libraries)
- ✅ Fallback mechanisms when dependencies are unavailable
- ✅ Factory functions for easy classifier creation

### External Dependencies
- ✅ Proper handling when dependencies are available
- ✅ Graceful fallback when dependencies are missing
- ✅ Mock testing of external libraries (transformers, detoxify, textstat, etc.)

### Edge Cases
- ✅ Empty text and whitespace-only input
- ✅ Very long text (1000+ words)
- ✅ Special characters and Unicode text
- ✅ Single words and short phrases
- ✅ Numbers and symbols

### Error Handling
- ✅ Invalid input validation
- ✅ External library failures
- ✅ Network/model loading errors
- ✅ Confidence threshold handling

### Performance Features
- ✅ Timing functionality
- ✅ Caching behavior
- ✅ Cache clearing
- ✅ Processing time recording

### Data Validation
- ✅ Classification result structure
- ✅ Confidence score validation (0.0-1.0)
- ✅ Metadata completeness
- ✅ Label consistency

## Running Tests

### Run All Classifier Tests
```bash
# Basic run
python -m pytest tests/unit_tests/classifiers/

# With verbose output
python -m pytest tests/unit_tests/classifiers/ -v

# With coverage reporting
python -m pytest tests/unit_tests/classifiers/ --cov=sifaka.classifiers --cov-report=html
```

### Run Specific Classifier Tests
```bash
# Test bias classifier
python -m pytest tests/unit_tests/classifiers/test_bias.py -v

# Test emotion classifier
python -m pytest tests/unit_tests/classifiers/test_emotion.py -v

# Test sentiment classifier
python -m pytest tests/unit_tests/classifiers/test_sentiment.py -v
```

### Using the Test Runner Script
```bash
# Run all classifier tests
python tests/run_classifier_tests.py

# Run with coverage
python tests/run_classifier_tests.py --coverage

# Run specific classifier
python tests/run_classifier_tests.py --classifier bias

# List available classifiers
python tests/run_classifier_tests.py --list
```

## Test Dependencies

The tests use mocking to avoid requiring actual external dependencies:

### Mocked Libraries
- **transformers** - Hugging Face transformers library
- **detoxify** - Toxicity detection library
- **textstat** - Text readability metrics
- **langdetect** - Language detection
- **textblob** - Text processing
- **sklearn** - Machine learning models

### Test Fixtures
- Sample text data for different scenarios
- Mock classification results
- Mock external library responses
- Async test utilities

## Test Patterns

### Initialization Testing
```python
def test_classifier_initialization(self):
    """Test classifier initialization with default parameters."""
    classifier = SomeClassifier()
    assert classifier.name == "expected_name"
    assert classifier.threshold == 0.7
```

### Async Classification Testing
```python
@pytest.mark.asyncio
async def test_classify_async(self):
    """Test async classification method."""
    classifier = SomeClassifier()
    result = await classifier.classify_async("test text")
    assert result.label in classifier.get_classes()
    assert 0.0 <= result.confidence <= 1.0
```

### Mock External Dependencies
```python
@patch('module.importlib.import_module')
async def test_with_external_library(self, mock_import):
    """Test classification with mocked external library."""
    mock_lib = Mock()
    mock_lib.some_method.return_value = expected_result
    mock_import.return_value = mock_lib
    
    classifier = SomeClassifier()
    result = await classifier.classify_async("test")
    assert result.metadata["method"] == "external_library"
```

### Fallback Testing
```python
async def test_fallback_method(self):
    """Test fallback when external dependencies unavailable."""
    with patch('module.importlib.import_module') as mock_import:
        mock_import.side_effect = ImportError("No dependencies")
        classifier = SomeClassifier()
    
    result = await classifier.classify_async("test")
    assert result.metadata["method"] == "fallback"
```

## Expected Test Results

When all tests pass, you should see:
- **200+ test cases** across all classifier files
- **90%+ code coverage** for classifier modules
- **All edge cases handled** gracefully
- **Consistent API behavior** across classifiers
- **Proper error handling** for all failure modes

## Contributing

When adding new classifiers or modifying existing ones:

1. **Create comprehensive tests** following the existing patterns
2. **Test all code paths** including fallbacks and error cases
3. **Mock external dependencies** to avoid test environment issues
4. **Include edge case testing** for unusual inputs
5. **Verify timing and caching** functionality works correctly
6. **Update this README** if adding new test categories

## Troubleshooting

### Common Issues

**Import Errors**: Ensure the test environment has access to the sifaka package
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Async Test Issues**: Make sure pytest-asyncio is installed
```bash
pip install pytest-asyncio
```

**Coverage Issues**: Install coverage tools
```bash
pip install pytest-cov coverage
```

**Mock Issues**: Verify unittest.mock is working correctly
```python
from unittest.mock import Mock, patch, AsyncMock
```
