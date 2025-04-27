# Sifaka Examples

This directory contains example scripts demonstrating various features and capabilities of the Sifaka library. Each example is designed to showcase different aspects of text analysis, pattern detection, and integration with various models and frameworks.

## Available Examples

### 1. Combined Classifiers (`combined_classifiers.py`)
Demonstrates the integration of multiple analysis techniques:
- Text Classification (sentiment and readability analysis)
- Pattern Analysis (symmetry and repetition detection)
- Multi-faceted Text Analysis with parallel processing
- Comprehensive logging and result aggregation

**Key Features:**
- Sentiment Analysis with confidence scoring
- Readability Assessment using standard metrics
- Symmetry Pattern Detection
- Repetition Identification
- Structured logging of results

### 2. OpenAI Integration (`openai_example.py`)
Shows how to integrate Sifaka with OpenAI's models:
- Pattern rule configuration with OpenAI
- Text validation and analysis
- Custom rule implementation
- Error handling and validation checks

### 3. Pydantic Integration (`pydantic_integration.py`)
Demonstrates how to use Sifaka with Pydantic for data validation:
- Film review generation and analysis
- Structured data validation
- Pattern matching on structured content
- Integration with type hints and data models

### 4. Reflector Usage (`reflector_usage.py`)
Showcases the pattern detection capabilities:
- Text symmetry analysis
- Repetition pattern detection
- Multiple text sample analysis
- Detailed pattern matching results

### 5. Basic Usage (`basic_usage.py`)
Provides fundamental examples of Sifaka's core functionality:
- Basic text analysis
- Simple pattern matching
- Rule configuration
- Result handling

## Requirements

- Python 3.7+
- Sifaka library with all dependencies installed
- Additional requirements per example (specified in each file)

## Usage

Each example can be run independently:

```bash
python openai_example.py
python combined_classifiers.py
python pydantic_integration.py
python reflector_usage.py
python basic_usage.py
```

## Important Notes

1. The `Reflector` class is deprecated and will be removed in version 2.0.0. Use `SymmetryRule` and `RepetitionRule` from `sifaka.rules.pattern_rules` instead.

2. Some examples may require additional configuration:
   - OpenAI examples require an API key
   - Certain examples may need specific model configurations
   - Check individual example files for specific requirements

3. The examples directory includes a `logs/` folder for output logging

## Best Practices

1. Review the docstrings in each example for detailed information
2. Check the configuration sections for customizable parameters
3. Use the logging output to understand the analysis results
4. Consider the confidence thresholds when interpreting results

## Contributing

When adding new examples:
1. Include comprehensive docstrings
2. Add logging for important operations
3. Document any special requirements
4. Follow the existing code structure
5. Update this README with the new example