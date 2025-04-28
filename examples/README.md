# Sifaka Examples

This directory contains example scripts that showcase the various capabilities of the Sifaka library. Each example is designed to help you understand how different components work together to create reliable and safe AI applications.

## Requirements

- Python 3.11+
- Sifaka library with dependencies installed
- API keys for respective providers (OpenAI/Anthropic) in environment variables
- Python dotenv for environment management

## Example Scripts

### Basic Usage and Core Features

1. **usage.py**
   This example serves as an introduction to Sifaka's core functionality. It demonstrates how to use the Chain class to validate and improve text using multiple rules. The script shows how to configure rules, set up a provider, and handle validation results effectively.

2. **openai_example.py**
   This script illustrates how to integrate OpenAI's GPT models with Sifaka's validation framework. It sets up a pipeline where text generation requests are first validated against safety rules, then improved if necessary using a PromptCritic. The example handles API authentication, rate limiting, and error cases.

### Advanced Classification

3. **advanced_classifiers_example.py**
   This example builds a sophisticated text classification system using Sifaka's ClassifierRule. It demonstrates how to create custom classifiers for genre detection and bias analysis, and how to integrate them into your validation pipeline using rule adapters.

4. **combined_classifiers.py**
   This script demonstrates the power of combining multiple classifiers for comprehensive text analysis. It creates a Chain that simultaneously analyzes sentiment, readability, and repetitive patterns, showing how to configure different thresholds and weights for each classifier.

### Safety and Content Validation

5. **toxicity_rule_example.py**
   This script implements a comprehensive toxicity detection system using Sifaka's ToxicityRule. It shows how to configure different types of toxicity detection, set appropriate thresholds, and provide actionable feedback for content improvement.

6. **comprehensive_content_validation.py**
   This example creates a complete content validation pipeline using Chain. It demonstrates how to combine multiple rules for format validation, structure checking, and content analysis into a single workflow, with detailed feedback for content improvement.

### Pattern Analysis

7. **symmetry_examples.py**
   This script demonstrates advanced pattern analysis using Sifaka's SymmetryRule and RepetitionRule. It shows how to detect various types of text patterns and structural symmetry, with examples of different pattern matching configurations and threshold adjustments.

### Domain-Specific Validation

8. **legal_rules_example.py**
   This example implements specialized validation rules for legal content using Sifaka's LegalRule components. It demonstrates how to validate legal citations, check document structure, and verify terminology usage across different jurisdictions.

9. **medical_rules_example.py**
   This script shows how to use Sifaka's MedicalRule for validating medical content. It includes examples of terminology validation, format checking, and ensuring compliance with medical documentation standards.

## Usage

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run an example:
   ```bash
   python examples/usage.py
   ```

## Contributing

Feel free to contribute additional examples by submitting a pull request. Please ensure your examples:
- Follow the established code style
- Include comprehensive documentation
- Demonstrate practical use cases
- Include appropriate error handling
- Are tested with the latest Sifaka version

## License

These examples are licensed under the same terms as the Sifaka library. See the LICENSE file in the root directory for details.