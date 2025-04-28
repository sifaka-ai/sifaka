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
   This example serves as an introduction to Sifaka's core functionality. It processes text through multiple analysis stages: first analyzing sentiment to understand emotional tone, then evaluating readability using Flesch-Kincaid metrics to assess complexity. The script then demonstrates how Sifaka can automatically improve text while preserving its original meaning. It uses a combination of classifiers to analyze the text from different angles, showing how these components work together in a real-world scenario.

2. **openai_example.py**
   This script illustrates how to integrate OpenAI's GPT models with Sifaka's validation framework. It sets up a pipeline where text generation requests are first validated against safety rules, then improved if necessary using a PromptCritic. The example handles API authentication, rate limiting, and error cases, showing how to build robust applications with external AI providers.

### Advanced Classification

3. **advanced_classifiers_example.py**
   This example builds a sophisticated text classification system that can identify both genre and potential bias in text. It creates two separate classifiers - one for genre detection and another for bias analysis - and shows how to train them on custom datasets. The script then combines these classifiers with rule adapters, allowing them to be used as validation rules in a larger pipeline. It includes a detailed walkthrough of how confidence scores are calculated and used in decision-making.

4. **combined_classifiers.py**
   This script demonstrates the power of combining multiple classifiers for comprehensive text analysis. It creates a pipeline that simultaneously analyzes sentiment, readability, and repetitive patterns. The example shows how to configure different thresholds for each classifier, how to weight their results, and how to make decisions based on combined analysis. It includes real-world examples of how this might be used in content moderation or quality assurance systems.

5. **topic_classifier_example.py**
   This example implements a sophisticated topic classification system. It begins by loading and preprocessing a custom dataset of categorized texts, then trains a classifier to identify topics across multiple domains. The script shows how to handle hierarchical topic structures, deal with multi-label classification, and adjust confidence thresholds. It includes methods for evaluating classifier performance and fine-tuning the model based on results.

6. **spam_classifier_example.py**
   This script builds a complete spam detection system using Sifaka's classification framework. It starts by training on a dataset of spam and legitimate messages, then implements custom preprocessing steps and feature extraction. The example shows how to balance precision and recall through confidence thresholds, and includes methods for handling edge cases and evolving spam patterns. It demonstrates real-time classification and includes performance monitoring.

7. **gpt2_classifier_example.py**
   This example shows how to leverage GPT-2's language understanding capabilities for classification tasks. It implements a custom classifier that uses GPT-2's contextual embeddings, demonstrating how to fine-tune the model on specific classification tasks. The script includes batch processing for efficiency, handling of long sequences, and methods for interpreting the model's decisions. It shows how to integrate transformer-based models into Sifaka's validation pipeline.

### Safety and Content Validation

8. **toxicity_rule_example.py**
   This script implements a comprehensive toxicity detection system. It creates a custom validation rule that combines multiple approaches to identify toxic content, including keyword matching, contextual analysis, and machine learning-based detection. The example shows how to handle different types of toxicity, set appropriate thresholds, and provide actionable feedback for content improvement. It includes integration with Sifaka's toxicity detection extras and demonstrates how to handle edge cases.

9. **comprehensive_content_validation.py**
   This example creates a complete content validation pipeline that checks multiple aspects of text quality. It combines format validation, structure checking, and content analysis into a single workflow. The script shows how to chain multiple validation rules, handle dependencies between rules, and aggregate validation results. It includes examples of custom validation rules and demonstrates how to provide detailed feedback for content improvement.

10. **multi_provider_safety_example.py**
    This script demonstrates how to use multiple AI providers for enhanced safety validation. It implements a system that cross-validates content across different providers, comparing their assessments to make more reliable decisions. The example includes fallback mechanisms for when providers fail, load balancing between providers, and methods for resolving conflicting assessments. It shows how to handle rate limits and costs while maintaining robust validation.

### Specialized Features

11. **advanced_rules_example.py**
    This example explores complex rule configurations and custom rule creation. It shows how to build rules with dependencies, implement custom validation logic, and chain rules together in sophisticated ways. The script demonstrates advanced features like rule priorities, cost-based execution ordering, and conditional rule activation. It includes examples of how to handle complex validation scenarios and provide detailed feedback.

12. **symmetry_examples.py**
    This script implements advanced pattern analysis using Sifaka's new SymmetryRule. It shows how to detect various types of text patterns, from simple repetition to complex structural symmetry. The example demonstrates different pattern matching configurations, threshold adjustments, and methods for handling nested patterns. It includes real-world applications like detecting content structure issues or identifying generated text patterns.

13. **test_legal_rules.py**
    This example implements specialized validation rules for legal content. It shows how to validate legal citations, check document structure against legal standards, and verify terminology usage. The script includes rules for different jurisdictions, handles complex legal document formats, and demonstrates how to provide legally relevant feedback. It shows how to integrate with legal document standards and maintain compliance requirements.

14. **language_critic_example.py**
    This script creates an advanced language improvement system using Sifaka's critic framework. It implements sophisticated grammar checking, style consistency validation, and tone adjustment across multiple languages. The example shows how to handle language-specific rules, maintain consistency across translations, and improve text while preserving legal or technical accuracy. It includes methods for handling idiomatic expressions and maintaining style guides.

## Usage

1. Install required dependencies:
   ```