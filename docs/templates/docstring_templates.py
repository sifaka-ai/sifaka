"""
Docstring templates for Sifaka components.

This module provides standardized docstring templates for different component types
in the Sifaka framework. Use these templates when creating or updating components
to ensure consistency across the codebase.

Usage:
    Copy the appropriate template and fill in the details for your component.
    Remove any sections that don't apply to your specific component.
"""

# Module Docstring Template
MODULE_DOCSTRING = '''
"""
{module_name} module for Sifaka.

This module provides components for {purpose}, including:
- {component1}: {brief_description1}
- {component2}: {brief_description2}
{additional_components}

Usage Example:
    ```python
    from sifaka.{module_path} import {component_name}
    
    {usage_example}
    ```
"""
'''

# Rule Class Docstring Template
RULE_CLASS_DOCSTRING = '''
"""
Rule for validating {validation_purpose}.

This rule validates {what_it_validates} by {validation_method}.

Lifecycle:
    1. Initialization: Set up with validation parameters
    2. Validation: Apply validation logic to input text
    3. Result: Return standardized validation results

Examples:
    ```python
    from sifaka.rules.{module_name} import create_{rule_name}_rule
    
    # Create a rule using the factory function
    rule = create_{rule_name}_rule(
        {param1}={value1},
        {param2}={value2}
    )
    
    # Validate text
    result = rule.validate("Text to validate")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''

# Validator Class Docstring Template
VALIDATOR_CLASS_DOCSTRING = '''
"""
Validator for {validation_purpose}.

This validator implements the validation logic for {what_it_validates}.
It {validation_method}.

Lifecycle:
    1. Initialization: Set up with validation parameters
    2. Validation: Apply validation logic to input
    3. Result: Return standardized validation results

Examples:
    ```python
    from sifaka.rules.{module_name} import create_{validator_name}_validator
    
    # Create a validator using the factory function
    validator = create_{validator_name}_validator(
        {param1}={value1},
        {param2}={value2}
    )
    
    # Validate text
    result = validator.validate("Text to validate")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''

# Classifier Class Docstring Template
CLASSIFIER_CLASS_DOCSTRING = '''
"""
A classifier for {classification_purpose}.

This classifier categorizes text into {categories} based on {classification_method}.

Lifecycle:
    1. Initialization: Set up with classification parameters
    2. Classification: Process input and apply classification logic
    3. Result: Return standardized classification results

Examples:
    ```python
    from sifaka.classifiers.{module_name} import create_{classifier_name}_classifier
    
    # Create a classifier using the factory function
    classifier = create_{classifier_name}_classifier(
        {param1}={value1},
        {param2}={value2}
    )
    
    # Classify text
    result = classifier.classify("Text to classify")
    print(f"Label: {result.label}, Confidence: {result.confidence}")
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''

# Critic Class Docstring Template
CRITIC_CLASS_DOCSTRING = '''
"""
A critic for {critique_purpose}.

This critic evaluates and improves text by {critique_method}.

Lifecycle:
    1. Initialization: Set up with critique parameters
    2. Critique: Analyze input text and generate feedback
    3. Improvement: Apply improvements based on feedback
    4. Memory: Maintain history of improvements (if applicable)

Examples:
    ```python
    from sifaka.critics.{module_name} import create_{critic_name}_critic
    
    # Create a critic using the factory function
    critic = create_{critic_name}_critic(
        {param1}={value1},
        {param2}={value2}
    )
    
    # Critique text
    critique = critic.critique("Text to critique")
    print(f"Feedback: {critique.feedback}")
    
    # Improve text
    improved_text = critic.improve("Text to improve", critique.feedback)
    print(f"Improved text: {improved_text}")
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''

# Chain Class Docstring Template
CHAIN_CLASS_DOCSTRING = '''
"""
A chain for {chain_purpose}.

This chain orchestrates the interaction between {components} to {chain_function}.

Lifecycle:
    1. Initialization: Set up with chain components
    2. Execution: Process input through the chain
    3. Result: Return standardized chain results

Examples:
    ```python
    from sifaka.chain.{module_name} import create_{chain_name}_chain
    from sifaka.models.openai import OpenAIProvider
    from sifaka.rules.formatting.length import create_length_rule
    
    # Create components
    model = OpenAIProvider(model_name="gpt-3.5-turbo")
    rules = [create_length_rule(min_chars=10, max_chars=100)]
    
    # Create a chain using the factory function
    chain = create_{chain_name}_chain(
        model=model,
        rules=rules,
        {param1}={value1},
        {param2}={value2}
    )
    
    # Run the chain
    result = chain.run("Input prompt")
    print(f"Output: {result.output}")
    print(f"Validation passed: {result.all_passed}")
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''

# Model Provider Class Docstring Template
MODEL_PROVIDER_CLASS_DOCSTRING = '''
"""
A model provider for {provider_name}.

This provider integrates with {service_name} to provide access to {model_types}.

Lifecycle:
    1. Initialization: Set up with API credentials and configuration
    2. Generation: Generate text from prompts
    3. Token Counting: Count tokens for input and output
    4. Error Handling: Handle API errors and rate limits

Examples:
    ```python
    from sifaka.models.{module_name} import {provider_class_name}
    from sifaka.models.base import ModelConfig
    
    # Create a provider
    provider = {provider_class_name}(
        model_name="{model_name}",
        config=ModelConfig(
            temperature=0.7,
            max_tokens=500
        )
    )
    
    # Generate text
    response = provider.generate("Write a short story about a robot.")
    print(f"Generated text: {response}")
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''

# Factory Function Docstring Template
FACTORY_FUNCTION_DOCSTRING = '''
"""
Create a {component_type} with the specified configuration.

This factory function creates a {component_name} configured with the provided parameters.

Args:
    {param1}: {description1}
    {param2}: {description2}
    **kwargs: Additional configuration parameters

Returns:
    Configured {component_type} instance

Examples:
    ```python
    from sifaka.{module_path} import create_{function_name}
    
    # Create component
    component = create_{function_name}(
        {param1}={value1},
        {param2}={value2}
    )
    
    # Use component
    result = component.{method}({input})
    ```
"""
'''

# Method Docstring Template
METHOD_DOCSTRING = '''
"""
{method_description}

{detailed_description}

Args:
    {param1}: {description1}
    {param2}: {description2}

Returns:
    {return_description}

Raises:
    {exception_type}: {exception_description}

Examples:
    ```python
    {example_code}
    ```
"""
'''

# Configuration Class Docstring Template
CONFIG_CLASS_DOCSTRING = '''
"""
Configuration for {component_name}.

This configuration class defines the parameters for {component_purpose}.

Examples:
    ```python
    from sifaka.{module_path} import {config_class_name}
    
    # Create configuration
    config = {config_class_name}(
        {param1}={value1},
        {param2}={value2},
        params={
            "option1": "value1",
            "option2": "value2"
        }
    )
    
    # Use with component
    component = {component_class_name}(config=config)
    ```

Attributes:
    {attribute1}: {description1}
    {attribute2}: {description2}
"""
'''
