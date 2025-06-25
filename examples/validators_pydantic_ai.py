"""Examples of using PydanticAI-based validators in Sifaka."""

import asyncio

from sifaka.agents.validators import (
    LengthValidatorAgent,
    ContentValidatorAgent,
    ComposableValidatorAgent,
)
from sifaka.agents.validators.pattern import (
    CodePatternValidator,
)
from sifaka.agents.validators.numeric import (
    FinancialDataValidator,
    ScientificDataValidator,
)
from sifaka.agents.validators.structured import (
    MultiCriteriaValidator,
    create_technical_documentation_validator,
)
from sifaka.agents.validators.adapter import ValidatorBuilder


async def example_basic_validators():
    """Example of basic validator usage."""
    print("\n=== Basic Validators Example ===\n")

    # Sample text
    text = """
    Artificial Intelligence is transforming the technology landscape.
    Machine learning algorithms are becoming more sophisticated.
    This revolution will continue to shape our future.
    """

    # Length validation
    length_validator = LengthValidatorAgent(
        min_length=50, max_length=100, count_type="words"
    )

    result = await length_validator.validate(text)
    print(f"Length Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Score: {result.score:.2f}")
    print(f"Details: {result.overall_assessment}\n")

    # Content validation
    content_validator = ContentValidatorAgent(
        required_topics=["AI", "machine learning"],
        tone="professional",
        target_audience="technology professionals",
    )

    result = await content_validator.validate(text)
    print(f"Content Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    if result.rules:
        for rule in result.rules:
            print(f"  - {rule.name}: {'✓' if rule.passed else '✗'}")


async def example_composable_validators():
    """Example of composable validators."""
    print("\n=== Composable Validators Example ===\n")

    # Create a blog post validator with multiple rules
    blog_validator = ComposableValidatorAgent(
        name="comprehensive_blog_validator",
        validation_rules=[
            "Must be 300-800 words long",
            "Include an engaging introduction that hooks the reader",
            "Use at least 3 subheadings to organize content",
            "Include practical examples or case studies",
            "End with a clear call-to-action",
            "Maintain a conversational yet professional tone",
            "Optimize for SEO with relevant keywords",
        ],
        strict_mode=False,  # Allow some flexibility
        pass_threshold=0.8,  # 80% of rules must pass
    )

    blog_text = """
    # The Future of AI in Healthcare

    Artificial Intelligence is revolutionizing healthcare in unprecedented ways.

    ## Diagnosis and Detection
    AI algorithms can now detect diseases earlier than ever before...

    ## Treatment Personalization
    Machine learning helps create personalized treatment plans...

    ## Challenges and Opportunities
    While AI brings immense benefits, we must address ethical concerns...

    Ready to embrace AI in your healthcare practice? Contact us today!
    """

    result = await blog_validator.validate(blog_text)
    print(f"Blog Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Score: {result.score:.2f}")
    print("\nValidation Rules:")
    for rule in result.rules:
        status = "✓" if rule.passed else "✗"
        print(f"  {status} {rule.name}")
        if not rule.passed and rule.suggestions:
            print(f"     Suggestion: {rule.suggestions[0]}")


async def example_structured_validation():
    """Example of structured document validation."""
    print("\n=== Structured Validation Example ===\n")

    # Create a technical documentation validator
    tech_doc_validator = create_technical_documentation_validator("API")

    api_doc = """
    # User Authentication API

    ## Overview
    This API provides secure user authentication for our platform.

    ## Prerequisites
    - API key from developer portal
    - HTTPS connection required

    ## Endpoints

    ### POST /auth/login
    Authenticates a user and returns a JWT token.

    **Parameters:**
    - username (string, required)
    - password (string, required)

    **Response:**
    ```json
    {
        "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "expires_in": 3600
    }
    ```

    ## Error Codes
    - 401: Invalid credentials
    - 429: Rate limit exceeded

    ## Examples
    ```bash
    curl -X POST https://api.example.com/auth/login \\
         -H "Content-Type: application/json" \\
         -d '{"username": "user", "password": "pass"}'
    ```
    """

    result = await tech_doc_validator.validate(api_doc)
    print(
        f"Technical Documentation Validation: {'PASSED' if result.is_valid else 'FAILED'}"
    )
    print(f"Score: {result.score:.2f}")
    print("\nPriority improvements:")
    for i, improvement in enumerate(result.improvement_priority[:3], 1):
        print(f"  {i}. {improvement}")


async def example_domain_specific_validators():
    """Example of domain-specific validators."""
    print("\n=== Domain-Specific Validators Example ===\n")

    # Financial data validation
    financial_text = """
    Q4 2023 Financial Results:

    Revenue: $5.2 million (up 23% YoY)
    Operating Expenses: $3.8 million
    Net Profit: $1.4 million (26.9% margin)

    Key metrics:
    - Customer Acquisition Cost: $125
    - Lifetime Value: $890
    - Monthly Recurring Revenue: $433,000
    """

    financial_validator = FinancialDataValidator(
        currency="USD", require_sources=True, fiscal_year=2023
    )

    result = await financial_validator.validate(financial_text)
    print("Financial Data Validation:")
    print(f"  Status: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"  Issues found: {len(result.errors)}")
    if result.errors:
        for error in result.errors[:2]:
            print(f"    - {error}")

    # Scientific data validation
    scientific_text = """
    Experimental Results:

    The reaction yielded 85.3% ± 2.1% of the theoretical maximum.
    Temperature was maintained at 25°C throughout.
    Sample size: n=50

    Statistical significance: p < 0.001
    Correlation coefficient: r = 0.92
    """

    scientific_validator = ScientificDataValidator(
        require_units=True, require_uncertainty=True, significant_figures=3
    )

    result = await scientific_validator.validate(scientific_text)
    print("\nScientific Data Validation:")
    print(f"  Status: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"  Score: {result.score:.2f}")


async def example_pattern_validation():
    """Example of pattern-based validation."""
    print("\n=== Pattern Validation Example ===\n")

    # Code pattern validation
    python_code = '''
    def calculate_average(numbers):
        """Calculate the average of a list of numbers."""
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")

        total = sum(numbers)
        count = len(numbers)
        return total / count

    # Example usage
    data = [1, 2, 3, 4, 5]
    avg = calculate_average(data)
    print(f"Average: {avg}")
    '''

    code_validator = CodePatternValidator(
        language="python",
        required_constructs=["function_definition", "docstring", "error_handling"],
        style_guide="PEP8",
    )

    result = await code_validator.validate(python_code)
    print(f"Code Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    print("Pattern checks:")
    for rule in result.rules[:3]:
        print(f"  - {rule.name}: {'✓' if rule.passed else '✗'}")


async def example_multi_criteria_validation():
    """Example of weighted multi-criteria validation."""
    print("\n=== Multi-Criteria Validation Example ===\n")

    # Create a weighted validator for product descriptions
    product_validator = MultiCriteriaValidator(
        criteria={
            "clarity": 0.25,  # 25% weight
            "persuasiveness": 0.20,  # 20% weight
            "technical_accuracy": 0.20,  # 20% weight
            "seo_optimization": 0.15,  # 15% weight
            "brand_voice": 0.10,  # 10% weight
            "completeness": 0.10,  # 10% weight
        },
        pass_threshold=0.75,  # Need 75% weighted score to pass
    )

    product_description = """
    Introducing the TechPro X1 Laptop

    Experience unparalleled performance with our latest flagship laptop.
    Powered by the cutting-edge Intel Core i9 processor and 32GB RAM,
    the TechPro X1 handles any task with ease.

    Key Features:
    - 15.6" 4K OLED display
    - 1TB NVMe SSD storage
    - All-day battery life (up to 12 hours)
    - Weighs only 3.5 lbs

    Perfect for professionals, creators, and power users who demand the best.
    """

    result = await product_validator.validate(product_description)
    print(
        f"Product Description Validation: {'PASSED' if result.is_valid else 'FAILED'}"
    )
    print(f"Weighted Score: {result.score:.2%}")
    print("\nCriteria Scores:")
    for rule in result.rules:
        print(f"  - {rule.name}: {rule.importance:.2f}")


async def example_adapter_usage():
    """Example of using validators with the legacy adapter."""
    print("\n=== Adapter Usage Example ===\n")

    # Build validators using the adapter
    validators = (
        ValidatorBuilder()
        .add_length(min_length=100, max_length=500)
        .add_content(
            required_topics=["innovation", "technology"],
            forbidden_topics=["outdated", "deprecated"],
        )
        .add_format(
            document_type="article",
            required_sections=["introduction", "main_points", "conclusion"],
        )
        .build()
    )

    sample_text = """
    # Innovation in Technology

    ## Introduction
    Technology continues to drive innovation across all industries.

    ## Main Points
    - AI and ML are transforming business processes
    - Cloud computing enables global collaboration
    - IoT connects our physical and digital worlds

    ## Conclusion
    The pace of technological innovation shows no signs of slowing.
    """

    # These validators work with the legacy Sifaka interface
    from sifaka.core.models import SifakaResult

    mock_result = SifakaResult(
        id="test", original_text=sample_text, final_text=sample_text
    )

    print("Running validators through adapter:")
    for validator in validators:
        result = await validator.validate(sample_text, mock_result)
        print(
            f"  - {validator.name}: {'PASSED' if result.passed else 'FAILED'} (score: {result.score:.2f})"
        )


async def main():
    """Run all examples."""
    await example_basic_validators()
    await example_composable_validators()
    await example_structured_validation()
    await example_domain_specific_validators()
    await example_pattern_validation()
    await example_multi_criteria_validation()
    await example_adapter_usage()


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
