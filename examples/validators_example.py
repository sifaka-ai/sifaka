"""Examples of using Sifaka validators.

This demonstrates various validators and how they can be combined
to ensure text quality meets specific requirements.
"""

import asyncio
from sifaka import improve
from sifaka.validators import (
    LengthValidator,
    ContentValidator,
    FormatValidator,
    PatternValidator,
    NumericRangeValidator,
    create_percentage_validator,
    create_price_validator,
    create_age_validator,
)


async def length_validation_example():
    """Example: Ensure text meets length requirements."""
    print("\n=== Length Validation Example ===")
    
    # Create validator for tweet-length content
    tweet_validator = LengthValidator(min_length=10, max_length=280)
    
    result = await improve(
        "Write a brief product announcement",
        validators=[tweet_validator],
        max_iterations=3
    )
    
    print(f"Original: {result.original_text}")
    print(f"Final: {result.final_text}")
    print(f"Length: {len(result.final_text)} characters")
    
    # Check validation results
    for v in result.validations:
        print(f"Validator: {v.validator}, Passed: {v.passed}, Score: {v.score}")


async def content_validation_example():
    """Example: Ensure text includes required content."""
    print("\n=== Content Validation Example ===")
    
    # Ensure marketing copy includes key elements
    marketing_validator = ContentValidator(
        required_keywords=["innovative", "solution"],
        forbidden_words=["cheap", "low-quality"],
        min_sentences=3
    )
    
    result = await improve(
        "Write marketing copy for our new software",
        validators=[marketing_validator],
        max_iterations=3
    )
    
    print(f"Final text: {result.final_text}")
    
    # Show validation details
    for v in result.validations:
        if v.validator == "ContentValidator":
            print(f"Content validation: {v.details}")


async def pattern_validation_example():
    """Example: Validate text matches specific patterns."""
    print("\n=== Pattern Validation Example ===")
    
    # Validate email format in contact information
    email_pattern = PatternValidator(
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        description="Must include a valid email address"
    )
    
    result = await improve(
        "Add contact information for our support team",
        validators=[email_pattern],
        max_iterations=2
    )
    
    print(f"Final text: {result.final_text}")
    
    # Check if pattern was found
    for v in result.validations:
        if v.validator == "PatternValidator":
            print(f"Email pattern found: {v.passed}")


async def numeric_validation_example():
    """Example: Validate numeric content in text."""
    print("\n=== Numeric Validation Example ===")
    
    # Validate percentage values
    percentage_validator = create_percentage_validator()
    
    # Validate price values
    price_validator = create_price_validator(min_price=10.0, max_price=1000.0)
    
    result = await improve(
        "Write a sales pitch with pricing and discount information",
        validators=[percentage_validator, price_validator],
        max_iterations=3
    )
    
    print(f"Final text: {result.final_text}")
    
    # Show numeric validations
    for v in result.validations:
        if "Numeric" in v.validator:
            print(f"{v.validator}: {v.details}")


async def combined_validation_example():
    """Example: Combine multiple validators for comprehensive checks."""
    print("\n=== Combined Validation Example ===")
    
    # Create a comprehensive set of validators for a product description
    validators = [
        # Length requirements
        LengthValidator(min_length=100, max_length=500),
        
        # Content requirements
        ContentValidator(
            required_keywords=["features", "benefits"],
            min_sentences=5
        ),
        
        # Must include a price
        create_price_validator(min_price=0.0, max_price=10000.0),
        
        # Format requirements
        FormatValidator(
            require_punctuation=True,
            allow_urls=True,
            max_caps_ratio=0.1  # Max 10% caps
        )
    ]
    
    result = await improve(
        "Write a product description for a new laptop",
        validators=validators,
        max_iterations=4,
        critics=["reflexion", "constitutional"]
    )
    
    print(f"Final text: {result.final_text}")
    print(f"\nValidation Summary:")
    
    passed = sum(1 for v in result.validations if v.passed)
    total = len(result.validations)
    
    print(f"Passed: {passed}/{total} validators")
    
    # Show failed validations
    for v in result.validations:
        if not v.passed:
            print(f"❌ {v.validator}: {v.details}")
        else:
            print(f"✅ {v.validator}: Score {v.score:.2f}")


async def custom_validation_example():
    """Example: Create custom validation logic."""
    print("\n=== Custom Validation Example ===")
    
    # Use pattern validator for custom format
    date_validator = PatternValidator(
        pattern=r'\d{4}-\d{2}-\d{2}',
        description="Must include date in YYYY-MM-DD format"
    )
    
    # Use numeric validator for custom ranges
    word_count_validator = NumericRangeValidator(
        min_value=50,
        max_value=100,
        value_extractor=lambda text: len(text.split()),
        description="Word count between 50-100"
    )
    
    result = await improve(
        "Write a daily report summary",
        validators=[date_validator, word_count_validator],
        max_iterations=3
    )
    
    print(f"Final text: {result.final_text}")
    print(f"Word count: {len(result.final_text.split())}")


async def main():
    """Run all validator examples."""
    examples = [
        length_validation_example,
        content_validation_example,
        pattern_validation_example,
        numeric_validation_example,
        combined_validation_example,
        custom_validation_example,
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())