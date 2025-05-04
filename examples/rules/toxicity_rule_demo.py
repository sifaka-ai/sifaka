"""
Example demonstrating the use of harmful content validation rules.

This example shows how to:
1. Create and use harmful content rule
2. Validate text against it
3. Extract information about harmful content detection
"""

from sifaka.rules.content.safety import create_harmful_content_rule

def run_harmful_content_example():
    """Run a demonstration of harmful content detection."""
    print("=== Harmful Content Rule Example ===")

    # Create a harmful content rule with multiple categories of terms
    safety_rule = create_harmful_content_rule(
        name="safety_rule",
        description="Validates text for harmful content",
        threshold=0.1,  # Very low threshold to be highly sensitive
        categories={
            "toxic_language": [
                "hate",
                "offensive",
                "vulgar",
                "profanity",
                "obscene",
                "racist",
                "sexist",
                "discriminatory",
                "threatening",
                "harassing",
                "insult",
                "stupid",
                "idiot",
            ],
            "profanity": [
                "fuck",
                "shit",
                "damn",
                "hell",
            ],
        },
        fail_if_any=True,
    )

    # Test with various examples
    test_texts = [
        "Machine learning is a fascinating field of artificial intelligence.",
        "I hate people who don't understand simple concepts, they're such idiots.",
        "This is a damn good example of how to detect harmful content.",
        "Natural language processing helps computers understand human language.",
        "What the hell is going on with this stupid code? It's driving me crazy!"
    ]

    # Validate each text
    for i, text in enumerate(test_texts):
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {text}")

        # Validate with the rule
        result = safety_rule.validate(text)

        # Display results
        if result.passed:
            print("✅ Passed validation: No harmful content detected")
        else:
            print("❌ Failed validation: Harmful content detected")
            print(f"Message: {result.message}")

            # Show detailed category matches if available
            if result.metadata and "category_matches" in result.metadata:
                print("\nCategory matches:")
                for category, matches in result.metadata["category_matches"].items():
                    if matches:
                        print(f"  {category}: {', '.join(matches)}")


if __name__ == "__main__":
    run_harmful_content_example()