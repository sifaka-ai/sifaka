"""Test script for mock provider."""

try:
    print("Importing modules...")
    from sifaka.models.mock import create_mock_provider
    from sifaka.models.base import ModelConfig

    print("Creating mock provider...")
    mock_model = create_mock_provider(
        model_name="test-mock",
        temperature=0.7,
        max_tokens=100,
    )

    print("Mock provider created successfully.")
    print(f"Model name: {mock_model.model_name}")

    print("Testing generate method...")
    response = mock_model.generate("Hello, world!")
    print(f"Response: {response}")

    print("Test completed successfully.")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
