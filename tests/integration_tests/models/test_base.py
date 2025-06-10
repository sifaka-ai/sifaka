"""Test basic model functionality.

This module tests basic model creation and functionality.
Note: The create_model function has been deprecated in favor of PydanticAI agents.
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_basic_model_functionality():
    """Test that we can import basic model components."""
    # Test that we can import the models module

    # Test that we can import critic results
    from sifaka.models.critic_results import CriticResult, CritiqueFeedback

    assert CriticResult is not None
    assert CritiqueFeedback is not None
