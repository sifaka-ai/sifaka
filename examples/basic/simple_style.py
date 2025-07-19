"""Transform bad code comments into good ones using the Style critic."""

import asyncio

from dotenv import load_dotenv

from sifaka import improve
from sifaka.core.config import Config, CriticConfig, LLMConfig
from sifaka.core.types import CriticType

# Load environment variables from .env file
load_dotenv()

# Bad code comment example - vague, no structure, missing context
bad_comment = """
This function processes data and returns results. It takes some parameters
and does validation. There might be errors. Use carefully.
"""

# Good code comment examples showing best practices
good_comment_examples = """
def process_user_data(user_id: int, validate: bool = True) -> Dict[str, Any]:
    \"\"\"Process and validate user data from the database.

    Retrieves user information, performs validation checks, and returns
    a formatted response with user details and validation status.

    Args:
        user_id: Unique identifier for the user
        validate: Whether to run validation checks (default: True)

    Returns:
        Dictionary containing:
        - 'user': User object with profile data
        - 'status': Validation status ('valid', 'invalid', 'pending')
        - 'errors': List of validation errors (empty if valid)

    Raises:
        UserNotFoundError: If user_id doesn't exist in database
        ValidationError: If validation fails and strict mode is enabled

    Example:
        >>> result = process_user_data(12345, validate=True)
        >>> print(result['status'])
        'valid'
    \"\"\"
"""

# Another example showing class/module documentation
module_doc_example = """
\"\"\"User authentication and session management module.

This module provides secure authentication mechanisms including:
- Password hashing with bcrypt
- JWT token generation and validation
- Session management with Redis backend
- Rate limiting for login attempts

Security considerations:
- All passwords are hashed using bcrypt with cost factor 12
- JWT tokens expire after 24 hours
- Failed login attempts are rate-limited to 5 per hour

Usage:
    from auth import authenticate_user, create_session

    user = authenticate_user(email, password)
    session = create_session(user)
\"\"\"
"""


async def main():
    print("🔧 Code Comment Improvement Example")
    print("=" * 60)

    # Configure Sifaka with the new structured config
    config = Config(
        llm=LLMConfig(model="gpt-4o-mini", temperature=0.7),
        critic=CriticConfig(
            critics=[CriticType.STYLE]
            # Note: Style-specific settings like style_description would be
            # handled differently in the new architecture
        ),
    )

    # Transform the bad comment
    result = await improve(
        bad_comment,
        critics=[CriticType.STYLE],
        max_iterations=2,
        config=config,
    )

    print("❌ Bad Comment:")
    print(bad_comment.strip())
    print("\n✅ Improved Comment:")
    print(result.final_text.strip())

    # Show the transformation details
    print(f"\n📊 Iterations: {result.iteration}")
    if result.critiques:
        last_critique = result.critiques[-1]
        print(
            f"📈 Final alignment score: {getattr(last_critique, 'alignment_score', 'N/A')}"
        )

    print("\n💡 Key improvements made:")
    print("- Added specific parameter documentation")
    print("- Clarified return values and types")
    print("- Documented error conditions")
    print("- Removed vague language")
    print("- Added proper docstring structure")


if __name__ == "__main__":
    asyncio.run(main())
