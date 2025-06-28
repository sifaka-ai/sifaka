"""Transform bad code comments into good ones using the Style critic."""

import asyncio
from sifaka import improve, Config


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
    
    # Transform the bad comment
    result = await improve(
        bad_comment,
        critics=["style"],
        max_iterations=2,
        config=Config(
            model="gpt-4o-mini",
            critic_model="gpt-4o-mini",
            temperature=0.7,
            style_description="""
            Professional code documentation style following these principles:
            - Clear, concise description of what the code does
            - Specify parameters/arguments with types and descriptions
            - Document return values and their structure
            - Include error conditions and exceptions
            - Add usage examples when helpful
            - Use proper formatting (Args:, Returns:, Raises:, Example:)
            - Be specific about data types and edge cases
            - Avoid vague terms like "some", "might", "various"
            """,
            style_reference_text=good_comment_examples + module_doc_example,
            style_examples=[
                "Retrieves user information from the database",
                "Args:\\n    user_id: Unique identifier for the user",
                "Returns:\\n    Dictionary containing user data and status",
                "Raises:\\n    UserNotFoundError: If user_id doesn't exist",
                "Example:\\n    >>> result = process_data(123)"
            ],
            force_improvements=True,
        ),
    )
    
    print("❌ Bad Comment:")
    print(bad_comment.strip())
    print("\n✅ Improved Comment:")
    print(result.final_text.strip())
    
    # Show the transformation details
    print(f"\n📊 Iterations: {result.iteration}")
    if result.critiques:
        last_critique = result.critiques[-1]
        print(f"📈 Final alignment score: {getattr(last_critique, 'alignment_score', 'N/A')}")
    
    print("\n💡 Key improvements made:")
    print("- Added specific parameter documentation")
    print("- Clarified return values and types")
    print("- Documented error conditions")
    print("- Removed vague language")
    print("- Added proper docstring structure")


if __name__ == "__main__":
    asyncio.run(main())