"""
Pattern matching utilities for Sifaka.

This module provides standardized pattern matching utilities for the Sifaka framework,
including regex compilation, caching, and pattern matching functions.

## Pattern Matching

The module provides standardized pattern matching:

1. **compile_pattern**: Compile a regex pattern with caching
2. **match_pattern**: Match a pattern against text
3. **find_patterns**: Find all matches of patterns in text

## Pattern Types

The module supports different pattern types:

1. **Regex Patterns**: Standard regular expressions
2. **Glob Patterns**: File path matching patterns
3. **Wildcard Patterns**: Simple wildcard patterns with * and ?

## Usage Examples

```python
from sifaka.utils.patterns import (
    compile_pattern, match_pattern, find_patterns
)

# Compile a pattern
pattern = compile_pattern(
    r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b",
    case_sensitive=False
)

# Match a pattern
is_match = match_pattern(
    "user@example.com",
    r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b",
    case_sensitive=False
)

# Find patterns
matches = find_patterns(
    "Contact us at user@example.com or support@example.com",
    {
        "email": r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b",
        "phone": r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b"
    },
    case_sensitive=False
)
```
"""

import fnmatch
import re
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Pattern, Set, Tuple, Union, cast

from sifaka.utils.errors import ValidationError


# Pattern cache for compiled regex patterns
_pattern_cache: Dict[str, Pattern] = {}


def compile_pattern(
    pattern: str,
    case_sensitive: bool = True,
    multiline: bool = False,
    dotall: bool = False,
    unicode: bool = True,
    cache_size: int = 100,
) -> Pattern:
    """
    Compile a regex pattern with caching.

    This function compiles a regex pattern with the specified flags and
    caches the result for improved performance.

    Args:
        pattern: Regex pattern string
        case_sensitive: Whether the pattern is case-sensitive
        multiline: Whether to use multiline mode
        dotall: Whether to use dotall mode (. matches newlines)
        unicode: Whether to use unicode mode
        cache_size: Maximum number of patterns to cache

    Returns:
        Compiled regex pattern

    Raises:
        ValidationError: If the pattern is invalid
    """
    # Create flags based on options
    flags = 0
    if not case_sensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL
    if unicode:
        flags |= re.UNICODE

    # Create cache key
    cache_key = f"{pattern}:{flags}"

    # Check cache
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]

    # Compile pattern
    try:
        compiled = re.compile(pattern, flags)

        # Cache pattern (with simple LRU behavior)
        if len(_pattern_cache) >= cache_size:
            # Remove a random entry if cache is full
            _pattern_cache.pop(next(iter(_pattern_cache)))
        _pattern_cache[cache_key] = compiled

        return compiled
    except re.error as e:
        raise ValidationError(
            f"Invalid regex pattern: {str(e)}", metadata={"pattern": pattern, "error": str(e)}
        )


@lru_cache(maxsize=100)
def _compile_pattern_cached(
    pattern: str,
    case_sensitive: bool = True,
    multiline: bool = False,
    dotall: bool = False,
    unicode: bool = True,
) -> Pattern:
    """
    Compile a regex pattern with LRU caching.

    This is an alternative implementation using Python's built-in LRU cache.

    Args:
        pattern: Regex pattern string
        case_sensitive: Whether the pattern is case-sensitive
        multiline: Whether to use multiline mode
        dotall: Whether to use dotall mode (. matches newlines)
        unicode: Whether to use unicode mode

    Returns:
        Compiled regex pattern

    Raises:
        ValidationError: If the pattern is invalid
    """
    # Create flags based on options
    flags = 0
    if not case_sensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL
    if unicode:
        flags |= re.UNICODE

    # Compile pattern
    try:
        return re.compile(pattern, flags)
    except re.error as e:
        raise ValidationError(
            f"Invalid regex pattern: {str(e)}", metadata={"pattern": pattern, "error": str(e)}
        )


def match_pattern(
    text: str,
    pattern: Union[str, Pattern],
    case_sensitive: bool = True,
    multiline: bool = False,
    dotall: bool = False,
    unicode: bool = True,
    match_type: str = "search",
) -> bool:
    """
    Match a pattern against text.

    This function matches a pattern against text and returns whether it matches.

    Args:
        text: Text to match against
        pattern: Regex pattern string or compiled pattern
        case_sensitive: Whether the pattern is case-sensitive
        multiline: Whether to use multiline mode
        dotall: Whether to use dotall mode (. matches newlines)
        unicode: Whether to use unicode mode
        match_type: Type of match to perform ("search", "match", or "fullmatch")

    Returns:
        True if the pattern matches, False otherwise

    Raises:
        ValidationError: If the pattern is invalid or match_type is invalid
    """
    # Validate match_type
    if match_type not in ("search", "match", "fullmatch"):
        raise ValidationError(
            f"Invalid match_type: {match_type}",
            metadata={"valid_types": ["search", "match", "fullmatch"]},
        )

    # Compile pattern if needed
    if isinstance(pattern, str):
        compiled = compile_pattern(
            pattern,
            case_sensitive=case_sensitive,
            multiline=multiline,
            dotall=dotall,
            unicode=unicode,
        )
    else:
        compiled = pattern

    # Perform match based on match_type
    if match_type == "search":
        return bool(compiled.search(text))
    elif match_type == "match":
        return bool(compiled.match(text))
    else:  # fullmatch
        return bool(compiled.fullmatch(text))


def find_patterns(
    text: str,
    patterns: Dict[str, Union[str, Pattern]],
    case_sensitive: bool = True,
    multiline: bool = False,
    dotall: bool = False,
    unicode: bool = True,
    return_matches: bool = True,
) -> Dict[str, Union[bool, List[str]]]:
    """
    Find all matches of patterns in text.

    This function finds all matches of the specified patterns in the text
    and returns a dictionary of results.

    Args:
        text: Text to search in
        patterns: Dictionary of pattern names to patterns
        case_sensitive: Whether the patterns are case-sensitive
        multiline: Whether to use multiline mode
        dotall: Whether to use dotall mode (. matches newlines)
        unicode: Whether to use unicode mode
        return_matches: Whether to return the matched strings or just boolean

    Returns:
        Dictionary of pattern names to match results

    Raises:
        ValidationError: If any pattern is invalid
    """
    results: Dict[str, Union[bool, List[str]]] = {}

    # Process each pattern
    for name, pattern in patterns.items():
        # Compile pattern if needed
        if isinstance(pattern, str):
            compiled = compile_pattern(
                pattern,
                case_sensitive=case_sensitive,
                multiline=multiline,
                dotall=dotall,
                unicode=unicode,
            )
        else:
            compiled = pattern

        # Find matches
        matches = compiled.findall(text)

        # Store results
        if return_matches:
            results[name] = matches
        else:
            results[name] = bool(matches)

    return results


def count_patterns(
    text: str,
    patterns: Dict[str, Union[str, Pattern]],
    case_sensitive: bool = True,
    multiline: bool = False,
    dotall: bool = False,
    unicode: bool = True,
) -> Dict[str, int]:
    """
    Count matches of patterns in text.

    This function counts the number of matches of the specified patterns
    in the text and returns a dictionary of counts.

    Args:
        text: Text to search in
        patterns: Dictionary of pattern names to patterns
        case_sensitive: Whether the patterns are case-sensitive
        multiline: Whether to use multiline mode
        dotall: Whether to use dotall mode (. matches newlines)
        unicode: Whether to use unicode mode

    Returns:
        Dictionary of pattern names to match counts

    Raises:
        ValidationError: If any pattern is invalid

    Examples:
        ```python
        from sifaka.utils.patterns import count_patterns

        # Count email addresses and phone numbers in text
        text = "Contact us at user@example.com or support@example.com. Call us at 555-123-4567."
        counts = count_patterns(
            text,
            {
                "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
            },
            case_sensitive=False
        )
        print(f"Found {counts['email']} email addresses and {counts['phone']} phone numbers")

        # Count word occurrences
        text = "The quick brown fox jumps over the lazy dog. The fox is quick."
        counts = count_patterns(
            text,
            {
                "the": r"\bthe\b",
                "fox": r"\bfox\b",
                "quick": r"\bquick\b"
            },
            case_sensitive=False
        )
        print(f"Word counts: {counts}")  # {'the': 2, 'fox': 2, 'quick': 2}

        # Count with compiled patterns
        import re
        email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
        phone_pattern = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

        counts = count_patterns(
            text,
            {
                "email": email_pattern,
                "phone": phone_pattern
            }
        )
        ```
    """
    results: Dict[str, int] = {}

    # Process each pattern
    for name, pattern in patterns.items():
        # Compile pattern if needed
        if isinstance(pattern, str):
            compiled = compile_pattern(
                pattern,
                case_sensitive=case_sensitive,
                multiline=multiline,
                dotall=dotall,
                unicode=unicode,
            )
        else:
            compiled = pattern

        # Count matches
        matches = compiled.findall(text)
        results[name] = len(matches)

    return results


def match_glob(text: str, pattern: str, case_sensitive: bool = True) -> bool:
    """
    Match a glob pattern against text.

    This function matches a glob pattern (like "*.txt") against text.

    Args:
        text: Text to match against
        pattern: Glob pattern string
        case_sensitive: Whether the pattern is case-sensitive

    Returns:
        True if the pattern matches, False otherwise

    Examples:
        ```python
        from sifaka.utils.patterns import match_glob

        # Match file extensions
        match_glob("document.txt", "*.txt")  # Returns True
        match_glob("document.pdf", "*.txt")  # Returns False
        match_glob("document.TXT", "*.txt", case_sensitive=False)  # Returns True

        # Match file names
        match_glob("config.json", "config.*")  # Returns True
        match_glob("settings.json", "config.*")  # Returns False

        # Match with multiple patterns
        match_glob("image.jpg", "*.jpg")  # Returns True
        match_glob("image.jpg", "*.png")  # Returns False
        match_glob("image.jpg", "image.*")  # Returns True

        # Match with character sets
        match_glob("file1.txt", "file[1-3].txt")  # Returns True
        match_glob("file4.txt", "file[1-3].txt")  # Returns False
        match_glob("fileA.txt", "file[A-C].txt")  # Returns True
        ```
    """
    if case_sensitive:
        return fnmatch.fnmatchcase(text, pattern)
    else:
        return fnmatch.fnmatch(text, pattern)


def match_wildcard(text: str, pattern: str, case_sensitive: bool = True) -> bool:
    """
    Match a wildcard pattern against text.

    This function matches a simple wildcard pattern against text.
    Wildcards are:
    - *: Matches any number of characters
    - ?: Matches a single character

    Args:
        text: Text to match against
        pattern: Wildcard pattern string
        case_sensitive: Whether the pattern is case-sensitive

    Returns:
        True if the pattern matches, False otherwise

    Examples:
        ```python
        from sifaka.utils.patterns import match_wildcard

        # Match with * wildcard (any number of characters)
        match_wildcard("hello world", "hello*")  # Returns True
        match_wildcard("hello world", "*world")  # Returns True
        match_wildcard("hello world", "h*d")  # Returns True
        match_wildcard("hello world", "hello universe")  # Returns False

        # Match with ? wildcard (single character)
        match_wildcard("file1.txt", "file?.txt")  # Returns True
        match_wildcard("file10.txt", "file?.txt")  # Returns False
        match_wildcard("cat", "c?t")  # Returns True
        match_wildcard("coat", "c?t")  # Returns False

        # Match with combined wildcards
        match_wildcard("hello.txt", "*.???")  # Returns True
        match_wildcard("hello.text", "*.???")  # Returns False
        match_wildcard("hello.text", "h?llo.*")  # Returns True

        # Case sensitivity
        match_wildcard("Hello", "hello", case_sensitive=False)  # Returns True
        match_wildcard("Hello", "hello", case_sensitive=True)  # Returns False
        ```
    """
    # Convert wildcard pattern to regex
    regex_pattern = pattern
    regex_pattern = regex_pattern.replace(".", "\\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = regex_pattern.replace("?", ".")
    regex_pattern = f"^{regex_pattern}$"

    # Match using regex
    return match_pattern(text, regex_pattern, case_sensitive=case_sensitive)
