"""
Examples demonstrating the use of SymmetryRule for various text patterns.
"""

from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.rules.pattern_rules import SymmetryRule


def test_horizontal_symmetry():
    """Test horizontal symmetry with various examples."""
    rule = SymmetryRule(
        name="horizontal_symmetry",
        description="Check horizontal text symmetry",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cache_size=100,
            cost=1.0,
            metadata={
                "mirror_mode": "horizontal",
                "preserve_whitespace": False,
                "preserve_case": False,
                "ignore_punctuation": True,
                "symmetry_threshold": 0.7,
            },
        ),
    )

    # Test palindromes
    palindromes = [
        "A man a plan a canal Panama",
        "Never odd or even",
        "Do geese see God?",
        "Madam, I'm Adam",
    ]

    print("\nTesting Palindromes:")
    print("-------------------")
    for text in palindromes:
        result = rule.validate(text)
        print(f"Text: {text}")
        print(f"Symmetry Score: {result.metadata['symmetry_score']:.2f}")
        print(f"Passed: {result.passed}\n")


def test_vertical_symmetry():
    """Test vertical symmetry with ASCII art and layouts."""
    rule = SymmetryRule(
        name="vertical_symmetry",
        description="Check vertical text symmetry",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cache_size=100,
            cost=1.0,
            metadata={
                "mirror_mode": "vertical",
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": False,
                "symmetry_threshold": 0.7,
            },
        ),
    )

    # Test ASCII art
    diamond = """
    /\\
   /  \\
  /    \\
 /      \\
----------
 \\      /
  \\    /
   \\  /
    \\/
"""

    ui_layout = """
+-------------+
|    Title    |
|  Subtitle   |
|-------------|
| Left | Right|
| Data | Info |
|-------------|
|   Footer    |
+-------------+
"""

    print("\nTesting ASCII Art:")
    print("----------------")
    result = rule.validate(diamond)
    print("Diamond Pattern:")
    print(diamond)
    print(f"Symmetry Score: {result.metadata['symmetry_score']:.2f}")
    print(f"Passed: {result.passed}\n")

    print("\nTesting UI Layout:")
    print("----------------")
    result = rule.validate(ui_layout)
    print("UI Layout:")
    print(ui_layout)
    print(f"Symmetry Score: {result.metadata['symmetry_score']:.2f}")
    print(f"Passed: {result.passed}\n")


def test_logo_symmetry():
    """Test symmetry in potential logo designs."""
    rule = SymmetryRule(
        name="logo_symmetry",
        description="Check logo text symmetry",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            cache_size=100,
            cost=1.0,
            metadata={
                "mirror_mode": "horizontal",
                "preserve_whitespace": True,
                "preserve_case": True,
                "ignore_punctuation": False,
                "symmetry_threshold": 0.8,
            },
        ),
    )

    logos = [
        "NOON",
        "TOYOTA",
        "AXA",
        "XEROX",
    ]

    print("\nTesting Logo Symmetry:")
    print("--------------------")
    for logo in logos:
        result = rule.validate(logo)
        print(f"Logo: {logo}")
        print(f"Symmetry Score: {result.metadata['symmetry_score']:.2f}")
        print(f"Passed: {result.passed}\n")


if __name__ == "__main__":
    test_horizontal_symmetry()
    test_vertical_symmetry()
    test_logo_symmetry()
