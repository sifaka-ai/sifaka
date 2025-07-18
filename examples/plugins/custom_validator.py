"""Example custom validator plugin for Sifaka.

This example shows how to create domain-specific validators
that ensure generated text meets specific requirements.
"""

import re
from typing import Any, Dict, List, Optional

from sifaka.core.models import ValidationResult
from sifaka.validators.base import BaseValidator


class SEOValidator(BaseValidator):
    """Validator for SEO-optimized content.

    Ensures content meets SEO best practices:
    - Keyword density
    - Meta description length
    - Header structure
    - Readability score
    """

    def __init__(
        self,
        target_keywords: List[str],
        min_keyword_density: float = 0.01,  # 1%
        max_keyword_density: float = 0.03,  # 3%
        ideal_length: tuple = (300, 2000),  # words
        require_headers: bool = True,
        **kwargs,
    ):
        """Initialize SEO validator.

        Args:
            target_keywords: Keywords to check for
            min_keyword_density: Minimum keyword density
            max_keyword_density: Maximum keyword density
            ideal_length: Ideal word count range (min, max)
            require_headers: Whether to require header tags
        """
        super().__init__(**kwargs)
        self.target_keywords = [kw.lower() for kw in target_keywords]
        self.min_keyword_density = min_keyword_density
        self.max_keyword_density = max_keyword_density
        self.ideal_length = ideal_length
        self.require_headers = require_headers

    async def validate(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate text for SEO optimization."""
        issues = []
        suggestions = []

        # Word count check
        words = text.split()
        word_count = len(words)

        if word_count < self.ideal_length[0]:
            issues.append(
                f"Content too short: {word_count} words (minimum: {self.ideal_length[0]})"
            )
            suggestions.append("Expand content with more detailed information")
        elif word_count > self.ideal_length[1]:
            issues.append(
                f"Content too long: {word_count} words (maximum: {self.ideal_length[1]})"
            )
            suggestions.append("Consider breaking into multiple pages or sections")

        # Keyword density check
        text_lower = text.lower()
        for keyword in self.target_keywords:
            keyword_count = text_lower.count(keyword)
            density = keyword_count / word_count if word_count > 0 else 0

            if density < self.min_keyword_density:
                issues.append(f"Low keyword density for '{keyword}': {density:.2%}")
                suggestions.append(f"Naturally incorporate '{keyword}' more frequently")
            elif density > self.max_keyword_density:
                issues.append(f"High keyword density for '{keyword}': {density:.2%}")
                suggestions.append(
                    f"Reduce repetition of '{keyword}' to avoid keyword stuffing"
                )

        # Header structure check
        if self.require_headers:
            # Check for H1
            h1_pattern = r"#\s+[^#\n]+|<h1[^>]*>.*?</h1>"
            h1_matches = re.findall(h1_pattern, text, re.IGNORECASE)

            if not h1_matches:
                issues.append("Missing H1 header")
                suggestions.append("Add a main title (H1) at the beginning")
            elif len(h1_matches) > 1:
                issues.append("Multiple H1 headers found")
                suggestions.append("Use only one H1 per page")

            # Check for H2/H3 structure
            h2_pattern = r"##\s+[^#\n]+|<h2[^>]*>.*?</h2>"
            h2_matches = re.findall(h2_pattern, text, re.IGNORECASE)

            if len(h2_matches) < 2:
                suggestions.append(
                    "Consider adding more subheadings (H2) for better structure"
                )

        # Meta description check (if in metadata)
        if metadata and "meta_description" in metadata:
            meta_desc = metadata["meta_description"]
            if len(meta_desc) < 120:
                issues.append("Meta description too short")
                suggestions.append("Expand meta description to 120-160 characters")
            elif len(meta_desc) > 160:
                issues.append("Meta description too long")
                suggestions.append("Shorten meta description to under 160 characters")

        # Create validation result
        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            validator_name="seo_validator",
            issues=issues,
            suggestions=suggestions,
            metadata={
                "word_count": word_count,
                "keywords_checked": self.target_keywords,
                "header_count": (
                    len(h1_matches) + len(h2_matches) if self.require_headers else 0
                ),
            },
        )


class CodeQualityValidator(BaseValidator):
    """Validator for code snippets in documentation.

    Ensures code examples are:
    - Syntactically correct
    - Follow style guidelines
    - Include proper error handling
    - Have appropriate comments
    """

    def __init__(
        self,
        language: str = "python",
        check_syntax: bool = True,
        require_docstrings: bool = True,
        max_line_length: int = 88,
        **kwargs,
    ):
        """Initialize code quality validator."""
        super().__init__(**kwargs)
        self.language = language
        self.check_syntax = check_syntax
        self.require_docstrings = require_docstrings
        self.max_line_length = max_line_length

    async def validate(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate code quality in text."""
        issues = []
        suggestions = []

        # Extract code blocks
        code_blocks = self._extract_code_blocks(text)

        if not code_blocks:
            return ValidationResult(
                is_valid=True,
                validator_name="code_quality",
                issues=[],
                suggestions=["No code blocks found to validate"],
                metadata={"code_blocks": 0},
            )

        for i, code in enumerate(code_blocks):
            block_issues = self._validate_code_block(code, i)
            issues.extend(block_issues)

        # General suggestions
        if issues:
            suggestions.append("Fix code quality issues before proceeding")
            suggestions.append("Consider running a linter on code examples")

        return ValidationResult(
            is_valid=len(issues) == 0,
            validator_name="code_quality",
            issues=issues,
            suggestions=suggestions,
            metadata={"code_blocks": len(code_blocks), "language": self.language},
        )

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text."""
        # Markdown code blocks
        pattern = r"```(?:\w+)?\n(.*?)\n```"
        blocks = re.findall(pattern, text, re.DOTALL)

        # Also check for indented code blocks
        lines = text.split("\n")
        current_block = []
        in_block = False

        for line in lines:
            if line.startswith("    ") and line.strip():
                in_block = True
                current_block.append(line[4:])
            else:
                if in_block and current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                in_block = False

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    def _validate_code_block(self, code: str, block_num: int) -> List[str]:
        """Validate a single code block."""
        issues = []

        if self.language == "python":
            # Basic Python validation
            if self.check_syntax:
                try:
                    compile(code, f"<block_{block_num}>", "exec")
                except SyntaxError as e:
                    issues.append(f"Syntax error in code block {block_num + 1}: {e}")

            # Line length check
            for line_num, line in enumerate(code.split("\n")):
                if len(line) > self.max_line_length:
                    issues.append(
                        f"Line too long in block {block_num + 1}, "
                        f"line {line_num + 1}: {len(line)} characters"
                    )

            # Docstring check for functions/classes
            if self.require_docstrings:
                if "def " in code or "class " in code:
                    if '"""' not in code and "'''" not in code:
                        issues.append(
                            f"Missing docstrings in code block {block_num + 1}"
                        )

        return issues


class AccessibilityValidator(BaseValidator):
    """Validator for web content accessibility.

    Ensures content meets WCAG guidelines:
    - Alt text for images
    - Proper heading hierarchy
    - Link text clarity
    - Color contrast mentions
    """

    def __init__(self, wcag_level: str = "AA", **kwargs):
        """Initialize accessibility validator."""
        super().__init__(**kwargs)
        self.wcag_level = wcag_level

    async def validate(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate text for accessibility."""
        issues = []
        suggestions = []

        # Check for images without alt text
        img_pattern = r"<img[^>]+>"
        img_tags = re.findall(img_pattern, text, re.IGNORECASE)

        for img in img_tags:
            if "alt=" not in img:
                issues.append("Image missing alt text")
                suggestions.append("Add descriptive alt text to all images")

        # Check link text
        link_pattern = r"<a[^>]+>(.*?)</a>|\[([^\]]+)\]\([^)]+\)"
        links = re.findall(link_pattern, text, re.IGNORECASE)

        for link in links:
            link_text = link[0] if link[0] else link[1]
            if link_text.lower() in ["click here", "here", "read more", "link"]:
                issues.append(f"Non-descriptive link text: '{link_text}'")
                suggestions.append(
                    "Use descriptive link text that explains the destination"
                )

        # Check heading hierarchy
        headings = re.findall(r"<h(\d)[^>]*>|^(#{1,6})\s", text, re.MULTILINE)
        if headings:
            levels = [int(h[0]) if h[0] else len(h[1]) for h in headings]
            for i in range(1, len(levels)):
                if levels[i] - levels[i - 1] > 1:
                    issues.append("Skipped heading level")
                    suggestions.append(
                        "Maintain proper heading hierarchy (don't skip levels)"
                    )

        return ValidationResult(
            is_valid=len(issues) == 0,
            validator_name="accessibility",
            issues=issues,
            suggestions=suggestions,
            metadata={
                "wcag_level": self.wcag_level,
                "images_found": len(img_tags),
                "links_found": len(links),
            },
        )


# To register these validators:
# 1. Manual registration:
# from sifaka.validators import register_validator
# register_validator("seo", SEOValidator)
# register_validator("code_quality", CodeQualityValidator)
# register_validator("accessibility", AccessibilityValidator)

# 2. Or via entry points in setup.py:
#    entry_points={
#        "sifaka.validators": [
#            "seo = my_plugin:SEOValidator",
#            "code_quality = my_plugin:CodeQualityValidator",
#            "accessibility = my_plugin:AccessibilityValidator",
#        ],
#    }


if __name__ == "__main__":
    # Example usage
    import asyncio

    # Example 1: SEO Validation
    seo_validator = SEOValidator(
        target_keywords=["renewable energy", "solar power"],
        min_keyword_density=0.01,
        max_keyword_density=0.03,
    )

    content = """
    # Renewable Energy: The Future of Power

    Renewable energy is transforming how we generate electricity.
    Solar power and wind energy are becoming increasingly affordable.

    ## Benefits of Solar Power

    Solar panels convert sunlight directly into electricity...
    """

    print("SEO Validation Example:")
    result = asyncio.run(seo_validator.validate(content))
    print(f"Valid: {result.is_valid}")
    print(f"Issues: {result.issues}")
    print(f"Suggestions: {result.suggestions}")

    # Example 2: Code Quality Validation
    code_validator = CodeQualityValidator(
        language="python", check_syntax=True, require_docstrings=True
    )

    doc_with_code = """
    Here's how to use the function:

    ```python
    def calculate_sum(numbers):
        total = 0
        for num in numbers:
            total += num
        return total
    ```
    """

    print("\nCode Quality Validation Example:")
    result = asyncio.run(code_validator.validate(doc_with_code))
    print(f"Valid: {result.is_valid}")
    print(f"Issues: {result.issues}")
