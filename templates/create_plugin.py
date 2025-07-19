#!/usr/bin/env python3
"""Simple plugin template generator for Sifaka.

Usage:
    python create_plugin.py critic my_critic "My Critic Plugin" "John Doe"
    python create_plugin.py validator my_validator "My Validator Plugin" "Jane Smith"
"""

import os
import sys

CRITIC_TEMPLATE = '''"""{{description}} critic plugin for Sifaka.

This plugin implements a critic that analyzes text for specific issues.
"""

import logging
from typing import Dict, Any, List

from sifaka.core.plugin_interfaces import CriticPlugin, PluginMetadata, PluginType
from sifaka.core.models import CritiqueResult, SifakaResult

logger = logging.getLogger(__name__)


class {{class_name}}(CriticPlugin):
    """{{description}} critic plugin."""

    def __init__(self) -> None:
        """Initialize the {{name}} critic plugin."""
        super().__init__()

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="{{name}}_critic",
            version="1.0.0",
            author="{{author}}",
            description="{{description}}",
            plugin_type=PluginType.CRITIC,
            dependencies=[],
            sifaka_version=">=0.1.0",
            python_version=">=3.10",
            license="MIT",
            keywords=["{{name}}", "critic", "sifaka"],
            default_config={
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        )

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Analyze text and provide feedback.

        Args:
            text: The text to analyze
            result: The complete SifakaResult with history

        Returns:
            CritiqueResult with feedback and suggestions
        """
        try:
            # TODO: Implement your critique logic here
            issues = self._analyze_text(text)
            suggestions = self._generate_suggestions(issues)

            needs_improvement = len(issues) > 0
            confidence = 0.8 if needs_improvement else 0.9

            if needs_improvement:
                feedback = f"Found {len(issues)} issues: {', '.join(issues)}"
            else:
                feedback = "Text meets quality standards."

            return CritiqueResult(
                critic=self.name,
                feedback=feedback,
                suggestions=suggestions,
                needs_improvement=needs_improvement,
                confidence=confidence,
                metadata={"issues_found": len(issues)}
            )

        except Exception as e:
            logger.error(f"Error in {{name}} critic: {e}")
            return CritiqueResult(
                critic=self.name,
                feedback=f"Error during analysis: {str(e)}",
                suggestions=["Please check the input text and try again."],
                needs_improvement=False,
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _analyze_text(self, text: str) -> List[str]:
        """Analyze text for issues.

        TODO: Replace this with your actual analysis logic.
        """
        issues = []

        # Example checks - replace with your logic
        if len(text.split()) < 10:
            issues.append("text is too short")

        if not text.strip():
            issues.append("text is empty")

        return issues

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        for issue in issues:
            if "too short" in issue:
                suggestions.append("Consider expanding the text with more detail.")
            elif "empty" in issue:
                suggestions.append("Please provide some text content.")

        return suggestions
'''

VALIDATOR_TEMPLATE = '''"""{{description}} validator plugin for Sifaka.

This plugin validates text against specific quality criteria.
"""

import logging
from typing import Dict, Any

from sifaka.core.plugin_interfaces import ValidatorPlugin, PluginMetadata, PluginType
from sifaka.core.models import ValidationResult, SifakaResult

logger = logging.getLogger(__name__)


class {{class_name}}(ValidatorPlugin):
    """{{description}} validator plugin."""

    def __init__(self) -> None:
        """Initialize the {{name}} validator plugin."""
        super().__init__()
        self._min_words = 10

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="{{name}}_validator",
            version="1.0.0",
            author="{{author}}",
            description="{{description}}",
            plugin_type=PluginType.VALIDATOR,
            dependencies=[],
            sifaka_version=">=0.1.0",
            python_version=">=3.10",
            license="MIT",
            keywords=["{{name}}", "validator", "sifaka"],
            default_config={
                "min_words": 10,
            }
        )

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text quality.

        Args:
            text: The text to validate
            result: The complete SifakaResult with history

        Returns:
            ValidationResult with validation status
        """
        try:
            issues = []
            score = 1.0

            # TODO: Implement your validation logic here
            words = text.split()
            if len(words) < self._min_words:
                issues.append(f"Text has only {len(words)} words (minimum: {self._min_words})")
                score -= 0.5

            # Ensure score doesn't go below 0
            score = max(0.0, score)
            passed = score > 0.5

            if issues:
                details = f"Validation issues: {'; '.join(issues)}"
            else:
                details = "Text passes all validation checks."

            return ValidationResult(
                validator=self.name,
                passed=passed,
                score=score,
                details=details,
                metadata={
                    "word_count": len(words),
                    "issues_count": len(issues),
                }
            )

        except Exception as e:
            logger.error(f"Error in {{name}} validator: {e}")
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details=f"Error during validation: {str(e)}",
                metadata={"error": str(e)}
            )

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        if "min_words" in config:
            if not isinstance(config["min_words"], int) or config["min_words"] < 1:
                raise ValueError("min_words must be a positive integer")
        return True

    def _on_initialize(self) -> None:
        """Initialize plugin-specific settings."""
        self._min_words = self.validation_config.get("min_words", 10)
        logger.info(f"{{class_name}} initialized: min_words={self._min_words}")
'''

TEST_TEMPLATE = '''"""Tests for {{name}} {{plugin_type}} plugin."""

import pytest
from datetime import datetime

from {{name}}_{{plugin_type}} import {{class_name}}
from sifaka.core.models import SifakaResult
from sifaka.core.plugin_interfaces import PluginType, PluginStatus


class Test{{class_name}}:
    """Test {{name}} {{plugin_type}} plugin."""

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        plugin = {{class_name}}()
        metadata = plugin.metadata

        assert metadata.name == "{{name}}_{{plugin_type}}"
        assert metadata.version == "1.0.0"
        assert metadata.author == "{{author}}"
        assert metadata.plugin_type == PluginType.{{plugin_type_upper}}

    def test_plugin_lifecycle(self):
        """Test plugin lifecycle management."""
        plugin = {{class_name}}()

        # Initial state
        assert plugin.status == PluginStatus.LOADED

        # Initialize
        plugin.initialize()
        assert plugin.status == PluginStatus.INITIALIZED

        # Activate
        plugin.activate()
        assert plugin.status == PluginStatus.ACTIVE

        # Deactivate
        plugin.deactivate()
        assert plugin.status == PluginStatus.INITIALIZED

        # Cleanup
        plugin.cleanup()
        assert plugin.status == PluginStatus.DISABLED

    @pytest.mark.asyncio
    async def test_{{plugin_type}}_functionality(self):
        """Test {{plugin_type}} functionality."""
        plugin = {{class_name}}()
        plugin.initialize()

        result = self._create_mock_result()

        {% if plugin_type == "critic" %}
        critique = await plugin.critique("Test text.", result)

        assert critique.critic == "{{name}}_critic"
        assert isinstance(critique.needs_improvement, bool)
        assert isinstance(critique.confidence, float)
        assert 0.0 <= critique.confidence <= 1.0
        {% else %}
        validation = await plugin.validate("Test text.", result)

        assert validation.validator == "{{name}}_validator"
        assert isinstance(validation.passed, bool)
        assert isinstance(validation.score, float)
        assert 0.0 <= validation.score <= 1.0
        {% endif %}

    def _create_mock_result(self) -> SifakaResult:
        """Create a mock SifakaResult for testing."""
        return SifakaResult(
            id="test_id",
            original_text="test text",
            final_text="test text",
            iteration=1,
            processing_time=0.1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            generations=[],
            critiques=[],
            validations=[],
        )
'''

PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{{name}}-{{plugin_type}}"
version = "1.0.0"
description = "{{description}}"
authors = [
    {name = "{{author}}", email = "your.email@example.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
keywords = ["{{name}}", "{{plugin_type}}", "sifaka", "ai"]

dependencies = [
    "sifaka>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0",
    "black>=22.0",
    "mypy>=1.0",
    "ruff>=0.3.0",
]

[project.entry-points."sifaka.{{plugin_type}}s"]
{{name}}_{{plugin_type}} = "{{name}}_{{plugin_type}}:{{class_name}}"

[tool.setuptools.packages.find]
where = ["."]
include = ["{{name}}_{{plugin_type}}*"]

[tool.black]
line-length = 88

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = ["--strict-markers", "--cov={{name}}_{{plugin_type}}"]
testpaths = ["tests"]
asyncio_mode = "auto"
"""

README_TEMPLATE = """# {{name.title()}} {{plugin_type.title()}} Plugin

{{description}}

## Installation

```bash
pip install {{name}}-{{plugin_type}}
```

## Usage

Once installed, this plugin will be automatically discovered by Sifaka:

```python
from sifaka.api import improve_text
from sifaka.core.config import Config

config = Config(
    {% if plugin_type == "critic" %}critics=["{{name}}_{{plugin_type}}"]{% else %}validators=["{{name}}_{{plugin_type}}"]{% endif %},
    model="gpt-4o-mini"
)

result = improve_text("Your text here", config=config)
```

## Development

```bash
git clone <your-repo>
cd {{name}}-{{plugin_type}}
pip install -e ".[dev]"
pytest
```

## License

MIT
"""


def to_class_name(name):
    """Convert plugin name to class name."""
    parts = name.replace("-", "_").replace(" ", "_").split("_")
    return "".join(word.capitalize() for word in parts)


def create_plugin(plugin_type, name, description, author):
    """Create a plugin from template."""
    if plugin_type not in ["critic", "validator"]:
        print("Error: plugin_type must be 'critic' or 'validator'")
        sys.exit(1)

    # Create directory structure
    plugin_dir = f"{name}_{plugin_type}"
    package_dir = plugin_dir
    tests_dir = f"{plugin_dir}/tests"

    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)

    class_name = to_class_name(name) + plugin_type.title()

    # Template variables
    vars = {
        "name": name,
        "class_name": class_name,
        "description": description,
        "author": author,
        "plugin_type": plugin_type,
        "plugin_type_upper": plugin_type.upper(),
    }

    # Choose template
    template = CRITIC_TEMPLATE if plugin_type == "critic" else VALIDATOR_TEMPLATE

    # Generate files
    files = {
        f"{package_dir}/__init__.py": '"""{description}"""\n\nfrom .plugin import {class_name}\n\n__version__ = "1.0.0"\n__all__ = ["{class_name}"]',
        f"{package_dir}/plugin.py": template,
        f"{package_dir}/py.typed": "# Marker file for PEP 561",
        f"{tests_dir}/__init__.py": '"""Tests for {{name}} {{plugin_type}} plugin."""',
        f"{tests_dir}/test_{name}_{plugin_type}.py": TEST_TEMPLATE,
        f"{plugin_dir}/pyproject.toml": PYPROJECT_TEMPLATE,
        f"{plugin_dir}/README.md": README_TEMPLATE,
    }

    for file_path, content in files.items():
        # Replace template variables
        for key, value in vars.items():
            content = content.replace("{{" + key + "}}", str(value))

        # Handle conditional blocks
        if plugin_type == "critic":
            content = (
                content.replace('{% if plugin_type == "critic" %}', "")
                .replace("{% else %}", "<!--")
                .replace("{% endif %}", "-->")
            )
        else:
            content = (
                content.replace('{% if plugin_type == "critic" %}', "<!--")
                .replace("{% else %}", "")
                .replace("{% endif %}", "-->")
            )

        # Clean up HTML comments
        content = content.replace("<!--", "").replace("-->", "")

        with open(file_path, "w") as f:
            f.write(content)

    print(f"Created {plugin_type} plugin '{name}' in directory: {plugin_dir}")
    print("To get started:")
    print(f"  cd {plugin_dir}")
    print("  pip install -e '.[dev]'")
    print("  pytest")


def main():
    """Main entry point."""
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    _, plugin_type, name, description, author = sys.argv
    create_plugin(plugin_type, name, description, author)


if __name__ == "__main__":
    main()
