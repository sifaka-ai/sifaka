"""Tests for wrapper rules."""

import pytest

from sifaka.rules.wrapper import (
    DEFAULT_LANGUAGE_CONFIGS,
    DEFAULT_TEMPLATES,
    CodeBlockConfig,
    CodeBlockRule,
    WrapperConfig,
    WrapperRule,
    create_code_block_rule,
    create_wrapper_rule,
)


@pytest.fixture
def wrapper_config() -> WrapperConfig:
    """Create a test wrapper configuration."""
    return WrapperConfig(
        prefix="<start>",
        suffix="</end>",
        template=None,
        strip_whitespace=True,
        preserve_newlines=True,
        cache_size=100,
        priority=1,
        cost=1.0,
    )

@pytest.fixture
def code_block_config() -> CodeBlockConfig:
    """Create a test code block configuration."""
    return CodeBlockConfig(
        language="python",
        indent_size=4,
        add_syntax_markers=True,
        preserve_indentation=True,
        cache_size=100,
        priority=1,
        cost=1.0,
    )

def test_wrapper_config_validation():
    """Test wrapper configuration validation."""
    # Valid configuration with prefix/suffix
    config = WrapperConfig(prefix="<", suffix=">")
    assert config.prefix == "<"
    assert config.suffix == ">"

    # Valid configuration with template
    config = WrapperConfig(template="[{content}]")
    assert config.template == "[{content}]"

    # Invalid template
    with pytest.raises(ValueError, match="Template must contain '{content}' placeholder"):
        WrapperConfig(template="[invalid]")

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        WrapperConfig(cache_size=-1)

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        WrapperConfig(priority=-1)

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        WrapperConfig(cost=-1)

def test_code_block_config_validation():
    """Test code block configuration validation."""
    # Valid configuration
    config = CodeBlockConfig(language="python")
    assert config.language == "python"
    assert config.indent_size == 4

    # Empty language
    with pytest.raises(ValueError, match="Language must be specified"):
        CodeBlockConfig(language="")

    # Invalid indent size
    with pytest.raises(ValueError, match="Indent size must be non-negative"):
        CodeBlockConfig(language="python", indent_size=-1)

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        CodeBlockConfig(language="python", cache_size=-1)

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        CodeBlockConfig(language="python", priority=-1)

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        CodeBlockConfig(language="python", cost=-1)

def test_wrapper_rule_validation(wrapper_config):
    """Test wrapper rule validation."""
    rule = create_wrapper_rule(
        name="test_wrapper",
        description="Test wrapper rule",
        prefix=wrapper_config.prefix,
        suffix=wrapper_config.suffix,
    )

    # Test basic wrapping
    result = rule._validate_impl("Hello World")
    assert result.passed
    assert "Text wrapped successfully" in result.message
    assert result.metadata["wrapped_text"] == "<start>Hello World</end>"
    assert result.metadata["has_prefix"]
    assert result.metadata["has_suffix"]
    assert not result.metadata["used_template"]

    # Test with whitespace
    result = rule._validate_impl("  Hello World  ")
    assert result.passed
    assert result.metadata["wrapped_text"] == "<start>Hello World</end>"

    # Test with newlines
    result = rule._validate_impl("Hello\nWorld")
    assert result.passed
    assert "Hello\nWorld" in result.metadata["wrapped_text"]

    # Test with template
    template_rule = create_wrapper_rule(
        name="template_test",
        description="Template test rule",
        template="[{content}]",
    )
    result = template_rule._validate_impl("Hello World")
    assert result.passed
    assert result.metadata["wrapped_text"] == "[Hello World]"
    assert result.metadata["used_template"]

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore

def test_code_block_rule_validation(code_block_config):
    """Test code block rule validation."""
    rule = create_code_block_rule(
        name="test_code_block",
        description="Test code block rule",
        language=code_block_config.language,
        indent_size=code_block_config.indent_size,
    )

    # Test basic code block
    code = "def hello():\n    print('Hello World')"
    result = rule._validate_impl(code)
    assert result.passed
    assert "Code block formatted successfully" in result.message
    assert "```python" in result.metadata["formatted_text"]
    assert result.metadata["line_count"] == 2
    assert result.metadata["has_syntax_markers"]

    # Test indentation preservation
    code = "def hello():\n        print('Hello World')"  # Extra indentation
    result = rule._validate_impl(code)
    assert result.passed
    assert "    print('Hello World')" in result.metadata["formatted_text"]

    # Test without syntax markers
    no_markers_rule = create_code_block_rule(
        name="no_markers",
        description="No markers rule",
        language="python",
        add_syntax_markers=False,
    )
    result = no_markers_rule._validate_impl(code)
    assert result.passed
    assert "```" not in result.metadata["formatted_text"]
    assert not result.metadata["has_syntax_markers"]

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore

def test_create_wrapper_rule():
    """Test wrapper rule factory function."""
    # Test with default configuration
    rule = create_wrapper_rule(
        name="test_wrapper",
        description="Test wrapper rule",
    )
    assert isinstance(rule, WrapperRule)
    assert rule.name == "test_wrapper"
    assert rule.description == "Test wrapper rule"

    # Test with custom configuration
    custom_rule = create_wrapper_rule(
        name="custom_wrapper",
        description="Custom wrapper rule",
        prefix="<<",
        suffix=">>",
        strip_whitespace=False,
    )
    assert isinstance(custom_rule, WrapperRule)
    assert custom_rule.name == "custom_wrapper"
    assert custom_rule.description == "Custom wrapper rule"

def test_create_code_block_rule():
    """Test code block rule factory function."""
    # Test with default configuration
    rule = create_code_block_rule(
        name="test_code_block",
        description="Test code block rule",
        language="python",
    )
    assert isinstance(rule, CodeBlockRule)
    assert rule.name == "test_code_block"
    assert rule.description == "Test code block rule"

    # Test with custom configuration
    custom_rule = create_code_block_rule(
        name="custom_code_block",
        description="Custom code block rule",
        language="javascript",
        indent_size=2,
        add_syntax_markers=False,
    )
    assert isinstance(custom_rule, CodeBlockRule)
    assert custom_rule.name == "custom_code_block"
    assert custom_rule.description == "Custom code block rule"

def test_default_templates():
    """Test default templates."""
    assert isinstance(DEFAULT_TEMPLATES, dict)
    assert len(DEFAULT_TEMPLATES) > 0
    assert "{content}" in DEFAULT_TEMPLATES["quote"]
    assert "{content}" in DEFAULT_TEMPLATES["bold"]
    assert "{content}" in DEFAULT_TEMPLATES["code"]

    # Test template formatting
    text = "Hello World"
    assert DEFAULT_TEMPLATES["quote"].format(content=text) == "> Hello World"
    assert DEFAULT_TEMPLATES["bold"].format(content=text) == "**Hello World**"
    assert DEFAULT_TEMPLATES["code"].format(content=text) == "`Hello World`"

def test_default_language_configs():
    """Test default language configurations."""
    assert isinstance(DEFAULT_LANGUAGE_CONFIGS, dict)
    assert len(DEFAULT_LANGUAGE_CONFIGS) > 0

    # Test Python config
    python_config = DEFAULT_LANGUAGE_CONFIGS["python"]
    assert python_config["indent_size"] == 4
    assert python_config["add_syntax_markers"] is True

    # Test JavaScript config
    js_config = DEFAULT_LANGUAGE_CONFIGS["javascript"]
    assert js_config["indent_size"] == 2
    assert js_config["add_syntax_markers"] is True

    # Test Markdown config
    md_config = DEFAULT_LANGUAGE_CONFIGS["markdown"]
    assert md_config["indent_size"] == 2
    assert md_config["add_syntax_markers"] is False

def test_template_based_wrapping():
    """Test wrapping with different templates."""
    for template_name, template in DEFAULT_TEMPLATES.items():
        rule = create_wrapper_rule(
            name=f"test_{template_name}",
            description=f"Test {template_name} template",
            template=template,
        )
        result = rule._validate_impl("Test content")
        assert result.passed
        assert result.metadata["used_template"]
        assert "Test content" in result.metadata["wrapped_text"]

def test_language_specific_formatting():
    """Test formatting with different language configurations."""
    for language, config in DEFAULT_LANGUAGE_CONFIGS.items():
        rule = create_code_block_rule(
            name=f"test_{language}",
            description=f"Test {language} formatting",
            language=language,
        )
        code = "function test() {\n  console.log('test');\n}"
        result = rule._validate_impl(code)
        assert result.passed
        assert result.metadata["language"] == language
        if config["add_syntax_markers"]:
            assert f"```{language}" in result.metadata["formatted_text"]
