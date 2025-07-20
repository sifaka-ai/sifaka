"""Simple tests to improve coverage."""


def test_api_module_imports():
    """Test that API module can be imported."""
    from sifaka import api

    # Test that functions are accessible
    assert hasattr(api, "improve")
    assert hasattr(api, "improve_sync")
    assert callable(api.improve)
    assert callable(api.improve_sync)


def test_config_module_imports():
    """Test config module structure."""
    from sifaka.core import config

    # Test that config classes are accessible
    assert hasattr(config, "Config")
    assert hasattr(config, "LLMConfig")
    assert hasattr(config, "CriticConfig")


def test_exception_module_imports():
    """Test exception module structure."""
    from sifaka.core import exceptions

    # Test that main exceptions are accessible
    assert hasattr(exceptions, "SifakaError")
    assert hasattr(exceptions, "ValidationError")
    assert hasattr(exceptions, "ModelProviderError")


def test_tools_module_structure():
    """Test tools module structure."""
    from sifaka import tools

    # Test that registry is accessible
    assert hasattr(tools, "registry")
    assert hasattr(tools, "types")


def test_storage_module_structure():
    """Test storage module imports."""
    from sifaka import storage

    # Test that storage classes are accessible
    assert hasattr(storage, "MemoryStorage")
    assert hasattr(storage, "FileStorage")


def test_critics_module_structure():
    """Test critics module imports."""
    from sifaka import critics

    # Test that main critics are accessible
    assert hasattr(critics, "ReflexionCritic")
    assert hasattr(critics, "SelfRefineCritic")


def test_validators_module_structure():
    """Test validators module imports."""
    from sifaka import validators

    # Test that validators are accessible
    assert hasattr(validators, "LengthValidator")
    assert hasattr(validators, "ContentValidator")


def test_type_defs_simple():
    """Test type definitions module."""
    from sifaka.core.type_defs import ToolResultItem

    # Test that type definition is accessible
    assert ToolResultItem is not None


def test_constants_module():
    """Test constants module structure."""
    from sifaka.core import constants

    # Test that constants are accessible
    assert hasattr(constants, "DEFAULT_MAX_ITERATIONS")
    assert hasattr(constants, "DEFAULT_TEMPERATURE")

    # Test that values are reasonable
    assert constants.DEFAULT_MAX_ITERATIONS > 0
    assert 0 <= constants.DEFAULT_TEMPERATURE <= 2.0


def test_retry_module_structure():
    """Test retry module structure."""
    from sifaka.core import retry

    # Test that retry functions are accessible
    assert hasattr(retry, "with_retry")
    assert callable(retry.with_retry)

    # Test that retry configs are accessible
    assert hasattr(retry, "RETRY_STANDARD")
    assert hasattr(retry, "RETRY_QUICK")


def test_llm_config_defaults():
    """Test LLM config default values."""
    from sifaka.core.config import LLMConfig

    # Test that we can create default config
    config = LLMConfig()
    assert config is not None
    assert hasattr(config, "model")
    assert hasattr(config, "temperature")


def test_critic_config_defaults():
    """Test critic config default values."""
    from sifaka.core.config import CriticConfig

    # Test that we can create default config
    config = CriticConfig()
    assert config is not None
    assert hasattr(config, "critics")


def test_engine_config_defaults():
    """Test engine config default values."""
    from sifaka.core.config import EngineConfig

    # Test that we can create default config
    config = EngineConfig()
    assert config is not None
    assert hasattr(config, "max_iterations")
