"""Test module imports work correctly."""


def test_main_imports():
    """Test that main API functions can be imported."""
    from sifaka import improve, improve_sync

    assert callable(improve)
    assert callable(improve_sync)


def test_core_imports():
    """Test core module imports."""
    from sifaka.core import Config, SifakaEngine
    from sifaka.core.models import SifakaResult
    from sifaka.core.types import CriticType

    assert Config is not None
    assert SifakaEngine is not None
    assert SifakaResult is not None
    assert CriticType is not None


def test_critic_imports():
    """Test critic imports."""
    from sifaka.critics import (
        ConstitutionalCritic,
        ReflexionCritic,
        SelfRefineCritic,
        StyleCritic,
    )

    assert ReflexionCritic is not None
    assert SelfRefineCritic is not None
    assert ConstitutionalCritic is not None
    assert StyleCritic is not None


def test_validator_imports():
    """Test validator imports."""
    from sifaka.validators import (
        ComposableValidator,
        ContentValidator,
        LengthValidator,
        PatternValidator,
    )

    assert LengthValidator is not None
    assert ContentValidator is not None
    assert PatternValidator is not None
    assert ComposableValidator is not None


def test_storage_imports():
    """Test storage imports."""
    from sifaka.storage import FileStorage, MemoryStorage

    assert FileStorage is not None
    assert MemoryStorage is not None


def test_exception_imports():
    """Test exception imports."""
    from sifaka.core.exceptions import (
        ConfigurationError,
        ModelProviderError,
        SifakaError,
        ValidationError,
    )

    assert SifakaError is not None
    assert ConfigurationError is not None
    assert ValidationError is not None
    assert ModelProviderError is not None
