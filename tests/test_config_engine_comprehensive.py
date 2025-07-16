"""Comprehensive tests for Config class and SifakaEngine.

This test suite covers:
- Config class initialization and validation
- Config factory methods (fast, quality, creative)
- Config serialization and deserialization
- SifakaEngine initialization and configuration
- Engine-config interactions
- Component integration through engine
- Error handling and edge cases
"""

import asyncio
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from sifaka.core.config import Config
from sifaka.core.engine import SifakaEngine
from sifaka.core.models import SifakaResult
from sifaka.storage import MemoryStorage
from sifaka.validators import LengthValidator, ContentValidator


class TestConfigInitialization:
    """Test Config class initialization and validation."""

    def test_config_default_initialization(self):
        """Test Config with default values."""
        config = Config()

        # Check default values
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_iterations == 3
        assert config.timeout_seconds == 300
        assert config.critics == ["reflexion"]
        assert config.force_improvements is False

    def test_config_custom_initialization(self):
        """Test Config with custom values."""
        config = Config(
            model="gpt-4o",
            temperature=0.9,
            max_iterations=5,
            timeout_seconds=600,
            critics=["constitutional", "self_refine"],
            force_improvements=True,
            style_description="Academic writing style",
        )

        assert config.model == "gpt-4o"
        assert config.temperature == 0.9
        assert config.max_iterations == 5
        assert config.timeout_seconds == 600
        assert config.critics == ["constitutional", "self_refine"]
        assert config.force_improvements is True
        assert config.style_description == "Academic writing style"

    def test_config_validation_temperature(self):
        """Test temperature validation."""
        # Valid temperatures
        Config(temperature=0.0)
        Config(temperature=1.0)
        Config(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValueError):
            Config(temperature=-0.1)

        with pytest.raises(ValueError):
            Config(temperature=2.1)

    def test_config_validation_max_iterations(self):
        """Test max_iterations validation."""
        # Valid iterations
        Config(max_iterations=1)
        Config(max_iterations=10)

        # Invalid iterations
        with pytest.raises(ValueError):
            Config(max_iterations=0)

        with pytest.raises(ValueError):
            Config(max_iterations=11)

    def test_config_validation_timeout(self):
        """Test timeout validation."""
        # Valid timeouts
        Config(timeout_seconds=1)
        Config(timeout_seconds=3600)

        # Invalid timeouts
        with pytest.raises(ValueError):
            Config(timeout_seconds=0)

        with pytest.raises(ValueError):
            Config(timeout_seconds=-1)

    def test_config_validation_critics(self):
        """Test critics validation."""
        # Valid critics
        Config(critics=["reflexion"])
        Config(critics=["constitutional", "self_refine"])

        # Empty critics list should be valid
        Config(critics=[])

    def test_config_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            Config(invalid_field="value")


class TestConfigFactoryMethods:
    """Test Config factory methods."""

    def test_config_fast(self):
        """Test Config.fast() factory method."""
        config = Config.fast()

        assert config.model == "gpt-3.5-turbo"
        assert config.critic_model == "gpt-3.5-turbo"
        assert config.max_iterations == 2
        assert config.timeout_seconds == 60
        assert config.force_improvements is False

    def test_config_quality(self):
        """Test Config.quality() factory method."""
        config = Config.quality()

        assert config.model == "gpt-4o"
        assert config.critic_model == "gpt-4o-mini"
        assert config.max_iterations == 5
        assert config.timeout_seconds == 600
        assert config.force_improvements is True

    def test_config_creative(self):
        """Test Config.creative() factory method."""
        config = Config.creative()

        assert config.temperature >= 0.8
        assert config.critic_temperature >= 0.7
        assert config.force_improvements is True

    def test_factory_methods_immutability(self):
        """Test that factory methods return independent instances."""
        config1 = Config.fast()
        config2 = Config.fast()

        # Modify one config
        config1.temperature = 0.5

        # Other config should be unchanged
        assert config2.temperature != 0.5


class TestConfigSerialization:
    """Test Config serialization and deserialization."""

    def test_config_model_dump(self):
        """Test Config serialization to dict."""
        config = Config(
            model="gpt-4o",
            temperature=0.8,
            max_iterations=4,
            critics=["reflexion", "constitutional"],
        )

        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["model"] == "gpt-4o"
        assert data["temperature"] == 0.8
        assert data["max_iterations"] == 4
        assert data["critics"] == ["reflexion", "constitutional"]

    def test_config_model_dump_exclude(self):
        """Test Config serialization with exclusions."""
        config = Config(model="gpt-4o", temperature=0.8, logfire_token="secret-token")

        data = config.model_dump(exclude={"logfire_token"})

        assert "model" in data
        assert "temperature" in data
        assert "logfire_token" not in data

    def test_config_json_serialization(self):
        """Test Config JSON serialization."""
        config = Config(model="gpt-4o", temperature=0.8, critics=["reflexion"])

        json_str = config.model_dump_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["model"] == "gpt-4o"
        assert data["temperature"] == 0.8

    def test_config_from_dict(self):
        """Test Config creation from dictionary."""
        data = {
            "model": "gpt-4o",
            "temperature": 0.9,
            "max_iterations": 5,
            "critics": ["constitutional", "self_refine"],
        }

        config = Config(**data)

        assert config.model == "gpt-4o"
        assert config.temperature == 0.9
        assert config.max_iterations == 5
        assert config.critics == ["constitutional", "self_refine"]


class TestSifakaEngineInitialization:
    """Test SifakaEngine initialization."""

    def test_engine_default_initialization(self):
        """Test SifakaEngine with default config."""
        engine = SifakaEngine()

        assert engine.config is not None
        assert isinstance(engine.config, Config)
        assert engine.storage is not None
        assert hasattr(engine, "generator")
        assert hasattr(engine, "orchestrator")
        assert hasattr(engine, "validator")

    def test_engine_custom_config(self):
        """Test SifakaEngine with custom config."""
        config = Config(model="gpt-4o", temperature=0.9, max_iterations=5)

        engine = SifakaEngine(config=config)

        assert engine.config == config
        assert engine.config.model == "gpt-4o"
        assert engine.config.temperature == 0.9

    def test_engine_custom_storage(self):
        """Test SifakaEngine with custom storage."""
        storage = MemoryStorage()
        engine = SifakaEngine(storage=storage)

        assert engine.storage == storage
        assert isinstance(engine.storage, MemoryStorage)

    def test_engine_both_custom(self):
        """Test SifakaEngine with both custom config and storage."""
        config = Config(model="gpt-4o", temperature=0.8)
        storage = MemoryStorage()

        engine = SifakaEngine(config=config, storage=storage)

        assert engine.config == config
        assert engine.storage == storage

    def test_engine_component_initialization(self):
        """Test that engine components are properly initialized."""
        config = Config(
            model="gpt-4o",
            temperature=0.7,
            critics=["reflexion", "constitutional"],
            critic_model="gpt-4o-mini",
            critic_temperature=0.5,
        )

        engine = SifakaEngine(config=config)

        # Check generator configuration
        assert engine.generator.model == "gpt-4o"
        assert engine.generator.temperature == 0.7

        # Check orchestrator configuration
        assert engine.orchestrator.critic_names == ["reflexion", "constitutional"]
        assert engine.orchestrator.model == "gpt-4o"
        assert engine.orchestrator.critic_model == "gpt-4o-mini"


class TestSifakaEngineImprove:
    """Test SifakaEngine improve method."""

    @pytest.mark.asyncio
    async def test_engine_improve_basic(self):
        """Test basic engine improve functionality."""
        with (
            patch("sifaka.core.engine.generation.TextGenerator") as mock_gen,
            patch("sifaka.core.engine.orchestration.CriticOrchestrator") as mock_orch,
            patch("sifaka.core.engine.validation.ValidationRunner") as mock_val,
        ):

            # Mock components
            mock_generator = MagicMock()
            mock_generator.generate = AsyncMock(return_value="Improved text")
            mock_gen.return_value = mock_generator

            mock_orchestrator = MagicMock()
            mock_orchestrator.run_critics = AsyncMock(return_value=[])
            mock_orch.return_value = mock_orchestrator

            mock_validator_instance = MagicMock()
            mock_validator_instance.run_validators = AsyncMock(return_value=[])
            mock_val.return_value = mock_validator_instance

            engine = SifakaEngine()
            result = await engine.improve("Test text")

            assert isinstance(result, SifakaResult)
            assert result.original_text == "Test text"

    @pytest.mark.asyncio
    async def test_engine_improve_with_validators(self):
        """Test engine improve with custom validators."""
        with (
            patch("sifaka.core.engine.generation.TextGenerator") as mock_gen,
            patch("sifaka.core.engine.orchestration.CriticOrchestrator") as mock_orch,
            patch("sifaka.core.engine.validation.ValidationRunner") as mock_val,
        ):

            # Mock components
            mock_generator = MagicMock()
            mock_generator.generate = AsyncMock(return_value="Improved text")
            mock_gen.return_value = mock_generator

            mock_orchestrator = MagicMock()
            mock_orchestrator.run_critics = AsyncMock(return_value=[])
            mock_orch.return_value = mock_orchestrator

            mock_validator_instance = MagicMock()
            mock_validator_instance.run_validators = AsyncMock(return_value=[])
            mock_val.return_value = mock_validator_instance

            validators = [
                LengthValidator(min_length=10, max_length=100),
                ContentValidator(required_terms=["test"]),
            ]

            engine = SifakaEngine()
            await engine.improve("Test text", validators=validators)

            # Verify validators were passed to validation runner
            mock_validator_instance.run_validators.assert_called()

    @pytest.mark.asyncio
    async def test_engine_improve_timeout_handling(self):
        """Test engine timeout handling."""
        config = Config(timeout_seconds=0.1)  # Very short timeout

        with patch("sifaka.core.engine.generation.TextGenerator") as mock_gen:
            # Mock slow generator
            mock_generator = MagicMock()

            async def slow_generate(*args, **kwargs):
                await asyncio.sleep(1)  # Longer than timeout
                return "Should not reach here"

            mock_generator.generate = slow_generate
            mock_gen.return_value = mock_generator

            engine = SifakaEngine(config=config)

            with pytest.raises(TimeoutError):
                await engine.improve("Test text")

    @pytest.mark.asyncio
    async def test_engine_improve_storage_integration(self):
        """Test engine integration with storage."""
        storage = MemoryStorage()

        with (
            patch("sifaka.core.engine.generation.TextGenerator") as mock_gen,
            patch("sifaka.core.engine.orchestration.CriticOrchestrator") as mock_orch,
            patch("sifaka.core.engine.validation.ValidationRunner") as mock_val,
        ):

            # Mock components
            mock_generator = MagicMock()
            mock_generator.generate = AsyncMock(return_value="Improved text")
            mock_gen.return_value = mock_generator

            mock_orchestrator = MagicMock()
            mock_orchestrator.run_critics = AsyncMock(return_value=[])
            mock_orch.return_value = mock_orchestrator

            mock_validator_instance = MagicMock()
            mock_validator_instance.run_validators = AsyncMock(return_value=[])
            mock_val.return_value = mock_validator_instance

            engine = SifakaEngine(storage=storage)
            result = await engine.improve("Test text")

            # Result should be stored
            stored_result = await storage.load(result.id)
            assert stored_result is not None
            assert stored_result.id == result.id


class TestConfigEngineIntegration:
    """Test Config and Engine integration scenarios."""

    def test_config_propagation_to_components(self):
        """Test that config values propagate to engine components."""
        config = Config(
            model="gpt-4o",
            temperature=0.9,
            critic_model="gpt-4o-mini",
            critic_temperature=0.3,
            critics=["reflexion", "constitutional"],
            timeout_seconds=120,
        )

        engine = SifakaEngine(config=config)

        # Check config propagation
        assert engine.config.model == "gpt-4o"
        assert engine.config.temperature == 0.9
        assert engine.config.critic_model == "gpt-4o-mini"
        assert engine.config.critic_temperature == 0.3
        assert engine.config.timeout_seconds == 120

    def test_config_modification_after_engine_creation(self):
        """Test that modifying config after engine creation doesn't affect engine."""
        config = Config(model="gpt-4o", temperature=0.7)
        engine = SifakaEngine(config=config)

        original_model = engine.config.model

        # Modify original config
        config.model = "gpt-3.5-turbo"

        # Engine config should be unchanged (if properly copied)
        # Note: This test depends on whether Config is copied or referenced
        assert engine.config.model == original_model

    def test_factory_config_with_engine(self):
        """Test using factory configs with engine."""
        # Test fast config
        fast_engine = SifakaEngine(config=Config.fast())
        assert fast_engine.config.model == "gpt-3.5-turbo"
        assert fast_engine.config.max_iterations == 2

        # Test quality config
        quality_engine = SifakaEngine(config=Config.quality())
        assert quality_engine.config.model == "gpt-4o"
        assert quality_engine.config.max_iterations == 5

        # Test creative config
        creative_engine = SifakaEngine(config=Config.creative())
        assert creative_engine.config.temperature >= 0.8

    @pytest.mark.asyncio
    async def test_config_validation_in_engine(self):
        """Test that engine respects config validation settings."""
        # This would test actual engine behavior with different configs
        # For now, we'll test that the config is properly set

        config = Config(max_iterations=1, force_improvements=False)  # Single iteration

        engine = SifakaEngine(config=config)
        assert engine.config.max_iterations == 1
        assert engine.config.force_improvements is False


class TestConfigErrorHandling:
    """Test Config error handling and edge cases."""

    def test_config_invalid_model(self):
        """Test config with invalid model name."""
        # Note: Config might not validate model names, but this tests the pattern
        config = Config(model="invalid-model-name")
        assert config.model == "invalid-model-name"  # Config accepts any string

    def test_config_boundary_values(self):
        """Test config with boundary values."""
        # Test minimum values
        config = Config(temperature=0.0, max_iterations=1, timeout_seconds=1)
        assert config.temperature == 0.0
        assert config.max_iterations == 1
        assert config.timeout_seconds == 1

        # Test maximum values
        config = Config(temperature=2.0, max_iterations=10, timeout_seconds=3600)
        assert config.temperature == 2.0
        assert config.max_iterations == 10
        assert config.timeout_seconds == 3600

    def test_config_none_values(self):
        """Test config with None values where allowed."""
        config = Config(max_tokens=None, style_description=None, style_examples=None)
        assert config.max_tokens is None
        assert config.style_description is None
        assert config.style_examples is None

    def test_config_empty_lists(self):
        """Test config with empty lists."""
        config = Config(critics=[], style_examples=[])
        assert config.critics == []
        assert config.style_examples == []


class TestEngineErrorHandling:
    """Test SifakaEngine error handling."""

    def test_engine_invalid_config_type(self):
        """Test engine with invalid config type."""
        with pytest.raises(TypeError):
            SifakaEngine(config="invalid-config")

    def test_engine_invalid_storage_type(self):
        """Test engine with invalid storage type."""
        with pytest.raises(TypeError):
            SifakaEngine(storage="invalid-storage")

    @pytest.mark.asyncio
    async def test_engine_improve_invalid_text_type(self):
        """Test engine improve with invalid text type."""
        engine = SifakaEngine()

        with pytest.raises(TypeError):
            await engine.improve(123)  # Not a string

    @pytest.mark.asyncio
    async def test_engine_improve_invalid_validators(self):
        """Test engine improve with invalid validators."""
        engine = SifakaEngine()

        with pytest.raises(TypeError):
            await engine.improve("test", validators="invalid")  # Not a list

    @pytest.mark.asyncio
    async def test_engine_improve_empty_text(self):
        """Test engine improve with empty text."""
        with (
            patch("sifaka.core.engine.generation.TextGenerator") as mock_gen,
            patch("sifaka.core.engine.orchestration.CriticOrchestrator") as mock_orch,
            patch("sifaka.core.engine.validation.ValidationRunner") as mock_val,
        ):

            # Mock components
            mock_generator = MagicMock()
            mock_generator.generate = AsyncMock(return_value="")
            mock_gen.return_value = mock_generator

            mock_orchestrator = MagicMock()
            mock_orchestrator.run_critics = AsyncMock(return_value=[])
            mock_orch.return_value = mock_orchestrator

            mock_validator_instance = MagicMock()
            mock_validator_instance.run_validators = AsyncMock(return_value=[])
            mock_val.return_value = mock_validator_instance

            engine = SifakaEngine()
            result = await engine.improve("")

            assert isinstance(result, SifakaResult)
            assert result.original_text == ""
