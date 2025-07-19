"""Tests for the plugin system."""

from datetime import datetime
from unittest.mock import patch

import pytest

from sifaka.core.models import CritiqueResult, SifakaResult, ValidationResult
from sifaka.core.plugin_interfaces import (
    CriticPlugin,
    PluginMetadata,
    PluginRegistry,
    PluginStatus,
    PluginType,
    ValidatorPlugin,
    get_plugin_registry,
)
from sifaka.core.plugin_loader import PluginLoader


class TestCriticPlugin(CriticPlugin):
    """Test critic plugin for testing."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_critic",
            version="1.0.0",
            author="Test Author",
            description="A test critic plugin",
            plugin_type=PluginType.CRITIC,
            dependencies=[],
            sifaka_version=">=0.1.0",
        )

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Test critique method."""
        return CritiqueResult(
            critic=self.name,
            feedback="Test feedback",
            suggestions=["Test suggestion"],
            needs_improvement=True,
            confidence=0.8,
        )


class TestValidatorPlugin(ValidatorPlugin):
    """Test validator plugin for testing."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_validator",
            version="1.0.0",
            author="Test Author",
            description="A test validator plugin",
            plugin_type=PluginType.VALIDATOR,
            dependencies=[],
            sifaka_version=">=0.1.0",
        )

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Test validate method."""
        return ValidationResult(
            validator=self.name,
            passed=True,
            score=1.0,
            details="Test validation passed",
        )


class TestPluginInterfaces:
    """Test plugin interfaces."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            plugin_type=PluginType.CRITIC,
            dependencies=["dep1", "dep2"],
            sifaka_version=">=0.1.0",
        )

        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "A test plugin"
        assert metadata.plugin_type == PluginType.CRITIC
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.sifaka_version == ">=0.1.0"

    def test_critic_plugin_creation(self):
        """Test creating a critic plugin."""
        plugin = TestCriticPlugin()

        assert plugin.name == "test_critic"
        assert plugin.status == PluginStatus.LOADED
        assert plugin.metadata.plugin_type == PluginType.CRITIC
        assert plugin.error is None

    def test_validator_plugin_creation(self):
        """Test creating a validator plugin."""
        plugin = TestValidatorPlugin()

        assert plugin.name == "test_validator"
        assert plugin.status == PluginStatus.LOADED
        assert plugin.metadata.plugin_type == PluginType.VALIDATOR
        assert plugin.error is None

    @pytest.mark.asyncio
    async def test_critic_plugin_critique(self):
        """Test critic plugin critique method."""
        plugin = TestCriticPlugin()

        # Create a mock result
        result = SifakaResult(
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

        critique = await plugin.critique("test text", result)

        assert critique.critic == "test_critic"
        assert critique.feedback == "Test feedback"
        assert critique.suggestions == ["Test suggestion"]
        assert critique.needs_improvement is True
        assert critique.confidence == 0.8

    @pytest.mark.asyncio
    async def test_validator_plugin_validate(self):
        """Test validator plugin validate method."""
        plugin = TestValidatorPlugin()

        # Create a mock result
        result = SifakaResult(
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

        validation = await plugin.validate("test text", result)

        assert validation.validator == "test_validator"
        assert validation.passed is True
        assert validation.score == 1.0
        assert validation.details == "Test validation passed"


class TestPluginRegistry:
    """Test plugin registry."""

    def test_registry_creation(self):
        """Test creating a plugin registry."""
        registry = PluginRegistry()

        assert len(registry.list_plugins()) == 0
        assert registry.get_plugin("nonexistent") is None

    def test_plugin_registration(self):
        """Test registering a plugin."""
        registry = PluginRegistry()
        plugin = TestCriticPlugin()

        registry.register(plugin)

        assert len(registry.list_plugins()) == 1
        assert registry.get_plugin("test_critic") is plugin
        assert registry.get_metadata("test_critic") == plugin.metadata

    def test_plugin_unregistration(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry()
        plugin = TestCriticPlugin()

        registry.register(plugin)
        assert len(registry.list_plugins()) == 1

        registry.unregister("test_critic")
        assert len(registry.list_plugins()) == 0
        assert registry.get_plugin("test_critic") is None

    def test_plugin_filtering_by_type(self):
        """Test filtering plugins by type."""
        registry = PluginRegistry()
        critic_plugin = TestCriticPlugin()
        validator_plugin = TestValidatorPlugin()

        registry.register(critic_plugin)
        registry.register(validator_plugin)

        assert len(registry.list_plugins()) == 2
        assert len(registry.list_plugins(PluginType.CRITIC)) == 1
        assert len(registry.list_plugins(PluginType.VALIDATOR)) == 1
        assert len(registry.list_plugins(PluginType.STORAGE)) == 0

    def test_plugin_status_tracking(self):
        """Test plugin status tracking."""
        registry = PluginRegistry()
        plugin = TestCriticPlugin()

        registry.register(plugin)

        assert registry.get_status("test_critic") == PluginStatus.LOADED

        plugin.initialize()
        assert registry.get_status("test_critic") == PluginStatus.INITIALIZED

        plugin.activate()
        assert registry.get_status("test_critic") == PluginStatus.ACTIVE

    def test_plugin_health_status(self):
        """Test plugin health status."""
        registry = PluginRegistry()
        plugin = TestCriticPlugin()

        registry.register(plugin)

        all_status = registry.get_all_status()
        assert "test_critic" in all_status

        status = all_status["test_critic"]
        assert status["name"] == "test_critic"
        assert status["version"] == "1.0.0"
        assert status["status"] == PluginStatus.LOADED.value
        assert status["error"] is None


class TestPluginLoader:
    """Test plugin loader."""

    def test_loader_creation(self):
        """Test creating a plugin loader."""
        loader = PluginLoader()

        assert len(loader.get_loaded_plugins()) == 0
        assert len(loader.get_failed_plugins()) == 0

    def test_plugin_validation(self):
        """Test plugin validation."""
        loader = PluginLoader()
        plugin = TestCriticPlugin()

        # Should validate successfully
        assert loader.validate_plugin(plugin) is True

    def test_plugin_validation_failure(self):
        """Test plugin validation failure."""
        loader = PluginLoader()

        # Create a mock plugin with invalid metadata
        class InvalidPlugin(CriticPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="",  # Invalid empty name
                    version="1.0.0",
                    author="Test Author",
                    description="Invalid plugin",
                    plugin_type=PluginType.CRITIC,
                )

            async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
                return CritiqueResult(
                    critic="invalid",
                    feedback="Test feedback",
                    suggestions=["Test suggestion"],
                    needs_improvement=True,
                    confidence=0.8,
                )

        plugin = InvalidPlugin()

        # Should raise an error
        with pytest.raises(Exception):
            loader.validate_plugin(plugin)

    def test_dependency_resolution(self):
        """Test dependency resolution."""
        loader = PluginLoader()

        # Create plugins with dependencies
        class PluginA(CriticPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="plugin_a",
                    version="1.0.0",
                    author="Test Author",
                    description="Plugin A",
                    plugin_type=PluginType.CRITIC,
                    dependencies=["plugin_b"],
                )

            async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
                return CritiqueResult(
                    critic="plugin_a",
                    feedback="Test feedback",
                    suggestions=["Test suggestion"],
                    needs_improvement=True,
                    confidence=0.8,
                )

        class PluginB(CriticPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="plugin_b",
                    version="1.0.0",
                    author="Test Author",
                    description="Plugin B",
                    plugin_type=PluginType.CRITIC,
                    dependencies=[],
                )

            async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
                return CritiqueResult(
                    critic="plugin_b",
                    feedback="Test feedback",
                    suggestions=["Test suggestion"],
                    needs_improvement=True,
                    confidence=0.8,
                )

        plugin_a = PluginA()
        plugin_b = PluginB()

        # Resolve dependencies
        ordered_plugins = loader.resolve_dependencies([plugin_a, plugin_b])

        # plugin_b should come before plugin_a
        assert len(ordered_plugins) == 2
        assert ordered_plugins[0].metadata.name == "plugin_b"
        assert ordered_plugins[1].metadata.name == "plugin_a"

    def test_global_registry_access(self):
        """Test accessing the global plugin registry."""
        registry = get_plugin_registry()

        # Should be the same instance
        assert registry is get_plugin_registry()

    def test_plugin_lifecycle(self):
        """Test plugin lifecycle management."""
        plugin = TestCriticPlugin()

        # Initial state
        assert plugin.status == PluginStatus.LOADED
        assert plugin.error is None

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

    def test_plugin_configuration(self):
        """Test plugin configuration."""
        plugin = TestCriticPlugin()

        # Initialize with config
        config = {"model": "gpt-4o-mini", "temperature": 0.5}
        plugin.initialize(config)

        assert plugin.config == config
        assert plugin.model_config["model"] == "gpt-4o-mini"
        assert plugin.model_config["temperature"] == 0.5

    def test_plugin_error_handling(self):
        """Test plugin error handling."""
        plugin = TestCriticPlugin()

        # Mock an error during initialization
        with patch.object(
            plugin, "_on_initialize", side_effect=ValueError("Test error")
        ):
            with pytest.raises(Exception):
                plugin.initialize()

            assert plugin.status == PluginStatus.ERROR
            assert plugin.error is not None
            assert "Test error" in str(plugin.error)
