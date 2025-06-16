"""Tests for SifakaRegistry plugin system."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from sifaka.core.registry import (
    SifakaRegistry, CriticInterface, ValidatorInterface, 
    PluginMetadata, get_registry, reset_registry
)


class MockCritic(CriticInterface):
    """Mock critic for testing."""
    
    def __init__(self, name: str = "mock_critic"):
        self._name = name
    
    async def critique(self, text: str, context: dict) -> dict:
        return {
            "feedback": f"Mock feedback for: {text}",
            "suggestions": ["Mock suggestion 1", "Mock suggestion 2"],
            "confidence": 0.8
        }
    
    @property
    def name(self) -> str:
        return self._name


class MockValidator(ValidatorInterface):
    """Mock validator for testing."""
    
    def __init__(self, name: str = "mock_validator"):
        self._name = name
    
    async def validate(self, text: str, context: dict) -> dict:
        return {
            "passed": len(text) > 10,
            "score": len(text) / 100.0,
            "details": {"length": len(text)}
        }
    
    @property
    def name(self) -> str:
        return self._name


class TestPluginMetadata:
    """Test cases for PluginMetadata."""
    
    def test_plugin_metadata_creation(self):
        """Test PluginMetadata creation."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            plugin_type="critic",
            dependencies=["dep1", "dep2"],
            min_sifaka_version="0.1.0",
            registered_at=datetime.now(),
            instance=MockCritic()
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == "critic"
        assert len(metadata.dependencies) == 2
        assert isinstance(metadata.instance, MockCritic)


class TestSifakaRegistry:
    """Test cases for SifakaRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = SifakaRegistry()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.registry.clear()
    
    def test_registry_initialization(self):
        """Test registry initializes with built-in components."""
        assert isinstance(self.registry._created_at, datetime)
        
        # Should have built-in critics registered
        plugins = self.registry.list_plugins("critic")
        assert len(plugins) > 0
        
        # Check for some expected built-in critics
        critic_names = [p.name for p in plugins]
        assert "reflexion" in critic_names
        assert "constitutional" in critic_names
    
    def test_register_critic_instance(self):
        """Test registering a critic instance."""
        critic = MockCritic("test_critic")
        self.registry.register_critic("test_critic", critic)
        
        # Check it's registered
        plugins = self.registry.list_plugins("critic")
        test_plugins = [p for p in plugins if p.name == "test_critic"]
        assert len(test_plugins) == 1
        
        # Check we can retrieve it
        retrieved_critic = self.registry.get_critic("test_critic")
        assert retrieved_critic is critic
    
    def test_register_critic_class(self):
        """Test registering a critic class."""
        self.registry.register_critic("test_critic", MockCritic)
        
        # Should be able to get an instance
        critic = self.registry.get_critic("test_critic")
        assert isinstance(critic, MockCritic)
        assert critic.name == "mock_critic"  # Default name
    
    def test_register_critic_with_metadata(self):
        """Test registering critic with custom metadata."""
        metadata = {
            "version": "2.0.0",
            "description": "Custom test critic",
            "author": "Test Author",
            "dependencies": ["dep1"]
        }
        
        self.registry.register_critic("test_critic", MockCritic, metadata)
        
        plugins = self.registry.list_plugins("critic")
        test_plugin = next(p for p in plugins if p.name == "test_critic")
        
        assert test_plugin.version == "2.0.0"
        assert test_plugin.description == "Custom test critic"
        assert test_plugin.author == "Test Author"
        assert test_plugin.dependencies == ["dep1"]
    
    def test_register_validator_instance(self):
        """Test registering a validator instance."""
        validator = MockValidator("test_validator")
        self.registry.register_validator("test_validator", validator)
        
        # Check it's registered
        plugins = self.registry.list_plugins("validator")
        test_plugins = [p for p in plugins if p.name == "test_validator"]
        assert len(test_plugins) == 1
        
        # Check we can retrieve it
        retrieved_validator = self.registry.get_validator("test_validator")
        assert retrieved_validator is validator
    
    def test_register_validator_class(self):
        """Test registering a validator class."""
        self.registry.register_validator("test_validator", MockValidator)
        
        # Should be able to get an instance
        validator = self.registry.get_validator("test_validator")
        assert isinstance(validator, MockValidator)
    
    def test_get_critic_not_found(self):
        """Test getting non-existent critic raises error."""
        with pytest.raises(ValueError, match="Critic 'nonexistent' is not registered"):
            self.registry.get_critic("nonexistent")
    
    def test_get_validator_not_found(self):
        """Test getting non-existent validator raises error."""
        with pytest.raises(ValueError, match="Validator 'nonexistent' is not registered"):
            self.registry.get_validator("nonexistent")
    
    def test_register_factory(self):
        """Test registering factory functions."""
        def critic_factory():
            return MockCritic("factory_critic")
        
        self.registry.register_factory("factory_critic", critic_factory, "critic")
        
        # Factory should be registered
        assert "critic:factory_critic" in self.registry._factories
    
    def test_list_plugins_all(self):
        """Test listing all plugins."""
        self.registry.register_critic("test_critic", MockCritic)
        self.registry.register_validator("test_validator", MockValidator)
        
        all_plugins = self.registry.list_plugins()
        critic_plugins = [p for p in all_plugins if p.plugin_type == "critic"]
        validator_plugins = [p for p in all_plugins if p.plugin_type == "validator"]
        
        assert len(critic_plugins) > 0  # Built-ins + test critic
        assert len(validator_plugins) >= 1  # Test validator
    
    def test_list_plugins_by_type(self):
        """Test listing plugins by type."""
        self.registry.register_critic("test_critic", MockCritic)
        self.registry.register_validator("test_validator", MockValidator)
        
        critic_plugins = self.registry.list_plugins("critic")
        validator_plugins = self.registry.list_plugins("validator")
        
        assert all(p.plugin_type == "critic" for p in critic_plugins)
        assert all(p.plugin_type == "validator" for p in validator_plugins)
    
    def test_get_registry_stats(self):
        """Test registry statistics."""
        self.registry.register_critic("test_critic", MockCritic)
        self.registry.register_validator("test_validator", MockValidator)
        
        # Get instances to populate caches
        self.registry.get_critic("test_critic")
        self.registry.get_validator("test_validator")
        
        stats = self.registry.get_registry_stats()
        
        assert "created_at" in stats
        assert "total_plugins" in stats
        assert "plugin_types" in stats
        assert "instantiated_critics" in stats
        assert "instantiated_validators" in stats
        
        assert stats["plugin_types"]["critic"] > 0
        assert stats["plugin_types"]["validator"] >= 1
        assert stats["instantiated_critics"] >= 1
        assert stats["instantiated_validators"] >= 1
    
    @patch('sifaka.core.registry.importlib.import_module')
    def test_discover_plugins_success(self, mock_import):
        """Test successful plugin discovery."""
        # Mock a package with plugin classes
        mock_package = Mock()
        mock_package.__dict__ = {
            "TestCritic": type("TestCritic", (CriticInterface,), {
                "critique": lambda self, text, context: {},
                "name": property(lambda self: "test")
            }),
            "TestValidator": type("TestValidator", (ValidatorInterface,), {
                "validate": lambda self, text, context: {},
                "name": property(lambda self: "test")
            }),
            "NotAPlugin": str  # Should be ignored
        }
        mock_import.return_value = mock_package
        
        discovered = self.registry.discover_plugins("test_package")
        
        assert "TestCritic" in discovered
        assert "TestValidator" in discovered
        assert "NotAPlugin" not in discovered
    
    @patch('sifaka.core.registry.importlib.import_module')
    def test_discover_plugins_import_error(self, mock_import):
        """Test plugin discovery with import error."""
        mock_import.side_effect = ImportError("Package not found")
        
        discovered = self.registry.discover_plugins("nonexistent_package")
        
        assert discovered == []
    
    def test_clear_registry(self):
        """Test clearing registry."""
        self.registry.register_critic("test_critic", MockCritic)
        self.registry.register_validator("test_validator", MockValidator)
        
        assert len(self.registry._plugins) > 0
        assert len(self.registry._critics) == 0  # Not instantiated yet
        
        # Get instances
        self.registry.get_critic("test_critic")
        self.registry.get_validator("test_validator")
        
        assert len(self.registry._critics) > 0
        assert len(self.registry._validators) > 0
        
        self.registry.clear()
        
        assert len(self.registry._plugins) == 0
        assert len(self.registry._critics) == 0
        assert len(self.registry._validators) == 0


class TestGlobalRegistry:
    """Test cases for global registry functions."""
    
    def teardown_method(self):
        """Clean up global registry after each test."""
        reset_registry()
    
    def test_get_registry_singleton(self):
        """Test global registry is singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        assert registry1 is registry2
    
    def test_reset_registry(self):
        """Test resetting global registry."""
        registry1 = get_registry()
        registry1.register_critic("test", MockCritic)
        
        reset_registry()
        
        registry2 = get_registry()
        assert registry1 is not registry2
        assert len(registry2._critics) == 0


class TestInterfaces:
    """Test cases for plugin interfaces."""
    
    def test_critic_interface_abstract(self):
        """Test CriticInterface is abstract."""
        with pytest.raises(TypeError):
            CriticInterface()
    
    def test_validator_interface_abstract(self):
        """Test ValidatorInterface is abstract."""
        with pytest.raises(TypeError):
            ValidatorInterface()
    
    def test_mock_critic_implementation(self):
        """Test MockCritic implements interface correctly."""
        critic = MockCritic("test")
        assert critic.name == "test"
        
        # Test async critique method
        import asyncio
        
        async def test_critique():
            result = await critic.critique("test text", {})
            return result
        
        result = asyncio.run(test_critique())
        assert "feedback" in result
        assert "suggestions" in result
        assert "confidence" in result
    
    def test_mock_validator_implementation(self):
        """Test MockValidator implements interface correctly."""
        validator = MockValidator("test")
        assert validator.name == "test"
        
        # Test async validate method
        import asyncio
        
        async def test_validate():
            result = await validator.validate("short", {})
            return result
        
        result = asyncio.run(test_validate())
        assert "passed" in result
        assert "score" in result
        assert "details" in result
        assert result["passed"] is False  # "short" is <= 10 chars
