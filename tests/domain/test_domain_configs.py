"""Tests for domain configurations and validation."""

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from sifaka.domain.base import DomainConfig


class TestDomainConfigs(unittest.TestCase):
    """Tests for domain configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = DomainConfig(
            name="test_domain",
            description="Test domain configuration",
            params={
                "enabled": True,
                "threshold": 0.8,
                "required_terms": ["term1", "term2"],
                "prohibited_terms": ["term3", "term4"]
            }
        )

    def test_base_config_initialization(self):
        """Test base domain configuration initialization."""
        self.assertIsNotNone(self.base_config)
        self.assertEqual(self.base_config.name, "test_domain")
        self.assertEqual(self.base_config.description, "Test domain configuration")
        self.assertEqual(self.base_config.params["enabled"], True)
        self.assertEqual(self.base_config.params["threshold"], 0.8)
        self.assertEqual(self.base_config.params["required_terms"], ["term1", "term2"])
        self.assertEqual(self.base_config.params["prohibited_terms"], ["term3", "term4"])

    def test_base_config_validation(self):
        """Test base domain configuration validation."""
        # Test valid config
        config = DomainConfig(
            name="valid_domain",
            description="Valid domain configuration",
            params={}
        )
        self.assertIsNotNone(config)

        # Test invalid config
        with self.assertRaises(ValueError):
            DomainConfig(
                name="",  # Empty name
                description="Invalid domain configuration",
                params={}
            )

    def test_base_config_serialization(self):
        """Test base domain configuration serialization."""
        # Test serialization to dict
        config_dict = self.base_config.model_dump()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["name"], "test_domain")
        self.assertEqual(config_dict["description"], "Test domain configuration")

        # Test deserialization from dict
        new_config = DomainConfig(**config_dict)
        self.assertEqual(new_config.name, "test_domain")
        self.assertEqual(new_config.description, "Test domain configuration")

    def test_domain_config_factory_methods(self):
        """Test domain configuration factory methods."""
        # Test create_with_config
        config_dict = {
            "name": "factory_domain",
            "description": "Factory domain configuration",
            "params": {"test": "value"}
        }
        config = DomainConfig.create_with_config(config_dict)
        self.assertEqual(config.name, "factory_domain")
        self.assertEqual(config.description, "Factory domain configuration")
        self.assertEqual(config.params["test"], "value")

        # Test create (alias for create_with_config)
        config = DomainConfig.create(config_dict)
        self.assertEqual(config.name, "factory_domain")

        # Test create_with_params
        config = DomainConfig.create_with_params(
            name="param_domain",
            description="Param domain configuration",
            params={"test": "value"}
        )
        self.assertEqual(config.name, "param_domain")
        self.assertEqual(config.description, "Param domain configuration")
        self.assertEqual(config.params["test"], "value")


if __name__ == "__main__":
    unittest.main()