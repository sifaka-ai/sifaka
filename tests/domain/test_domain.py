"""Tests for domain-specific components."""

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from sifaka.domain.base import DomainConfig


class TestDomainConfig(unittest.TestCase):
    """Tests for domain configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DomainConfig(
            name="test_domain",
            description="Test domain configuration",
            params={
                "required_terms": ["term1", "term2"],
                "prohibited_terms": ["term3", "term4"],
                "style": "professional"
            }
        )

    def test_domain_config_initialization(self):
        """Test domain configuration initialization."""
        self.assertIsNotNone(self.config)
        self.assertEqual(self.config.name, "test_domain")
        self.assertEqual(self.config.description, "Test domain configuration")
        self.assertIn("required_terms", self.config.params)
        self.assertIn("prohibited_terms", self.config.params)
        self.assertIn("style", self.config.params)

    def test_domain_config_validation(self):
        """Test domain configuration validation."""
        # Test valid config
        config = DomainConfig(
            name="valid_domain",
            description="Valid domain configuration",
            params={
                "required_terms": ["term1"],
                "prohibited_terms": ["term2"],
                "style": "professional"
            }
        )
        self.assertIsNotNone(config)

        # Test invalid config
        with self.assertRaises(ValueError):
            DomainConfig(
                name="",  # Empty name
                description="Invalid domain configuration",
                params={}
            )

    def test_domain_config_serialization(self):
        """Test domain configuration serialization."""
        # Test serialization to dict
        config_dict = self.config.model_dump()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["name"], "test_domain")
        self.assertEqual(config_dict["description"], "Test domain configuration")
        self.assertIn("params", config_dict)

        # Test deserialization from dict
        new_config = DomainConfig(**config_dict)
        self.assertEqual(new_config.name, "test_domain")
        self.assertEqual(new_config.description, "Test domain configuration")
        self.assertEqual(new_config.params, self.config.params)

    def test_domain_config_factory_methods(self):
        """Test domain configuration factory methods."""
        # Test create_with_config
        config_dict = {
            "name": "factory_domain",
            "description": "Factory domain configuration",
            "params": {
                "required_terms": ["term1"],
                "prohibited_terms": ["term2"],
                "style": "professional"
            }
        }
        config = DomainConfig.create_with_config(config_dict)
        self.assertEqual(config.name, "factory_domain")
        self.assertEqual(config.description, "Factory domain configuration")
        self.assertEqual(config.params["required_terms"], ["term1"])
        self.assertEqual(config.params["prohibited_terms"], ["term2"])
        self.assertEqual(config.params["style"], "professional")

        # Test create (alias for create_with_config)
        config = DomainConfig.create(config_dict)
        self.assertEqual(config.name, "factory_domain")

        # Test create_with_params
        config = DomainConfig.create_with_params(
            name="param_domain",
            description="Param domain configuration",
            params={
                "required_terms": ["term1"],
                "prohibited_terms": ["term2"],
                "style": "professional"
            }
        )
        self.assertEqual(config.name, "param_domain")
        self.assertEqual(config.description, "Param domain configuration")
        self.assertEqual(config.params["required_terms"], ["term1"])
        self.assertEqual(config.params["prohibited_terms"], ["term2"])
        self.assertEqual(config.params["style"], "professional")


if __name__ == "__main__":
    unittest.main()