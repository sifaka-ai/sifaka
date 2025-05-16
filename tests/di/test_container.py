"""
Tests for the dependency container.
"""

import unittest
from unittest.mock import MagicMock, patch

from sifaka.di import (
    DependencyScope,
    get_container,
    mock_container,
)
from sifaka.di.core import DependencyContainer
from sifaka.di.errors import (
    CircularDependencyError,
    DependencyNotFoundError,
)


class TestDependencyContainer(unittest.TestCase):
    """Test the dependency container."""

    def setUp(self):
        """Set up the test environment."""
        # Reset the container before each test
        DependencyContainer.reset_instance()
        self.container = get_container()

    def test_singleton_pattern(self):
        """Test that the container follows the singleton pattern."""
        container1 = get_container()
        container2 = get_container()
        self.assertIs(container1, container2)

    def test_register_and_resolve(self):
        """Test registration and resolution of dependencies."""
        # Register a dependency
        self.container.register("test", "value")

        # Resolve the dependency
        value = self.container.resolve("test")
        self.assertEqual(value, "value")

    def test_register_factory(self):
        """Test registration and resolution of factory functions."""
        # Register a factory function
        mock_factory = MagicMock(return_value="factory_value")
        self.container.register_factory("test_factory", mock_factory)

        # Resolve the dependency
        value = self.container.resolve("test_factory")
        self.assertEqual(value, "factory_value")
        mock_factory.assert_called_once()

    def test_register_type(self):
        """Test registration and resolution by type."""

        # Define a simple class for testing
        class TestClass:
            pass

        # Create an instance
        instance = TestClass()

        # Register by type
        self.container.register_type(TestClass, instance)

        # Resolve by type
        resolved = self.container.resolve_type(TestClass)
        self.assertIs(resolved, instance)

    def test_dependency_not_found(self):
        """Test behavior when a dependency is not found."""
        # Resolve a non-existent dependency
        with self.assertRaises(DependencyNotFoundError):
            self.container.resolve("non_existent")

        # Resolve with a default value
        value = self.container.resolve("non_existent", default="default")
        self.assertEqual(value, "default")

    def test_circular_dependencies(self):
        """Test detection of circular dependencies."""
        # Create circular dependencies
        with self.assertRaises(CircularDependencyError):
            self.container.register_with_dependencies("a", "a_value", ["b"])
            self.container.register_with_dependencies("b", "b_value", ["c"])
            self.container.register_with_dependencies("c", "c_value", ["a"])

    def test_scopes(self):
        """Test dependency scopes."""
        # Register dependencies with different scopes
        self.container.register("singleton", "singleton_value", DependencyScope.SINGLETON)
        self.container.register_factory(
            "transient", lambda: {"value": "transient"}, DependencyScope.TRANSIENT
        )

        # Test singleton scope
        value1 = self.container.resolve("singleton")
        value2 = self.container.resolve("singleton")
        self.assertIs(value1, value2)

        # Test transient scope
        value1 = self.container.resolve("transient")
        value2 = self.container.resolve("transient")
        self.assertIsNot(value1, value2)

        # Test session scope
        with self.container.session_scope("test_session") as _:
            self.container.register("session", "session_value", DependencyScope.SESSION)
            value1 = self.container.resolve("session")
            value2 = self.container.resolve("session")
            self.assertIs(value1, value2)

        # Session-scoped dependency should not be available after session ends
        self.assertIsNone(self.container.resolve("session", None))

    def test_clear(self):
        """Test clearing the container."""
        # Register some dependencies
        self.container.register("test", "value")
        self.assertTrue(self.container.has_dependency("test"))

        # Clear the container
        self.container.clear()
        self.assertFalse(self.container.has_dependency("test"))

    def test_verify_dependencies(self):
        """Test verification of dependencies."""
        # Register a dependency
        self.container.register("test", "value")

        # Verify the dependency exists
        self.assertTrue(self.container.verify_dependencies(["test"]))

        # Verify a non-existent dependency
        self.assertFalse(self.container.verify_dependencies(["non_existent"]))


class TestMockContainer(unittest.TestCase):
    """Test the mock container for testing."""

    def test_mock_container(self):
        """Test the mock container context manager."""
        # Register a real dependency
        real_container = get_container()
        real_container.register("real", "real_value")

        # Use a mock container
        with mock_container() as container:
            # Mock container should not have the real dependency
            self.assertFalse(container.has_dependency("real"))

            # Register a mock dependency
            container.register("mock", "mock_value")
            self.assertTrue(container.has_dependency("mock"))

            # Mock a dependency
            mock_dep = container.mock_dependency("logger")
            self.assertIsInstance(mock_dep, MagicMock)

            # Resolve the mocked dependency
            logger = container.resolve("logger")
            self.assertIs(logger, mock_dep)

        # After the context manager exits, the real container should be restored
        self.assertTrue(real_container.has_dependency("real"))
        self.assertFalse(real_container.has_dependency("mock"))


if __name__ == "__main__":
    unittest.main()
