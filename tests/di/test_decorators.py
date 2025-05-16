"""
Tests for the dependency injection decorators.
"""

import unittest
from unittest.mock import MagicMock

from sifaka.di import (
    inject,
    inject_by_type,
    injectable,
    mock_container,
)
from sifaka.di.core import DependencyContainer, DependencyScope


class TestInjectDecorator(unittest.TestCase):
    """Test the @inject decorator."""

    def setUp(self):
        """Set up the test environment."""
        DependencyContainer.reset_instance()

    def test_positional_injection(self):
        """Test injection of positional dependencies."""
        with mock_container() as container:
            # Register a mock dependency
            logger_mock = container.mock_dependency("logger")

            # Define a function with injection
            @inject("logger")
            def log_message(message, logger):
                logger.log(message)
                return True

            # Call the function
            result = log_message("test message")

            # Verify the dependency was injected
            self.assertTrue(result)
            logger_mock.log.assert_called_once_with("test message")

    def test_keyword_injection(self):
        """Test injection of keyword dependencies."""
        with mock_container() as container:
            # Register mock dependencies
            config_mock = {"debug": True}
            db_mock = container.mock_dependency("database")
            container.register("config", config_mock)

            # Define a function with injection
            @inject(config="config", db="database")
            def save_data(data, config, db):
                if config["debug"]:
                    print("Debug mode")
                db.save(data)
                return True

            # Call the function
            result = save_data("test data")

            # Verify the dependencies were injected
            self.assertTrue(result)
            db_mock.save.assert_called_once_with("test data")

    def test_mixed_injection(self):
        """Test injection of both positional and keyword dependencies."""
        with mock_container() as container:
            # Register mock dependencies
            logger_mock = container.mock_dependency("logger")
            config_mock = {"debug": True}
            container.register("config", config_mock)

            # Define a function with injection
            @inject("logger", config="config")
            def process_data(data, logger, config):
                logger.log(f"Processing data in debug mode: {config['debug']}")
                return True

            # Call the function
            result = process_data("test data")

            # Verify the dependencies were injected
            self.assertTrue(result)
            logger_mock.log.assert_called_once()

    def test_method_injection(self):
        """Test injection into class methods."""
        with mock_container() as container:
            # Register mock dependencies
            db_mock = container.mock_dependency("database")

            # Define a class with method injection
            class UserService:
                @inject(db="database")
                def get_user(self, user_id, db):
                    return db.get_user(user_id)

            # Create an instance and call the method
            service = UserService()
            service.get_user(123)

            # Verify the dependency was injected
            db_mock.get_user.assert_called_once_with(123)

    def test_override_injection(self):
        """Test that explicitly provided arguments override injection."""
        with mock_container() as container:
            # Register mock dependencies
            container_logger = container.mock_dependency("logger")

            # Define a function with injection
            @inject("logger")
            def log_message(message, logger):
                logger.log(message)
                return logger

            # Create a different logger to pass explicitly
            explicit_logger = MagicMock()

            # Call the function with explicit argument
            result = log_message("test message", explicit_logger)

            # Verify the explicit argument was used, not the injected one
            self.assertIs(result, explicit_logger)
            explicit_logger.log.assert_called_once_with("test message")
            container_logger.log.assert_not_called()


class TestInjectByTypeDecorator(unittest.TestCase):
    """Test the @inject_by_type decorator."""

    def setUp(self):
        """Set up the test environment."""
        DependencyContainer.reset_instance()

    def test_type_injection(self):
        """Test injection by parameter type."""
        with mock_container() as container:
            # Define a type to inject
            class Logger:
                def log(self, message):
                    pass

            # Register a mock implementation
            logger_mock = MagicMock(spec=Logger)
            container.register_type(Logger, logger_mock)

            # Define a function with type injection
            @inject_by_type()
            def process_data(data: str, logger: Logger):
                logger.log(f"Processing: {data}")
                return True

            # Call the function
            result = process_data("test data")

            # Verify the dependency was injected
            self.assertTrue(result)
            logger_mock.log.assert_called_once_with("Processing: test data")

    def test_multiple_type_injection(self):
        """Test injection of multiple types."""
        with mock_container() as container:
            # Define types to inject
            class Logger:
                def log(self, message):
                    pass

            class Database:
                def save(self, data):
                    pass

            class Config:
                def __init__(self):
                    self.debug = True

            # Register mock implementations
            logger_mock = MagicMock(spec=Logger)
            db_mock = MagicMock(spec=Database)
            config_mock = MagicMock(spec=Config)
            config_mock.debug = True

            container.register_type(Logger, logger_mock)
            container.register_type(Database, db_mock)
            container.register_type(Config, config_mock)

            # Define a function with type injection
            @inject_by_type()
            def process_data(data: str, logger: Logger, db: Database, config: Config):
                logger.log(f"Processing: {data}")
                if config.debug:
                    logger.log("Debug mode")
                db.save(data)
                return True

            # Call the function
            result = process_data("test data")

            # Verify the dependencies were injected
            self.assertTrue(result)
            logger_mock.log.assert_called()
            db_mock.save.assert_called_once_with("test data")

    def test_type_and_name_injection(self):
        """Test type injection with use_names=True."""
        with mock_container() as container:
            # Register a dependency by name only
            logger_mock = container.mock_dependency("logger")

            # Define a function with type injection that falls back to names
            @inject_by_type(use_names=True)
            def process_data(data: str, logger):
                logger.log(f"Processing: {data}")
                return True

            # Call the function
            result = process_data("test data")

            # Verify the dependency was injected by name
            self.assertTrue(result)
            logger_mock.log.assert_called_once_with("Processing: test data")


class TestInjectableDecorator(unittest.TestCase):
    """Test the @injectable decorator."""

    def setUp(self):
        """Set up the test environment."""
        DependencyContainer.reset_instance()

    def test_injectable_class(self):
        """Test that a class decorated with @injectable can be injected by type."""

        # Define an injectable class
        @injectable
        class Logger:
            def __init__(self):
                self.messages = []

            def log(self, message):
                self.messages.append(message)

        # Define a class that uses the injectable
        class Service:
            @inject_by_type()
            def __init__(self, logger: Logger):
                self.logger = logger

            def do_something(self):
                self.logger.log("Did something")

        # Create a service instance (should get an injected logger)
        service = Service()

        # Verify the logger was injected
        self.assertIsInstance(service.logger, Logger)

        # Use the service
        service.do_something()

        # Verify the logger was used
        self.assertEqual(service.logger.messages, ["Did something"])

    def test_injectable_with_singleton_scope(self):
        """Test that injectable classes are singleton by default."""

        # Define an injectable class
        @injectable
        class Counter:
            def __init__(self):
                self.count = 0

            def increment(self):
                self.count += 1
                return self.count

        # Define two classes that use the injectable
        class ServiceA:
            @inject_by_type()
            def __init__(self, counter: Counter):
                self.counter = counter

        class ServiceB:
            @inject_by_type()
            def __init__(self, counter: Counter):
                self.counter = counter

        # Create service instances
        service_a = ServiceA()
        service_b = ServiceB()

        # Verify they got the same counter instance
        self.assertIs(service_a.counter, service_b.counter)

        # Use the counter
        value_a = service_a.counter.increment()
        value_b = service_b.counter.increment()

        # Verify the counter was shared
        self.assertEqual(value_a, 1)
        self.assertEqual(value_b, 2)
        self.assertEqual(service_a.counter.count, 2)
        self.assertEqual(service_b.counter.count, 2)


if __name__ == "__main__":
    unittest.main()
