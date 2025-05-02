"""
Tests for the RetryStrategy classes.
"""

import unittest
from unittest.mock import Mock, patch

from sifaka.chain.strategies.retry import SimpleRetryStrategy, BackoffRetryStrategy
from sifaka.chain.result import ChainResult
from sifaka.validation import ValidationResult, RuleResult


class TestSimpleRetryStrategy(unittest.TestCase):
    """Tests for the SimpleRetryStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SimpleRetryStrategy[str](max_attempts=3)
        
        # Create mocks
        self.generator = Mock()
        self.validation_manager = Mock()
        self.prompt_manager = Mock()
        self.result_formatter = Mock()
        self.critic = Mock()
        
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.strategy.max_attempts, 3)
        
        # Test with different max_attempts
        strategy = SimpleRetryStrategy[str](max_attempts=5)
        self.assertEqual(strategy.max_attempts, 5)
        
    def test_run_with_passing_validation(self):
        """Test run with passing validation."""
        # Configure mocks
        self.generator.generate.return_value = "Generated text"
        
        validation_result = ValidationResult(
            output="Generated text",
            rule_results=[RuleResult(passed=True, message="Passed")],
            all_passed=True,
        )
        self.validation_manager.validate.return_value = validation_result
        
        chain_result = ChainResult(
            output="Generated text",
            rule_results=[RuleResult(passed=True, message="Passed")],
        )
        self.result_formatter.format_result.return_value = chain_result
        
        # Run the strategy
        result = self.strategy.run(
            prompt="Test prompt",
            generator=self.generator,
            validation_manager=self.validation_manager,
            prompt_manager=self.prompt_manager,
            result_formatter=self.result_formatter,
        )
        
        # Verify the result
        self.assertEqual(result, chain_result)
        
        # Verify the mocks were called correctly
        self.generator.generate.assert_called_once_with("Test prompt")
        self.validation_manager.validate.assert_called_once_with("Generated text")
        self.result_formatter.format_result.assert_called_once_with(
            output="Generated text",
            validation_result=validation_result,
            critique_details=None,
        )
        
    def test_run_with_failing_validation_then_passing(self):
        """Test run with failing validation then passing."""
        # Configure mocks
        self.generator.generate.side_effect = ["First output", "Second output"]
        
        # First validation fails, second passes
        first_validation_result = ValidationResult(
            output="First output",
            rule_results=[RuleResult(passed=False, message="Failed")],
            all_passed=False,
        )
        second_validation_result = ValidationResult(
            output="Second output",
            rule_results=[RuleResult(passed=True, message="Passed")],
            all_passed=True,
        )
        self.validation_manager.validate.side_effect = [
            first_validation_result,
            second_validation_result,
        ]
        
        # Get error messages from validation
        self.validation_manager.get_error_messages.return_value = ["Failed"]
        
        # Format feedback from validation
        self.result_formatter.format_feedback_from_validation.return_value = "Feedback"
        
        # Create prompt with feedback
        self.prompt_manager.create_prompt_with_feedback.return_value = "Updated prompt"
        
        # Format result
        chain_result = ChainResult(
            output="Second output",
            rule_results=[RuleResult(passed=True, message="Passed")],
        )
        self.result_formatter.format_result.return_value = chain_result
        
        # Run the strategy
        result = self.strategy.run(
            prompt="Test prompt",
            generator=self.generator,
            validation_manager=self.validation_manager,
            prompt_manager=self.prompt_manager,
            result_formatter=self.result_formatter,
        )
        
        # Verify the result
        self.assertEqual(result, chain_result)
        
        # Verify the mocks were called correctly
        self.assertEqual(self.generator.generate.call_count, 2)
        self.generator.generate.assert_any_call("Test prompt")
        self.generator.generate.assert_any_call("Updated prompt")
        
        self.assertEqual(self.validation_manager.validate.call_count, 2)
        self.validation_manager.validate.assert_any_call("First output")
        self.validation_manager.validate.assert_any_call("Second output")
        
        self.result_formatter.format_feedback_from_validation.assert_called_once_with(
            first_validation_result
        )
        
        self.prompt_manager.create_prompt_with_feedback.assert_called_once_with(
            "Test prompt", "Feedback"
        )
        
        self.result_formatter.format_result.assert_called_once_with(
            output="Second output",
            validation_result=second_validation_result,
            critique_details=None,
        )
        
    def test_run_with_critic(self):
        """Test run with a critic."""
        # Configure mocks
        self.generator.generate.side_effect = ["First output", "Second output"]
        
        # First validation fails, second passes
        first_validation_result = ValidationResult(
            output="First output",
            rule_results=[RuleResult(passed=False, message="Failed")],
            all_passed=False,
        )
        second_validation_result = ValidationResult(
            output="Second output",
            rule_results=[RuleResult(passed=True, message="Passed")],
            all_passed=True,
        )
        self.validation_manager.validate.side_effect = [
            first_validation_result,
            second_validation_result,
        ]
        
        # Critique
        critique_details = {
            "feedback": "Needs improvement",
            "score": 0.5,
        }
        self.critic.critique.return_value = critique_details
        
        # Format feedback from critique
        self.result_formatter.format_feedback_from_critique.return_value = "Critique feedback"
        
        # Create prompt with feedback
        self.prompt_manager.create_prompt_with_feedback.return_value = "Updated prompt"
        
        # Format result
        chain_result = ChainResult(
            output="Second output",
            rule_results=[RuleResult(passed=True, message="Passed")],
            critique_details=critique_details,
        )
        self.result_formatter.format_result.return_value = chain_result
        
        # Run the strategy
        result = self.strategy.run(
            prompt="Test prompt",
            generator=self.generator,
            validation_manager=self.validation_manager,
            prompt_manager=self.prompt_manager,
            result_formatter=self.result_formatter,
            critic=self.critic,
        )
        
        # Verify the result
        self.assertEqual(result, chain_result)
        
        # Verify the mocks were called correctly
        self.critic.critique.assert_called_once_with("First output")
        
        self.result_formatter.format_feedback_from_critique.assert_called_once_with(
            critique_details
        )
        
        self.prompt_manager.create_prompt_with_feedback.assert_called_once_with(
            "Test prompt", "Critique feedback"
        )
        
    def test_run_with_max_attempts_exceeded(self):
        """Test run with max attempts exceeded."""
        # Configure mocks
        self.generator.generate.return_value = "Generated text"
        
        # All validations fail
        validation_result = ValidationResult(
            output="Generated text",
            rule_results=[RuleResult(passed=False, message="Failed")],
            all_passed=False,
        )
        self.validation_manager.validate.return_value = validation_result
        
        # Get error messages from validation
        self.validation_manager.get_error_messages.return_value = ["Failed"]
        
        # Format feedback from validation
        self.result_formatter.format_feedback_from_validation.return_value = "Feedback"
        
        # Create prompt with feedback
        self.prompt_manager.create_prompt_with_feedback.return_value = "Updated prompt"
        
        # Run the strategy - should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.strategy.run(
                prompt="Test prompt",
                generator=self.generator,
                validation_manager=self.validation_manager,
                prompt_manager=self.prompt_manager,
                result_formatter=self.result_formatter,
            )
            
        # Verify the error message
        self.assertIn("Validation failed after 3 attempts", str(context.exception))
        
        # Verify the mocks were called correctly
        self.assertEqual(self.generator.generate.call_count, 3)
        self.assertEqual(self.validation_manager.validate.call_count, 3)
        self.assertEqual(self.result_formatter.format_feedback_from_validation.call_count, 2)
        self.assertEqual(self.prompt_manager.create_prompt_with_feedback.call_count, 2)


class TestBackoffRetryStrategy(unittest.TestCase):
    """Tests for the BackoffRetryStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = BackoffRetryStrategy[str](
            max_attempts=3,
            initial_backoff=1.0,
            backoff_factor=2.0,
            max_backoff=60.0,
        )
        
        # Create mocks
        self.generator = Mock()
        self.validation_manager = Mock()
        self.prompt_manager = Mock()
        self.result_formatter = Mock()
        self.critic = Mock()
        
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.strategy.max_attempts, 3)
        
        # Test with different parameters
        strategy = BackoffRetryStrategy[str](
            max_attempts=5,
            initial_backoff=2.0,
            backoff_factor=3.0,
            max_backoff=120.0,
        )
        self.assertEqual(strategy.max_attempts, 5)
        
    @patch('time.sleep')
    def test_run_with_passing_validation(self, mock_sleep):
        """Test run with passing validation."""
        # Configure mocks
        self.generator.generate.return_value = "Generated text"
        
        validation_result = ValidationResult(
            output="Generated text",
            rule_results=[RuleResult(passed=True, message="Passed")],
            all_passed=True,
        )
        self.validation_manager.validate.return_value = validation_result
        
        chain_result = ChainResult(
            output="Generated text",
            rule_results=[RuleResult(passed=True, message="Passed")],
        )
        self.result_formatter.format_result.return_value = chain_result
        
        # Run the strategy
        result = self.strategy.run(
            prompt="Test prompt",
            generator=self.generator,
            validation_manager=self.validation_manager,
            prompt_manager=self.prompt_manager,
            result_formatter=self.result_formatter,
        )
        
        # Verify the result
        self.assertEqual(result, chain_result)
        
        # Verify the mocks were called correctly
        self.generator.generate.assert_called_once_with("Test prompt")
        self.validation_manager.validate.assert_called_once_with("Generated text")
        self.result_formatter.format_result.assert_called_once_with(
            output="Generated text",
            validation_result=validation_result,
            critique_details=None,
        )
        
        # Verify sleep was not called
        mock_sleep.assert_not_called()
        
    @patch('time.sleep')
    def test_run_with_failing_validation_then_passing(self, mock_sleep):
        """Test run with failing validation then passing."""
        # Configure mocks
        self.generator.generate.side_effect = ["First output", "Second output"]
        
        # First validation fails, second passes
        first_validation_result = ValidationResult(
            output="First output",
            rule_results=[RuleResult(passed=False, message="Failed")],
            all_passed=False,
        )
        second_validation_result = ValidationResult(
            output="Second output",
            rule_results=[RuleResult(passed=True, message="Passed")],
            all_passed=True,
        )
        self.validation_manager.validate.side_effect = [
            first_validation_result,
            second_validation_result,
        ]
        
        # Get error messages from validation
        self.validation_manager.get_error_messages.return_value = ["Failed"]
        
        # Format feedback from validation
        self.result_formatter.format_feedback_from_validation.return_value = "Feedback"
        
        # Create prompt with feedback
        self.prompt_manager.create_prompt_with_feedback.return_value = "Updated prompt"
        
        # Format result
        chain_result = ChainResult(
            output="Second output",
            rule_results=[RuleResult(passed=True, message="Passed")],
        )
        self.result_formatter.format_result.return_value = chain_result
        
        # Run the strategy
        result = self.strategy.run(
            prompt="Test prompt",
            generator=self.generator,
            validation_manager=self.validation_manager,
            prompt_manager=self.prompt_manager,
            result_formatter=self.result_formatter,
        )
        
        # Verify the result
        self.assertEqual(result, chain_result)
        
        # Verify the mocks were called correctly
        self.assertEqual(self.generator.generate.call_count, 2)
        self.generator.generate.assert_any_call("Test prompt")
        self.generator.generate.assert_any_call("Updated prompt")
        
        self.assertEqual(self.validation_manager.validate.call_count, 2)
        self.validation_manager.validate.assert_any_call("First output")
        self.validation_manager.validate.assert_any_call("Second output")
        
        self.result_formatter.format_feedback_from_validation.assert_called_once_with(
            first_validation_result
        )
        
        self.prompt_manager.create_prompt_with_feedback.assert_called_once_with(
            "Test prompt", "Feedback"
        )
        
        self.result_formatter.format_result.assert_called_once_with(
            output="Second output",
            validation_result=second_validation_result,
            critique_details=None,
        )
        
        # Verify sleep was called with the correct backoff
        mock_sleep.assert_called_once_with(1.0)
        
    @patch('time.sleep')
    def test_run_with_multiple_failures(self, mock_sleep):
        """Test run with multiple failures."""
        # Configure mocks
        self.generator.generate.side_effect = ["First output", "Second output", "Third output"]
        
        # All validations fail
        validation_result = ValidationResult(
            output="Generated text",
            rule_results=[RuleResult(passed=False, message="Failed")],
            all_passed=False,
        )
        self.validation_manager.validate.return_value = validation_result
        
        # Get error messages from validation
        self.validation_manager.get_error_messages.return_value = ["Failed"]
        
        # Format feedback from validation
        self.result_formatter.format_feedback_from_validation.return_value = "Feedback"
        
        # Create prompt with feedback
        self.prompt_manager.create_prompt_with_feedback.return_value = "Updated prompt"
        
        # Run the strategy - should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.strategy.run(
                prompt="Test prompt",
                generator=self.generator,
                validation_manager=self.validation_manager,
                prompt_manager=self.prompt_manager,
                result_formatter=self.result_formatter,
            )
            
        # Verify the error message
        self.assertIn("Validation failed after 3 attempts", str(context.exception))
        
        # Verify the mocks were called correctly
        self.assertEqual(self.generator.generate.call_count, 3)
        self.assertEqual(self.validation_manager.validate.call_count, 3)
        self.assertEqual(self.result_formatter.format_feedback_from_validation.call_count, 2)
        self.assertEqual(self.prompt_manager.create_prompt_with_feedback.call_count, 2)
        
        # Verify sleep was called with the correct backoffs
        mock_sleep.assert_any_call(1.0)  # First backoff
        mock_sleep.assert_any_call(2.0)  # Second backoff (1.0 * 2.0)


if __name__ == "__main__":
    unittest.main()
