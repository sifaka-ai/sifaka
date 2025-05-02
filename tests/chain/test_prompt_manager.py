"""
Tests for the PromptManager class.
"""

import unittest

from sifaka.chain.managers.prompt import PromptManager


class TestPromptManager(unittest.TestCase):
    """Tests for the PromptManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prompt_manager = PromptManager()
        
    def test_create_prompt_with_feedback(self):
        """Test create_prompt_with_feedback method."""
        original_prompt = "Write a short story about a robot."
        feedback = "Make it more emotional."
        
        result = self.prompt_manager.create_prompt_with_feedback(original_prompt, feedback)
        
        self.assertEqual(
            result,
            "Write a short story about a robot.\n\nPrevious attempt feedback:\nMake it more emotional."
        )
        
    def test_create_prompt_with_history(self):
        """Test create_prompt_with_history method."""
        original_prompt = "Write a short story about a robot."
        history = ["First attempt", "Second attempt"]
        
        result = self.prompt_manager.create_prompt_with_history(original_prompt, history)
        
        self.assertEqual(
            result,
            "Write a short story about a robot.\n\nPrevious attempts:\nFirst attempt\nSecond attempt"
        )
        
    def test_create_prompt_with_context(self):
        """Test create_prompt_with_context method."""
        original_prompt = "Write a short story about a robot."
        context = "The robot is named R2D2 and is very friendly."
        
        result = self.prompt_manager.create_prompt_with_context(original_prompt, context)
        
        self.assertEqual(
            result,
            "Context:\nThe robot is named R2D2 and is very friendly.\n\nPrompt:\nWrite a short story about a robot."
        )
        
    def test_create_prompt_with_examples(self):
        """Test create_prompt_with_examples method."""
        original_prompt = "Write a short story about a robot."
        examples = ["R2D2 was a friendly robot.", "C3PO was a protocol droid."]
        
        result = self.prompt_manager.create_prompt_with_examples(original_prompt, examples)
        
        self.assertEqual(
            result,
            "Write a short story about a robot.\n\nExamples:\nExample 1: R2D2 was a friendly robot.\nExample 2: C3PO was a protocol droid."
        )
        
    def test_empty_inputs(self):
        """Test with empty inputs."""
        # Empty original prompt
        result = self.prompt_manager.create_prompt_with_feedback("", "Feedback")
        self.assertEqual(result, "\n\nPrevious attempt feedback:\nFeedback")
        
        # Empty feedback
        result = self.prompt_manager.create_prompt_with_feedback("Prompt", "")
        self.assertEqual(result, "Prompt\n\nPrevious attempt feedback:\n")
        
        # Empty history
        result = self.prompt_manager.create_prompt_with_history("Prompt", [])
        self.assertEqual(result, "Prompt\n\nPrevious attempts:\n")
        
        # Empty context
        result = self.prompt_manager.create_prompt_with_context("Prompt", "")
        self.assertEqual(result, "Context:\n\n\nPrompt:\nPrompt")
        
        # Empty examples
        result = self.prompt_manager.create_prompt_with_examples("Prompt", [])
        self.assertEqual(result, "Prompt\n\nExamples:\n")


if __name__ == "__main__":
    unittest.main()
