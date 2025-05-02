"""
Tests for API documentation examples.

These tests verify that the examples in API documentation work as documented.
"""

import pytest
import importlib
import re
import inspect
import doctest
from typing import Dict, Any, List, Callable, Type, Optional

from sifaka.adapters.rules.base import BaseAdapter, Adaptable
from sifaka.adapters.rules.classifier import ClassifierAdapter, create_classifier_rule
from sifaka.rules.base import Rule, RuleResult
from sifaka.classifiers.base import ClassificationResult


def extract_docstring_examples(obj) -> List[str]:
    """Extract code examples from a docstring.

    Args:
        obj: Object with docstring to extract examples from

    Returns:
        List of code example strings
    """
    docstring = inspect.getdoc(obj)
    if not docstring:
        return []

    # Look for code examples between triple backticks or after "Example:" or "Examples:"
    examples = []

    # Triple backtick examples
    backtick_pattern = r'```(?:python)?\s*(.*?)\s*```'
    for match in re.finditer(backtick_pattern, docstring, re.DOTALL):
        examples.append(match.group(1).strip())

    # Examples after "Example:" or "Examples:"
    example_pattern = r'(?:Example|Examples):\s*\n((?:\s{4}.*\n)+)'
    for match in re.finditer(example_pattern, docstring, re.DOTALL):
        # Remove the leading 4 spaces from each line
        code = '\n'.join(line[4:] for line in match.group(1).split('\n'))
        examples.append(code.strip())

    return examples


class TestDocStringExamples:
    """Tests for examples in docstrings."""

    def test_run_doctest_examples(self):
        """Test that doctest examples in docstrings execute without errors."""
        modules_to_test = [
            "sifaka.adapters.rules.base",
            "sifaka.adapters.rules.classifier",
            "sifaka.rules.base",
            "sifaka.classifiers.base",
            "sifaka.critics.base",
            "sifaka.models.base",
        ]

        for module_name in modules_to_test:
            # Import the module
            module = importlib.import_module(module_name)

            # Run doctests on the module
            finder = doctest.DocTestFinder()
            runner = doctest.DocTestRunner(verbose=False)

            tests = finder.find(module)
            for test in tests:
                failures, _ = runner.run(test)
                assert failures == 0, f"Doctest failures in {module_name}: {test.name}"

    def test_base_adapter_examples(self):
        """Test examples in BaseAdapter docstrings."""
        # Extract examples from BaseAdapter docstrings
        examples = extract_docstring_examples(BaseAdapter)

        # Ensure there are examples
        assert examples, "No examples found in BaseAdapter docstrings"

        # Test the first example by creating a minimal implementation
        class SimpleSentimentClassifier:
            @property
            def name(self) -> str:
                return "simple_sentiment"

            @property
            def description(self) -> str:
                return "Detects sentiment"

            def classify(self, text: str) -> Dict[str, Any]:
                # Simplified for testing
                return {
                    "label": "positive" if "good" in text.lower() else "negative",
                    "confidence": 0.9
                }

        class SentimentAdapter(BaseAdapter):
            def __init__(self, adaptee, valid_labels=None):
                super().__init__(adaptee)
                self.valid_labels = valid_labels or ["positive"]

            def validate(self, text: str, **kwargs) -> RuleResult:
                # Simple implementation
                result = self.adaptee.classify(text)
                valid = result["label"] in self.valid_labels
                return RuleResult(
                    passed=valid,
                    message=f"Sentiment validation {'passed' if valid else 'failed'}",
                    metadata=result
                )

        # Test the adapter
        classifier = SimpleSentimentClassifier()
        adapter = SentimentAdapter(classifier)

        # Test with positive and negative examples
        positive_result = adapter.validate("This is good")
        assert positive_result.passed

        negative_result = adapter.validate("This is bad")
        assert not negative_result.passed

    def test_classifier_adapter_examples(self):
        """Test examples in ClassifierAdapter docstrings."""
        # Extract examples from ClassifierAdapter docstrings
        examples = extract_docstring_examples(ClassifierAdapter)

        # Ensure there are examples
        assert examples, "No examples found in ClassifierAdapter docstrings"

        # Implement a simple classifier following the example
        class SimpleClassifier:
            @property
            def name(self) -> str:
                return "simple_classifier"

            @property
            def description(self) -> str:
                return "A simple classifier"

            @property
            def config(self) -> Dict[str, Any]:
                return {"labels": ["positive", "negative", "neutral"]}

            def classify(self, text: str) -> ClassificationResult:
                if "positive" in text.lower():
                    label = "positive"
                elif "negative" in text.lower():
                    label = "negative"
                else:
                    label = "neutral"

                return ClassificationResult(
                    label=label,
                    confidence=0.9,
                    metadata={}
                )

            def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
                return [self.classify(text) for text in texts]

        # Create classifier and adapter
        classifier = SimpleClassifier()
        adapter = ClassifierAdapter(
            classifier,
            valid_labels=["positive", "neutral"],
            threshold=0.5
        )

        # Test functionality
        positive_result = adapter.validate("This is positive")
        assert positive_result.passed

        negative_result = adapter.validate("This is negative")
        assert not negative_result.passed

        neutral_result = adapter.validate("This is neutral")
        assert neutral_result.passed

    def test_create_classifier_rule_examples(self):
        """Test examples for create_classifier_rule function."""
        # Extract examples
        examples = extract_docstring_examples(create_classifier_rule)

        # Ensure there are examples
        assert examples, "No examples found in create_classifier_rule docstrings"

        # Implement a simple classifier
        class ToxicityClassifier:
            @property
            def name(self) -> str:
                return "toxicity_classifier"

            @property
            def description(self) -> str:
                return "Classifies toxicity"

            @property
            def config(self) -> Dict[str, Any]:
                return {"labels": ["toxic", "safe"]}

            def classify(self, text: str) -> ClassificationResult:
                is_toxic = any(word in text.lower() for word in ["bad", "toxic", "hate"])
                return ClassificationResult(
                    label="toxic" if is_toxic else "safe",
                    confidence=0.9,
                    metadata={}
                )

            def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
                return [self.classify(text) for text in texts]

        # Create classifier and rule
        classifier = ToxicityClassifier()
        rule = create_classifier_rule(
            classifier=classifier,
            valid_labels=["safe"],
            name="safety_rule",
            description="Ensures text is safe"
        )

        # Test the rule
        safe_result = rule.validate("This is safe text")
        assert safe_result.passed

        toxic_result = rule.validate("This is bad and toxic text")
        assert not toxic_result.passed


class TestDocumentationConsistency:
    """Tests for consistency between documentation and implementation."""

    def test_parameter_documentation(self):
        """Test that documented parameters match actual parameters."""
        # List of functions to check
        functions_to_check = [
            create_classifier_rule
        ]

        for func in functions_to_check:
            # Get actual parameters
            signature = inspect.signature(func)
            actual_params = list(signature.parameters.keys())

            # Get documented parameters from docstring
            docstring = inspect.getdoc(func)
            param_pattern = r'(?:Args|Parameters):(.*?)(?:\n\n|\Z)'
            param_match = re.search(param_pattern, docstring, re.DOTALL)

            if param_match:
                param_section = param_match.group(1)
                documented_params = []

                # Extract parameter names from docstring
                param_name_pattern = r'\n\s+(\w+):'
                for match in re.finditer(param_name_pattern, param_section):
                    documented_params.append(match.group(1))

                # Check if all actual parameters are documented
                for param in actual_params:
                    if param != 'self':  # Skip self
                        assert param in documented_params, \
                               f"Parameter '{param}' is not documented in {func.__name__}"

                # Check if all documented parameters exist
                for param in documented_params:
                    assert param in actual_params, \
                           f"Documented parameter '{param}' doesn't exist in {func.__name__}"

    def test_return_type_documentation(self):
        """Test that documented return types match actual return annotations."""
        # List of functions to check
        functions_to_check = [
            create_classifier_rule
        ]

        for func in functions_to_check:
            # Get actual return type
            signature = inspect.signature(func)
            actual_return = signature.return_annotation

            # Check if the return type is annotated
            if actual_return is not inspect.Signature.empty:
                # Get documented return type from docstring
                docstring = inspect.getdoc(func)
                return_pattern = r'Returns:(.*?)(?:\n\n|\Z)'
                return_match = re.search(return_pattern, docstring, re.DOTALL)

                if return_match:
                    # Documented return exists, should mention the type
                    return_doc = return_match.group(1).strip()

                    # Get the actual return type name
                    if hasattr(actual_return, "__name__"):
                        return_type_name = actual_return.__name__
                    else:
                        return_type_name = str(actual_return)

                    # Check if the return type is mentioned in the docs
                    assert return_type_name in return_doc, \
                           f"Return type {return_type_name} not mentioned in docs for {func.__name__}"