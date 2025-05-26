"""Shared utilities for validator implementations.

This module provides common functionality that can be shared across different
validator implementations to reduce code duplication.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import validation_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ValidationMixin

logger = get_logger(__name__)


class BaseValidator(ValidationMixin):
    """Base class for validator implementations with common functionality.
    
    This class provides shared functionality for validator implementations including:
    - Standardized validation result creation
    - Error handling patterns
    - Empty text checking
    - Common validation patterns
    
    Subclasses should implement the _validate_content method and can override
    other methods as needed.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the base validator.
        
        Args:
            name: Optional name for the validator (defaults to class name).
        """
        super().__init__()
        self.name = name or self.__class__.__name__
        logger.debug(f"Initialized validator: {self.name}")
    
    def validate(self, thought: Thought) -> ValidationResult:
        """Validate text with standardized error handling.
        
        Args:
            thought: The Thought container with the text to validate.
            
        Returns:
            A ValidationResult with information about the validation.
        """
        with validation_context(
            validator_name=self.name,
            operation="validation",
            message_prefix=f"Failed to validate with {self.name}"
        ):
            # Check if text is available
            if not thought.text:
                return self.create_empty_text_result(self.name)
            
            # Perform the actual validation
            try:
                return self._validate_content(thought)
            except Exception as e:
                return self.create_error_result(e, self.name)
    
    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Perform the actual content validation.
        
        This method should be implemented by subclasses to provide
        the specific validation logic.
        
        Args:
            thought: The Thought container with the text to validate.
            
        Returns:
            A ValidationResult with the validation outcome.
        """
        raise NotImplementedError("Subclasses must implement _validate_content")


class LengthValidatorBase(BaseValidator):
    """Base class for length-based validators.
    
    This class provides common functionality for validators that check
    text length in various units (characters, words, sentences, etc.).
    """
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "characters",
        name: Optional[str] = None
    ):
        """Initialize the length validator.
        
        Args:
            min_length: Minimum required length.
            max_length: Maximum allowed length.
            unit: Unit of measurement ("characters", "words", "sentences").
            name: Optional name for the validator.
        """
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
        
        if min_length is None and max_length is None:
            raise ValueError("At least one of min_length or max_length must be specified")
    
    def _get_length(self, text: str) -> int:
        """Get the length of text in the specified unit.
        
        Args:
            text: The text to measure.
            
        Returns:
            The length in the specified unit.
        """
        if self.unit == "characters":
            return len(text)
        elif self.unit == "words":
            return len(text.split())
        elif self.unit == "sentences":
            # Simple sentence counting (can be improved)
            import re
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")
    
    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text length.
        
        Args:
            thought: The Thought container with the text to validate.
            
        Returns:
            A ValidationResult with the length validation outcome.
        """
        text = thought.text
        length = self._get_length(text)
        
        issues = []
        suggestions = []
        
        # Check minimum length
        if self.min_length is not None and length < self.min_length:
            issues.append(f"Text is too short: {length} {self.unit} (minimum: {self.min_length})")
            suggestions.append(f"Add more content to reach at least {self.min_length} {self.unit}")
        
        # Check maximum length
        if self.max_length is not None and length > self.max_length:
            issues.append(f"Text is too long: {length} {self.unit} (maximum: {self.max_length})")
            suggestions.append(f"Reduce content to stay within {self.max_length} {self.unit}")
        
        # Determine if validation passed
        passed = len(issues) == 0
        
        if passed:
            message = f"Text length is valid: {length} {self.unit}"
        else:
            message = f"Text length validation failed: {length} {self.unit}"
        
        return self.create_validation_result(
            passed=passed,
            message=message,
            score=1.0 if passed else 0.0,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "validator": self.name,
                "length": length,
                "unit": self.unit,
                "min_length": self.min_length,
                "max_length": self.max_length
            }
        )


class RegexValidatorBase(BaseValidator):
    """Base class for regex-based validators.
    
    This class provides common functionality for validators that use
    regular expressions to check text patterns.
    """
    
    def __init__(
        self,
        patterns: Dict[str, str],
        mode: str = "require_all",
        name: Optional[str] = None
    ):
        """Initialize the regex validator.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns.
            mode: Validation mode ("require_all", "require_any", "forbid_all", "forbid_any").
            name: Optional name for the validator.
        """
        super().__init__(name)
        self.patterns = patterns
        self.mode = mode
        
        # Compile patterns for efficiency
        import re
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in patterns.items()
        }
    
    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text against regex patterns.
        
        Args:
            thought: The Thought container with the text to validate.
            
        Returns:
            A ValidationResult with the regex validation outcome.
        """
        text = thought.text
        matches = {}
        
        # Check each pattern
        for name, pattern in self.compiled_patterns.items():
            matches[name] = bool(pattern.search(text))
        
        # Determine validation outcome based on mode
        issues = []
        suggestions = []
        
        if self.mode == "require_all":
            missing_patterns = [name for name, matched in matches.items() if not matched]
            if missing_patterns:
                issues.extend([f"Required pattern '{name}' not found" for name in missing_patterns])
                suggestions.extend([f"Include content matching pattern '{name}'" for name in missing_patterns])
        
        elif self.mode == "require_any":
            if not any(matches.values()):
                pattern_names = list(self.patterns.keys())
                issues.append(f"None of the required patterns found: {', '.join(pattern_names)}")
                suggestions.append(f"Include content matching at least one of: {', '.join(pattern_names)}")
        
        elif self.mode == "forbid_all":
            found_patterns = [name for name, matched in matches.items() if matched]
            if found_patterns:
                issues.extend([f"Forbidden pattern '{name}' found" for name in found_patterns])
                suggestions.extend([f"Remove content matching pattern '{name}'" for name in found_patterns])
        
        elif self.mode == "forbid_any":
            if any(matches.values()):
                found_patterns = [name for name, matched in matches.items() if matched]
                issues.append(f"Forbidden patterns found: {', '.join(found_patterns)}")
                suggestions.append(f"Remove content matching any of: {', '.join(found_patterns)}")
        
        else:
            raise ValueError(f"Unsupported validation mode: {self.mode}")
        
        # Determine if validation passed
        passed = len(issues) == 0
        
        if passed:
            message = f"Text passes regex validation ({self.mode})"
        else:
            message = f"Text fails regex validation ({self.mode})"
        
        return self.create_validation_result(
            passed=passed,
            message=message,
            score=1.0 if passed else 0.0,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "validator": self.name,
                "mode": self.mode,
                "pattern_matches": matches
            }
        )


class ClassifierValidatorBase(BaseValidator):
    """Base class for classifier-based validators.
    
    This class provides common functionality for validators that use
    machine learning classifiers to validate text.
    """
    
    def __init__(
        self,
        classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """Initialize the classifier validator.
        
        Args:
            classifier: The classifier instance to use.
            threshold: Confidence threshold for classification.
            valid_labels: List of labels that indicate valid text.
            invalid_labels: List of labels that indicate invalid text.
            name: Optional name for the validator.
        """
        super().__init__(name)
        self.classifier = classifier
        self.threshold = threshold
        self.valid_labels = valid_labels or []
        self.invalid_labels = invalid_labels or []
        
        if not valid_labels and not invalid_labels:
            raise ValueError("At least one of valid_labels or invalid_labels must be specified")
    
    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text using the classifier.
        
        Args:
            thought: The Thought container with the text to validate.
            
        Returns:
            A ValidationResult with the classification validation outcome.
        """
        text = thought.text
        
        try:
            # Classify the text
            result = self.classifier.classify(text)
            predicted_label = result.label
            confidence = result.confidence
            
            # Check if confidence meets threshold
            if confidence < self.threshold:
                return self.create_validation_result(
                    passed=False,
                    message=f"Classification confidence too low: {confidence:.3f} < {self.threshold}",
                    score=confidence,
                    issues=[f"Low confidence classification: {predicted_label} ({confidence:.3f})"],
                    suggestions=[
                        "Provide clearer, more definitive text",
                        "Add more context to improve classification confidence"
                    ],
                    metadata={
                        "validator": self.name,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "threshold": self.threshold
                    }
                )
            
            # Check against valid/invalid labels
            issues = []
            suggestions = []
            passed = True
            
            if self.valid_labels and predicted_label not in self.valid_labels:
                passed = False
                issues.append(f"Text classified as '{predicted_label}' which is not in valid labels: {self.valid_labels}")
                suggestions.append(f"Modify text to align with valid categories: {self.valid_labels}")
            
            if self.invalid_labels and predicted_label in self.invalid_labels:
                passed = False
                issues.append(f"Text classified as '{predicted_label}' which is in invalid labels: {self.invalid_labels}")
                suggestions.append(f"Modify text to avoid invalid categories: {self.invalid_labels}")
            
            if passed:
                message = f"Text classified as '{predicted_label}' with confidence {confidence:.3f}"
                score = confidence
            else:
                message = f"Text classified as invalid label '{predicted_label}'"
                score = 1.0 - confidence  # Invert score for invalid labels
            
            return self.create_validation_result(
                passed=passed,
                message=message,
                score=score,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    "validator": self.name,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "valid_labels": self.valid_labels,
                    "invalid_labels": self.invalid_labels
                }
            )
            
        except Exception as e:
            return self.create_error_result(e, self.name, "classification")
