"""ValidationResult model for Sifaka.

This module contains the ValidationResult model that tracks validation operations
with detailed information for debugging and improvement.

Extracted from the monolithic thought.py file to improve maintainability.
"""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of a validation operation.

    Records validation outcome with detailed information for debugging and improvement.
    
    This model tracks:
    - Validation pass/fail status
    - Detailed validation information and metrics
    - Validator identification and iteration tracking
    - Timestamp for audit trail
    
    Example:
        ```python
        validation = ValidationResult(
            iteration=1,
            validator="length_validator",
            passed=True,
            details={
                "word_count": 150,
                "character_count": 890,
                "min_required": 100,
                "max_allowed": 1000
            },
            timestamp=datetime.now()
        )
        ```
    """

    iteration: int
    validator: str
    passed: bool
    details: Dict[str, Any]
    timestamp: datetime

    def get_metric_value(self, metric_name: str) -> Any:
        """Get a specific metric value from the validation details.
        
        Args:
            metric_name: Name of the metric to retrieve
            
        Returns:
            The metric value, or None if not found
        """
        return self.details.get(metric_name)

    def get_numeric_metrics(self) -> Dict[str, float]:
        """Extract numeric metrics from validation details.
        
        Returns:
            Dictionary of numeric metrics
        """
        numeric_metrics = {}
        
        for key, value in self.details.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key] = float(value)
                
        return numeric_metrics

    def get_threshold_info(self) -> Dict[str, Any]:
        """Extract threshold information from validation details.
        
        Returns:
            Dictionary with threshold information
        """
        threshold_info = {}
        
        # Common threshold field patterns
        threshold_patterns = [
            ("min", ["min", "minimum", "min_required", "min_threshold"]),
            ("max", ["max", "maximum", "max_allowed", "max_threshold"]),
            ("target", ["target", "expected", "goal"]),
            ("actual", ["actual", "current", "value", "score"])
        ]
        
        for threshold_type, field_names in threshold_patterns:
            for field_name in field_names:
                if field_name in self.details:
                    threshold_info[threshold_type] = self.details[field_name]
                    break
                    
        return threshold_info

    def get_failure_reason(self) -> str:
        """Get the reason for validation failure.
        
        Returns:
            Human-readable failure reason, or empty string if passed
        """
        if self.passed:
            return ""
            
        # Look for common failure reason fields
        reason_fields = ["reason", "error", "message", "failure_reason", "description"]
        
        for field in reason_fields:
            if field in self.details and isinstance(self.details[field], str):
                return self.details[field]
                
        # Generate reason from threshold info if available
        threshold_info = self.get_threshold_info()
        if "actual" in threshold_info:
            actual = threshold_info["actual"]
            
            if "min" in threshold_info and actual < threshold_info["min"]:
                return f"Value {actual} is below minimum threshold {threshold_info['min']}"
            elif "max" in threshold_info and actual > threshold_info["max"]:
                return f"Value {actual} exceeds maximum threshold {threshold_info['max']}"
                
        return "Validation failed (no specific reason provided)"

    def get_validation_score(self) -> float:
        """Get a normalized validation score (0.0 to 1.0).
        
        Returns:
            Validation score, or 1.0 if passed, 0.0 if failed with no score
        """
        if self.passed:
            return 1.0
            
        # Look for score fields
        score_fields = ["score", "confidence", "rating", "percentage"]
        
        for field in score_fields:
            if field in self.details:
                value = self.details[field]
                if isinstance(value, (int, float)):
                    # Normalize to 0-1 range
                    if value <= 1.0:
                        return max(0.0, min(1.0, float(value)))
                    elif value <= 100.0:
                        return max(0.0, min(1.0, float(value) / 100.0))
                        
        return 0.0

    def is_critical_failure(self) -> bool:
        """Check if this is a critical validation failure.
        
        Returns:
            True if this is marked as a critical failure
        """
        if self.passed:
            return False
            
        # Look for criticality indicators
        critical_indicators = [
            ("critical", True),
            ("severity", "critical"),
            ("level", "error"),
            ("priority", "high")
        ]
        
        for field, critical_value in critical_indicators:
            if field in self.details and self.details[field] == critical_value:
                return True
                
        return False

    def get_suggestions(self) -> List[str]:
        """Get improvement suggestions from validation details.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Look for suggestion fields
        suggestion_fields = ["suggestions", "recommendations", "tips", "advice"]
        
        for field in suggestion_fields:
            if field in self.details:
                value = self.details[field]
                if isinstance(value, list):
                    suggestions.extend([str(s) for s in value])
                elif isinstance(value, str):
                    suggestions.append(value)
                    
        return suggestions

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this validation result.
        
        Returns:
            Dictionary with validation summary
        """
        return {
            "validator": self.validator,
            "iteration": self.iteration,
            "passed": self.passed,
            "score": self.get_validation_score(),
            "is_critical": self.is_critical_failure(),
            "failure_reason": self.get_failure_reason() if not self.passed else None,
            "numeric_metrics": self.get_numeric_metrics(),
            "suggestions_count": len(self.get_suggestions()),
            "timestamp": self.timestamp
        }
