"""Pydantic models for critic results and feedback.

This module defines structured models for critic feedback, violations,
suggestions, and results to ensure consistent data representation
across the Sifaka critic system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    """Severity levels for violations and issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceScore(BaseModel):
    """Detailed confidence scoring for critic feedback."""
    
    overall: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    content_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    grammar_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    factual_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    coherence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    calculation_method: Optional[str] = None
    factors_considered: List[str] = Field(default_factory=list)
    uncertainty_sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class ViolationReport(BaseModel):
    """Detailed report of a specific violation or issue."""
    
    violation_type: str
    description: str
    severity: SeverityLevel
    
    location: Optional[str] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    rule_violated: Optional[str] = None
    evidence: Optional[str] = None
    suggested_fix: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class ImprovementSuggestion(BaseModel):
    """Structured improvement suggestion with metadata."""
    
    suggestion: str
    category: str
    priority: Optional[int] = Field(None, ge=1, le=10)
    expected_impact: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    depends_on: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class CritiqueFeedback(BaseModel):
    """Main structured critique feedback model."""
    
    message: str
    needs_improvement: bool
    violations: List[ViolationReport] = Field(default_factory=list)
    suggestions: List[ImprovementSuggestion] = Field(default_factory=list)
    confidence: ConfidenceScore
    critic_name: str
    critic_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class CriticResult(BaseModel):
    """Complete result of a critic operation."""
    
    feedback: CritiqueFeedback
    operation_type: str
    success: bool = True
    total_processing_time_ms: float
    input_text_length: int
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('error_message')
    @classmethod
    def validate_error_message(cls, v, info):
        """Validate that error message is provided when success=False."""
        if not info.data.get('success', True) and not v:
            raise ValueError("Error message is required when success=False")
        return v
    
    class Config:
        frozen = True
