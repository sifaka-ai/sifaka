"""Response parsing for critic outputs."""

import re
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CriticResponse(BaseModel):
    """Standardized response format for all critics."""

    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(..., description="Main feedback about the text")
    suggestions: List[str] = Field(
        default_factory=list, description="Specific improvement suggestions"  
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional critic-specific data"
    )


class ResponseParser:
    """Handles parsing of LLM responses into structured format."""
    
    def parse(self, response: str, format_type: str = "json") -> CriticResponse:
        """Parse response based on format type.
        
        Args:
            response: Raw LLM response
            format_type: Type of response format (json only supported now)
            
        Returns:
            Parsed CriticResponse
        """
        if format_type != "json":
            # Force JSON format for consistency
            format_type = "json"
            
        return self.parse_json(response)
    
    def parse_json(self, response: str) -> CriticResponse:
        """Parse JSON-formatted response.
        
        Args:
            response: Raw JSON response from LLM
            
        Returns:
            Parsed CriticResponse
        """
        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )
            if json_match:
                response = json_match.group(1)
            
            # Parse JSON
            data = json.loads(response)
            
            # Ensure all required fields
            return CriticResponse(
                feedback=data.get("feedback", "No feedback provided"),
                suggestions=data.get("suggestions", []),
                needs_improvement=data.get("needs_improvement", True),
                confidence=data.get("confidence", 0.7),
                metadata=data.get("metadata", {})
            )
            
        except (json.JSONDecodeError, ValueError):
            # Fallback to extracting what we can
            return self._extract_from_text(response)
    
    def _extract_from_text(self, response: str) -> CriticResponse:
        """Extract critique information from unstructured text.
        
        Args:
            response: Unstructured text response
            
        Returns:
            Best-effort CriticResponse
        """
        # Use the full response as feedback
        feedback = response.strip()
        
        # Try to extract suggestions from bullet points or numbered lists
        suggestions = self._extract_list_items(response)
        
        # Look for improvement indicators
        needs_improvement = self._detect_needs_improvement(feedback)
        
        # Conservative confidence for fallback parsing
        confidence = 0.5
        
        return CriticResponse(
            feedback=feedback,
            suggestions=suggestions or ["Consider the feedback provided above"],
            needs_improvement=needs_improvement,
            confidence=confidence,
            metadata={"parse_method": "text_fallback"}
        )
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract numbered or bulleted list items from text."""
        items = []
        
        # Match numbered lists (1. 2. etc) or bullet points (-, *, •)
        list_patterns = [
            r"^\s*\d+[\.)]\s*(.+)$",  # Numbered
            r"^\s*[-*•]\s*(.+)$",      # Bullets
        ]
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    items.append(match.group(1).strip())
                    break
        
        return items
    
    def _detect_needs_improvement(self, feedback: str) -> bool:
        """Detect if the text needs improvement based on feedback."""
        feedback_lower = feedback.lower()
        
        # Negative indicators suggest improvement needed
        negative_indicators = [
            "needs improvement", "could be better", "should", "lacking",
            "insufficient", "weak", "poor", "missing", "unclear", "confusing",
            "needs more", "requires", "must", "fix", "revise", "rework",
            "not enough", "inadequate", "fails to", "doesn't"
        ]
        
        # Positive indicators suggest no improvement needed
        positive_indicators = [
            "excellent", "perfect", "outstanding", "no issues", "well-written",
            "comprehensive", "clear", "effective", "strong", "solid",
            "good job", "well done", "great", "no changes needed"
        ]
        
        # Count indicators
        negative_count = sum(1 for ind in negative_indicators if ind in feedback_lower)
        positive_count = sum(1 for ind in positive_indicators if ind in feedback_lower)
        
        # More negative than positive = needs improvement
        return negative_count > positive_count