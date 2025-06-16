"""CritiqueResult model for Sifaka.

This module contains the CritiqueResult model that captures critic feedback
with structured suggestions for iterative improvement.

Extracted from the monolithic thought.py file to improve maintainability.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CritiqueResult(BaseModel):
    """Result of a critique operation.

    Captures critic feedback with structured suggestions for iterative improvement.
    Enhanced with rich metadata for observability and debugging.
    
    This model tracks:
    - Critic feedback and improvement suggestions
    - Confidence and reasoning information
    - Performance metrics and research methodology
    - Tool usage and retrieval context
    
    Example:
        ```python
        critique = CritiqueResult(
            iteration=1,
            critic="reflexion_critic",
            feedback="The explanation could be more detailed...",
            suggestions=["Add specific examples", "Include statistics"],
            timestamp=datetime.now(),
            confidence=0.85,
            reasoning="Based on clarity and completeness analysis",
            needs_improvement=True,
            processing_time_ms=1250.5,
            model_name="openai:gpt-4",
            paper_reference="Reflexion: Language Agents with Verbal Reinforcement Learning"
        )
        ```
    """

    iteration: int
    critic: str
    feedback: str
    suggestions: List[str]
    timestamp: datetime

    # Rich metadata for observability
    confidence: Optional[float] = None  # 0.0 to 1.0
    reasoning: Optional[str] = None
    needs_improvement: bool = True

    # Critic-specific metadata
    critic_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Performance metrics
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None

    # Research paper information
    paper_reference: Optional[str] = None
    methodology: Optional[str] = None

    # Tool usage tracking
    tools_used: List[str] = Field(default_factory=list)
    retrieval_context: Optional[Dict[str, Any]] = None

    def get_confidence_level(self) -> str:
        """Get a human-readable confidence level.
        
        Returns:
            Confidence level as string (very_low, low, medium, high, very_high)
        """
        if self.confidence is None:
            return "unknown"
            
        if self.confidence >= 0.9:
            return "very_high"
        elif self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        elif self.confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    def get_processing_time_seconds(self) -> Optional[float]:
        """Get processing time in seconds.
        
        Returns:
            Processing time in seconds, or None if not available
        """
        if self.processing_time_ms is None:
            return None
        return self.processing_time_ms / 1000.0

    def has_actionable_suggestions(self) -> bool:
        """Check if this critique has actionable suggestions.
        
        Returns:
            True if there are non-empty suggestions
        """
        return bool(self.suggestions and any(s.strip() for s in self.suggestions))

    def get_suggestion_categories(self) -> Dict[str, List[str]]:
        """Categorize suggestions by type.
        
        Returns:
            Dictionary with categorized suggestions
        """
        categories = {
            "content": [],
            "structure": [],
            "style": [],
            "accuracy": [],
            "clarity": [],
            "other": []
        }
        
        # Keywords for categorization
        category_keywords = {
            "content": ["add", "include", "mention", "discuss", "explain", "detail", "example"],
            "structure": ["organize", "structure", "order", "flow", "paragraph", "section"],
            "style": ["tone", "voice", "style", "formal", "informal", "concise", "verbose"],
            "accuracy": ["correct", "accurate", "fact", "verify", "check", "error"],
            "clarity": ["clear", "clarify", "confusing", "ambiguous", "understand", "readable"]
        }
        
        for suggestion in self.suggestions:
            suggestion_lower = suggestion.lower()
            categorized = False
            
            for category, keywords in category_keywords.items():
                if any(keyword in suggestion_lower for keyword in keywords):
                    categories[category].append(suggestion)
                    categorized = True
                    break
                    
            if not categorized:
                categories["other"].append(suggestion)
                
        return {k: v for k, v in categories.items() if v}  # Remove empty categories

    def get_research_info(self) -> Dict[str, Any]:
        """Get research methodology information.
        
        Returns:
            Dictionary with research information
        """
        return {
            "paper_reference": self.paper_reference,
            "methodology": self.methodology,
            "has_research_backing": bool(self.paper_reference or self.methodology)
        }

    def get_tool_usage_summary(self) -> Dict[str, Any]:
        """Get summary of tool usage during critique.
        
        Returns:
            Dictionary with tool usage information
        """
        return {
            "tools_count": len(self.tools_used),
            "tools_used": self.tools_used,
            "has_retrieval_context": bool(self.retrieval_context),
            "retrieval_context_size": len(self.retrieval_context) if self.retrieval_context else 0
        }

    def get_improvement_priority(self) -> str:
        """Get improvement priority based on confidence and suggestions.
        
        Returns:
            Priority level (critical, high, medium, low)
        """
        if not self.needs_improvement:
            return "none"
            
        if not self.has_actionable_suggestions():
            return "low"
            
        confidence_level = self.get_confidence_level()
        suggestion_count = len(self.suggestions)
        
        if confidence_level in ["very_high", "high"] and suggestion_count >= 3:
            return "critical"
        elif confidence_level in ["high", "medium"] and suggestion_count >= 2:
            return "high"
        elif confidence_level in ["medium"] or suggestion_count >= 1:
            return "medium"
        else:
            return "low"

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary of all metadata.
        
        Returns:
            Dictionary with metadata summary
        """
        return {
            "has_confidence": self.confidence is not None,
            "has_reasoning": bool(self.reasoning),
            "has_performance_metrics": self.processing_time_ms is not None,
            "has_model_info": bool(self.model_name),
            "has_research_info": bool(self.paper_reference or self.methodology),
            "has_tool_usage": bool(self.tools_used),
            "metadata_fields": len(self.critic_metadata)
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of this critique result.
        
        Returns:
            Dictionary with critique summary
        """
        return {
            "critic": self.critic,
            "iteration": self.iteration,
            "needs_improvement": self.needs_improvement,
            "confidence_level": self.get_confidence_level(),
            "suggestions_count": len(self.suggestions),
            "has_actionable_suggestions": self.has_actionable_suggestions(),
            "improvement_priority": self.get_improvement_priority(),
            "processing_time_seconds": self.get_processing_time_seconds(),
            "suggestion_categories": list(self.get_suggestion_categories().keys()),
            "research_backed": bool(self.paper_reference),
            "tools_used_count": len(self.tools_used),
            "timestamp": self.timestamp
        }
