from typing import Any, Dict, Optional, List
from sifaka.critics.base import BaseCritic
from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

class SelfEvaluationCritic(BaseCritic):
    """Critic that enables agents to evaluate their own outputs.
    
    Features:
    - Task completion verification
    - Output quality assessment
    - Failure detection
    - Improvement suggestions
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        evaluation_criteria: Optional[List[str]] = None,
        min_score_threshold: float = 0.7,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.evaluation_criteria = evaluation_criteria or [
            "Task Completion: Has the task been fully completed?",
            "Output Quality: Is the output clear, coherent, and well-structured?",
            "Accuracy: Is all information accurate and well-supported?",
            "Edge Cases: Have potential edge cases been considered?",
            "Failure Modes: Are there any potential points of failure?"
        ]
        self.min_score_threshold = min_score_threshold

    async def critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform self-evaluation of the agent's output."""
        
        evaluation_prompt = self._build_evaluation_prompt(thought)
        evaluation_result = await self.model.generate_async(evaluation_prompt)
        
        try:
            parsed_result = self._parse_evaluation(evaluation_result)
            metrics = self._calculate_metrics(parsed_result)
            
            return {
                "score": metrics["overall_score"],
                "passed": metrics["overall_score"] >= self.min_score_threshold,
                "metrics": metrics,
                "issues": parsed_result["issues"],
                "suggestions": parsed_result["suggestions"],
                "evaluation_details": parsed_result["criteria_scores"]
            }
        except Exception as e:
            logger.error(f"Self-evaluation failed: {e}")
            return self._create_error_feedback(str(e))

    def _build_evaluation_prompt(self, thought: Thought) -> str:
        return f"""
        As an AI self-evaluation system, analyze the following output:

        Original Task: {thought.prompt}
        Generated Output: {thought.text}

        Evaluate based on these criteria:
        {self._format_criteria()}

        Provide a structured evaluation following this format:
        CRITERIA SCORES:
        - [Criterion]: [Score 0-10] - [Justification]

        ISSUES:
        - [List specific issues found]

        SUGGESTIONS:
        - [List specific improvement suggestions]

        OVERALL ASSESSMENT:
        [Provide a brief overall assessment]
        """

    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse the evaluation response into structured data."""
        # Implementation for parsing the evaluation text
        # Returns dict with scores, issues, suggestions
