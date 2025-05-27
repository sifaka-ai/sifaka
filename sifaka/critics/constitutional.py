"""Constitutional critic for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on "Constitutional AI: Harmlessness from AI Feedback":
https://arxiv.org/abs/2212.08073

@misc{bai2022constitutionalaiharmlessnessai,
      title={Constitutional AI: Harmlessness from AI Feedback},
      author={Yuntao Bai and Saurav Kadavath and Sandipan Kundu and Amanda Askell and Jackson Kernion and Andy Jones and Anna Chen and Anna Goldie and Azalia Mirhoseini and Cameron McKinnon and Carol Chen and Catherine Olsson and Christopher Olah and Danny Hernandez and Dawn Drain and Deep Ganguli and Dustin Li and Eli Tran-Johnson and Ethan Perez and Jamie Kerr and Jared Mueller and Jeffrey Ladish and Joshua Landau and Kamal Ndousse and Kamile Lukosuite and Liane Lovitt and Michael Sellitto and Nelson Elhage and Nicholas Schiefer and Noemi Mercado and Nova DasSarma and Robert Lasenby and Robin Larson and Sam Ringer and Scott Johnston and Shauna Kravec and Sheer El Showk and Stanislav Fort and Tamera Lanham and Timothy Telleen-Lawton and Tom Conerly and Tom Henighan and Tristan Hume and Samuel R. Bowman and Zac Hatfield-Dodds and Ben Mann and Dario Amodei and Nicholas Joseph and Sam McCandlish and Tom Brown and Jared Kaplan},
      year={2022},
      eprint={2212.08073},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.08073},
}

The ConstitutionalCritic implements key Constitutional AI concepts:
1. Principle-based evaluation against a written constitution
2. Natural language feedback on principle violations
3. Iterative improvement through constitutional critique
4. Harmlessness assessment through AI feedback
5. Learning from principle violation patterns and contextual importance (enhanced)
6. Adaptive principle weighting based on task types and effectiveness (enhanced)

Note: This implementation captures core Constitutional AI principles with enhanced
learning capabilities through integration with the Sifaka thoughts system.
The critic learns which principles are most important for different contexts.
"""

import json
import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ConstitutionalCritic(BaseCritic):
    """Critic that evaluates text against constitutional principles.

    This critic implements the Constitutional AI approach by evaluating text
    against a set of predefined principles (a "constitution"). It provides
    detailed feedback on principle violations and suggests improvements.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        principles: Optional[List[str]] = None,
        strict_mode: bool = False,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Constitutional critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            principles: List of constitutional principles to evaluate against.
            strict_mode: Whether to require all principles to be satisfied.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        # Set up principles (constitution)
        self.principles = principles or [
            "Do not provide harmful, offensive, or biased content.",
            "Explain reasoning in a clear and truthful manner.",
            "Respect user autonomy and avoid manipulative language.",
            "Provide accurate and factual information.",
            "Be helpful and constructive in responses.",
        ]

        self.strict_mode = strict_mode

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "Evaluate the following text against the constitutional principles provided.\n\n"
            "Constitutional Principles:\n{principles}\n\n"
            "Original task: {prompt}\n\n"
            "Text to evaluate:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please analyze the text against each principle and provide:\n\n"
            "Issues:\n- [List any principle violations here]\n\n"
            "Suggestions:\n- [List specific suggestions for addressing violations]\n\n"
            "Overall Assessment: [Brief assessment of constitutional compliance]\n\n"
            "If the text adheres to all principles, please state that clearly.\n"
            "Be specific and constructive in your feedback."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text to better align with the constitutional principles.\n\n"
            "Constitutional Principles:\n{principles}\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Constitutional critique:\n{critique}\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses all principle violations identified in the critique\n"
            "2. Maintains the core message and usefulness of the response\n"
            "3. Fully adheres to all constitutional principles\n"
            "4. Remains helpful and relevant to the original task\n"
            "5. Better incorporates factual information from the context (if available)\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Constitutional AI approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Extract learning context from thought for enhanced constitutional evaluation
        learning_context = self._extract_constitutional_learning_context(thought)

        # Format principles for the prompt with learned weights
        principles_text = self._format_principles_with_learning(learning_context)

        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create enhanced critique prompt with learning context
        critique_prompt = self._build_enhanced_constitutional_prompt(
            thought.prompt, thought.text, context, principles_text, learning_context
        )

        # Generate critique with enhanced constitutional awareness
        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_message="You are an expert constitutional AI evaluator with learning from past principle violation patterns.",
        )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)
        violations = self._extract_violations(critique_response)

        # Determine if improvement is needed
        needs_improvement = len(violations) > 0 or len(issues) > 0
        if self.strict_mode:
            needs_improvement = needs_improvement or "concern" in critique_response.lower()

        # Calculate confidence based on violations found
        confidence = 1.0 - (len(violations) / len(self.principles)) if self.principles else 1.0
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Store constitutional learning outcomes for future evaluations
        self._store_constitutional_outcomes(
            thought, learning_context, violations, confidence, issues
        )

        logger.debug(
            f"ConstitutionalCritic: Found {len(violations)} violations, confidence: {confidence:.2f} with learning integration"
        )

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": confidence,
            "metadata": {
                "principle_violations": violations,
                "principles_evaluated": len(self.principles),
                "strict_mode": self.strict_mode,
                "learning_applied": bool(learning_context.get("principle_weights")),
                "task_type": learning_context.get("task_type", "general"),
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text to align with constitutional principles.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that better aligns with constitutional principles.

        Raises:
            ImproverError: If the improvement fails.
        """
        start_time = time.time()

        with critic_context(
            critic_name="ConstitutionalCritic",
            operation="improve",
            message_prefix="Failed to improve text with Constitutional principles",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="ConstitutionalCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            critique = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "ConstitutionalCritic":
                        critique = feedback.feedback
                        break

            # If no critique available, generate one
            if not critique:
                logger.debug("No critique found in thought, generating new critique")
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._perform_critique_async(thought))
                        critique_result = future.result()
                except RuntimeError:
                    critique_result = asyncio.run(self._perform_critique_async(thought))

                critique = critique_result["message"]

            # Format principles for the prompt
            principles_text = "\n".join(
                f"{i+1}. {principle}" for i, principle in enumerate(self.principles)
            )

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with context
            improve_prompt = self.improve_prompt_template.format(
                principles=principles_text,
                prompt=thought.prompt,
                text=thought.text,
                critique=critique,
                context=context,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor improving text to align with constitutional principles.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"ConstitutionalCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _parse_critique(self, critique: str) -> tuple[List[str], List[str]]:
        """Parse critique text to extract issues and suggestions.

        Args:
            critique: The critique text to parse.

        Returns:
            A tuple of (issues, suggestions) lists.
        """
        issues = []
        suggestions = []

        # Simple parsing logic
        in_issues = False
        in_suggestions = False

        for line in critique.split("\n"):
            line = line.strip()
            if line.lower().startswith("issues:"):
                in_issues = True
                in_suggestions = False
                continue
            elif line.lower().startswith("suggestions:"):
                in_issues = False
                in_suggestions = True
                continue
            elif line.lower().startswith("overall assessment:"):
                in_issues = False
                in_suggestions = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        # If no structured format found, extract from general content
        if not issues and not suggestions:
            critique_lower = critique.lower()
            if any(
                word in critique_lower for word in ["violation", "violates", "fails to", "does not"]
            ):
                issues.append("Constitutional principle violations identified")
            if any(word in critique_lower for word in ["improve", "suggest", "should", "consider"]):
                suggestions.append("See critique for constitutional improvement suggestions")

        return issues, suggestions

    def _extract_violations(self, critique: str) -> List[Dict[str, Any]]:
        """Extract principle violations from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of violation dictionaries.
        """
        violations = []

        # Try to find JSON in the critique first
        try:
            if "{" in critique and "}" in critique:
                start = critique.find("{")
                end = critique.rfind("}") + 1
                json_str = critique[start:end]
                data = json.loads(json_str)

                # Extract violations from structured data
                if "violations" in data:
                    for violation in data["violations"]:
                        if isinstance(violation, dict):
                            violations.append(violation)
                        else:
                            violations.append(
                                {
                                    "type": "principle_violation",
                                    "description": str(violation),
                                    "severity": "unknown",
                                }
                            )
                return violations
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: simple text parsing
        lines = critique.split("\n")
        for line in lines:
            line_clean = line.strip().lower()
            if not line_clean:
                continue

            # Look for lines that clearly indicate violations
            if any(
                phrase in line_clean
                for phrase in [
                    "violation:",
                    "violates principle",
                    "does not adhere to",
                    "fails to meet",
                ]
            ):
                # Skip obvious non-violations
                if not any(
                    phrase in line_clean
                    for phrase in ["no violation", "violation: none", "no violations found"]
                ):
                    violations.append(
                        {
                            "type": "principle_violation",
                            "description": line.strip(),
                            "severity": "medium",
                        }
                    )

        return violations

    def _extract_constitutional_learning_context(self, thought: Thought) -> Dict[str, Any]:
        """Extract learning context from thought for enhanced constitutional evaluation.

        Args:
            thought: The Thought to extract learning context from.

        Returns:
            Dictionary with constitutional learning context.
        """
        learning_context = {
            "constitutional_sessions": 0,
            "principle_violations": {},
            "principle_weights": {},
            "violation_patterns": [],
            "task_type": self._classify_constitutional_task_type(thought.prompt),
        }

        # Extract from thought metadata
        if thought.metadata:
            constitutional_data = thought.metadata.get("constitutional_memory", {})
            if constitutional_data:
                learning_context["constitutional_sessions"] = len(
                    constitutional_data.get("sessions", [])
                )
                learning_context["principle_violations"] = constitutional_data.get(
                    "principle_violations", {}
                )
                learning_context["principle_weights"] = constitutional_data.get(
                    "principle_weights", {}
                )
                learning_context["violation_patterns"] = constitutional_data.get(
                    "violation_patterns", []
                )[
                    -10:
                ]  # Last 10

        # Extract from thought history
        if thought.history:
            learning_context["previous_attempts"] = len(thought.history)

        # Extract from critic feedback history
        if thought.critic_feedback:
            constitutional_feedback = [
                f for f in thought.critic_feedback if f.critic_name == "ConstitutionalCritic"
            ]
            if constitutional_feedback:
                learning_context["previous_feedback_count"] = len(constitutional_feedback)
                # Analyze principle violation patterns from previous feedback
                for feedback in constitutional_feedback[-5:]:  # Last 5 feedback instances
                    if feedback.metadata and "principle_violations" in feedback.metadata:
                        violations = feedback.metadata["principle_violations"]
                        for violation in violations:
                            violation_type = violation.get("type", "unknown")
                            if violation_type not in learning_context["principle_violations"]:
                                learning_context["principle_violations"][violation_type] = 0
                            learning_context["principle_violations"][violation_type] += 1

        return learning_context

    def _classify_constitutional_task_type(self, prompt: str) -> str:
        """Classify the task type for constitutional learning purposes.

        Args:
            prompt: The task prompt to classify.

        Returns:
            String representing the constitutional task type.
        """
        prompt_lower = prompt.lower()

        # Safety-critical tasks that need strong constitutional oversight
        if any(word in prompt_lower for word in ["advice", "recommendation", "should", "guidance"]):
            return "advisory"
        elif any(
            word in prompt_lower for word in ["sensitive", "controversial", "political", "religion"]
        ):
            return "sensitive"
        elif any(word in prompt_lower for word in ["medical", "health", "treatment", "diagnosis"]):
            return "medical"
        elif any(word in prompt_lower for word in ["legal", "law", "court", "lawsuit"]):
            return "legal"
        elif any(word in prompt_lower for word in ["financial", "investment", "money", "trading"]):
            return "financial"
        elif any(word in prompt_lower for word in ["personal", "private", "confidential"]):
            return "personal"
        elif any(word in prompt_lower for word in ["creative", "story", "fiction", "poem"]):
            return "creative"
        elif any(word in prompt_lower for word in ["technical", "code", "programming", "software"]):
            return "technical"
        else:
            return "general"

    def _format_principles_with_learning(self, learning_context: Dict[str, Any]) -> str:
        """Format principles with learned weights and importance.

        Args:
            learning_context: Learning context from past constitutional evaluations.

        Returns:
            Formatted principles string with learned emphasis.
        """
        task_type = learning_context.get("task_type", "general")
        principle_weights = learning_context.get("principle_weights", {})

        formatted_principles = []
        for i, principle in enumerate(self.principles):
            principle_key = f"principle_{i}"
            weight = principle_weights.get(task_type, {}).get(principle_key, 1.0)

            # Add emphasis based on learned importance
            if weight > 1.5:
                emphasis = " [HIGH PRIORITY]"
            elif weight > 1.2:
                emphasis = " [IMPORTANT]"
            elif weight < 0.8:
                emphasis = " [LOWER PRIORITY]"
            else:
                emphasis = ""

            formatted_principles.append(f"{i+1}. {principle}{emphasis}")

        return "\n".join(formatted_principles)

    def _build_enhanced_constitutional_prompt(
        self,
        prompt: str,
        text: str,
        context: str,
        principles_text: str,
        learning_context: Dict[str, Any],
    ) -> str:
        """Build enhanced constitutional prompt with learning context.

        Args:
            prompt: The original task prompt.
            text: The text to evaluate.
            context: Retrieved context.
            principles_text: Formatted principles with learned weights.
            learning_context: Learning context from past evaluations.

        Returns:
            Enhanced constitutional prompt string.
        """
        base_prompt = self.critique_prompt_template.format(
            principles=principles_text,
            prompt=prompt,
            text=text,
            context=context,
        )

        # Add learning enhancements if available
        task_type = learning_context.get("task_type", "general")
        violation_patterns = learning_context.get("violation_patterns", [])

        if violation_patterns or learning_context.get("constitutional_sessions", 0) > 3:
            learning_section = "\n\nConstitutional Learning Context:\n"

            if violation_patterns:
                learning_section += f"Common violation patterns for {task_type} tasks:\n"
                for pattern in violation_patterns[-5:]:  # Last 5 patterns
                    learning_section += f"- {pattern}\n"

            if learning_context.get("principle_violations"):
                learning_section += "Previously violated principles to pay special attention to:\n"
                violations = learning_context["principle_violations"]
                for violation_type, count in sorted(
                    violations.items(), key=lambda x: x[1], reverse=True
                )[:3]:
                    learning_section += f"- {violation_type} (occurred {count} times)\n"

            learning_section += f"\nThis is a {task_type} task. Focus on principles most relevant to this context.\n"

            base_prompt += learning_section

        return base_prompt

    def _store_constitutional_outcomes(
        self,
        thought: Thought,
        learning_context: Dict[str, Any],
        violations: List[Dict[str, Any]],
        confidence: float,
        issues: List[str],
    ) -> None:
        """Store constitutional outcomes in thought metadata for future learning.

        Args:
            thought: The Thought to store outcomes in.
            learning_context: The learning context used.
            violations: The violations found.
            confidence: The confidence score.
            issues: The issues identified.
        """
        if not thought.metadata:
            thought.metadata = {}

        # Initialize constitutional memory if not exists
        if "constitutional_memory" not in thought.metadata:
            thought.metadata["constitutional_memory"] = {
                "sessions": [],
                "principle_violations": {},
                "principle_weights": {},
                "violation_patterns": [],
            }

        # Analyze this constitutional session
        task_type = learning_context.get("task_type", "general")
        session_data = {
            "session_id": f"constitutional_session_{int(time.time())}",
            "task_type": task_type,
            "violations_found": len(violations),
            "confidence": confidence,
            "issues_count": len(issues),
            "principles_evaluated": len(self.principles),
            "timestamp": time.time(),
        }

        # Update principle violation tracking
        for violation in violations:
            violation_type = violation.get("type", "unknown")
            if (
                violation_type
                not in thought.metadata["constitutional_memory"]["principle_violations"]
            ):
                thought.metadata["constitutional_memory"]["principle_violations"][
                    violation_type
                ] = 0
            thought.metadata["constitutional_memory"]["principle_violations"][violation_type] += 1

        # Update principle weights based on violations and task type
        if task_type not in thought.metadata["constitutional_memory"]["principle_weights"]:
            thought.metadata["constitutional_memory"]["principle_weights"][task_type] = {}

        weights = thought.metadata["constitutional_memory"]["principle_weights"][task_type]

        # Increase weight for principles that were violated (need more attention)
        for i, principle in enumerate(self.principles):
            principle_key = f"principle_{i}"
            if principle_key not in weights:
                weights[principle_key] = 1.0

            # Check if this principle was violated
            principle_violated = any(
                principle.lower() in violation.get("description", "").lower()
                for violation in violations
            )

            if principle_violated:
                weights[principle_key] = min(2.0, weights[principle_key] + 0.2)  # Increase weight
            elif confidence > 0.9:  # High confidence, no violations
                weights[principle_key] = max(
                    0.5, weights[principle_key] - 0.05
                )  # Slightly decrease weight

        # Store violation patterns for learning
        if violations:
            for violation in violations:
                pattern = f"{task_type}: {violation.get('type', 'unknown')} - {violation.get('description', '')[:100]}"
                thought.metadata["constitutional_memory"]["violation_patterns"].append(pattern)

        # Store this session
        thought.metadata["constitutional_memory"]["sessions"].append(session_data)

        # Keep only last 20 sessions
        if len(thought.metadata["constitutional_memory"]["sessions"]) > 20:
            thought.metadata["constitutional_memory"]["sessions"] = thought.metadata[
                "constitutional_memory"
            ]["sessions"][-20:]

        # Keep only last 30 violation patterns
        if len(thought.metadata["constitutional_memory"]["violation_patterns"]) > 30:
            thought.metadata["constitutional_memory"]["violation_patterns"] = thought.metadata[
                "constitutional_memory"
            ]["violation_patterns"][-30:]
