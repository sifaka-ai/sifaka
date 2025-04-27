"""
LLM-based classifier implementation.
"""

from typing import List, Dict, Any, Optional
import json

from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.models.base import ModelProvider


class LLMClassifier(Classifier):
    """
    A classifier that uses an LLM for predictions.

    This allows for flexible classification tasks using LLM capabilities.

    Attributes:
        model: The LLM provider to use
        system_prompt: System prompt for classification
        user_prompt_template: Template for user prompts
        output_format: Expected format of LLM output
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: ModelProvider,
        labels: List[str],
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        config: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LLM classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            model: The LLM provider to use
            labels: List of possible labels/classes
            system_prompt: System prompt for classification
            user_prompt_template: Template for user prompts
            config: Additional configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=labels,
            cost=5,  # Higher cost for LLM API calls
            **kwargs,
        )

        self.model = model
        self.system_prompt = system_prompt or (
            f"You are a classifier that assigns one of the following labels: {', '.join(labels)}. "
            "Respond with a JSON object containing 'label' and 'confidence' (0-1) fields."
        )
        self.user_prompt_template = user_prompt_template or (
            "Classify the following text:\n\n{text}\n\n"
            "Respond with a JSON object containing:\n"
            "- label: one of {labels}\n"
            "- confidence: number between 0 and 1\n"
            "- explanation: brief explanation of the classification"
        )

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.

        Args:
            response: Raw LLM response

        Returns:
            Dictionary with parsed data
        """
        try:
            # Try to find JSON in the response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data
            raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract label and confidence using simple heuristics
            lines = response.lower().split("\n")
            data = {}

            for line in lines:
                if "label" in line and ":" in line:
                    data["label"] = line.split(":")[1].strip().strip("\"'")
                elif "confidence" in line and ":" in line:
                    try:
                        conf = float(line.split(":")[1].strip().strip("\"'"))
                        data["confidence"] = min(max(conf, 0), 1)  # Clamp to [0,1]
                    except ValueError:
                        pass
                elif "explanation" in line and ":" in line:
                    data["explanation"] = line.split(":")[1].strip().strip("\"'")

            if "label" in data and "confidence" in data:
                return data

            # If all else fails, make a best effort guess
            return {
                "label": self.labels[0],
                "confidence": 0.5,
                "explanation": "Failed to parse LLM response",
            }

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text using the LLM.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with LLM's prediction
        """
        prompt = self.user_prompt_template.format(text=text, labels=self.labels)

        response = self.model.generate(
            prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,  # Low temperature for more consistent results
        )

        result = self._parse_llm_response(response)

        return ClassificationResult(
            label=result["label"],
            confidence=result["confidence"],
            metadata={"explanation": result.get("explanation", ""), "raw_response": response},
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts using the LLM.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
