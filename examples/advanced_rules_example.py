#!/usr/bin/env python3
"""
Advanced Rules Example for Sifaka.

This example demonstrates:
1. Creating custom rule validators with domain-specific logic
2. Composing multiple rules into composite validators
3. Using pattern matching for content validation
4. Implementing domain-specific rules (financial, medical, legal)
5. Using an LLM critic to improve non-compliant content

Usage:
    python advanced_rules_example.py

Requirements:
    - Python environment with Sifaka installed
    - OpenAI API key in OPENAI_API_KEY environment variable
"""

import os
import re
import sys
from typing import List, Optional, Set

# Add parent directory to system path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dotenv package. Install with: pip install python-dotenv")
    sys.exit(1)

from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RulePriority,
    RuleResult,
    RuleValidator,
)
from sifaka.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class FinancialContentValidator(RuleValidator[str]):
    """Validator for financial content compliance with regulatory requirements."""

    def __init__(self):
        """Initialize the validator with financial regulatory rules."""
        self.currency_pattern = re.compile(r"\$\d+(?:\.\d{2})?")
        self.percentage_pattern = re.compile(r"\d+(?:\.\d+)?%")
        self.disclaimers = {
            "past performance is not indicative of future results",
            "this is not financial advice",
            "investments may lose value",
            "consult a financial advisor",
            "consider your investment objectives",
            "read the prospectus carefully",
        }
        self.required_terms = {"risk", "investment", "return"}
        self.prohibited_terms = {
            "guaranteed returns",
            "risk-free",
            "certain profit",
            "guaranteed profit",
            "can't lose",
        }

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate financial content for regulatory compliance."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")

        output_lower = output.lower()

        # Extract numerical claims first
        currency_mentions = self.currency_pattern.findall(output)
        percentage_mentions = self.percentage_pattern.findall(output)

        # Check if content contains financial terms
        has_financial_terms = any(term in output_lower for term in self.required_terms)
        has_currency = bool(self.currency_pattern.search(output))
        has_percentages = bool(self.percentage_pattern.search(output))

        # If no financial content, pass automatically
        is_financial_content = has_financial_terms or has_currency or has_percentages
        if not is_financial_content:
            return RuleResult(
                passed=True,
                message="No financial content detected",
                metadata={"has_financial_content": False},
            )

        # Check for required disclaimers - only need one for basic content, two for specific claims
        found_disclaimers = [d for d in self.disclaimers if d in output_lower]

        # Check for specific financial claims that require more disclaimers
        has_specific_claims = (
            "profit" in output_lower
            or "return" in output_lower
            or has_percentages
            or len(currency_mentions) > 0
        )

        required_disclaimer_count = 2 if has_specific_claims else 1
        has_enough_disclaimers = len(found_disclaimers) >= required_disclaimer_count

        # Check for required terms
        found_terms = [t for t in self.required_terms if t in output_lower]
        self.required_terms - set(found_terms)

        # Check for prohibited terms
        prohibited_found = [t for t in self.prohibited_terms if t in output_lower]

        # Content passes if it has required disclaimers and no prohibited terms
        passed = has_enough_disclaimers and not prohibited_found

        # Create validation message
        if passed:
            message = "Financial content complies with regulatory requirements"
        else:
            failures = []
            if not has_enough_disclaimers:
                needed = "two" if has_specific_claims else "one"
                failures.append(
                    f"Need at least {needed} of these disclaimers: {', '.join(self.disclaimers)}"
                )
            if prohibited_found:
                failures.append(f"Found prohibited terms: {', '.join(prohibited_found)}")
            message = "; ".join(failures)

        return RuleResult(
            passed=passed,
            message=message,
            metadata={
                "has_financial_content": True,
                "found_disclaimers": found_disclaimers,
                "required_disclaimer_count": required_disclaimer_count,
                "has_specific_claims": has_specific_claims,
                "prohibited_found": prohibited_found,
                "currency_mentions": currency_mentions,
                "percentage_mentions": percentage_mentions,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can validate the output."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """Return the type of output that this validator can handle."""
        return str


class MedicalContentValidator(RuleValidator[str]):
    """Validator for medical content compliance with healthcare regulations."""

    def __init__(self, medical_terms_file: Optional[str] = None):
        """Initialize the validator with medical compliance rules."""
        self.medical_terms = self._load_medical_terms(medical_terms_file)

        # Group disclaimers by type
        self.disclaimer_groups = {
            "advisory": {
                "this is not medical advice",
                "not a substitute for professional medical advice",
            },
            "consultation": {
                "consult with a healthcare professional",
            },
            "results": {
                "results may vary",
            },
        }

        # Define patterns for medical claims
        self.treatment_pattern = re.compile(
            r"(?:cure|treat|heal|remedy|alleviate|eliminate|prevent)\s+(?:\w+\s+){0,3}(?:disease|condition|disorder|illness|symptoms?)",
            re.IGNORECASE,
        )
        self.diagnosis_pattern = re.compile(
            r"(?:diagnose|identify|detect|determine|confirm)\s+(?:\w+\s+){0,3}(?:disease|condition|disorder|illness)",
            re.IGNORECASE,
        )

    def _load_medical_terms(self, filepath: Optional[str]) -> Set[str]:
        """Load medical terms from file or use default terms."""
        default_terms = {
            # Specific medical procedures and treatments
            "surgery",
            "chemotherapy",
            "radiation",
            "transplant",
            "vaccination",
            "immunization",
            # Medical conditions and symptoms
            "disease",
            "syndrome",
            "disorder",
            "infection",
            "inflammation",
            "chronic",
            "acute",
            "pain",
            "symptoms",
            # Medical specialties and practitioners
            "physician",
            "surgeon",
            "oncologist",
            "pediatrician",
            "psychiatrist",
            # Medical facilities and institutions
            "hospital",
            "clinic",
            "pharmacy",
            "laboratory",
            # Medical procedures and tests
            "diagnosis",
            "treatment",
            "therapy",
            "examination",
            "screening",
            # Medications and drugs
            "medication",
            "prescription",
            "drug",
            "antibiotic",
            "vaccine",
            # Body systems and anatomy
            "cardiovascular",
            "respiratory",
            "nervous system",
            "immune system",
            # Medical devices and equipment
            "pacemaker",
            "prosthetic",
            "implant",
            "ventilator",
        }

        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return {line.strip().lower() for line in f if line.strip()}
            except Exception as e:
                logger.warning(f"Error loading medical terms file: {e}")

        return default_terms

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate medical content for compliance."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")

        output_lower = output.lower()

        # Check if content contains medical terms
        medical_terms_found = [term for term in self.medical_terms if term in output_lower]
        has_medical_content = len(medical_terms_found) > 0

        # Find treatment/cure claims
        treatment_claims = self.treatment_pattern.findall(output)
        diagnosis_claims = self.diagnosis_pattern.findall(output)
        has_claims = bool(treatment_claims or diagnosis_claims)

        # If no medical content and no claims, pass automatically
        if not has_medical_content and not has_claims:
            return RuleResult(
                passed=True,
                message="No medical content detected",
                metadata={
                    "has_medical_content": False,
                    "has_claims": has_claims,
                    "medical_terms_found": [],
                },
            )

        # Check for disclaimers by group
        found_disclaimers = {
            group: [d for d in disclaimers if d in output_lower]
            for group, disclaimers in self.disclaimer_groups.items()
        }

        # Determine required groups based on content
        required_groups = {"advisory"}  # Always require advisory disclaimer for medical content
        if has_claims or len(medical_terms_found) >= 3:
            required_groups.add("consultation")  # Require consultation for strong claims
        if has_claims:
            required_groups.add("results")  # Require results disclaimer for claims

        # Check if each required group has at least one disclaimer
        missing_groups = [
            group for group in required_groups if not found_disclaimers.get(group, [])
        ]

        if missing_groups:
            passed = False
            missing_disclaimers = [
                d for group in missing_groups for d in self.disclaimer_groups[group]
            ]
            message = (
                f"Medical content missing required disclaimers: {', '.join(missing_disclaimers)}"
            )
        else:
            passed = True
            message = "Medical content complies with requirements"

        return RuleResult(
            passed=passed,
            message=message,
            metadata={
                "has_medical_content": has_medical_content,
                "has_claims": has_claims,
                "medical_terms_found": medical_terms_found,
                "treatment_claims": treatment_claims,
                "diagnosis_claims": diagnosis_claims,
                "found_disclaimers": {k: v for k, v in found_disclaimers.items() if v},
                "required_groups": list(required_groups),
                "missing_groups": missing_groups,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can validate the output."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """Return the type of output that this validator can handle."""
        return str


class CompositeValidator(RuleValidator[str]):
    """Validator that combines multiple validators with configurable logic."""

    def __init__(self, validators: List[RuleValidator[str]], require_all: bool = True):
        """Initialize with a list of validators."""
        if not validators:
            raise ValueError("Must provide at least one validator")

        self._validators = validators
        self._require_all = require_all

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate using all child validators."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")

        results = []
        for validator in self._validators:
            if validator.can_validate(output):
                try:
                    result = validator.validate(output, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Validator failed: {e}")

        if not results:
            return RuleResult(
                passed=False,
                message="No validators could process the input",
                metadata={"validator_count": len(self._validators)},
            )

        if self._require_all:
            # All validators must pass (AND logic)
            passed = all(r.passed for r in results)
            if passed:
                message = "All validations passed"
            else:
                failures = [r.message for r in results if not r.passed]
                message = f"Failed validations: {'; '.join(failures)}"
        else:
            # Any validator can pass (OR logic)
            passed = any(r.passed for r in results)
            if passed:
                message = "At least one validation passed"
            else:
                message = "All validations failed"

        # Combine metadata from all results
        combined_metadata = {"results": []}
        for result in results:
            combined_metadata["results"].append(
                {
                    "passed": result.passed,
                    "message": result.message,
                    "metadata": result.metadata,
                }
            )

        return RuleResult(
            passed=passed,
            message=message,
            metadata=combined_metadata,
        )

    def can_validate(self, output: str) -> bool:
        """Check if any validator can validate the output."""
        return isinstance(output, str) and any(v.can_validate(output) for v in self._validators)

    @property
    def validation_type(self) -> type:
        """Return the type of output that this validator can handle."""
        return str


class FinancialContentRule(Rule):
    """Rule for validating financial content regulatory compliance."""

    def __init__(
        self,
        name: str = "financial_content_rule",
        description: str = "Validates financial content for regulatory compliance",
        config: Optional[RuleConfig] = None,
    ):
        """Initialize the financial content rule."""
        validator = FinancialContentValidator()

        if config is None:
            config = RuleConfig(
                priority=RulePriority.HIGH,
                cache_size=100,
                cost=2.0,
            )

        super().__init__(name=name, description=description, config=config, validator=validator)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate using the financial content validator."""
        return self._validator.validate(output, **kwargs)


class MedicalContentRule(Rule):
    """Rule for validating medical content compliance."""

    def __init__(
        self,
        name: str = "medical_content_rule",
        description: str = "Validates medical content for compliance",
        config: Optional[RuleConfig] = None,
        medical_terms_file: Optional[str] = None,
    ):
        """Initialize the medical content rule."""
        validator = MedicalContentValidator(medical_terms_file=medical_terms_file)

        if config is None:
            config = RuleConfig(
                priority=RulePriority.HIGH,
                cache_size=100,
                cost=2.0,
            )

        super().__init__(name=name, description=description, config=config, validator=validator)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate using the medical content validator."""
        return self._validator.validate(output, **kwargs)


class LegalPatternValidator(RuleValidator[str]):
    """Validator for checking legal compliance patterns."""

    def __init__(self):
        """Initialize the validator with legal compliance patterns."""
        self.legal_patterns = [
            # Terms and conditions variations
            r"terms?\s*(?:and|&)?\s*conditions?",
            r"terms?\s*of\s*(?:use|service)",
            # Privacy policy variations
            r"privacy\s*policy",
            r"privacy\s*notice",
            # Disclaimer variations
            r"disclaimers?",
            r"legal\s*notices?",
            r"important\s*notices?",
            # General legal statements
            r"all\s*rights?\s*reserved",
            r"subject\s*to\s*(?:terms|conditions)",
        ]
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.legal_patterns
        ]

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate text for legal compliance patterns."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")

        # Find all matches
        matches = []
        for pattern in self.compiled_patterns:
            matches.extend(pattern.findall(output))

        # Content passes if at least one legal pattern is found
        passed = len(matches) > 0

        if passed:
            message = "Legal compliance patterns found"
        else:
            message = "Missing required legal references"

        return RuleResult(
            passed=passed,
            message=message,
            metadata={"matches": matches, "patterns_checked": self.legal_patterns},
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can validate the output."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """Return the type of output that this validator can handle."""
        return str


class LegalPatternRule(Rule):
    """Rule for checking legal compliance patterns."""

    def __init__(
        self,
        name: str = "legal_compliance",
        description: str = "Checks for legal compliance patterns",
        config: Optional[RuleConfig] = None,
    ):
        """Initialize the legal pattern rule."""
        validator = LegalPatternValidator()

        if config is None:
            config = RuleConfig(
                priority=RulePriority.MEDIUM,
                cache_size=100,
                cost=1.0,
            )

        super().__init__(name=name, description=description, config=config, validator=validator)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate using the legal pattern validator."""
        return self._validator.validate(output, **kwargs)


class CompositeRule(Rule):
    """Rule that combines multiple rules with configurable logic."""

    def __init__(
        self,
        name: str,
        description: str,
        rules: List[Rule],
        require_all: bool = True,
        config: Optional[RuleConfig] = None,
    ):
        """Initialize with a list of rules."""
        if not rules:
            raise ValueError("Must provide at least one rule")

        # Create a composite validator from rule validators
        validators = [rule._validator for rule in rules]
        validator = CompositeValidator(validators, require_all=require_all)

        if config is None:
            # Use highest priority from child rules
            priority = max((rule.config.priority for rule in rules), key=lambda x: x.value)
            cost = sum(rule.config.cost for rule in rules)

            config = RuleConfig(
                priority=priority,
                cache_size=100,
                cost=cost,
            )

        super().__init__(name=name, description=description, config=config, validator=validator)
        self._rules = rules
        self._require_all = require_all

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate using the composite validator."""
        return self._validator.validate(output, **kwargs)


def setup_critic():
    """Set up the critic for improving content."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found. Content improvement disabled.")
        return None

    provider = OpenAIProvider(
        model_name="gpt-4-turbo",
        config=ModelConfig(
            api_key=api_key,
            temperature=0.5,
            max_tokens=1000,
        ),
    )

    return PromptCritic(
        model=provider,
        config=PromptCriticConfig(
            name="regulatory_content_critic",
            description="Improves content to meet regulatory requirements",
            system_prompt=(
                "You are an expert editor specializing in financial, medical, and legal content. "
                "Improve content to comply with regulatory requirements while preserving the "
                "original meaning. Add appropriate disclaimers, modify problematic claims, "
                "and ensure compliance with industry best practices."
            ),
            temperature=0.5,
            max_tokens=1000,
        ),
    )


def setup_rules():
    """Set up all the validation rules."""
    # Domain-specific rules
    financial_rule = FinancialContentRule()
    medical_rule = MedicalContentRule()

    # Legal compliance rule using pattern matching
    legal_rule = LegalPatternRule()

    # Composite rule (AND logic - content must comply with all applicable domain rules)
    composite_rule = CompositeRule(
        name="regulatory_compliance",
        description="Validates content for regulatory compliance across domains",
        rules=[financial_rule, medical_rule, legal_rule],
        require_all=True,  # Changed to True for AND logic
    )

    return {
        "financial": financial_rule,
        "medical": medical_rule,
        "legal": legal_rule,
        "composite": composite_rule,
    }


def validate_and_improve(text, rules, critic):
    """Validate content and improve if needed."""
    # Apply individual domain rules
    financial_result = rules["financial"].validate(text)
    medical_result = rules["medical"].validate(text)
    legal_result = rules["legal"].validate(text)

    # Apply composite rule
    composite_result = rules["composite"].validate(text)

    # Print results
    print(f"Financial Rule: {'PASSED' if financial_result.passed else 'FAILED'}")
    if not financial_result.passed and financial_result.metadata.get(
        "has_financial_content", False
    ):
        print(f"  Message: {financial_result.message}")

    print(f"Medical Rule: {'PASSED' if medical_result.passed else 'FAILED'}")
    if not medical_result.passed and medical_result.metadata.get("has_medical_content", False):
        print(f"  Message: {medical_result.message}")

    print(f"Legal Rule: {'PASSED' if legal_result.passed else 'FAILED'}")

    print(f"Composite Rule: {'PASSED' if composite_result.passed else 'FAILED'}")
    print(f"  Message: {composite_result.message}")

    # If the composite rule fails and critic is available, improve the content
    if not composite_result.passed and critic:
        print("\nImproving content with regulatory critic...")
        violations = [
            {
                "rule": "regulatory_compliance",
                "message": composite_result.message,
                "metadata": composite_result.metadata,
            }
        ]

        try:
            # Improve the content
            improved_text = critic.improve(text, violations)
            print(f'Improved text: "{improved_text}"\n')

            # Revalidate the improved text
            print("Revalidating improved content:")
            revalidation = rules["composite"].validate(improved_text)
            print(f"  Result: {'PASSED' if revalidation.passed else 'FAILED'}")
            print(f"  Message: {revalidation.message}")

            return improved_text

        except Exception as e:
            print(f"Error improving text: {e}")
    elif composite_result.passed:
        print("\nContent already complies with all relevant rules. No improvement needed.")
    else:
        print("\nContent needs improvement, but critic is not available.")

    return None


def main():
    """Run the advanced rules example."""
    # Load environment variables
    load_dotenv()

    # Set up rules and critic
    rules = setup_rules()
    critic = setup_critic()

    # Example texts to validate
    example_texts = [
        # Test case for legal patterns
        "Terms and Conditions: By using this service, you agree to our terms. Privacy Policy: We protect your data. All rights reserved.",
        # Test case for mixed content with proper disclaimers
        """Our healthcare investment fund focuses on medical technology.

        Financial Disclaimer: Past performance is not indicative of future results.
        Please consult a financial advisor. Investments may lose value.

        Medical Disclaimer: This is not medical advice. Results may vary.
        Please consult with a healthcare professional.

        Terms and Conditions apply. All rights reserved.""",
        # Test case for specific financial claims
        """Investment Opportunity: Historical returns of 12%.
        Consider your investment objectives carefully.
        This is not financial advice.
        Read the prospectus carefully before investing.
        Subject to terms and conditions.""",
        # Test case for medical content with proper disclaimers
        """Our research suggests potential benefits for joint health.
        This is not medical advice.
        Not a substitute for professional medical advice.
        Results may vary.
        Consult with a healthcare professional.
        Subject to terms and conditions.""",
        # Test case for minimal content with just legal requirements
        "Subject to terms and conditions. All rights reserved.",
        # Test case for actual medical content
        """Our clinic provides advanced therapy for chronic pain and inflammation.
        We use state-of-the-art diagnostic screening and treatment procedures.
        Our physicians specialize in cardiovascular and respiratory conditions.

        Note: This content is for informational purposes only and is not medical advice.
        Please consult with a healthcare professional for proper medical advice.
        Results may vary. All rights reserved.""",
    ]

    print("\n=== Advanced Rules Example ===\n")

    for i, text in enumerate(example_texts, 1):
        print(f"\n--- Example {i} ---")
        print(f'Original text: "{text}"\n')

        # Validate and potentially improve the content
        validate_and_improve(text, rules, critic)

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
