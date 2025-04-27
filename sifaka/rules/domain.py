"""
Domain-specific rules for Sifaka.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult
import re
import ast


class MedicalRule(Rule):
    """
    Rule that checks for medical content accuracy and safety.

    Attributes:
        medical_terms (Dict[str, List[str]]): Dictionary of medical terms and their correct variations
        warning_terms (List[str]): List of terms that require medical disclaimers
        disclaimer_required (bool): Whether medical disclaimers are required
    """

    medical_terms: Dict[str, List[str]] = Field(
        default={
            "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
            "treatment": ["treatment", "treat", "treating", "therapy"],
            "medication": ["medication", "drug", "prescription", "medicine"],
            "symptom": ["symptom", "symptoms", "sign", "signs"],
        }
    )
    warning_terms: List[str] = Field(
        default=[
            "diagnosis",
            "treatment",
            "medication",
            "prescription",
            "therapy",
            "cure",
            "heal",
            "remedy",
        ]
    )
    disclaimer_required: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains accurate medical information and proper disclaimers.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        output_lower = output.lower()
        issues = []
        found_warning_terms = []

        # Check for warning terms
        for term in self.warning_terms:
            if term in output_lower:
                found_warning_terms.append(term)

        # Check for disclaimer if warning terms are found
        if self.disclaimer_required and found_warning_terms:
            disclaimer_patterns = [
                r"not medical advice",
                r"consult.*doctor",
                r"seek.*professional",
                r"medical disclaimer",
            ]
            has_disclaimer = any(
                re.search(pattern, output_lower) for pattern in disclaimer_patterns
            )

            if not has_disclaimer:
                issues.append("Medical disclaimer required but not found")

        if issues:
            return RuleResult(
                passed=False,
                message="Medical content validation failed",
                metadata={"issues": issues, "warning_terms_found": found_warning_terms},
            )

        return RuleResult(
            passed=True,
            message="Medical content validation passed",
            metadata={"warning_terms_found": found_warning_terms},
        )


class LegalRule(Rule):
    """
    Rule that validates legal content.

    Attributes:
        legal_terms: Dictionary of legal terms and their variations
        citation_patterns: List of regex patterns for legal citations
        disclaimers: List of required legal disclaimers
        disclaimer_required: Whether legal disclaimers are required
    """

    legal_terms: Dict[str, List[str]] = Field(
        default={
            "jurisdiction": ["jurisdiction", "court", "venue", "forum", "tribunal"],
            "statute": ["statute", "law", "regulation", "code", "act", "bill", "ordinance"],
            "precedent": ["precedent", "case law", "ruling", "decision", "holding", "opinion"],
            "liability": ["liability", "responsibility", "duty", "obligation", "negligence"],
            "procedure": ["procedure", "motion", "pleading", "filing", "petition", "appeal"],
            "evidence": ["evidence", "proof", "exhibit", "testimony", "witness", "document"],
        },
        description="Dictionary of legal terms and their variations",
    )

    citation_patterns: List[str] = Field(
        default=[
            # Basic citation patterns
            r"\d+\s*(?:U\.?S\.?|F\.?(?:2d|3d)?|S\.?Ct\.?)\s*\d+",  # Federal cases
            r"\d+\s*[A-Z][a-z]*\.?\s*(?:2d|3d)?\s*\d+",  # State cases
            r"(?:\d+\s*)?U\.?S\.?C\.?\s*§*\s*\d+(?:\([a-z]\))?",  # U.S. Code
            # Extended citation patterns
            r"\d+\s*(?:Cal\.?|N\.?Y\.?|Tex\.?)\s*(?:2d|3d|4th)?\s*\d+",  # State reporters
            r"(?:pub\.?\s*l\.?|P\.?L\.?)\s*\d+[-‐]\d+",  # Public Laws
            r"(?:CFR|C\.F\.R\.)\s*§*\s*\d+\.\d+",  # Code of Federal Regulations
            r"\d+\s*L\.?\s*Ed\.?\s*(?:2d)?\s*\d+",  # Supreme Court (Lawyers' Edition)
        ],
        description="List of regex patterns for legal citations",
    )

    disclaimers: List[str] = Field(
        default=[
            r"(?i)not\s+(?:intended\s+as\s+)?legal\s+advice",
            r"(?i)consult\s+(?:(?:a|your)\s+)?(?:qualified\s+)?(?:attorney|lawyer|legal\s+counsel)",
            r"(?i)seek\s+legal\s+(?:counsel|advice|representation)",
            r"(?i)legal\s+disclaimer\s*[:\-]?",
            r"(?i)for\s+informational\s+purposes\s+only",
            r"(?i)does\s+not\s+constitute\s+(?:a|an)\s+attorney-client\s+relationship",
            r"(?i)not\s+a\s+substitute\s+for\s+legal\s+(?:counsel|advice)",
        ],
        description="List of required legal disclaimers",
    )

    disclaimer_required: bool = Field(
        default=True,
        description="Whether legal disclaimers are required",
    )

    common_phrases: List[str] = Field(
        default=[
            "legal information",
            "legal content",
            "legal text",
            "legal document",
            "legal advice",
            "legal services",
            "legal assistance",
        ],
        description="Common phrases that should not trigger legal term detection",
    )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains appropriate legal content and disclaimers.
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        metadata = {
            "citations": [],
            "issues": [],
            "legal_terms_found": [],
            "has_disclaimer": False,
            "terms_context": {},
            "citation_context": {},
        }

        # Check for legal terms with context
        for category, terms in self.legal_terms.items():
            for term in terms:
                term_lower = term.lower()
                # Skip if it's just a common phrase
                if any(
                    phrase in output_lower and term_lower in phrase
                    for phrase in self.common_phrases
                ):
                    continue

                # Look for the term with word boundaries
                pattern = rf"\b{re.escape(term_lower)}\b"
                matches = list(re.finditer(pattern, output_lower))

                if matches:
                    # Store the term and its context
                    metadata["legal_terms_found"].append(
                        {
                            "term": term,
                            "category": category,
                        }
                    )
                    metadata["terms_context"][term] = [
                        output[max(0, m.start() - 40) : min(len(output), m.end() + 40)]
                        for m in matches
                    ]

        has_legal_terms = len(metadata["legal_terms_found"]) > 0

        # Check for citations with improved pattern matching
        for pattern in self.citation_patterns:
            matches = list(re.finditer(pattern, output, re.IGNORECASE))
            for match in matches:
                citation = match.group(0)
                metadata["citations"].append(
                    {
                        "citation": citation,
                        "pattern_type": pattern,
                    }
                )
                # Store citation context
                metadata["citation_context"][citation] = output[
                    max(0, match.start() - 40) : min(len(output), match.end() + 40)
                ]

        has_citations = len(metadata["citations"]) > 0

        # Check for disclaimer with context awareness
        if self.disclaimer_required:
            disclaimer_matches = []
            for pattern in self.disclaimers:
                matches = list(re.finditer(pattern, output_lower))
                if matches:
                    disclaimer_matches.extend(
                        [
                            output[max(0, m.start() - 20) : min(len(output), m.end() + 20)]
                            for m in matches
                        ]
                    )

            metadata["has_disclaimer"] = len(disclaimer_matches) > 0
            metadata["disclaimer_matches"] = disclaimer_matches

            if not metadata["has_disclaimer"] and (has_legal_terms or has_citations):
                metadata["issues"].append(
                    {
                        "type": "missing_disclaimer",
                        "message": "Legal disclaimer required but not found",
                        "context": {
                            "has_legal_terms": has_legal_terms,
                            "has_citations": has_citations,
                        },
                    }
                )

        # Validate based on presence of legal content
        if has_legal_terms and not has_citations:
            metadata["issues"].append(
                {
                    "type": "missing_citations",
                    "message": "Legal terms found without supporting citations",
                    "terms": metadata["legal_terms_found"],
                }
            )
        elif has_citations and not has_legal_terms:
            metadata["issues"].append(
                {
                    "type": "missing_legal_terms",
                    "message": "Citations found without supporting legal discussion",
                    "citations": metadata["citations"],
                }
            )

        # If no legal content is found at all, pass
        if not has_legal_terms and not has_citations:
            return RuleResult(passed=True, message="No legal content found", metadata=metadata)

        # Legal content is present, check if it passes all requirements
        passed = (has_legal_terms == has_citations) and (  # Both present or both absent
            not self.disclaimer_required or metadata["has_disclaimer"]
        )  # Disclaimer if needed

        message = "Legal content validation " + ("passed" if passed else "failed")
        if not passed:
            message += ": " + ", ".join(issue["message"] for issue in metadata["issues"])

        return RuleResult(passed=passed, message=message, metadata=metadata)


class PythonRule(Rule):
    """
    Rule that checks for Python code quality and best practices.

    Attributes:
        code_style_patterns: Dictionary of code style patterns to check
        security_patterns: Dictionary of security-related patterns to check
        performance_patterns: Dictionary of performance-related patterns to check
    """

    code_style_patterns: Dict[str, str] = Field(
        default={
            # Import patterns
            "pep8_imports": r"^(?:from\s+[a-zA-Z0-9_.]+\s+)?import\s+(?:[a-zA-Z0-9_]+(?:\s*,\s*[a-zA-Z0-9_]+)*|\s*\([^)]+\))",
            # Class patterns
            "pep8_classes": r"^class\s+[A-Z][a-zA-Z0-9]*(?:\([^)]+\))?:",
            # Function patterns
            "pep8_functions": r"^(?:async\s+)?def\s+[a-z_][a-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
            # Variable patterns
            "pep8_variables": r"^[a-z_][a-z0-9_]*\s*(?::\s*[^=\n]+)?\s*=",
            # Docstring patterns (supports all formats)
            "docstring": r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')',
            # Type hints
            "type_hints": r":\s*(?:[a-zA-Z_][a-zA-Z0-9_]*(?:\[.*?\])?|None|Any|Optional\[.*?\])",
        }
    )

    security_patterns: Dict[str, str] = Field(
        default={
            # Code execution
            "eval": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\beval\s*\(",
            "exec": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\bexec\s*\(",
            # Unsafe deserialization
            "pickle": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:pickle|marshal|shelve)\.(?:load|loads)\b",
            # Shell injection
            "shell": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:os\.system|subprocess\.(?:call|Popen|run))\s*\(.*(?:shell\s*=\s*True|`|\$|;)",
            # SQL injection
            "sql": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:execute|executemany)\s*\(.*(?:%s|\+)",
            # File operations
            "unsafe_file": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:open|file)\s*\([^)]*(?:mode\s*=\s*['\"]w|\s*,\s*['\"]w)",
            # Network
            "unsafe_network": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:urllib\.request\.urlopen|requests\.(?:get|post|put|delete))\s*\([^)]*verify\s*=\s*False",
        }
    )

    performance_patterns: Dict[str, str] = Field(
        default={
            # Global variables
            "global": r"^global\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*",
            # Nested loops
            "nested_loop": r"(?:^|\s)for\s+[^:]+:[^:]*\bfor\s+[^:]+:",
            # List/dict/set comprehensions
            "comprehension": r"\[(?:[^][]|\[[^]]*\])*for.*?\]|\{(?:[^{}]|\{[^}]*\})*for.*?\}",
            # Generator expressions
            "generator": r"\((?:[^()]|\([^)]*\))*for.*?\)",
            # Memory leaks
            "memory_leak": r"(?:^|\s)(?:while\s+True|for\s+.*?in\s+.*?):[^:]*(?:append|extend|add)\(",
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output is valid Python code and follows style guidelines.

        Args:
            output (str): The code to validate
            **kwargs: Additional validation options

        Returns:
            RuleResult: The validation result
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        # Remove comments and string literals for security checks
        def remove_comments_and_strings(code: str) -> str:
            code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)  # Remove single-line comments
            code = re.sub(r'"""[\s\S]*?"""', "", code)  # Remove triple-quoted strings
            code = re.sub(r"'''[\s\S]*?'''", "", code)  # Remove triple-quoted strings
            code = re.sub(r'"[^"]*"', '""', code)  # Replace double-quoted strings
            code = re.sub(r"'[^']*'", "''", code)  # Replace single-quoted strings
            return code

        clean_code = remove_comments_and_strings(output)

        issues = []
        metadata = {
            "style_issues": [],
            "security_issues": [],
            "performance_issues": [],
            "ast_valid": False,
            "patterns_checked": {
                "style": list(self.code_style_patterns.keys()),
                "security": list(self.security_patterns.keys()),
                "performance": list(self.performance_patterns.keys()),
            },
        }

        # Check AST validity
        try:
            ast.parse(output)
            metadata["ast_valid"] = True
        except SyntaxError as e:
            issues.append(
                {
                    "type": "syntax_error",
                    "message": f"Invalid Python syntax: {str(e)}",
                    "line": e.lineno,
                    "offset": e.offset,
                }
            )
            return RuleResult(
                passed=False,
                message="Code validation failed: syntax error",
                metadata=metadata,
            )

        # Code style checks
        for check_name, pattern in self.code_style_patterns.items():
            if check_name == "docstring":
                # Special handling for docstrings
                if not re.search(pattern, output):
                    metadata["style_issues"].append(
                        {
                            "type": check_name,
                            "message": "Missing docstring",
                        }
                    )
            else:
                violations = []
                lines = output.split("\n")
                for i, line in enumerate(lines, 1):
                    if line.strip() and not line.strip().startswith("#"):
                        if not re.match(pattern, line.strip()):
                            violations.append(i)

                if violations:
                    metadata["style_issues"].append(
                        {
                            "type": check_name,
                            "message": f"Style violation: {check_name}",
                            "lines": violations,
                        }
                    )

        # Security checks on cleaned code
        for check_name, pattern in self.security_patterns.items():
            matches = list(re.finditer(pattern, clean_code, re.MULTILINE))
            if matches:
                metadata["security_issues"].append(
                    {
                        "type": check_name,
                        "message": f"Security issue: {check_name}",
                        "matches": [
                            {
                                "line": clean_code.count("\n", 0, m.start()) + 1,
                                "code": output.split("\n")[
                                    clean_code.count("\n", 0, m.start())
                                ].strip(),
                            }
                            for m in matches
                        ],
                    }
                )

        # Performance checks
        for check_name, pattern in self.performance_patterns.items():
            matches = list(re.finditer(pattern, clean_code, re.MULTILINE))
            if matches:
                metadata["performance_issues"].append(
                    {
                        "type": check_name,
                        "message": f"Performance concern: {check_name}",
                        "matches": [
                            {
                                "line": clean_code.count("\n", 0, m.start()) + 1,
                                "code": output.split("\n")[
                                    clean_code.count("\n", 0, m.start())
                                ].strip(),
                            }
                            for m in matches
                        ],
                    }
                )

        # Aggregate all issues
        if metadata["style_issues"]:
            issues.append(
                {
                    "type": "style",
                    "message": "Code style violations found",
                    "details": metadata["style_issues"],
                }
            )

        if metadata["security_issues"]:
            issues.append(
                {
                    "type": "security",
                    "message": "Security issues found",
                    "details": metadata["security_issues"],
                }
            )

        if metadata["performance_issues"]:
            issues.append(
                {
                    "type": "performance",
                    "message": "Performance concerns found",
                    "details": metadata["performance_issues"],
                }
            )

        if issues:
            return RuleResult(
                passed=False,
                message="Code validation failed: "
                + "; ".join(issue["message"] for issue in issues),
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message="Code validation passed",
            metadata=metadata,
        )


class ConsistencyRule(Rule):
    """
    Rule that checks for consistency in the output.

    Attributes:
        consistency_patterns (Dict[str, str]): Dictionary of patterns to check for consistency
        contradiction_indicators (List[Tuple[str, str]]): List of tuples indicating contradictions
        repetition_threshold (float): Threshold for considering text repetitive (0.0 to 1.0)
    """

    consistency_patterns: Dict[str, str] = Field(
        default={
            # Tense patterns
            "present": r"\b(?:is|are|am|has|have|do|does)\b",
            "past": r"\b(?:was|were|had|did)\b",
            "future": r"\b(?:will|shall|going to)\b",
            # Person patterns
            "first_person": r"\b(?:I|we|my|our|myself|ourselves)\b",
            "second_person": r"\b(?:you|your|yourself|yourselves)\b",
            "third_person": r"\b(?:he|she|it|they|his|her|its|their|himself|herself|itself|themselves)\b",
            # Voice patterns
            "active": r"\b(?:subject)\s+(?:verb)\b",
            "passive": r"\b(?:is|are|was|were)\s+(?:\w+ed|\w+en)\b",
            # Format patterns
            "list_marker": r"(?m)^[-*•]\s+|\d+\.\s+",
            "code_block": r"```[\s\S]*?```|`[^`]+`",
            "table_marker": r"\|[^|]+\|",
            "heading": r"(?m)^#{1,6}\s+\w+",
        }
    )

    contradiction_indicators: List[Tuple[str, str]] = Field(
        default=[
            # Direct contradictions
            (r"\b(?:is|are)\b", r"\b(?:is not|are not|isn't|aren't)\b"),
            (r"\b(?:will|shall)\b", r"\b(?:will not|shall not|won't|shan't)\b"),
            (r"\b(?:must|should)\b", r"\b(?:must not|should not|shouldn't)\b"),
            # Temporal contradictions
            (r"\b(?:always|never)\b", r"\b(?:sometimes|occasionally)\b"),
            (r"\b(?:all|every)\b", r"\b(?:some|few|none)\b"),
            # Logical contradictions
            (r"\b(?:increase|rise)\b", r"\b(?:decrease|fall)\b"),
            (r"\b(?:more|greater)\b", r"\b(?:less|fewer)\b"),
            (r"\b(?:begin|start)\b", r"\b(?:end|finish)\b"),
        ]
    )

    repetition_threshold: float = Field(default=0.3)

    synonyms: Dict[str, List[str]] = Field(
        default={
            "increase": ["rise", "grow", "expand", "enhance"],
            "decrease": ["fall", "shrink", "reduce", "decline"],
            "important": ["crucial", "essential", "vital", "critical"],
            "good": ["great", "excellent", "outstanding", "superior"],
            "bad": ["poor", "inferior", "subpar", "inadequate"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["gradual", "leisurely", "unhurried", "sluggish"],
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def _find_contradictions(self, text: str) -> List[Dict[str, Any]]:
        """Find contradictions in the text using pattern matching and context analysis."""
        contradictions = []
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Check for direct contradictions within the same sentence
        for sentence in sentences:
            for pattern_pair in self.contradiction_indicators:
                pattern1, pattern2 = pattern_pair
                if re.search(pattern1, sentence, re.IGNORECASE) and re.search(
                    pattern2, sentence, re.IGNORECASE
                ):
                    contradictions.append(
                        {
                            "type": "direct_contradiction",
                            "sentence": sentence,
                            "patterns": [pattern1, pattern2],
                        }
                    )

        # Check for contradictions between sentences
        for i, s1 in enumerate(sentences):
            for j, s2 in enumerate(sentences[i + 1 :], i + 1):
                for pattern_pair in self.contradiction_indicators:
                    pattern1, pattern2 = pattern_pair
                    if (
                        re.search(pattern1, s1, re.IGNORECASE)
                        and re.search(pattern2, s2, re.IGNORECASE)
                    ) or (
                        re.search(pattern2, s1, re.IGNORECASE)
                        and re.search(pattern1, s2, re.IGNORECASE)
                    ):
                        contradictions.append(
                            {
                                "type": "cross_sentence_contradiction",
                                "sentence1": s1,
                                "sentence2": s2,
                                "patterns": [pattern1, pattern2],
                            }
                        )

        return contradictions

    def _check_consistency(self, text: str) -> Dict[str, Any]:
        """Check for consistency in various aspects of the text."""
        results = {
            "tense": {"present": 0, "past": 0, "future": 0},
            "person": {"first": 0, "second": 0, "third": 0},
            "voice": {"active": 0, "passive": 0},
            "format": {"list": 0, "code": 0, "table": 0, "heading": 0},
        }

        # Count occurrences of each pattern
        for aspect, pattern in self.consistency_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            category = next(
                (
                    cat
                    for cat in results.keys()
                    if any(aspect.startswith(p) for p in [f"{cat}_", cat])
                ),
                None,
            )
            if category:
                subcategory = aspect.split("_")[0] if "_" in aspect else aspect
                results[category][subcategory] = len(matches)

        return results

    def _find_repetitions(self, text: str) -> List[Dict[str, Any]]:
        """Find repeated phrases and similar expressions."""
        repetitions = []
        words = text.lower().split()

        # Find exact repetitions
        seen_phrases = {}
        for i in range(len(words)):
            for length in range(3, 8):  # Look for phrases of length 3-7 words
                if i + length <= len(words):
                    phrase = " ".join(words[i : i + length])
                    if phrase in seen_phrases:
                        repetitions.append(
                            {
                                "type": "exact_repetition",
                                "phrase": phrase,
                                "first_occurrence": seen_phrases[phrase],
                                "second_occurrence": i,
                            }
                        )
                    else:
                        seen_phrases[phrase] = i

        # Find similar expressions using synonyms
        for base_word, synonyms in self.synonyms.items():
            found_occurrences = []
            for word in [base_word] + synonyms:
                pattern = rf"\b{word}\b"
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    found_occurrences.append(
                        {
                            "word": word,
                            "position": match.start(),
                            "context": text[
                                max(0, match.start() - 20) : min(len(text), match.end() + 20)
                            ],
                        }
                    )

            if len(found_occurrences) > 1:
                repetitions.append(
                    {
                        "type": "similar_expression",
                        "base_word": base_word,
                        "occurrences": found_occurrences,
                    }
                )

        return repetitions

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate the consistency of the output text.

        Args:
            output (str): The text to validate
            **kwargs: Additional validation options

        Returns:
            RuleResult: The validation result
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        # Initialize metadata
        metadata = {
            "contradictions": [],
            "consistency_metrics": {},
            "repetitions": [],
            "format_changes": [],
            "issues": [],
        }

        # Find contradictions
        contradictions = self._find_contradictions(output)
        if contradictions:
            metadata["contradictions"] = contradictions
            metadata["issues"].append(
                {
                    "type": "contradiction",
                    "message": "Found contradicting statements",
                    "details": contradictions,
                }
            )

        # Check consistency
        consistency_metrics = self._check_consistency(output)
        metadata["consistency_metrics"] = consistency_metrics

        # Check for mixed tenses
        tense_counts = consistency_metrics["tense"]
        if sum(count > 0 for count in tense_counts.values()) > 1:
            metadata["issues"].append(
                {
                    "type": "mixed_tense",
                    "message": "Mixed tenses detected",
                    "details": tense_counts,
                }
            )

        # Check for mixed persons
        person_counts = consistency_metrics["person"]
        if sum(count > 0 for count in person_counts.values()) > 1:
            metadata["issues"].append(
                {
                    "type": "mixed_person",
                    "message": "Mixed grammatical persons detected",
                    "details": person_counts,
                }
            )

        # Find repetitions
        repetitions = self._find_repetitions(output)
        if repetitions:
            metadata["repetitions"] = repetitions
            if len(repetitions) > int(len(output.split()) * self.repetition_threshold):
                metadata["issues"].append(
                    {
                        "type": "excessive_repetition",
                        "message": "Excessive repetition detected",
                        "details": repetitions,
                    }
                )

        # Check format consistency
        format_metrics = consistency_metrics["format"]
        format_changes = []
        current_format = None
        for line in output.split("\n"):
            detected_format = None
            if re.match(self.consistency_patterns["list_marker"], line):
                detected_format = "list"
            elif re.match(self.consistency_patterns["code_block"], line):
                detected_format = "code"
            elif re.match(self.consistency_patterns["table_marker"], line):
                detected_format = "table"
            elif re.match(self.consistency_patterns["heading"], line):
                detected_format = "heading"

            if detected_format and detected_format != current_format:
                format_changes.append(
                    {
                        "from": current_format,
                        "to": detected_format,
                        "line": line,
                    }
                )
                current_format = detected_format

        if len(format_changes) > 1:
            metadata["format_changes"] = format_changes
            metadata["issues"].append(
                {
                    "type": "inconsistent_format",
                    "message": "Inconsistent formatting detected",
                    "details": format_changes,
                }
            )

        # Determine if the text passes validation
        passed = len(metadata["issues"]) == 0

        return RuleResult(
            passed=passed,
            message="Consistency validation " + ("passed" if passed else "failed"),
            metadata=metadata,
        )
