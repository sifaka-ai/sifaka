"""
Legal domain-specific validation rules for Sifaka.

This module provides validators and rules for checking legal content, citations, and terminology.
It enables validation of legal documents against various requirements such as disclaimer presence,
citation formatting, and legal terminology usage.

## Architecture Overview

The legal validation system follows a component-based architecture with three main rule types:

1. **Legal Content Rules**: Validate general legal content requirements
   - Disclaimer presence and format
   - Legal terminology usage
   - Content structure

2. **Legal Citation Rules**: Validate legal citations
   - Citation presence and count
   - Citation format validation
   - Citation requirements

3. **Legal Terms Rules**: Validate legal terminology
   - Required terms presence
   - Prohibited terms absence
   - Warning terms detection

Each rule type follows a consistent pattern:
- Configuration class (LegalConfig, LegalCitationConfig, LegalTermsConfig)
- Protocol interface (LegalValidator, LegalCitationValidator, LegalTermsValidator)
- Default validator implementation (DefaultLegalValidator, DefaultLegalCitationValidator, DefaultLegalTermsValidator)
- Rule implementation (LegalRule, LegalCitationRule, LegalTermsRule)
- Factory functions for creation (create_legal_rule, create_legal_citation_rule, create_legal_terms_rule)

## Component Lifecycle

### Configuration Classes
1. **Creation**: Instantiate with default or custom values
2. **Validation**: Values are validated by Pydantic
3. **Usage**: Pass to validator and rule constructors

### Validator Classes
1. **Initialization**: Set up with configuration
2. **Analyzer Creation**: Create specialized analyzers
3. **Validation**: Check text against requirements
4. **Result**: Return RuleResult with validation details

### Rule Classes
1. **Initialization**: Set up with configuration
2. **Validator Creation**: Create validator if not provided
3. **Validation**: Delegate to validator
4. **Result**: Return RuleResult from validator

## Error Handling Patterns

The legal validation system implements several error handling patterns:

1. **Configuration Validation**: Validates all configuration values
   - Ensures required fields are present
   - Validates field types and constraints
   - Provides clear error messages for invalid configuration

2. **Input Validation**: Validates input text
   - Checks for proper string type
   - Handles empty text appropriately
   - Provides clear error messages for invalid input

3. **Exception Handling**: Catches and handles exceptions
   - Wraps validation logic in try/except blocks
   - Returns failure results with error details
   - Preserves original exception information

## Usage Examples

### Basic Legal Rule Usage

```python
from sifaka.rules.domain.legal import create_legal_rule

# Create a legal rule
legal_rule = create_legal_rule(
    disclaimer_required=True,
    legal_terms={
        "jurisdiction": ["court", "venue", "forum"],
        "liability": ["liability", "responsibility", "duty"]
    }
)

# Validate text
result = legal_rule.validate("This legal document is subject to the jurisdiction of the court.")
print(f"Valid: {result.passed}")
print(f"Message: {result.message}")
print(f"Has disclaimer: {result.metadata.get('has_disclaimer')}")
print(f"Legal term counts: {result.metadata.get('legal_term_counts')}")
```

### Legal Citation Rule Usage

```python
from sifaka.rules.domain.legal import create_legal_citation_rule

# Create a legal citation rule
citation_rule = create_legal_citation_rule(
    citation_patterns=[r"\d+\s+U\.S\.\s+\d+", r"\d+\s+S\.\s*Ct\.\s+\d+"],
    require_citations=True,
    min_citations=1,
    max_citations=10
)

# Validate text
text = "As the Supreme Court held in 410 U.S. 113 and reaffirmed in 505 U.S. 833..."
result = citation_rule.validate(text)
print(f"Valid: {result.passed}")
print(f"Message: {result.message}")
print(f"Total citations: {result.metadata.get('total_citations')}")
```

### Legal Terms Rule Usage

```python
from sifaka.rules.domain.legal import create_legal_terms_rule

# Create a legal terms rule
terms_rule = create_legal_terms_rule(
    required_terms={"disclaimer", "notice"},
    prohibited_terms={"guarantee", "warranty"},
    case_sensitive=False
)

# Validate text
text = "DISCLAIMER: This document contains a legal notice and is provided for informational purposes only."
result = terms_rule.validate(text)
print(f"Valid: {result.passed}")
print(f"Message: {result.message}")
print(f"Legal terms found: {result.metadata.get('legal_terms_found')}")
print(f"Warning terms found: {result.metadata.get('warning_terms_found')}")
```

### Using Multiple Rules Together

```python
from sifaka.rules.domain.legal import (
    create_legal_rule,
    create_legal_citation_rule,
    create_legal_terms_rule
)

# Create rules
legal_rule = create_legal_rule(disclaimer_required=True)
citation_rule = create_legal_citation_rule(require_citations=True)
terms_rule = create_legal_terms_rule(required_terms={"notice"})

# Validate text
text = "NOTICE: This document contains legal information. As held in 410 U.S. 113..."
results = [
    legal_rule.validate(text),
    citation_rule.validate(text),
    terms_rule.validate(text)
]

# Check if all rules passed
all_passed = all(result.passed for result in results)
print(f"All rules passed: {all_passed}")
```

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- All rule-specific configuration is stored in specialized config classes
- The LegalConfig, LegalCitationConfig, and LegalTermsConfig classes provide type-safe access to parameters
- Factory functions (create_legal_rule, create_legal_citation_rule, create_legal_terms_rule) handle configuration
- Validator factory functions create standalone validators
- Rule factory functions create rules with validators
"""

# Standard library
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
    Pattern,
)

# Third-party
from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

# Sifaka
from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.domain.base import BaseDomainValidator


__all__ = [
    # Config classes
    "LegalConfig",
    "LegalCitationConfig",
    "LegalTermsConfig",
    # Protocol classes
    "LegalValidator",
    "LegalCitationValidator",
    "LegalTermsValidator",
    # Validator classes
    "DefaultLegalValidator",
    "DefaultLegalCitationValidator",
    "DefaultLegalTermsValidator",
    # Rule classes
    "LegalRule",
    "LegalCitationRule",
    "LegalTermsRule",
    # Factory functions
    "create_legal_validator",
    "create_legal_rule",
    "create_legal_citation_validator",
    "create_legal_citation_rule",
    "create_legal_terms_validator",
    "create_legal_terms_rule",
    # Internal helpers (non-exported)
    "_DisclaimerAnalyzer",
    "_LegalTermAnalyzer",
    "_CitationAnalyzer",
]


class LegalConfig(BaseModel):
    """
    Configuration for legal rules.

    This class defines the configuration options for legal content validation,
    including legal terms, citation patterns, disclaimers, and validation settings.
    It's used by DefaultLegalValidator and LegalRule to determine validation behavior.

    ## Architecture

    LegalConfig follows a component-based architecture:
    - Uses Pydantic for schema validation
    - Provides default values for all configuration options
    - Includes field validators for critical parameters
    - Supports immutability through frozen=True

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create through factory functions

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Custom validators for legal_terms, citation_patterns, and disclaimers
       - Range validation for numeric fields

    3. **Usage**: Pass to validators and rules
       - Used by DefaultLegalValidator
       - Used by LegalRule._create_default_validator
       - Used by create_legal_validator

    ## Error Handling

    - Type validation through Pydantic
    - Custom validators ensure legal_terms, citation_patterns, and disclaimers are not empty
    - Range validation for numeric fields (cache_size, priority, cost)
    - Immutability prevents accidental modification

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.domain.legal import LegalConfig

    # Create with default values
    config = LegalConfig()

    # Create with custom values
    config = LegalConfig(
        legal_terms=["copyright", "trademark", "patent"],
        citation_patterns=[r"\d+\s+U\.S\.\s+\d+", r"\d+\s+S\.\s*Ct\.\s+\d+"],
        disclaimers=["This is not legal advice", "Consult an attorney"],
        disclaimer_required=True,
        cache_size=200,
        priority=2,
        cost=1.5
    )

    # Access configuration values
    print(f"Legal terms: {config.legal_terms}")
    print(f"Disclaimer required: {config.disclaimer_required}")
    ```

    Using with validators:

    ```python
    from sifaka.rules.domain.legal import LegalConfig, DefaultLegalValidator

    # Create config
    config = LegalConfig(
        legal_terms=["copyright", "trademark"],
        disclaimer_required=True
    )

    # Create validator with config
    validator = DefaultLegalValidator(config)

    # Validate text
    result = validator.validate("This document contains copyright information.")
    print(f"Valid: {result.passed}")
    ```
    """

    model_config = ConfigDict(frozen=True)

    legal_terms: List[str] = Field(
        default_factory=lambda: [
            "copyright",
            "trademark",
            "patent",
            "license",
            "agreement",
            "contract",
            "terms",
            "conditions",
            "warranty",
            "liability",
        ],
        description="List of legal terms to validate",
    )
    citation_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\d+\s*(?:U\.?S\.?|F\.?(?:2d|3d)?|S\.?Ct\.?)\s*\d+",  # Federal cases
            r"\d+\s*[A-Z][a-z]*\.?\s*(?:2d|3d)?\s*\d+",  # State cases
            r"(?:\d+\s*)?U\.?S\.?C\.?\s*§*\s*\d+(?:\([a-z]\))?",  # U.S. Code
            r"\d+\s*(?:Cal\.?|N\.?Y\.?|Tex\.?)\s*(?:2d|3d|4th)?\s*\d+",  # State reporters
            r"(?:pub\.?\s*l\.?|P\.?L\.?)\s*\d+[-‐]\d+",  # Public Laws
            r"(?:CFR|C\.F\.R\.)\s*§*\s*\d+\.\d+",  # Code of Federal Regulations
            r"\d+\s*L\.?\s*Ed\.?\s*(?:2d)?\s*\d+",  # Supreme Court (Lawyers' Edition)
        ],
        description="List of regex patterns for legal citations",
    )
    disclaimers: List[str] = Field(
        default_factory=lambda: [
            "This is not legal advice",
            "Consult an attorney",
            "For informational purposes only",
            "Not a substitute for legal counsel",
        ],
        description="List of acceptable legal disclaimers",
    )
    disclaimer_required: bool = Field(
        default=True,
        description="Whether to require a legal disclaimer",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )

    @field_validator("legal_terms")
    @classmethod
    def validate_legal_terms(cls, v: List[str]) -> List[str]:
        """Validate that legal terms are not empty."""
        if not v:
            raise ValueError("Legal terms cannot be empty")
        return v

    @field_validator("citation_patterns")
    @classmethod
    def validate_citation_patterns(cls, v: List[str]) -> List[str]:
        """Validate that citation patterns are not empty."""
        if not v:
            raise ValueError("Must provide at least one citation pattern")
        return v

    @field_validator("disclaimers")
    @classmethod
    def validate_disclaimers(cls, v: List[str]) -> List[str]:
        """Validate that disclaimers are not empty."""
        if not v:
            raise ValueError("Must provide at least one disclaimer pattern")
        return v


class LegalCitationConfig(BaseModel):
    """
    Configuration for legal citation validation.

    This class defines the configuration options for legal citation validation,
    including citation patterns, requirements, and validation settings.
    It's used by DefaultLegalCitationValidator and LegalCitationRule to determine
    validation behavior.

    ## Architecture

    LegalCitationConfig follows a component-based architecture:
    - Uses Pydantic for schema validation
    - Provides default values for all configuration options
    - Includes field validators for critical parameters
    - Enforces logical constraints between parameters

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create through factory functions

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Custom validators for citation_patterns
       - Range validation for numeric fields
       - Logical validation (max_citations >= min_citations)

    3. **Usage**: Pass to validators and rules
       - Used by DefaultLegalCitationValidator
       - Used by LegalCitationRule._create_default_validator
       - Used by create_legal_citation_validator

    ## Error Handling

    - Type validation through Pydantic
    - Custom validators ensure citation_patterns contains only strings
    - Range validation for numeric fields (min_citations, max_citations, cache_size, priority, cost)
    - Logical validation ensures max_citations is greater than or equal to min_citations

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.domain.legal import LegalCitationConfig

    # Create with default values
    config = LegalCitationConfig()

    # Create with custom values
    config = LegalCitationConfig(
        citation_patterns=[r"\d+\s+U\.S\.\s+\d+", r"\d+\s+S\.\s*Ct\.\s+\d+"],
        require_citations=True,
        min_citations=1,
        max_citations=10,
        cache_size=200,
        priority=2,
        cost=1.5
    )

    # Access configuration values
    print(f"Citation patterns: {config.citation_patterns}")
    print(f"Require citations: {config.require_citations}")
    print(f"Min citations: {config.min_citations}")
    print(f"Max citations: {config.max_citations}")
    ```

    Using with validators:

    ```python
    from sifaka.rules.domain.legal import LegalCitationConfig, DefaultLegalCitationValidator

    # Create config
    config = LegalCitationConfig(
        citation_patterns=[r"\d+\s+U\.S\.\s+\d+"],
        require_citations=True,
        min_citations=1
    )

    # Create validator with config
    validator = DefaultLegalCitationValidator(config)

    # Validate text
    result = validator.validate("The Supreme Court held in 410 U.S. 113 that...")
    print(f"Valid: {result.passed}")
    print(f"Total citations: {result.metadata.get('total_citations')}")
    ```
    """

    citation_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
            r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
            r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
            r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
        ],
        description="List of regex patterns for legal citations",
    )
    require_citations: bool = Field(
        default=True,
        description="Whether citations are required in the text",
    )
    min_citations: int = Field(
        default=0,
        ge=0,
        description="Minimum number of citations required",
    )
    max_citations: int = Field(
        default=100,
        ge=0,
        description="Maximum number of citations allowed",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )

    @field_validator("citation_patterns")
    @classmethod
    def validate_citation_patterns(cls, v: List[str]) -> List[str]:
        """Validate that citation patterns are valid strings."""
        if not all(isinstance(p, str) for p in v):
            raise ValueError("citation_patterns must contain only strings")
        return v

    @field_validator("max_citations")
    @classmethod
    def validate_max_citations(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate that max_citations is greater than or equal to min_citations."""
        if "min_citations" in values and v < values["min_citations"]:
            raise ValueError("max_citations must be greater than or equal to min_citations")
        return v


class LegalTermsConfig(BaseModel):
    """
    Configuration for legal terms validation.

    This class defines the configuration options for legal terminology validation,
    including legal terms, warning terms, required terms, prohibited terms, and
    validation settings. It's used by DefaultLegalTermsValidator and LegalTermsRule
    to determine validation behavior.

    ## Architecture

    LegalTermsConfig follows a component-based architecture:
    - Uses Pydantic for schema validation
    - Provides default values for all configuration options
    - Supports case-sensitive and case-insensitive matching
    - Allows flexible term categorization

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create through factory functions

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Range validation for numeric fields
       - Set-based term storage for efficient lookups

    3. **Usage**: Pass to validators and rules
       - Used by DefaultLegalTermsValidator
       - Used by LegalTermsRule._create_default_validator
       - Used by create_legal_terms_validator

    ## Error Handling

    - Type validation through Pydantic
    - Range validation for numeric fields (cache_size, priority, cost)
    - Set-based term storage prevents duplicates

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.domain.legal import LegalTermsConfig

    # Create with default values
    config = LegalTermsConfig()

    # Create with custom values
    config = LegalTermsConfig(
        legal_terms={"copyright", "trademark", "patent"},
        warning_terms={"warning", "caution", "notice"},
        required_terms={"disclaimer", "notice"},
        prohibited_terms={"guarantee", "warranty"},
        case_sensitive=False,
        cache_size=200,
        priority=2,
        cost=1.5
    )

    # Access configuration values
    print(f"Legal terms: {config.legal_terms}")
    print(f"Required terms: {config.required_terms}")
    print(f"Prohibited terms: {config.prohibited_terms}")
    print(f"Case sensitive: {config.case_sensitive}")
    ```

    Using with validators:

    ```python
    from sifaka.rules.domain.legal import LegalTermsConfig, DefaultLegalTermsValidator

    # Create config
    config = LegalTermsConfig(
        required_terms={"disclaimer", "notice"},
        prohibited_terms={"guarantee", "warranty"}
    )

    # Create validator with config
    validator = DefaultLegalTermsValidator(config)

    # Validate text
    result = validator.validate("DISCLAIMER: This document contains a legal notice.")
    print(f"Valid: {result.passed}")
    print(f"Legal terms found: {result.metadata.get('legal_terms_found')}")
    ```
    """

    legal_terms: Set[str] = Field(
        default_factory=lambda: {
            "confidential",
            "proprietary",
            "restricted",
            "private",
            "sensitive",
        },
        description="Set of legal terms to check for",
    )
    warning_terms: Set[str] = Field(
        default_factory=lambda: {
            "warning",
            "caution",
            "notice",
            "disclaimer",
            "privileged",
        },
        description="Set of warning terms to check for",
    )
    required_terms: Set[str] = Field(
        default_factory=set,
        description="Set of terms that must be present",
    )
    prohibited_terms: Set[str] = Field(
        default_factory=set,
        description="Set of terms that must not be present",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether term matching should be case sensitive",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )


@runtime_checkable
class LegalValidator(Protocol):
    """
    Protocol for legal content validation.

    This protocol defines the interface that all legal content validators must implement.
    It ensures that validators provide a consistent interface for validation and
    configuration access.

    ## Interface Requirements

    Implementing classes must provide:
    - A validate method that accepts text and returns a RuleResult
    - A config property that returns a LegalConfig instance

    ## Usage

    This protocol enables runtime type checking and duck typing for legal validators:

    ```python
    from sifaka.rules.domain.legal import LegalValidator, DefaultLegalValidator, LegalConfig

    # Create a validator
    config = LegalConfig()
    validator = DefaultLegalValidator(config)

    # Check if an object implements the protocol
    if isinstance(validator, LegalValidator):
        # Use the validator
        result = validator.validate("This is a legal document.")
        print(f"Valid: {result.passed}")
        print(f"Config: {validator.config}")
    ```

    Custom validators can also implement this protocol:

    ```python
    class CustomLegalValidator:
        def __init__(self, config: LegalConfig):
            self._config = config

        @property
        def config(self) -> LegalConfig:
            return self._config

        def validate(self, text: str) -> RuleResult:
            # Custom validation logic
            return RuleResult(passed=True, message="Custom validation passed")

    # This will return True at runtime
    isinstance(CustomLegalValidator(LegalConfig()), LegalValidator)
    ```
    """

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalConfig: ...


@runtime_checkable
class LegalCitationValidator(Protocol):
    """
    Protocol for legal citation validation.

    This protocol defines the interface that all legal citation validators must implement.
    It ensures that validators provide a consistent interface for validation and
    configuration access.

    ## Interface Requirements

    Implementing classes must provide:
    - A validate method that accepts text and returns a RuleResult
    - A config property that returns a LegalCitationConfig instance

    ## Usage

    This protocol enables runtime type checking and duck typing for citation validators:

    ```python
    from sifaka.rules.domain.legal import (
        LegalCitationValidator,
        DefaultLegalCitationValidator,
        LegalCitationConfig
    )

    # Create a validator
    config = LegalCitationConfig()
    validator = DefaultLegalCitationValidator(config)

    # Check if an object implements the protocol
    if isinstance(validator, LegalCitationValidator):
        # Use the validator
        result = validator.validate("As held in 410 U.S. 113...")
        print(f"Valid: {result.passed}")
        print(f"Config: {validator.config}")
    ```

    Custom validators can also implement this protocol:

    ```python
    class CustomCitationValidator:
        def __init__(self, config: LegalCitationConfig):
            self._config = config

        @property
        def config(self) -> LegalCitationConfig:
            return self._config

        def validate(self, text: str) -> RuleResult:
            # Custom validation logic
            return RuleResult(passed=True, message="Custom validation passed")

    # This will return True at runtime
    isinstance(CustomCitationValidator(LegalCitationConfig()), LegalCitationValidator)
    ```
    """

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalCitationConfig: ...


@runtime_checkable
class LegalTermsValidator(Protocol):
    """
    Protocol for legal terms validation.

    This protocol defines the interface that all legal terminology validators must implement.
    It ensures that validators provide a consistent interface for validation and
    configuration access.

    ## Interface Requirements

    Implementing classes must provide:
    - A validate method that accepts text and returns a RuleResult
    - A config property that returns a LegalTermsConfig instance

    ## Usage

    This protocol enables runtime type checking and duck typing for terminology validators:

    ```python
    from sifaka.rules.domain.legal import (
        LegalTermsValidator,
        DefaultLegalTermsValidator,
        LegalTermsConfig
    )

    # Create a validator
    config = LegalTermsConfig()
    validator = DefaultLegalTermsValidator(config)

    # Check if an object implements the protocol
    if isinstance(validator, LegalTermsValidator):
        # Use the validator
        result = validator.validate("DISCLAIMER: This document contains legal terms.")
        print(f"Valid: {result.passed}")
        print(f"Config: {validator.config}")
    ```

    Custom validators can also implement this protocol:

    ```python
    class CustomTermsValidator:
        def __init__(self, config: LegalTermsConfig):
            self._config = config

        @property
        def config(self) -> LegalTermsConfig:
            return self._config

        def validate(self, text: str) -> RuleResult:
            # Custom validation logic
            return RuleResult(passed=True, message="Custom validation passed")

    # This will return True at runtime
    isinstance(CustomTermsValidator(LegalTermsConfig()), LegalTermsValidator)
    ```
    """

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalTermsConfig: ...


# ---------------------------------------------------------------------------
# Analyzer helpers (Single Responsibility, re-usable)
# ---------------------------------------------------------------------------


class _DisclaimerAnalyzer(BaseModel):
    """
    Detect whether a text contains at least one required disclaimer pattern.

    This internal helper class analyzes text for the presence of legal disclaimers
    using regular expression patterns. It follows the Single Responsibility Principle
    by focusing solely on disclaimer detection.

    ## Architecture

    _DisclaimerAnalyzer follows a component-based architecture:
    - Uses Pydantic for configuration validation
    - Compiles regex patterns during initialization
    - Provides a simple API for disclaimer detection

    ## Lifecycle

    1. **Initialization**: Set up with disclaimer patterns
       - Initialize with list of disclaimer patterns
       - Compile patterns with re.IGNORECASE for case-insensitive matching

    2. **Detection**: Check text for disclaimers
       - Search text for any compiled pattern
       - Return boolean indicating presence of at least one disclaimer

    ## Error Handling

    - Pattern compilation during initialization
    - Efficient pattern matching using pre-compiled regexes
    - Boolean return value simplifies error handling

    ## Examples

    ```python
    from sifaka.rules.domain.legal import _DisclaimerAnalyzer

    # Create analyzer with disclaimer patterns
    analyzer = _DisclaimerAnalyzer(
        patterns=[
            "This is not legal advice",
            "Consult an attorney",
            "For informational purposes only"
        ]
    )

    # Check if text contains a disclaimer
    has_disclaimer = analyzer.contains_disclaimer(
        "DISCLAIMER: This document is for informational purposes only."
    )
    print(f"Has disclaimer: {has_disclaimer}")  # True

    # Check text without disclaimer
    has_disclaimer = analyzer.contains_disclaimer(
        "This document contains legal information."
    )
    print(f"Has disclaimer: {has_disclaimer}")  # False
    ```
    """

    patterns: List[str] = Field(default_factory=list)

    _compiled: List[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = [re.compile(pat, re.IGNORECASE) for pat in self.patterns]

    # Public API -----------------------------------------------------------
    def contains_disclaimer(self, text: str) -> bool:
        return any(pat.search(text) for pat in self._compiled)


class _LegalTermAnalyzer(BaseModel):
    """
    Count occurrences of legal terms grouped by category.

    This internal helper class analyzes text for the presence and frequency of legal
    terms organized by category. It follows the Single Responsibility Principle by
    focusing solely on legal term analysis.

    ## Architecture

    _LegalTermAnalyzer follows a component-based architecture:
    - Uses Pydantic for configuration validation
    - Compiles regex patterns during initialization
    - Organizes terms by category for structured analysis
    - Uses word boundary matching for accurate term detection

    ## Lifecycle

    1. **Initialization**: Set up with categorized terms
       - Initialize with dictionary mapping categories to term lists
       - Compile patterns with word boundaries and case-insensitive matching
       - Store compiled patterns by category

    2. **Analysis**: Count term occurrences by category
       - Search text for all compiled patterns in each category
       - Count occurrences of each term
       - Return dictionary mapping categories to occurrence counts

    ## Error Handling

    - Pattern compilation during initialization
    - Efficient pattern matching using pre-compiled regexes
    - Dictionary return value provides structured results

    ## Examples

    ```python
    from sifaka.rules.domain.legal import _LegalTermAnalyzer

    # Create analyzer with categorized legal terms
    analyzer = _LegalTermAnalyzer(
        terms={
            "jurisdiction": ["court", "venue", "forum"],
            "liability": ["liability", "responsibility", "duty"],
            "confidentiality": ["confidential", "private", "sensitive"]
        }
    )

    # Analyze text for legal terms
    counts = analyzer.analyze(
        "This document is subject to the jurisdiction of the court. "
        "The parties accept no liability for any damages."
    )

    # Print counts by category
    for category, count in counts.items():
        print(f"{category}: {count}")
    # Output:
    # jurisdiction: 2
    # liability: 1
    # confidentiality: 0
    ```
    """

    terms: Dict[str, List[str]] = Field(default_factory=dict)

    _compiled: Dict[str, List[re.Pattern[str]]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {
            cat: [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in term_list]
            for cat, term_list in self.terms.items()
        }

    def analyze(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cat, patterns in self._compiled.items():
            counts[cat] = sum(len(p.findall(text)) for p in patterns)
        return counts


class _CitationAnalyzer(BaseModel):
    """
    Locate citations, validate formatting, and compute totals.

    This internal helper class analyzes text for legal citations using regular
    expression patterns. It follows the Single Responsibility Principle by
    focusing solely on citation extraction and validation.

    ## Architecture

    _CitationAnalyzer follows a component-based architecture:
    - Uses Pydantic for configuration validation
    - Compiles regex patterns during initialization
    - Provides methods for citation extraction and validation

    ## Lifecycle

    1. **Initialization**: Set up with citation patterns
       - Initialize with list of citation regex patterns
       - Compile patterns for efficient matching

    2. **Extraction**: Extract citations from text
       - Search text for all compiled patterns
       - Extract matching citations
       - Return list of found citations

    3. **Validation**: Validate citation format
       - Check if citations match expected patterns
       - Return list of invalid citations

    ## Error Handling

    - Pattern compilation during initialization
    - Efficient pattern matching using pre-compiled regexes
    - List return values provide all matched or invalid citations

    ## Examples

    ```python
    from sifaka.rules.domain.legal import _CitationAnalyzer

    # Create analyzer with citation patterns
    analyzer = _CitationAnalyzer(
        patterns=[
            r"\d+\s+U\.S\.\s+\d+",  # US Reports
            r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
            r"\d+\s+F\.\s*\d+d\s+\d+"  # Federal Reporter
        ]
    )

    # Extract citations from text
    text = "As the Supreme Court held in 410 U.S. 113 and reaffirmed in 505 U.S. 833..."
    citations = analyzer.extract(text)
    print(f"Citations found: {citations}")  # ['410 U.S. 113', '505 U.S. 833']

    # Check for invalid citations
    invalid = analyzer.invalid(citations)
    print(f"Invalid citations: {invalid}")  # []
    ```
    """

    patterns: List[str] = Field(default_factory=list)

    _compiled: List[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = [re.compile(p) for p in self.patterns]

    def extract(self, text: str) -> List[str]:
        citations: List[str] = []
        for pat in self._compiled:
            citations.extend(pat.findall(text))
        return citations

    def invalid(self, citations: List[str]) -> List[str]:
        return [c for c in citations if not any(p.match(c) for p in self._compiled)]


class DefaultLegalValidator(BaseDomainValidator):
    """
    Default implementation of legal content validation (delegates to analyzers).

    This class implements the LegalValidator interface for legal content validation.
    It delegates the actual validation logic to specialized analyzer components,
    following the standard Sifaka delegation pattern.

    ## Architecture

    DefaultLegalValidator follows a component-based architecture:
    - Inherits from BaseDomainValidator for common validation functionality
    - Uses LegalConfig for configuration
    - Delegates to _DisclaimerAnalyzer for disclaimer detection
    - Delegates to _LegalTermAnalyzer for legal term analysis
    - Implements LegalValidator protocol

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with LegalConfig
       - Create _DisclaimerAnalyzer with disclaimer patterns
       - Create _LegalTermAnalyzer with legal terms

    2. **Validation**: Check text for legal content
       - Validate input text type
       - Delegate to analyzers for content analysis
       - Check disclaimer requirements
       - Return RuleResult with validation results and metadata

    ## Error Handling

    - Input validation (text must be a string)
    - Exception handling with informative error messages
    - Detailed metadata for debugging and analysis

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.domain.legal import DefaultLegalValidator, LegalConfig

    # Create configuration
    config = LegalConfig(
        legal_terms={"jurisdiction": ["court", "venue", "forum"]},
        disclaimers=["This is not legal advice"],
        disclaimer_required=True
    )

    # Create validator
    validator = DefaultLegalValidator(config)

    # Validate text
    result = validator.validate(
        "DISCLAIMER: This is not legal advice. This document is subject to the jurisdiction of the court."
    )
    print(f"Valid: {result.passed}")
    print(f"Message: {result.message}")
    print(f"Has disclaimer: {result.metadata.get('has_disclaimer')}")
    print(f"Legal term counts: {result.metadata.get('legal_term_counts')}")
    ```

    Using with factory function:

    ```python
    from sifaka.rules.domain.legal import create_legal_validator

    # Create validator using factory function
    validator = create_legal_validator(
        legal_terms={"jurisdiction": ["court", "venue", "forum"]},
        disclaimers=["This is not legal advice"],
        disclaimer_required=True
    )

    # Validate text
    result = validator.validate("This document is subject to the jurisdiction of the court.")
    print(f"Valid: {result.passed}")
    ```
    """

    def __init__(self, config: LegalConfig) -> None:
        super().__init__(config)

        self._disclaimer_analyzer = _DisclaimerAnalyzer(patterns=config.disclaimers)
        self._term_analyzer = _LegalTermAnalyzer(terms=config.legal_terms)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> LegalConfig:  # type: ignore[override]
        return self._config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401 – simple desc
        """Validate *text* for legal content consistency and disclaimers."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            has_disclaimer = self._disclaimer_analyzer.contains_disclaimer(text)
            term_counts = self._term_analyzer.analyze(text)

            if self.config.disclaimer_required and not has_disclaimer:
                return RuleResult(
                    passed=False,
                    message="No legal disclaimer found when required",
                    metadata={
                        "legal_term_counts": term_counts,
                        "has_disclaimer": False,
                        "disclaimer_required": True,
                    },
                )

            return RuleResult(
                passed=True,
                message="Legal content validation passed",
                metadata={
                    "legal_term_counts": term_counts,
                    "has_disclaimer": has_disclaimer,
                    "disclaimer_required": self.config.disclaimer_required,
                },
            )
        except Exception as e:  # pragma: no cover
            return RuleResult(
                passed=False,
                message=f"Error validating legal content: {e}",
                metadata={"error": str(e)},
            )


class DefaultLegalCitationValidator(BaseDomainValidator):
    """Default implementation of legal citation validation using _CitationAnalyzer."""

    def __init__(self, config: LegalCitationConfig) -> None:
        super().__init__(config)
        self._citation_analyzer = _CitationAnalyzer(patterns=config.citation_patterns)

    @property
    def config(self) -> LegalCitationConfig:  # type: ignore[override]
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        """Validate *text* for citation presence, count, and correctness."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            citations = self._citation_analyzer.extract(text)
            invalid = self._citation_analyzer.invalid(citations)
            total = len(citations)

            # Requirements checks ------------------------------------------------
            if self.config.require_citations and total == 0:
                return RuleResult(
                    passed=False,
                    message="No citations found when required",
                    metadata={"total_citations": 0},
                )

            if total < self.config.min_citations:
                return RuleResult(
                    passed=False,
                    message=(
                        f"Found {total} citations; minimum required is {self.config.min_citations}"
                    ),
                    metadata={"total_citations": total},
                )

            if total > self.config.max_citations:
                return RuleResult(
                    passed=False,
                    message=(
                        f"Found {total} citations; maximum allowed is {self.config.max_citations}"
                    ),
                    metadata={"total_citations": total},
                )

            if invalid:
                return RuleResult(
                    passed=False,
                    message=f"Found {len(invalid)} invalid citations",
                    metadata={"invalid_citations": invalid, "total_citations": total},
                )

            return RuleResult(
                passed=True,
                message=f"Found {total} valid citations",
                metadata={"total_citations": total},
            )
        except Exception as e:  # pragma: no cover
            return RuleResult(
                passed=False,
                message=f"Error validating citations: {e}",
                metadata={"error": str(e)},
            )


class DefaultLegalTermsValidator(BaseDomainValidator):
    """Default implementation of legal terms validation with analyzers."""

    def __init__(self, config: LegalTermsConfig) -> None:
        super().__init__(config)

        flags = 0 if config.case_sensitive else re.IGNORECASE

        # Pre-compile sets for quick membership checks
        self._legal_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.legal_terms
        ]
        self._warning_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.warning_terms
        ]
        self._required_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.required_terms
        ]
        self._prohibited_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.prohibited_terms
        ]

    @property
    def config(self) -> LegalTermsConfig:  # type: ignore[override]
        return self._config

    def _matches(self, patterns: List[re.Pattern[str]], text: str) -> Set[str]:
        return {p.pattern.strip("\\b").strip("\\b") for p in patterns if p.search(text)}

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        """Validate legal term usage in *text*."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            legal_found = self._matches(self._legal_patterns, text)
            warning_found = self._matches(self._warning_patterns, text)
            missing_required = {
                p.pattern.strip("\\b").strip("\\b")
                for p in self._required_patterns
                if not p.search(text)
            }
            prohibited_found = self._matches(self._prohibited_patterns, text)

            if missing_required:
                return RuleResult(
                    passed=False,
                    message="Missing required legal terms",
                    metadata={
                        "missing_required_terms": missing_required,
                        "legal_terms_found": legal_found,
                        "warning_terms_found": warning_found,
                        "prohibited_terms_found": prohibited_found,
                    },
                )

            if prohibited_found:
                return RuleResult(
                    passed=False,
                    message="Found prohibited legal terms",
                    metadata={
                        "missing_required_terms": missing_required,
                        "legal_terms_found": legal_found,
                        "warning_terms_found": warning_found,
                        "prohibited_terms_found": prohibited_found,
                    },
                )

            return RuleResult(
                passed=True,
                message="Legal terms validation passed",
                metadata={
                    "legal_terms_found": legal_found,
                    "warning_terms_found": warning_found,
                },
            )
        except Exception as e:  # pragma: no cover
            return RuleResult(
                passed=False,
                message=f"Error validating legal terms: {e}",
                metadata={"error": str(e)},
            )


class LegalRule(Rule[str, RuleResult, DefaultLegalValidator, Any]):
    """Rule that validates legal content."""

    def __init__(
        self,
        name: str = "legal_rule",
        description: str = "Validates legal content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLegalValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the legal rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultLegalValidator:
        """Create a default validator from config."""
        legal_config = LegalConfig(**self._rule_params)
        return DefaultLegalValidator(legal_config)


class LegalCitationRule(Rule[str, RuleResult, DefaultLegalCitationValidator, Any]):
    """Rule that checks for legal citations."""

    def __init__(
        self,
        name: str = "legal_citation_rule",
        description: str = "Checks for legal citations",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLegalCitationValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the legal citation rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultLegalCitationValidator:
        """Create a default validator from config."""
        citation_config = LegalCitationConfig(**self._rule_params)
        return DefaultLegalCitationValidator(citation_config)


class LegalTermsRule(Rule[str, RuleResult, DefaultLegalTermsValidator, Any]):
    """Rule that validates legal terminology."""

    def __init__(
        self,
        name: str = "legal_terms_rule",
        description: str = "Validates legal terminology",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLegalTermsValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the legal terms rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultLegalTermsValidator:
        """Create a default validator from config."""
        terms_config = LegalTermsConfig(**self._rule_params)
        return DefaultLegalTermsValidator(terms_config)


def create_legal_validator(
    legal_terms: Optional[Dict[str, List[str]]] = None,
    citation_patterns: Optional[List[str]] = None,
    disclaimers: Optional[List[str]] = None,
    disclaimer_required: Optional[bool] = None,
    **kwargs,
) -> DefaultLegalValidator:
    """
    Create a legal validator with the specified configuration.

    This factory function creates a configured legal validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        legal_terms: Dictionary mapping categories to lists of legal terms
        citation_patterns: List of regex patterns for legal citations
        disclaimers: List of regex patterns for legal disclaimers
        disclaimer_required: Whether a disclaimer is required
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured legal validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if legal_terms is not None:
        config_params["legal_terms"] = legal_terms
    if citation_patterns is not None:
        config_params["citation_patterns"] = citation_patterns
    if disclaimers is not None:
        config_params["disclaimers"] = disclaimers
    if disclaimer_required is not None:
        config_params["disclaimer_required"] = disclaimer_required

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = LegalConfig(**config_params)

    # Return configured validator
    return DefaultLegalValidator(config)


def create_legal_rule(
    name: str = "legal_rule",
    description: str = "Validates text for legal content",
    legal_terms: Optional[Dict[str, List[str]]] = None,
    citation_patterns: Optional[List[str]] = None,
    disclaimers: Optional[List[str]] = None,
    disclaimer_required: Optional[bool] = None,
    **kwargs,
) -> LegalRule:
    """
    Create a legal rule with configuration.

    This factory function creates a configured LegalRule instance.
    It uses create_legal_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        legal_terms: Dictionary mapping categories to lists of legal terms
        citation_patterns: List of regex patterns for legal citations
        disclaimers: List of regex patterns for legal disclaimers
        disclaimer_required: Whether a disclaimer is required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured LegalRule instance
    """
    # Create validator using the validator factory
    validator = create_legal_validator(
        legal_terms=legal_terms,
        citation_patterns=citation_patterns,
        disclaimers=disclaimers,
        disclaimer_required=disclaimer_required,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return LegalRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


def create_legal_citation_validator(
    citation_patterns: Optional[List[str]] = None,
    require_citations: Optional[bool] = None,
    min_citations: Optional[int] = None,
    max_citations: Optional[int] = None,
    **kwargs,
) -> DefaultLegalCitationValidator:
    """
    Create a legal citation validator with the specified configuration.

    This factory function creates a configured legal citation validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        citation_patterns: List of regex patterns for legal citations
        require_citations: Whether citations are required
        min_citations: Minimum number of citations required
        max_citations: Maximum number of citations allowed
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured legal citation validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if citation_patterns is not None:
        config_params["citation_patterns"] = citation_patterns
    if require_citations is not None:
        config_params["require_citations"] = require_citations
    if min_citations is not None:
        config_params["min_citations"] = min_citations
    if max_citations is not None:
        config_params["max_citations"] = max_citations

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = LegalCitationConfig(**config_params)

    # Return configured validator
    return DefaultLegalCitationValidator(config)


def create_legal_citation_rule(
    name: str = "legal_citation_rule",
    description: str = "Validates legal citations",
    citation_patterns: Optional[List[str]] = None,
    require_citations: Optional[bool] = None,
    min_citations: Optional[int] = None,
    max_citations: Optional[int] = None,
    **kwargs,
) -> LegalCitationRule:
    """
    Create a legal citation validation rule.

    This factory function creates a configured LegalCitationRule instance.
    It uses create_legal_citation_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        citation_patterns: List of regex patterns for legal citations
        require_citations: Whether citations are required
        min_citations: Minimum number of citations required
        max_citations: Maximum number of citations allowed
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured LegalCitationRule
    """
    # Create validator using the validator factory
    validator = create_legal_citation_validator(
        citation_patterns=citation_patterns,
        require_citations=require_citations,
        min_citations=min_citations,
        max_citations=max_citations,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return LegalCitationRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


def create_legal_terms_validator(
    legal_terms: Optional[Set[str]] = None,
    warning_terms: Optional[Set[str]] = None,
    required_terms: Optional[Set[str]] = None,
    prohibited_terms: Optional[Set[str]] = None,
    case_sensitive: Optional[bool] = None,
    **kwargs,
) -> DefaultLegalTermsValidator:
    """
    Create a legal terms validator with the specified configuration.

    This factory function creates a configured legal terms validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        legal_terms: Set of legal terms to check for
        warning_terms: Set of warning terms to check for
        required_terms: Set of terms that must be present
        prohibited_terms: Set of terms that must not be present
        case_sensitive: Whether to make checks case-sensitive
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured legal terms validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if legal_terms is not None:
        config_params["legal_terms"] = legal_terms
    if warning_terms is not None:
        config_params["warning_terms"] = warning_terms
    if required_terms is not None:
        config_params["required_terms"] = required_terms
    if prohibited_terms is not None:
        config_params["prohibited_terms"] = prohibited_terms
    if case_sensitive is not None:
        config_params["case_sensitive"] = case_sensitive

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = LegalTermsConfig(**config_params)

    # Return configured validator
    return DefaultLegalTermsValidator(config)


def create_legal_terms_rule(
    name: str = "legal_terms_rule",
    description: str = "Validates legal terms",
    legal_terms: Optional[Set[str]] = None,
    warning_terms: Optional[Set[str]] = None,
    required_terms: Optional[Set[str]] = None,
    prohibited_terms: Optional[Set[str]] = None,
    case_sensitive: Optional[bool] = None,
    **kwargs,
) -> LegalTermsRule:
    """
    Create a legal terms validation rule.

    This factory function creates a configured LegalTermsRule instance.
    It uses create_legal_terms_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        legal_terms: Set of legal terms to check for
        warning_terms: Set of warning terms to check for
        required_terms: Set of terms that must be present
        prohibited_terms: Set of terms that must not be present
        case_sensitive: Whether to make checks case-sensitive
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured LegalTermsRule
    """
    # Create validator using the validator factory
    validator = create_legal_terms_validator(
        legal_terms=legal_terms,
        warning_terms=warning_terms,
        required_terms=required_terms,
        prohibited_terms=prohibited_terms,
        case_sensitive=case_sensitive,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return LegalTermsRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
