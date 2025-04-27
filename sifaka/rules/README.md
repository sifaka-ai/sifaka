# Sifaka Rules Documentation

This directory contains the implementation of various validation rules used in the Sifaka framework.

## Rule Categories

### Safety Rules (`safety.py`)

#### ToxicityRule
- **Purpose**: Validates text for toxic content
- **Implementation**:
  - Uses predefined toxic indicators and thresholds
  - Calculates toxicity score based on found indicators
  - Metadata includes found toxic indicators and overall score
- **Example Usage**:
```python
rule = ToxicityRule(threshold=0.7)
result = rule.validate("text")
# Metadata: {"toxic_indicators": ["found_indicators"], "toxicity_score": 0.8}
```

#### BiasRule
- **Purpose**: Detects biased content
- **Implementation**:
  - Checks for bias across multiple categories
  - Uses category-specific indicators
  - Returns detailed bias analysis in metadata
- **Example Usage**:
```python
rule = BiasRule(categories=["gender", "race"])
result = rule.validate("text")
# Metadata: {"bias_categories": {"gender": 0.6, "race": 0.2}}
```

#### HarmfulContentRule
- **Purpose**: Identifies harmful or dangerous content
- **Implementation**:
  - Categorizes harmful content types
  - Stores found categories in metadata
  - Provides detailed analysis of harmful elements
- **Example Usage**:
```python
rule = HarmfulContentRule()
result = rule.validate("text")
# Metadata: {"harmful_categories": ["violence", "self_harm"]}
```

### Content Rules (`content.py`)

#### ProhibitedContentRule
- **Purpose**: Enforces content restrictions
- **Implementation**:
  - Maintains list of prohibited terms/patterns
  - Case-sensitive and insensitive matching
  - Returns found prohibited content in metadata

#### ToneConsistencyRule
- **Purpose**: Ensures consistent tone
- **Implementation**:
  - Analyzes tone indicators
  - Tracks tone shifts
  - Provides tone consistency score

### Legal Rules (`legal.py`)

#### LegalCitationRule
- **Purpose**: Validates legal citations
- **Implementation**:
  - Uses regex patterns for citation formats
  - Validates citation structure
  - Returns invalid citations in metadata

### Factual Rules (`factual.py`)

#### FactualConsistencyRule
- **Purpose**: Ensures factual consistency
- **Implementation**:
  - Tracks factual claims
  - Validates consistency across text
  - Returns inconsistencies in metadata

#### ConfidenceRule
- **Purpose**: Validates confidence statements
- **Implementation**:
  - Analyzes confidence indicators
  - Validates confidence levels
  - Returns confidence analysis

### Base Rules

#### ClassifierRule (`classifier_rule.py`)
- **Purpose**: Base class for classifier-based rules
- **Implementation**:
  - Integrates with classifiers
  - Handles validation logic
  - Manages thresholds and labels

#### TemplateRule (`template.py`)
- **Purpose**: Template for new rules
- **Implementation**:
  - Provides base structure
  - Includes required methods
  - Documents extension points

## Error Handling

All rules implement consistent error handling:
- `None` input validation
- Type checking
- Appropriate error messages
- Exception propagation

## Metadata Structure

Rules follow a consistent metadata structure:
```python
{
    "score": float,          # Overall rule score
    "details": dict,         # Rule-specific details
    "found_patterns": list,  # Found patterns/indicators
    "analysis": dict         # Detailed analysis results
}
```

## Contributing

When adding new rules:
1. Follow the template structure
2. Implement comprehensive error handling
3. Include detailed metadata
4. Add unit tests
5. Update documentation

## Testing

Each rule has corresponding tests in `tests/rules/`:
- Initialization tests
- Validation tests
- Edge cases
- Error handling
- Metadata validation