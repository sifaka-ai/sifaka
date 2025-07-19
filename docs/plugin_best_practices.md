# Plugin Best Practices

This guide covers best practices for developing high-quality Sifaka plugins.

## General Principles

### 1. Follow the Interface

Always implement the complete interface:

```python
# Good: Complete implementation
class MyCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Implementation
        return CritiqueResult(...)

# Bad: Missing required method
class BadCritic(CriticPlugin):
    def analyze(self, text):  # Wrong method name
        pass
```

### 2. Handle Errors Gracefully

```python
class RobustCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        try:
            # Your logic here
            analysis = await self._analyze(text)
            return CritiqueResult(
                critic="robust_critic",
                feedback=analysis.feedback,
                suggestions=analysis.suggestions,
                needs_improvement=True,
                confidence=0.8
            )
        except Exception as e:
            # Return meaningful feedback even on error
            return CritiqueResult(
                critic="robust_critic",
                feedback=f"Analysis incomplete: {str(e)}",
                suggestions=["Please review the text manually"],
                needs_improvement=True,
                confidence=0.1  # Low confidence on error
            )
```

### 3. Be Async-First

```python
# Good: Async implementation
class AsyncCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Can use await for async operations
        data = await fetch_external_data()
        return CritiqueResult(...)

# Bad: Blocking operations
class BlockingCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # This blocks the event loop!
        time.sleep(5)
        data = requests.get("https://api.example.com").json()
        return CritiqueResult(...)
```

## Critic Plugin Best Practices

### 1. Provide Clear Feedback

```python
class ClearFeedbackCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        issues = []

        # Specific, actionable feedback
        if len(text.split()) < 50:
            issues.append("Expand the text to at least 50 words for better context")

        if "however" in text.lower() and "but" in text.lower():
            issues.append("Avoid using both 'however' and 'but' - choose one for consistency")

        return CritiqueResult(
            critic="clear_feedback",
            feedback="Style and length analysis complete",
            suggestions=issues,
            needs_improvement=len(issues) > 0,
            confidence=0.9 if len(issues) == 0 else 0.6
        )
```

### 2. Use Confidence Appropriately

```python
class ConfidenceCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Calculate confidence based on analysis certainty
        word_count = len(text.split())

        if word_count < 10:
            confidence = 0.3  # Low confidence for very short text
        elif word_count > 1000:
            confidence = 0.7  # Medium confidence for long text
        else:
            confidence = 0.9  # High confidence for typical text

        return CritiqueResult(
            critic="confidence_aware",
            feedback="Analysis complete",
            suggestions=[],
            needs_improvement=False,
            confidence=confidence
        )
```

### 3. Consider Context

```python
class ContextAwareCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Look at previous iterations
        if result.iteration > 1:
            # Check if issues from previous critiques were addressed
            prev_suggestions = []
            for critique in result.critiques:
                prev_suggestions.extend(critique.suggestions)

            # Adjust feedback based on progress
            if len(prev_suggestions) > len(current_issues):
                feedback = "Good progress on addressing previous issues"
            else:
                feedback = "Some previous issues remain unaddressed"

        return CritiqueResult(...)
```

## Validator Plugin Best Practices

### 1. Return Meaningful Scores

```python
class ScoringValidator(ValidatorPlugin):
    async def validate(self, text: str) -> ValidationResult:
        word_count = len(text.split())

        # Graduated scoring, not just pass/fail
        if word_count < 50:
            score = 0.3
            details = "Too short - aim for 50+ words"
        elif word_count < 100:
            score = 0.7
            details = "Acceptable length, could be expanded"
        else:
            score = 1.0
            details = "Excellent length"

        return ValidationResult(
            validator="length_scorer",
            passed=score >= 0.5,
            score=score,
            details=details
        )
```

### 2. Provide Actionable Details

```python
class DetailedValidator(ValidatorPlugin):
    async def validate(self, text: str) -> ValidationResult:
        issues = []

        # Check multiple criteria
        if not text[0].isupper():
            issues.append("Start with a capital letter")

        if text[-1] not in '.!?':
            issues.append("End with proper punctuation")

        sentences = text.split('.')
        if any(len(s.split()) > 30 for s in sentences):
            issues.append("Break up long sentences (30+ words)")

        return ValidationResult(
            validator="grammar_checker",
            passed=len(issues) == 0,
            score=1.0 - (len(issues) * 0.2),  # Deduct 20% per issue
            details="; ".join(issues) if issues else "All checks passed"
        )
```

## Storage Plugin Best Practices

### 1. Handle Concurrent Access

```python
class ThreadSafeStorage(StoragePlugin):
    def __init__(self):
        self._lock = asyncio.Lock()
        self._data = {}

    async def save(self, result: SifakaResult) -> str:
        async with self._lock:
            self._data[result.id] = result
            return result.id

    async def load(self, result_id: str) -> SifakaResult | None:
        async with self._lock:
            return self._data.get(result_id)
```

### 2. Implement Cleanup

```python
class ManagedStorage(StoragePlugin):
    def __init__(self, max_age_hours: int = 24):
        self.max_age_hours = max_age_hours

    async def cleanup(self):
        """Remove old results"""
        cutoff = datetime.now() - timedelta(hours=self.max_age_hours)

        for result_id in await self.list_results():
            result = await self.load(result_id)
            if result and result.created_at < cutoff:
                await self.delete(result_id)
```

### 3. Handle Large Data

```python
class EfficientStorage(StoragePlugin):
    async def save(self, result: SifakaResult) -> str:
        # Save large fields separately if needed
        if len(result.final_text) > 10000:
            # Store text separately
            text_id = await self._save_large_text(result.final_text)
            # Save reference in main result
            result_data = result.model_dump()
            result_data['final_text'] = f"ref:{text_id}"

        return await self._save_result(result_data)
```

## Performance Best Practices

### 1. Cache Expensive Operations

```python
class CachedCritic(CriticPlugin):
    def __init__(self):
        self._cache = {}

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Cache key based on text hash
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Return cached result with updated confidence
            return CritiqueResult(
                **cached,
                confidence=cached['confidence'] * 0.9  # Slightly lower
            )

        # Perform expensive analysis
        result = await self._expensive_analysis(text)
        self._cache[cache_key] = result.model_dump()

        return result
```

### 2. Batch Operations

```python
class BatchValidator(ValidatorPlugin):
    def __init__(self):
        self._queue = []
        self._results = {}

    async def validate(self, text: str) -> ValidationResult:
        # Add to queue
        text_id = str(uuid4())
        self._queue.append((text_id, text))

        # Batch process when queue is full
        if len(self._queue) >= 10:
            await self._process_batch()

        # Wait for result
        while text_id not in self._results:
            await asyncio.sleep(0.1)

        return self._results.pop(text_id)
```

## Testing Best Practices

### 1. Test Edge Cases

```python
import pytest

class TestMyCritic:
    @pytest.mark.asyncio
    async def test_empty_text(self):
        critic = MyCritic()
        result = await critic.critique("", SifakaResult(...))
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        critic = MyCritic()
        long_text = "word " * 10000
        result = await critic.critique(long_text, SifakaResult(...))
        assert result.feedback  # Should handle gracefully
```

### 2. Mock External Dependencies

```python
class TestExternalCritic:
    @pytest.mark.asyncio
    async def test_api_failure(self, mocker):
        # Mock external API
        mocker.patch('aiohttp.ClientSession.get',
                    side_effect=Exception("API Error"))

        critic = ExternalAPICritic()
        result = await critic.critique("test", SifakaResult(...))

        # Should handle failure gracefully
        assert result.confidence < 0.5
        assert "error" in result.feedback.lower()
```

## Documentation Best Practices

### 1. Include Examples

```python
class WellDocumentedCritic(CriticPlugin):
    """
    Checks text for technical accuracy.

    Example:
        >>> critic = TechnicalAccuracyCritic()
        >>> result = await critic.critique(
        ...     "Python uses tabs for indentation",
        ...     SifakaResult(...)
        ... )
        >>> print(result.suggestions)
        ['Python typically uses 4 spaces, not tabs']
    """

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Implementation
        pass
```

### 2. Document Parameters

```python
class ConfigurableCritic(CriticPlugin):
    """
    A critic that checks for specific patterns.

    Args:
        patterns: List of regex patterns to check
        severity: How strictly to evaluate (0.0-1.0)

    Raises:
        ValueError: If severity is not between 0 and 1
    """

    def __init__(self, patterns: list[str], severity: float = 0.5):
        if not 0 <= severity <= 1:
            raise ValueError("Severity must be between 0 and 1")
        self.patterns = patterns
        self.severity = severity
```

## Security Best Practices

### 1. Validate Input

```python
class SecureCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Validate input
        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        if len(text) > 1_000_000:  # 1MB limit
            raise ValueError("Text too large")

        # Sanitize if needed
        safe_text = self._sanitize(text)

        # Continue with analysis
        return await self._analyze(safe_text)
```

### 2. Don't Expose Sensitive Data

```python
class PrivacyCritic(CriticPlugin):
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        # Don't include sensitive data in feedback
        if self._contains_pii(text):
            return CritiqueResult(
                critic="privacy_critic",
                feedback="Text contains sensitive information",
                suggestions=["Remove personal identifiable information"],
                needs_improvement=True,
                confidence=1.0
            )

        # Regular analysis for safe text
        return await self._normal_critique(text, result)
```

## Summary

Key takeaways for plugin development:

1. **Be Async**: Use async/await properly
2. **Handle Errors**: Always return valid results
3. **Provide Value**: Clear, actionable feedback
4. **Test Thoroughly**: Edge cases and failures
5. **Document Well**: Examples and parameter docs
6. **Think Performance**: Cache and batch when appropriate
7. **Stay Secure**: Validate input and protect data
