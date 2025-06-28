# Things to Solve

## High Priority Issues

### LLM Non-Deterministic Feedback Quality
- **Issue**: Critics (especially N-Critics) produce inconsistent feedback quality with identical prompts and settings
- **Symptoms**:
  - Generic feedback: "The text contains several issues that need to be addressed from different critical perspectives"
  - vs Specific feedback: "The text promotes investing all money in cryptocurrency without considering potential risks..."
- **Root Cause**: LLM non-determinism with `gpt-3.5-turbo` at `temperature: 0.7`
- **Impact**: Unpredictable user experience, sometimes receiving unhelpful generic feedback
- **Evidence**: Two consecutive runs of n_critics_example.py with identical settings produced vastly different feedback quality
- **Potential Solutions**:
  - **Lower Temperature** (0.3-0.5): More consistent but potentially less creative responses
  - **Upgrade Model** (gpt-4o-mini/gpt-4): Better instruction following but slower and more expensive
  - **Quality Validation**: Detect generic responses and retry, adds latency and cost
  - **Enhanced Prompting**: Add explicit anti-generic instructions, may reduce creativity
- **Tradeoffs**:
  - **Consistency vs Cost**: Better models cost 3-10x more per token
  - **Reliability vs Speed**: Quality validation adds 2x retry overhead
  - **Determinism vs Creativity**: Lower temperature reduces creative variations
  - **Quality vs Latency**: Retries for quality increase response time
- **Files Affected**: `sifaka/critics/n_critics.py`, `sifaka/critics/core/base.py`
- **Thought Files**: Compare `n_critics_20250627_204627_d093b44c.json` (generic) vs `n_critics_20250627_204640_15a9774b.json` (specific)

### Memory Management in Large Documents
- **Issue**: Current memory bounds may not scale well for very large documents
- **Impact**: Potential performance degradation with extensive critique histories
- **Solution**: Implement sliding window or summarization for critique history

### Configuration Complexity
- **Issue**: Many configuration options can be overwhelming for new users
- **Solution**: Implement configuration presets for common use cases
- **Status**: Medium priority

## Medium Priority Issues

### Test Coverage Gaps
- **Issue**: Some edge cases and error handling paths lack test coverage
- **Target**: Maintain >80% test coverage
- **Status**: Ongoing

### Documentation Completeness
- **Issue**: Some advanced features lack comprehensive documentation
- **Solution**: Expand examples and API documentation
- **Status**: Ongoing

## Low Priority Issues

### Performance Optimization
- **Issue**: Potential optimizations for parallel critic execution
- **Impact**: Minor performance gains in specific scenarios
- **Status**: Future enhancement

### Additional Validator Types
- **Issue**: Limited set of built-in validators
- **Solution**: Add more domain-specific validators
- **Status**: Feature enhancement
