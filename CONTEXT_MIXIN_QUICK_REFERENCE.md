# ContextAwareMixin Quick Reference

## 🚀 3-Line Integration

Make any critic or model context-aware with exactly 3 lines:

```python
# 1. Add mixin inheritance
class MyCritic(ContextAwareMixin):

# 2. Get context in methods
context = self._prepare_context(thought)

# 3. Use context in prompts
prompt = template.format(text=thought.text, context=context)
```

## 📚 Core Methods

### Basic Context
```python
# Standard context preparation
context = self._prepare_context(thought, max_docs=5)

# Check if context is available
has_context = self._has_context(thought)

# Get human-readable summary
summary = self._get_context_summary(thought)
# Returns: "Context available: 2 pre-generation documents and 1 post-generation document (total: 3)"
```

### Advanced Context
```python
# Relevance-filtered context
context = self._prepare_context_with_relevance(
    thought, 
    query="AI healthcare diagnosis", 
    max_docs=3
)

# Embedding-based context (with fallback to keyword overlap)
context = self._prepare_context_with_embeddings(
    thought,
    similarity_threshold=0.7,
    max_docs=5
)

# Compressed context for long documents
context = self._compress_context(
    thought,
    max_length=2000,
    preserve_diversity=True
)
```

### Template Enhancement
```python
# Automatically add {context} to templates
enhanced_template = self._enhance_template_with_context(
    "Analyze this text: {text}"
)
# Returns: "Analyze this text: {text}\n\nRetrieved context:\n{context}"

# One-call template + context preparation
template, kwargs = self._context_aware_template(
    "Analyze this text: {text}",
    thought
)
result = template.format(text=thought.text, **kwargs)
```

### Model-Specific Methods
```python
# Context optimized for generation
context = self._prepare_context_for_generation(thought, max_docs=5)

# Complete contextualized prompt
full_prompt = self._build_contextualized_prompt(thought, max_docs=5)

# Context for system prompts
system_context = self._get_context_for_system_prompt(thought)
```

## 🎯 Usage Patterns

### Critic Pattern
```python
class MyCritic(ContextAwareMixin):
    def critique(self, thought: Thought) -> Dict[str, Any]:
        # Prepare context
        context = self._prepare_context(thought)
        
        # Log context usage
        if self._has_context(thought):
            summary = self._get_context_summary(thought)
            logger.debug(f"MyCritic using context: {summary}")
        
        # Use in critique prompt
        prompt = self.template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context
        )
        
        # Generate critique
        critique = self.model.generate(prompt)
        
        return {
            "needs_improvement": self._needs_improvement(critique),
            "message": critique,
            "critique": critique
        }
```

### Model Pattern
```python
class MyModel(ContextAwareMixin):
    def generate_with_thought(self, thought: Thought, **options) -> str:
        # Build contextualized prompt
        full_prompt = self._build_contextualized_prompt(thought, max_docs=5)
        
        # Log context usage
        if self._has_context(thought):
            summary = self._get_context_summary(thought)
            logger.debug(f"MyModel using context: {summary}")
        
        # Generate with context
        return self.generate(full_prompt, **options)
```

### Advanced Pattern
```python
class AdvancedCritic(ContextAwareMixin):
    def critique(self, thought: Thought) -> Dict[str, Any]:
        # Choose context method based on configuration
        if self.use_embeddings:
            context = self._prepare_context_with_embeddings(
                thought, similarity_threshold=0.75
            )
        elif self.use_relevance_filter:
            context = self._prepare_context_with_relevance(
                thought, query=self._extract_key_terms(thought.prompt)
            )
        else:
            context = self._prepare_context(thought)
        
        # Compress if too long
        if len(context) > 2000:
            context = self._compress_context(thought, max_length=2000)
        
        # Enhanced template with automatic context
        template, kwargs = self._context_aware_template(
            self.base_template, thought
        )
        
        # Generate critique
        prompt = template.format(
            prompt=thought.prompt,
            text=thought.text,
            **kwargs
        )
        
        return self._process_critique(prompt)
```

## 🔧 Template Updates

### Before (Manual Context)
```python
template = """
Critique this text: {text}

Original prompt: {prompt}

Please provide feedback.
"""

# Manual context handling (inconsistent, error-prone)
if thought.pre_generation_context:
    context_parts = []
    for i, doc in enumerate(thought.pre_generation_context):
        context_parts.append(f"Document {i+1}: {doc.text}")
    context = "\n\n".join(context_parts)
    template += f"\n\nContext:\n{context}"
```

### After (ContextAwareMixin)
```python
template = """
Critique this text: {text}

Original prompt: {prompt}

Retrieved context: {context}

Please provide feedback.
"""

# Automatic context handling (consistent, robust)
context = self._prepare_context(thought)
prompt = template.format(
    text=thought.text,
    prompt=thought.prompt,
    context=context
)
```

## 📊 Context Output Examples

### Basic Context
```
Document 1: Artificial Intelligence (AI) is transforming healthcare through diagnostic tools, personalized medicine, and robotic surgery.

Document 2: Machine learning algorithms can analyze medical images with accuracy comparable to human radiologists.

Document 3 (post-generation): Recent studies show AI-powered drug discovery can reduce development time by 30%.
```

### Relevance-Filtered Context
```
Document 1 (pre-generation, relevance: 0.67): Artificial Intelligence (AI) is transforming healthcare through diagnostic tools, personalized medicine, and robotic surgery.

Document 2 (pre-generation, relevance: 0.83): Machine learning algorithms can analyze medical images with accuracy comparable to human radiologists.
```

### Embedding-Based Context
```
Document 1 (pre-generation, similarity: 0.892): Artificial Intelligence (AI) is transforming healthcare through diagnostic tools, personalized medicine, and robotic surgery.

Document 2 (post-generation, similarity: 0.756): Recent studies show AI-powered drug discovery can reduce development time by 30%.
```

### Compressed Context
```
Document 1 (pre-generation): Artificial Intelligence (AI) is transforming healthcare through diagnostic tools, personalized medicine...

Document 2 (pre-generation): Machine learning algorithms can analyze medical images with accuracy comparable to human...

Document 3 (post-generation): Recent studies show AI-powered drug discovery can reduce development time by 30%.
```

## ⚡ Performance Tips

### Optimization
```python
# Cache context for repeated use
if not hasattr(self, '_cached_context'):
    self._cached_context = self._prepare_context(thought)
context = self._cached_context

# Use compression for long contexts
if len(context) > 1500:
    context = self._compress_context(thought, max_length=1500)

# Limit documents for performance
context = self._prepare_context(thought, max_docs=3)
```

### Debugging
```python
# Check context availability
if not self._has_context(thought):
    logger.warning("No context available for processing")
    return self._process_without_context(thought)

# Log context summary
summary = self._get_context_summary(thought)
logger.debug(f"Processing with context: {summary}")

# Log context length
context = self._prepare_context(thought)
logger.debug(f"Context length: {len(context)} characters")
```

## 🎯 Common Patterns

### Error Handling
```python
try:
    context = self._prepare_context_with_embeddings(thought)
except Exception as e:
    logger.warning(f"Embedding-based context failed: {e}")
    context = self._prepare_context(thought)  # Fallback
```

### Conditional Context
```python
# Use different context methods based on task
if task_type == "factual":
    context = self._prepare_context_with_relevance(thought, query="facts")
elif task_type == "creative":
    context = self._prepare_context(thought, max_docs=2)  # Less context
else:
    context = self._prepare_context_with_embeddings(thought)
```

### Context Validation
```python
context = self._prepare_context(thought)
if len(context.strip()) < 50:  # Too little context
    logger.warning("Very little context available")
elif len(context) > 3000:  # Too much context
    context = self._compress_context(thought, max_length=2000)
```

---

**Need help? Check the full documentation or ask in our Discord community! 🚀**
