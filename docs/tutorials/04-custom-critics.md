# Custom Critics Tutorial - Build Your Own Improvement Logic

**Goal**: Create sophisticated custom critics that implement your specific improvement strategies.

**Prerequisites**: Complete [Advanced Features Tutorial](03-advanced-features.md)

## Understanding Critics

Critics are the heart of Sifaka's improvement system. They analyze generated content and provide specific feedback for enhancement. Sifaka includes research-backed critics (Reflexion, Constitutional AI, Self-RAG), but you can create your own for domain-specific needs.

## Critic Architecture

```python
from sifaka.critics.base import BaseCritic, CritiqueResult

class YourCustomCritic(BaseCritic):
    """Your custom improvement logic."""
    
    def __init__(self, model_name: str = "openai:gpt-4"):
        super().__init__(model_name=model_name)
    
    async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
        # Your improvement logic here
        pass
```

## Example 1: Domain-Specific Critic

### Medical Content Critic

```python
import asyncio
from sifaka.critics.base import BaseCritic, CritiqueResult
from sifaka.advanced import SifakaEngine, SifakaConfig

class MedicalContentCritic(BaseCritic):
    """Critic specialized for medical and healthcare content."""
    
    def __init__(self, model_name: str = "openai:gpt-4"):
        super().__init__(model_name=model_name)
        self.medical_guidelines = [
            "Include appropriate medical disclaimers",
            "Use precise medical terminology",
            "Cite reputable medical sources",
            "Avoid giving direct medical advice",
            "Include warnings about consulting healthcare professionals"
        ]
    
    async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
        prompt = f"""
        Analyze this medical/healthcare content for accuracy, safety, and compliance:
        
        {text}
        
        Evaluation criteria:
        1. Medical accuracy and terminology
        2. Appropriate disclaimers and warnings
        3. Ethical considerations
        4. Patient safety implications
        5. Regulatory compliance
        
        Guidelines to follow:
        {chr(10).join(f"- {guideline}" for guideline in self.medical_guidelines)}
        
        Provide specific, actionable feedback for improvement.
        Rate the content's medical appropriateness from 1-10.
        """
        
        response = await self._generate_critique(prompt)
        
        # Extract confidence score from response
        confidence = self._extract_confidence_score(response)
        
        return CritiqueResult(
            feedback=response,
            suggestions=self._extract_suggestions(response),
            confidence=confidence,
            critic_name="medical_content",
            metadata={
                "domain": "healthcare",
                "guidelines_checked": len(self.medical_guidelines)
            }
        )
    
    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from critic response."""
        import re
        
        # Look for rating pattern like "8/10" or "Rating: 7"
        patterns = [
            r'(\d+)/10',
            r'Rating:\s*(\d+)',
            r'Score:\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return float(match.group(1)) / 10.0
        
        return 0.7  # Default confidence

# Usage
async def medical_critic_example():
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=5,
        critics={"medical": MedicalContentCritic()}
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think(
        "Write about the benefits of regular exercise for heart health"
    )
    
    print(f"Final text: {result.final_text}")
    
    # Check medical critic feedback
    for critique in result.critiques:
        if critique.critic_name == "medical_content":
            print(f"Medical critic confidence: {critique.confidence}")
            print(f"Medical feedback: {critique.feedback}")

asyncio.run(medical_critic_example())
```

## Example 2: Style and Tone Critic

### Brand Voice Critic

```python
class BrandVoiceCritic(BaseCritic):
    """Ensures content matches specific brand voice and style guidelines."""
    
    def __init__(self, brand_guidelines: dict, model_name: str = "openai:gpt-4"):
        super().__init__(model_name=model_name)
        self.brand_guidelines = brand_guidelines
    
    async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
        guidelines_text = "\n".join([
            f"- {key}: {value}" for key, value in self.brand_guidelines.items()
        ])
        
        prompt = f"""
        Evaluate this content against our brand voice guidelines:
        
        Content:
        {text}
        
        Brand Guidelines:
        {guidelines_text}
        
        Check for:
        1. Tone consistency with brand voice
        2. Appropriate language level
        3. Brand personality alignment
        4. Messaging consistency
        5. Target audience appropriateness
        
        Provide specific suggestions to better align with brand guidelines.
        """
        
        response = await self._generate_critique(prompt)
        
        return CritiqueResult(
            feedback=response,
            suggestions=self._extract_suggestions(response),
            confidence=0.8,
            critic_name="brand_voice",
            metadata={
                "brand": self.brand_guidelines.get("brand_name", "Unknown"),
                "guidelines_count": len(self.brand_guidelines)
            }
        )

# Usage
async def brand_voice_example():
    brand_guidelines = {
        "brand_name": "TechFlow",
        "tone": "Professional yet approachable",
        "voice": "Confident and knowledgeable",
        "language_level": "Technical but accessible",
        "personality": "Innovative, reliable, forward-thinking",
        "avoid": "Jargon, overly casual language, uncertainty"
    }
    
    config = SifakaConfig(
        model="openai:gpt-4",
        critics={"brand": BrandVoiceCritic(brand_guidelines)}
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write a product announcement for our new AI platform")

asyncio.run(brand_voice_example())
```

## Example 3: Multi-Perspective Critic

### Stakeholder Perspective Critic

```python
class StakeholderCritic(BaseCritic):
    """Evaluates content from multiple stakeholder perspectives."""
    
    def __init__(self, stakeholders: list, model_name: str = "openai:gpt-4"):
        super().__init__(model_name=model_name)
        self.stakeholders = stakeholders
    
    async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
        stakeholder_analyses = []
        
        for stakeholder in self.stakeholders:
            prompt = f"""
            Analyze this content from the perspective of: {stakeholder['role']}
            
            Content:
            {text}
            
            Stakeholder Profile:
            - Role: {stakeholder['role']}
            - Concerns: {', '.join(stakeholder.get('concerns', []))}
            - Priorities: {', '.join(stakeholder.get('priorities', []))}
            
            Questions to consider:
            1. Does this content address their concerns?
            2. Is it relevant to their priorities?
            3. What questions might they have?
            4. What improvements would they suggest?
            
            Provide feedback from this stakeholder's perspective.
            """
            
            analysis = await self._generate_critique(prompt)
            stakeholder_analyses.append({
                "stakeholder": stakeholder['role'],
                "analysis": analysis
            })
        
        # Combine all perspectives
        combined_feedback = "STAKEHOLDER PERSPECTIVES:\n\n"
        suggestions = []
        
        for analysis in stakeholder_analyses:
            combined_feedback += f"{analysis['stakeholder']}:\n{analysis['analysis']}\n\n"
            suggestions.extend(self._extract_suggestions(analysis['analysis']))
        
        return CritiqueResult(
            feedback=combined_feedback,
            suggestions=list(set(suggestions)),  # Remove duplicates
            confidence=0.85,
            critic_name="stakeholder_perspective",
            metadata={
                "stakeholders_analyzed": len(self.stakeholders),
                "perspectives": [s['role'] for s in self.stakeholders]
            }
        )

# Usage
async def stakeholder_critic_example():
    stakeholders = [
        {
            "role": "Technical Team",
            "concerns": ["Implementation complexity", "Technical accuracy", "Maintainability"],
            "priorities": ["Clear specifications", "Feasibility", "Best practices"]
        },
        {
            "role": "Business Stakeholders", 
            "concerns": ["Cost", "Timeline", "ROI", "Risk"],
            "priorities": ["Business value", "Market impact", "Competitive advantage"]
        },
        {
            "role": "End Users",
            "concerns": ["Usability", "Performance", "Reliability"],
            "priorities": ["User experience", "Ease of use", "Value delivery"]
        }
    ]
    
    config = SifakaConfig(
        model="openai:gpt-4",
        critics={"stakeholder": StakeholderCritic(stakeholders)}
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think("Write a proposal for implementing AI-powered customer service")

asyncio.run(stakeholder_critic_example())
```

## Example 4: Data-Driven Critic

### SEO Optimization Critic

```python
import re
from typing import List

class SEOCritic(BaseCritic):
    """Optimizes content for search engine optimization."""
    
    def __init__(self, target_keywords: List[str], model_name: str = "openai:gpt-4"):
        super().__init__(model_name=model_name)
        self.target_keywords = target_keywords
    
    async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
        # Analyze current SEO metrics
        seo_analysis = self._analyze_seo_metrics(text)
        
        prompt = f"""
        Optimize this content for SEO while maintaining quality and readability:
        
        Content:
        {text}
        
        Target Keywords: {', '.join(self.target_keywords)}
        
        Current SEO Analysis:
        - Word count: {seo_analysis['word_count']}
        - Keyword density: {seo_analysis['keyword_density']}%
        - Keywords found: {seo_analysis['keywords_found']}
        - Missing keywords: {seo_analysis['missing_keywords']}
        
        SEO Optimization Guidelines:
        1. Include target keywords naturally (2-3% density)
        2. Use keywords in headings and subheadings
        3. Optimize for readability (short paragraphs, clear structure)
        4. Include semantic variations of keywords
        5. Ensure content length is appropriate (300+ words for blog posts)
        
        Provide specific suggestions to improve SEO while maintaining content quality.
        """
        
        response = await self._generate_critique(prompt)
        
        return CritiqueResult(
            feedback=response,
            suggestions=self._extract_suggestions(response),
            confidence=0.9,
            critic_name="seo_optimization",
            metadata={
                "target_keywords": self.target_keywords,
                "current_metrics": seo_analysis
            }
        )
    
    def _analyze_seo_metrics(self, text: str) -> dict:
        """Analyze current SEO metrics of the text."""
        words = text.lower().split()
        word_count = len(words)
        
        keywords_found = []
        missing_keywords = []
        
        for keyword in self.target_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text.lower():
                keywords_found.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Calculate keyword density
        total_keyword_occurrences = sum(
            text.lower().count(kw.lower()) for kw in keywords_found
        )
        keyword_density = (total_keyword_occurrences / word_count * 100) if word_count > 0 else 0
        
        return {
            "word_count": word_count,
            "keyword_density": round(keyword_density, 2),
            "keywords_found": keywords_found,
            "missing_keywords": missing_keywords
        }

# Usage
async def seo_critic_example():
    target_keywords = [
        "artificial intelligence",
        "machine learning",
        "AI automation",
        "business intelligence"
    ]
    
    config = SifakaConfig(
        model="openai:gpt-4",
        critics={"seo": SEOCritic(target_keywords)}
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think(
        "Write a blog post about how AI is transforming business operations"
    )

asyncio.run(seo_critic_example())
```

## Combining Multiple Custom Critics

```python
async def multi_critic_workflow():
    """Combine multiple custom critics for comprehensive improvement."""
    
    # Define brand guidelines
    brand_guidelines = {
        "brand_name": "InnovateTech",
        "tone": "Professional and innovative",
        "voice": "Authoritative yet accessible"
    }
    
    # Define stakeholders
    stakeholders = [
        {"role": "CTO", "concerns": ["Technical feasibility"], "priorities": ["Innovation"]},
        {"role": "Marketing", "concerns": ["Brand alignment"], "priorities": ["Market appeal"]}
    ]
    
    # Define SEO keywords
    seo_keywords = ["AI innovation", "technology solutions", "digital transformation"]
    
    # Create configuration with multiple custom critics
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=7,
        critics={
            "brand": BrandVoiceCritic(brand_guidelines),
            "stakeholder": StakeholderCritic(stakeholders),
            "seo": SEOCritic(seo_keywords),
            "medical": MedicalContentCritic()  # If relevant
        },
        validation_weight=0.6,
        critic_weight=0.4,
        always_apply_critics=True
    )
    
    engine = SifakaEngine(config=config)
    result = await engine.think(
        "Write a company blog post about our new AI-powered healthcare platform"
    )
    
    # Analyze results from each critic
    for critique in result.critiques:
        print(f"\n{critique.critic_name.upper()} FEEDBACK:")
        print(f"Confidence: {critique.confidence}")
        print(f"Suggestions: {len(critique.suggestions)}")
        if critique.metadata:
            print(f"Metadata: {critique.metadata}")

asyncio.run(multi_critic_workflow())
```

## Best Practices for Custom Critics

### 1. Clear Evaluation Criteria
```python
class WellDefinedCritic(BaseCritic):
    def __init__(self):
        super().__init__()
        # Define clear, measurable criteria
        self.criteria = {
            "clarity": "Is the content easy to understand?",
            "completeness": "Does it cover all necessary points?",
            "accuracy": "Is the information correct?",
            "relevance": "Is it relevant to the target audience?"
        }
```

### 2. Consistent Feedback Format
```python
async def critique_async(self, text: str, context: dict = None) -> CritiqueResult:
    # Use consistent prompt structure
    prompt = f"""
    EVALUATION CRITERIA: {self.criteria}
    
    CONTENT TO ANALYZE:
    {text}
    
    ANALYSIS:
    [Provide structured analysis]
    
    SUGGESTIONS:
    1. [Specific suggestion]
    2. [Another suggestion]
    
    CONFIDENCE: [1-10 rating]
    """
```

### 3. Metadata for Tracking
```python
return CritiqueResult(
    feedback=response,
    suggestions=suggestions,
    confidence=confidence,
    critic_name="your_critic",
    metadata={
        "version": "1.0",
        "criteria_count": len(self.criteria),
        "analysis_timestamp": datetime.now().isoformat()
    }
)
```

## What's Next?

🎉 **You've mastered custom critic creation!**

**Next Steps**:
- **[API Reference](../API_REFERENCE.md)** - Complete API documentation
- **[Examples Directory](../../examples/)** - Real-world usage examples

## Key Takeaways

✅ **Domain-specific critics** for specialized improvement logic  
✅ **Multi-perspective analysis** for comprehensive feedback  
✅ **Data-driven optimization** for measurable improvements  
✅ **Consistent feedback format** for reliable results  
✅ **Metadata tracking** for analysis and debugging  

**You can now build sophisticated improvement systems with Sifaka! 🚀**

---

**Previous**: 📚 **[Advanced Features](03-advanced-features.md)**  
**Complete Tutorial Series**: 📚 **[Quick Start](01-quick-start.md)** → **[Basic Usage](02-basic-usage.md)** → **[Advanced Features](03-advanced-features.md)** → **Custom Critics**
