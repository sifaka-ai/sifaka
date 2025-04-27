"""
Example of using Sifaka with Pydantic models for structured data validation.
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Add the parent directory to the path so we can import sifaka
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sifaka import Reflector
from sifaka.models import OpenAIProvider
from sifaka.rules import Rule, RuleResult

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a Pydantic model for structured data validation
class LegalCitation(BaseModel):
    """
    A legal citation in a document.
    
    Attributes:
        text (str): The full text of the citation
        case_name (Optional[str]): The name of the case
        reporter (Optional[str]): The reporter where the case was published
        volume (Optional[int]): The volume number
        page (Optional[int]): The page number
        year (Optional[int]): The year of the decision
    """
    text: str
    case_name: Optional[str] = None
    reporter: Optional[str] = None
    volume: Optional[int] = None
    page: Optional[int] = None
    year: Optional[int] = None


class LegalDocument(BaseModel):
    """
    A legal document with citations.
    
    Attributes:
        title (str): The title of the document
        content (str): The content of the document
        citations (List[LegalCitation]): The citations in the document
    """
    title: str
    content: str
    citations: List[LegalCitation] = Field(default_factory=list)


class PydanticCitationRule(Rule):
    """
    Rule that validates legal citations using Pydantic models.
    """
    min_citations: int = 1
    
    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains valid legal citations.
        
        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation
            
        Returns:
            RuleResult: The result of the validation
        """
        # In a real implementation, we would parse the output to extract citations
        # For this example, we'll use a simple approach
        import re
        
        # Extract citations using regex
        citation_pattern = r'\d+ [A-Za-z\.]+ \d+'
        citations_text = re.findall(citation_pattern, output)
        
        # Create LegalCitation objects
        citations = []
        for text in citations_text:
            try:
                # Parse the citation text (simplified for example)
                parts = text.split()
                if len(parts) >= 3:
                    citation = LegalCitation(
                        text=text,
                        volume=int(parts[0]),
                        reporter=parts[1],
                        page=int(parts[2])
                    )
                    citations.append(citation)
            except Exception as e:
                return RuleResult(
                    passed=False,
                    message=f"Invalid citation format: {text}",
                    metadata={"error": str(e)}
                )
        
        # Check if we have enough citations
        if len(citations) < self.min_citations:
            return RuleResult(
                passed=False,
                message=f"Not enough citations. Found {len(citations)}, expected at least {self.min_citations}",
                metadata={"citations": [c.dict() for c in citations]}
            )
        
        # Create a LegalDocument
        document = LegalDocument(
            title="Generated Legal Document",
            content=output,
            citations=citations
        )
        
        return RuleResult(
            passed=True,
            message=f"Found {len(citations)} valid citations",
            metadata={"document": document.dict()}
        )


def main():
    # Initialize the model provider
    # Replace with your API key or set the OPENAI_API_KEY environment variable
    model = OpenAIProvider(
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Create a rule that requires at least 2 citations
    citation_rule = PydanticCitationRule(min_citations=2)
    
    # Create a reflector with the rule and critique
    reflector = Reflector(
        rules=[citation_rule],
        critique=True,
        trace=True
    )
    
    # Example prompt
    prompt = """
    Write a brief summary of two landmark Supreme Court cases: 
    Brown v. Board of Education and Roe v. Wade.
    Include proper legal citations for both cases.
    """
    
    # Run the reflector
    print(f"Running reflector with prompt: {prompt}")
    result = reflector.run(model, prompt)
    
    # Print the results
    print("\nOriginal output:")
    print(result["original_output"])
    
    if result["rule_violations"]:
        print("\nRule violations:")
        for violation in result["rule_violations"]:
            print(f"- {violation['rule']}: {violation['message']}")
    else:
        print("\nNo rule violations found.")
        # Access the structured data
        document = result["rule_violations"][0]["metadata"]["document"]
        print(f"\nDocument title: {document['title']}")
        print(f"Citations found: {len(document['citations'])}")
        for citation in document["citations"]:
            print(f"- {citation['text']} ({citation.get('year', 'N/A')})")
    
    print("\nFinal output:")
    print(result["final_output"])


if __name__ == "__main__":
    main()
