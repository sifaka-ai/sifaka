# Critic Comparison Example

This example demonstrates how to compare the performance of different critics in Sifaka, measuring metrics such as execution time, number of revisions, and output quality.

## Overview

The Critic Comparison Example:
1. Creates instances of different critic types (PromptCritic, ReflexionCritic, SelfRefineCritic, SelfRAGCritic, ConstitutionalCritic)
2. Tests each critic with different model providers (OpenAI GPT-4, Anthropic Claude-3-Opus)
3. Measures performance metrics for each critic
4. Displays comparison results

This example is useful for understanding the strengths and weaknesses of different critics and choosing the most appropriate one for your specific use case.

## Usage

To run the example:

```bash
python examples/critic_comparison_example.py
```

Make sure you have the required API keys set as environment variables:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models

## Implementation Details

The example implements the following components:

1. **Performance Metrics**:
   - Time spent (execution time)
   - Number of revisions/iterations
   - Output quality (using a simple heuristic)
   - Output length

2. **Critic Testing Functions**:
   - Each critic type has a dedicated testing function that creates the critic, runs it on the input text, and returns the improved text and number of revisions.

3. **Quality Evaluation**:
   - A simple heuristic function evaluates the quality of the output based on length, structure, and complexity.

4. **Results Display**:
   - Results are displayed in a formatted table showing metrics for each critic.
   - Average metrics by critic type are calculated and displayed.
   - The output text from each critic is shown for comparison.

## Example Output

The example produces output similar to the following:

```
Critic Comparison Results:
----------------------------------------------------------------------------------------------------
Critic               Model                Time (s)   Revisions  Quality    Length    
----------------------------------------------------------------------------------------------------
PromptCritic         OpenAI GPT-4         1.70       1          0.20       33        
ReflexionCritic      OpenAI GPT-4         4.30       1          0.24       274       
SelfRefineCritic     OpenAI GPT-4         17.60      1          0.75       2375      
SelfRAGCritic        OpenAI GPT-4         2.37       1          0.18       30        
ConstitutionalCritic OpenAI GPT-4         3.61       1          0.59       916       
PromptCritic         Anthropic Claude-3-Opus 17.59      1          0.55       1171      
ReflexionCritic      Anthropic Claude-3-Opus 26.85      1          0.51       879       
SelfRefineCritic     Anthropic Claude-3-Opus 104.32     1          0.94       3477      
SelfRAGCritic        Anthropic Claude-3-Opus 15.66      1          0.18       30        
ConstitutionalCritic Anthropic Claude-3-Opus 22.78      1          0.74       1947      
----------------------------------------------------------------------------------------------------

Average Metrics by Critic Type:
----------------------------------------------------------------------------------------------------
Critic Type          Time (s)   Quality    Length    
----------------------------------------------------------------------------------------------------
PromptCritic         9.65       0.38       602       
ReflexionCritic      15.58      0.38       576       
SelfRefineCritic     60.96      0.85       2926      
SelfRAGCritic        9.02       0.18       30        
ConstitutionalCritic 13.20      0.66       1432      
----------------------------------------------------------------------------------------------------
```

## Key Findings

Based on the example results, we can observe the following:

1. **Performance Comparison**:
   - **SelfRefineCritic** produces the highest quality outputs but takes the longest time.
   - **ConstitutionalCritic** provides a good balance between quality and execution time.
   - **SelfRAGCritic** is fast but doesn't significantly improve the input text for this particular task.
   - **PromptCritic** and **ReflexionCritic** are relatively fast but produce lower quality outputs compared to SelfRefineCritic and ConstitutionalCritic.

2. **Model Comparison**:
   - Anthropic Claude-3-Opus generally produces higher quality and longer outputs than OpenAI GPT-4.
   - Anthropic Claude-3-Opus takes significantly more time to generate outputs than OpenAI GPT-4.

3. **Use Case Recommendations**:
   - For high-quality outputs where time is not a constraint, use **SelfRefineCritic**.
   - For a balance between quality and speed, use **ConstitutionalCritic**.
   - For tasks requiring retrieval-augmented generation, ensure that **SelfRAGCritic** has relevant documents in its retriever.

## Customization

You can customize the example by:

1. Changing the input text and task
2. Adding or removing critics
3. Modifying the quality evaluation function
4. Adding more model providers
5. Adding more performance metrics

## Implementation Notes

When implementing critics, keep in mind:

1. **Metadata Handling**:
   - Some critics require task information in the metadata.
   - Pass metadata as a dictionary with the "task" key.

2. **Feedback Handling**:
   - PromptCritic and ReflexionCritic require a critique step to generate feedback before improvement.
   - Use the critique method to get feedback and then pass it to the improve method.

3. **State Management**:
   - Different critics manage state differently.
   - Some critics track revisions internally, while others don't.

4. **Error Handling**:
   - Wrap critic calls in try-except blocks to handle potential errors.
   - Provide fallback mechanisms for when critics fail.

## Related Examples

- [Prompt Critic Example](prompt_critic_example.md)
- [Reflexion Critic Example](reflexion_critic_example.md)
- [Self-Refine Critic Example](self_refine_critic_example.md)
- [Self-RAG Critic Example](self_rag_critic_example.md)
- [Constitutional Critic Example](constitutional_critic_example.md)
