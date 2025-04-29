"""
Integrations for Sifaka with external frameworks.

This module provides integrations with various external frameworks:
- LangChain: Integration with the LangChain framework
- LangGraph: Integration with the LangGraph framework

Example usage:
    ```python
    # LangChain integration
    from langchain.chains import LLMChain
    from sifaka.integrations.langchain import wrap_chain
    
    # Create a LangChain chain
    chain = LLMChain(...)
    
    # Wrap it with Sifaka features
    sifaka_chain = wrap_chain(chain)
    
    # Use it like a regular LangChain chain
    result = sifaka_chain.run("Hello, world!")
    ```
    
    ```python
    # LangGraph integration
    from langgraph.graph import StateGraph
    from sifaka.integrations.langgraph import wrap_graph
    
    # Create a LangGraph graph
    graph = StateGraph()
    
    # Wrap it with Sifaka features
    sifaka_graph = wrap_graph(graph)
    
    # Use it like a regular LangGraph graph
    result = sifaka_graph.run({"input": "Hello, world!"})
    ```
"""

# Import LangChain integration
try:
    from sifaka.integrations.langchain import (
        ChainValidator,
        ChainOutputProcessor,
        ChainMemory,
        ChainConfig,
        SifakaChain,
        RuleBasedValidator,
        SifakaMemory,
        wrap_chain,
        wrap_memory,
    )
except ImportError:
    # LangChain might not be installed
    pass

# Import LangGraph integration
try:
    from sifaka.integrations.langgraph import (
        GraphValidator,
        GraphProcessor,
        GraphNode,
        GraphConfig,
        SifakaGraph,
        SifakaNode,
        wrap_graph,
        wrap_node,
    )
except ImportError:
    # LangGraph might not be installed
    pass
