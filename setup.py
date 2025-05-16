"""
Setup script for Sifaka.
"""

from setuptools import find_packages, setup

# Core dependencies (required)
core_requirements = [
    "pydantic>=2.11.3",
    "typing-extensions>=4.10.0",
    "python-dotenv>=1.0.1",
    "tqdm>=4.66.2",
    "requests>=2.31.0",
    "httpx>=0.28.1",
    "tenacity>=8.2.3",
]

# Optional dependencies
extras_require = {
    # Model providers
    "openai": ["openai>=1.76.0", "tiktoken>=0.9.0"],
    "anthropic": ["anthropic>=0.50.0"],
    # Classifiers
    "toxicity": ["detoxify>=0.5.1", "torch>=2.2.1", "transformers>=4.38.2"],
    "sentiment": ["vaderSentiment>=3.3.2"],
    "profanity": ["better-profanity>=0.7.0"],
    "language": ["langdetect>=1.0.9"],
    "readability": ["textstat>=0.7.3"],
    "ner": ["spacy>=3.8.0"],
    # Integrations
    "pydantic-ai": ["pydantic-ai>=0.7.1"],
    "guardrails": ["guardrails-ai>=0.6.6"],
    # Development
    "dev": [
        "pytest>=8.0.2",
        "black>=24.2.0",
        "isort>=5.13.2",
        "mypy>=1.9.0",
        "ruff>=0.3.0",
        "pytest-cov>=4.1.0",
        "flake8>=5.0.0",
    ],
    # Benchmarking
    "benchmark": [
        "memory-profiler>=0.60.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
}

# Add classifier group that includes all classifier dependencies
extras_require["classifiers"] = [
    dep
    for name, deps in extras_require.items()
    if name in ["toxicity", "sentiment", "profanity", "language", "readability", "ner"]
    for dep in deps
]

# Add integrations group that includes all integration dependencies
extras_require["integrations"] = [
    dep
    for name, deps in extras_require.items()
    if name in ["pydantic-ai", "guardrails"]
    for dep in deps
]

# Add an 'all' extra that includes everything except 'dev'
extras_require["all"] = [
    dep for name, deps in extras_require.items() if name != "dev" for dep in deps
]

setup(
    name="sifaka",
    version="0.1.0",
    description="Sifaka - AI Text Processing Framework",
    author="Sifaka Team",
    author_email="info@sifaka.ai",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
