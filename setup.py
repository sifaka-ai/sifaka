"""
Setup script for Sifaka.

This script defines the package dependencies and installation options for Sifaka.
Dependencies are organized into logical groups for better maintainability and
to allow users to install only what they need.
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

# Group related dependencies
model_providers = {
    "openai": ["openai>=1.76.0", "tiktoken>=0.9.0"],
    "anthropic": ["anthropic>=0.50.0"],
    "gemini": ["google-generativeai>=0.3.2"],
}

classifiers = {
    "toxicity": ["detoxify>=0.5.1", "torch>=2.2.1", "transformers>=4.38.2"],
    "sentiment": ["vaderSentiment>=3.3.2"],
    "profanity": ["better-profanity>=0.7.0"],
    "language": ["langdetect>=1.0.9"],
    "readability": ["textstat>=0.7.3"],
    "ner": ["spacy>=3.8.0"],
}

integrations = {
    "pydantic-ai": ["pydantic-ai>=0.7.1"],
    "guardrails": ["guardrails-ai>=0.6.6"],
}

development = {
    "dev": [
        "pytest>=8.0.2",
        "black>=24.2.0",
        "isort>=5.13.2",
        "mypy>=1.9.0",
        "ruff>=0.3.0",
        "pytest-cov>=4.1.0",
        "flake8>=5.0.0",
    ],
    "benchmark": [
        "memory-profiler>=0.60.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
}

# Combine all dependencies into extras_require
extras_require = {}

# Add individual dependencies
extras_require.update(model_providers)
extras_require.update(classifiers)
extras_require.update(integrations)
extras_require.update(development)

# Create logical groups
extras_require["all_models"] = [dep for deps in model_providers.values() for dep in deps]

extras_require["all_classifiers"] = [dep for deps in classifiers.values() for dep in deps]

extras_require["all_integrations"] = [dep for deps in integrations.values() for dep in deps]

# Add an 'all' extra that includes everything except development tools
extras_require["all"] = [
    dep
    for group in [model_providers, classifiers, integrations]
    for deps in group.values()
    for dep in deps
]

setup(
    name="sifaka",
    version="0.1.1",
    description="A framework for building reliable and reflective AI systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Evan Volgas",
    author_email="evan.volgas@gmail.com",
    url="https://github.com/sifaka-ai/sifaka",
    packages=find_packages(),
    install_requires=core_requirements,
    extras_require=extras_require,
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
