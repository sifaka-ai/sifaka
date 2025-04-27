from setuptools import setup, find_packages

setup(
    name="sifaka",
    version="0.1.0",
    description="A framework for adding reflection and reliability to LLM applications",
    author="Evan Volgas",
    author_email="evan.volgas@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "openai": ["openai>=0.27.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "all": ["openai>=0.27.0", "anthropic>=0.5.0"],
        "dev": [
            "black>=23.1.0",
            "isort>=5.12.0",
            "mypy>=1.0.1",
            "pytest>=7.2.2",
            "pytest-cov>=4.1.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
