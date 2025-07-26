"""
Setup script for Multilingual RAG System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="multilingual-rag-system",
    version="1.0.0",
    author="RAG System Developer",
    author_email="developer@example.com",
    description="A comprehensive Retrieval-Augmented Generation system for Bengali and English queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emonhossennn/multilingual-rag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "api": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "python-multipart>=0.0.6",
        ],
        "evaluation": [
            "matplotlib>=3.7.2",
            "seaborn>=0.12.2",
            "jupyter>=1.0.0",
        ],
        "cloud": [
            "pinecone-client>=2.2.4",
            "openai>=1.3.5",
        ]
    },
    entry_points={
        "console_scripts": [
            "multilingual-rag=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    keywords=[
        "rag", "retrieval-augmented-generation", "multilingual", "bengali", 
        "nlp", "ai", "education", "question-answering", "semantic-search"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/multilingual-rag-system/issues",
        "Source": "https://github.com/your-username/multilingual-rag-system",
        "Documentation": "https://github.com/your-username/multilingual-rag-system/wiki",
    },
)