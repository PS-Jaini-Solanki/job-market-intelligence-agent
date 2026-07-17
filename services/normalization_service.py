"""
services/normalization_service.py
==================================
Skill name normalization.

The problem this solves:
- Job postings use different names for the same technology.
- "Amazon Web Services", "AWS", "amazon aws" all mean the same thing.
- If we don't normalize, we'd undercount skill demand.

How it works:
1. A dictionary maps all known aliases → canonical name.
2. We also apply simple cleanup (lowercase, strip whitespace).
3. Unknown skills pass through unchanged.

Adding new normalizations:
- Simply add an entry to SKILL_ALIASES below.
- The key is the alias (lowercase), the value is the canonical name.
"""

from __future__ import annotations

import re

# ============================================================
# Skill alias dictionary
# ============================================================
# Format: "alias (lowercase)" → "Canonical Name"
#
# Rules for canonical names:
# - Use the most common industry abbreviation where it exists (e.g. AWS not Amazon)
# - Use title case for multi-word names (e.g. "Large Language Models")
# - Keep acronyms fully uppercase (e.g. LLM, RAG, MCP)
# ============================================================

SKILL_ALIASES: dict[str, str] = {
    # Cloud providers
    "amazon web services": "AWS",
    "amazon aws": "AWS",
    "aws": "AWS",
    "google cloud platform": "GCP",
    "google cloud": "GCP",
    "gcp": "GCP",
    "microsoft azure": "Azure",
    "azure": "Azure",

    # AI / ML frameworks
    "langchain": "LangChain",
    "lang chain": "LangChain",
    "langgraph": "LangGraph",
    "lang graph": "LangGraph",
    "llamaindex": "LlamaIndex",
    "llama index": "LlamaIndex",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "transformers": "Hugging Face Transformers",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "openai": "OpenAI API",
    "openai api": "OpenAI API",
    "anthropic": "Anthropic Claude",
    "claude": "Anthropic Claude",
    "groq": "Groq",

    # AI Concepts
    "large language models": "LLMs",
    "large language model": "LLMs",
    "llms": "LLMs",
    "llm": "LLMs",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "rag": "RAG",
    "model context protocol": "MCP",
    "mcp": "MCP",
    "generative ai": "Generative AI",
    "gen ai": "Generative AI",
    "genai": "Generative AI",
    "agentic ai": "Agentic AI",
    "ai agents": "Agentic AI",
    "prompt engineering": "Prompt Engineering",
    "fine tuning": "Fine-Tuning",
    "fine-tuning": "Fine-Tuning",
    "rlhf": "RLHF",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    "vector db": "Vector Databases",
    "embeddings": "Vector Embeddings",
    "vector embeddings": "Vector Embeddings",

    # Vector Databases
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "qdrant": "Qdrant",
    "chroma": "ChromaDB",
    "chromadb": "ChromaDB",
    "milvus": "Milvus",
    "faiss": "FAISS",

    # Programming languages
    "python": "Python",
    "python3": "Python",
    "py": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "golang": "Go",
    "go lang": "Go",
    "rust": "Rust",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "scala": "Scala",
    "r": "R",

    # Web frameworks
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "express.js": "Express.js",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "vue": "Vue.js",
    "vue.js": "Vue.js",
    "angular": "Angular",

    # Databases
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch",
    "elastic search": "Elasticsearch",
    "cassandra": "Apache Cassandra",

    # DevOps / Infrastructure
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "jenkins": "Jenkins",
    "github actions": "GitHub Actions",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",

    # Data engineering
    "apache spark": "Apache Spark",
    "spark": "Apache Spark",
    "apache kafka": "Apache Kafka",
    "kafka": "Apache Kafka",
    "airflow": "Apache Airflow",
    "apache airflow": "Apache Airflow",
    "dbt": "dbt",
    "data build tool": "dbt",

    # MLOps
    "mlflow": "MLflow",
    "ml flow": "MLflow",
    "kubeflow": "Kubeflow",
    "bentoml": "BentoML",
    "ray": "Ray",
    "ray serve": "Ray Serve",
    "wandb": "Weights & Biases",
    "weights and biases": "Weights & Biases",
    "weights & biases": "Weights & Biases",

    # Other tools
    "git": "Git",
    "linux": "Linux",
    "bash": "Bash",
    "shell scripting": "Bash",
    "rest api": "REST APIs",
    "restful api": "REST APIs",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "microservices": "Microservices",
}


def normalize_skill(raw_name: str) -> str:
    """
    Normalize a skill name to its canonical form.

    Steps:
    1. Clean up whitespace and special characters.
    2. Look up in the alias dictionary.
    3. Return canonical form if found, otherwise return cleaned original.

    Args:
        raw_name: The raw skill name from a job posting or resume.

    Returns:
        Canonical skill name string.

    Examples:
        >>> normalize_skill("Amazon Web Services")
        "AWS"
        >>> normalize_skill("  python3  ")
        "Python"
        >>> normalize_skill("some-unknown-tech")
        "some-unknown-tech"
    """
    if not raw_name:
        return raw_name

    # Step 1: Clean up the input
    cleaned = _clean_skill_name(raw_name)

    # Step 2: Look up in alias dict (case-insensitive)
    lookup_key = cleaned.lower()
    if lookup_key in SKILL_ALIASES:
        return SKILL_ALIASES[lookup_key]

    # Step 3: Return the cleaned (but not normalized) name
    return cleaned


def normalize_skills(raw_names: list[str]) -> list[str]:
    """
    Normalize a list of skill names and deduplicate.

    Args:
        raw_names: List of raw skill name strings.

    Returns:
        Deduplicated list of canonical skill names.
    """
    normalized = [normalize_skill(name) for name in raw_names if name.strip()]
    # Deduplicate while preserving order (use dict to maintain insertion order)
    return list(dict.fromkeys(normalized))


def _clean_skill_name(raw: str) -> str:
    """
    Apply basic text cleanup to a skill name.

    Args:
        raw: Raw skill name string.

    Returns:
        Cleaned string.
    """
    # Strip leading/trailing whitespace
    cleaned = raw.strip()
    # Collapse multiple spaces into one
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Remove trailing punctuation that's not part of the name
    cleaned = cleaned.rstrip(".,;:")
    return cleaned


def get_all_canonical_skills() -> list[str]:
    """
    Return a deduplicated list of all canonical skill names.

    Useful for building skill detection patterns.

    Returns:
        Sorted list of unique canonical skill names.
    """
    return sorted(set(SKILL_ALIASES.values()))
