"""
tools/skill_extractor_tool.py
==============================
Extracts technology skills from job descriptions and text.

HOW IT WORKS:
1. We maintain a comprehensive list of known tech skills (from normalization_service.py).
2. We scan text for mentions of these skills using keyword matching.
3. We normalize found skills using the normalization service.
4. We count how many postings mention each skill.

WHY KEYWORD MATCHING (not LLM extraction)?
- More reliable for well-defined tech terms.
- No risk of the LLM inventing skills.
- Much faster and cheaper (no API call needed).
- The LLM can be used for ambiguous cases separately.

ACCURACY NOTES:
- False positives: "Go" might match "going", "R" might match "or".
  We use word-boundary matching to minimize this.
- Unknown skills: If a skill isn't in our dictionary, we miss it.
  This is better than inventing skills (false negatives > false positives).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

from langchain_core.tools import tool

from models.job import JobSearchResult
from models.skill import Skill, SkillCategory, SkillStatistic
from services.normalization_service import (
    SKILL_ALIASES,
    get_all_canonical_skills,
    normalize_skill,
)

logger = logging.getLogger(__name__)

# Pre-compile regex patterns for all known skills
# This is computed once at module load time for efficiency
_SKILL_PATTERNS: dict[str, re.Pattern] = {}


def _get_skill_patterns() -> dict[str, re.Pattern]:
    """
    Return compiled regex patterns for all known skills.

    Patterns use word boundaries (\\b) to avoid partial matches.
    For example, "Go" won't match "going" or "Django".

    Returns:
        Dictionary of alias → compiled regex pattern.
    """
    # Single-letter/short-word aliases are too ambiguous for plain
    # case-insensitive word-boundary matching — "r" matches "R&D", and
    # "go" matches ordinary English ("go to", "go through", "going
    # forward" is blocked by \b but bare "go" as its own word is still
    # extremely common outside of the Go language). These get dedicated,
    # case-SENSITIVE patterns with negative lookaheads below instead of
    # the generic loop.
    _AMBIGUOUS_ALIASES = {"r", "go"}

    global _SKILL_PATTERNS
    if not _SKILL_PATTERNS:
        for alias in SKILL_ALIASES:
            if alias in _AMBIGUOUS_ALIASES:
                continue
            # Escape the alias for regex safety, then add word boundaries
            escaped = re.escape(alias)
            try:
                # Word-boundary matching, case-insensitive
                _SKILL_PATTERNS[alias] = re.compile(
                    rf"\b{escaped}\b", re.IGNORECASE
                )
            except re.error:
                # Skip patterns that fail to compile (e.g., "c++" has special chars)
                pass

        # Handle special cases with custom patterns
        # "C++" needs special handling because + is a regex special char
        _SKILL_PATTERNS["c++"] = re.compile(r"\bC\+\+\b", re.IGNORECASE)
        _SKILL_PATTERNS["c#"] = re.compile(r"\bC#\b", re.IGNORECASE)
        _SKILL_PATTERNS[".net"] = re.compile(r"\.NET\b", re.IGNORECASE)
        # "R" the language: require the capitalized standalone token and
        # exclude "R&D" / "R & D", by far the most common false positive.
        _SKILL_PATTERNS["r"] = re.compile(r"\bR\b(?!\s*&\s*D\b)")
        # "Go" the language: require the capitalized standalone token and
        # exclude it when followed by common English phrasal-verb
        # continuations ("go to the careers page", "go live", etc.).
        _SKILL_PATTERNS["go"] = re.compile(
            r"\bGo\b(?!\s+(?:to|through|live|beyond|ahead|forward|further|back|public)\b)"
        )

    return _SKILL_PATTERNS


def extract_skills_from_text(text: str) -> list[str]:
    """
    Extract technology skills from a text string.

    Scans the text for known skill keywords and returns
    their normalized canonical names.

    Args:
        text: Any text to scan (job description, resume, etc.).

    Returns:
        List of normalized skill names found in the text.
        May contain duplicates if a skill is mentioned multiple times.
    """
    if not text:
        return []

    patterns = _get_skill_patterns()
    found_normalized: list[str] = []

    for alias, pattern in patterns.items():
        if pattern.search(text):
            canonical = normalize_skill(alias)
            found_normalized.append(canonical)

    # Deduplicate while preserving the most common form
    return list(dict.fromkeys(found_normalized))


def compute_skill_statistics(
    job_result: JobSearchResult,
) -> list[SkillStatistic]:
    """
    Compute skill frequency statistics from a job search result.

    For each skill, we count how many postings mention it.
    This gives us honest "X out of Y postings" statistics.

    Args:
        job_result: JobSearchResult containing all collected postings.

    Returns:
        List of SkillStatistic objects sorted by occurrence count (descending).
    """
    total_postings = len(job_result.postings)
    if total_postings == 0:
        logger.warning("No postings to analyze for skill statistics")
        return []

    # Count skill occurrences across all postings
    skill_counter: Counter[str] = Counter()

    # Only skills present in our curated vocabulary are counted.
    # Raw tags from job sources (e.g. arbeitnow German tags like "Produktion")
    # are NOT in the vocabulary and must not pollute the statistics.
    known_skills: set[str] = set(get_all_canonical_skills())

    for posting in job_result.postings:
        # Extract skills from description if not already extracted
        description_text = posting.description or ""
        title_text = posting.title or ""
        full_text = f"{title_text} {description_text}"

        # Combine explicitly listed skills with extracted ones
        explicit_skills = [
            s
            for s in (normalize_skill(x) for x in posting.skills)
            if s in known_skills
        ]
        extracted_skills = extract_skills_from_text(full_text)

        all_skills = list(dict.fromkeys(explicit_skills + extracted_skills))
        for skill_name in all_skills:
            skill_counter[skill_name] += 1

    # Build SkillStatistic objects
    statistics: list[SkillStatistic] = []
    for canonical_name, count in skill_counter.most_common():
        skill = Skill(
            raw_name=canonical_name,
            normalized_name=canonical_name,
            category=_infer_skill_category(canonical_name),
        )
        stat = SkillStatistic(
            skill=skill,
            occurrence_count=count,
            total_postings_analyzed=total_postings,
            sources=job_result.sources_queried,
        )
        statistics.append(stat)

    logger.info(
        "Skill statistics computed",
        extra={
            "unique_skills": len(statistics),
            "total_postings": total_postings,
        },
    )
    return statistics


def _infer_skill_category(skill_name: str) -> SkillCategory:
    """
    Infer the category of a skill based on its name.

    This is a heuristic — not perfectly accurate, but good enough
    for display and filtering purposes.

    Args:
        skill_name: Normalized skill name.

    Returns:
        SkillCategory enum value.
    """
    name_lower = skill_name.lower()

    cloud_keywords = ["aws", "gcp", "azure", "cloud"]
    if any(kw in name_lower for kw in cloud_keywords):
        return SkillCategory.CLOUD

    ai_keywords = ["llm", "llms", "rag", "mcp", "langchain", "langgraph", "openai",
                   "hugging", "pytorch", "tensorflow", "generative", "agentic",
                   "embeddings", "fine-tuning", "prompt", "vector", "pinecone",
                   "weaviate", "chromadb", "qdrant", "anthropic", "groq"]
    if any(kw in name_lower for kw in ai_keywords):
        return SkillCategory.AI_ML

    db_keywords = ["postgresql", "mysql", "mongodb", "redis", "elasticsearch",
                   "cassandra", "sqlite"]
    if any(kw in name_lower for kw in db_keywords):
        return SkillCategory.DATABASE

    devops_keywords = ["docker", "kubernetes", "terraform", "ansible", "jenkins",
                       "github actions", "ci/cd"]
    if any(kw in name_lower for kw in devops_keywords):
        return SkillCategory.DEVOPS

    lang_keywords = ["python", "javascript", "typescript", "java", "go", "rust",
                     "c++", "c#", "scala", "r"]
    if any(kw in name_lower for kw in lang_keywords):
        return SkillCategory.PROGRAMMING_LANGUAGE

    framework_keywords = ["fastapi", "django", "flask", "react", "node", "express",
                          "next.js", "vue", "angular", "spring"]
    if any(kw in name_lower for kw in framework_keywords):
        return SkillCategory.FRAMEWORK

    data_keywords = ["spark", "kafka", "airflow", "dbt", "pandas", "numpy",
                     "data engineering", "etl", "pipeline"]
    if any(kw in name_lower for kw in data_keywords):
        return SkillCategory.DATA_ENGINEERING

    return SkillCategory.OTHER


@tool
def extract_skills(job_search_result: dict) -> dict:
    """
    Extract and count skills from job search results.

    This LangChain tool takes a serialized JobSearchResult and returns
    skill frequency statistics. Every statistic is traceable to the
    actual job postings it came from.

    Args:
        job_search_result: Serialized JobSearchResult dictionary
                           (from the search_jobs tool output).

    Returns:
        Dictionary with skill statistics sorted by frequency.
    """
    # Deserialize the input
    result = JobSearchResult.model_validate(job_search_result)

    # First, enrich postings with extracted skills from descriptions
    for posting in result.postings:
        if posting.description and not posting.skills:
            posting.skills = extract_skills_from_text(posting.description)

    # Compute statistics
    statistics = compute_skill_statistics(result)

    # Serialize output
    return {
        "total_postings_analyzed": len(result.postings),
        "unique_skills_found": len(statistics),
        "sources": result.sources_queried,
        "statistics": [
            {
                "skill_name": stat.skill.normalized_name,
                "category": stat.skill.category.value,
                "occurrence_count": stat.occurrence_count,
                "total_postings": stat.total_postings_analyzed,
                "percentage": stat.percentage,
                "evidence": stat.as_evidence_string(),
            }
            for stat in statistics[:50]  # Top 50 skills
        ],
    }
