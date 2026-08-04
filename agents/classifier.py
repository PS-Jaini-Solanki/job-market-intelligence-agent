"""
agents/classifier.py
=====================
Query Intent Classifier — the entry point of the agent graph.

WHAT IT DOES
------------
Before anything else runs, we decide WHAT the user is asking. Every
question maps to one of two families:

  1. ANALYTICAL intents  -> route to the Planner -> full multi-agent pipeline
  2. CONVERSATIONAL      -> route to the ChatResponder -> direct answer

WHY DETERMINISTIC FIRST
-----------------------
Routing must never depend on an LLM being available. This module uses
pure keyword / regex rules so the graph ALWAYS knows where to go, even
when Groq is down or a rate limit is hit.

RESPONSIBILITIES
----------------
- classify_query()        -> QueryIntent (deterministic)
- run_query_classifier()  -> LangGraph node function
- _extract_role_from_query()       -> job role mention (or None)
- _extract_country_from_query()    -> ISO country code (or None)
- _extract_technologies_from_query() -> tech names mentioned (or [])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from langchain_core.messages import AIMessage

from agents.state import GraphState

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Every way a user can talk to the system."""

    JOB_MARKET = "job_market"
    TECH_TREND = "tech_trend"
    COUNTRY_COMPARISON = "country_comparison"
    SALARY = "salary"
    RESUME_ANALYSIS = "resume_analysis"
    GENERAL_CHAT = "general_chat"
    OUT_OF_SCOPE = "out_of_scope"


# Intents that need the full analytical pipeline (report is produced).
ANALYTICAL_INTENTS: set[str] = {
    QueryIntent.JOB_MARKET.value,
    QueryIntent.TECH_TREND.value,
    QueryIntent.COUNTRY_COMPARISON.value,
    QueryIntent.SALARY.value,
    QueryIntent.RESUME_ANALYSIS.value,
}

# Conversational intents (direct answer, no report).
CONVERSATIONAL_INTENTS: set[str] = {
    QueryIntent.GENERAL_CHAT.value,
    QueryIntent.OUT_OF_SCOPE.value,
}


# ============================================================
# QueryIntent helper properties
# ============================================================

def is_analytical(intent: Optional[str]) -> bool:
    """True if the intent requires the multi-agent pipeline."""
    return intent in ANALYTICAL_INTENTS


def is_conversational(intent: Optional[str]) -> bool:
    """True if the intent should be answered directly."""
    return intent in CONVERSATIONAL_INTENTS


# ============================================================
# Detection patterns
# ============================================================

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|hii+|hlo|yo|namaste|hola|good (morning|afternoon|evening)"
    r"|howdy|what'?s up|sup)\b",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"\b(thanks|thank you|thankyou|thx|tyvm|appreciate(d)?|grateful)\b",
    re.IGNORECASE,
)

_CAPABILITY_RE = re.compile(
    r"\b(what can you do|what do you do|how (can|do) you help|your (features|capabilities|abilities)"
    r"|who are you|what are you|what is your purpose|how (does|do) (this|the|you) work"
    r"|help me|show me what you can do|what can i ask|capabilit(y|ies))\b",
    re.IGNORECASE,
)

_RESUME_RE = re.compile(
    r"\b(resume|resume analysis|analy[sz]e my resume|review my resume|cv|my skills|my profile"
    r"|skill gap|gap analysis|missing skills|am i ready|am i a good fit|fit for)\b",
    re.IGNORECASE,
)

_SALARY_RE = re.compile(
    r"\b(salary|salaries|pay|compensation|earn|earning|income|ctc|remuneration"
    r"|how much.*(make|earn|paid)|average.*(package|lpa|k))\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|vs\.?|versus|difference between|both countries|versus)\b",
    re.IGNORECASE,
)

_TREND_RE = re.compile(
    r"\b(trend|trending|popular|popularity|should i learn|worth learning|hot(tech| skill)?"
    r"|future|growing|rise|emerging|top technologies|what.?s the next big|demand for)\b",
    re.IGNORECASE,
)

# Concrete job roles the system understands.
_ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("AI Engineer", ["ai engineer", "ai/ml engineer", "ai ml engineer", "artificial intelligence engineer",
                     "genai engineer", "genai", "gen ai", "generative ai engineer", "llm engineer",
                     "ai job", "ai jobs", "ai job market", "ai market", "ai role", "ai roles",
                     "ai career", "ai careers", "ai position", "ai positions", "ai hiring"]),
    ("Data Scientist", ["data scientist"]),
    ("ML Engineer", ["ml engineer", "machine learning engineer", "mlops engineer"]),
    ("Data Engineer", ["data engineer"]),
    ("Data Analyst", ["data analyst", "business analyst"]),
    ("Backend Engineer", ["backend engineer", "backend developer"]),
    ("Software Engineer", ["software engineer", "software developer", "sde"]),
    ("Full Stack Engineer", ["full stack", "fullstack"]),
    ("DevOps Engineer", ["devops", "sre", "site reliability engineer", "platform engineer"]),
    ("Frontend Engineer", ["frontend engineer", "front end engineer", "frontend developer"]),
    ("Cloud Engineer", ["cloud engineer", "cloud architect", "solutions architect"]),
    ("Security Engineer", ["security engineer", "cybersecurity", "cyber security"]),
    ("QA Engineer", ["qa engineer", "test engineer", "sdets"]),
    ("Python Developer", ["python developer"]),
    ("Java Developer", ["java developer"]),
    ("Mobile Developer", ["mobile developer", "ios developer", "android developer", "flutter developer"]),
    ("Product Manager", ["product manager", "technical program manager"]),
    ("Data Architect", ["data architect", "database architect"]),
    ("Tech Lead", ["tech lead", "engineering manager", "lead engineer"]),
]

# Generic job-market language (used when no explicit role is mentioned).
_JOB_MARKET_RE = re.compile(
    r"\b(job|jobs|job market|posting|postings|vacanc|hiring|recruit|opening"
    r"|skills required|required skills|in demand|market demand|career path|interview"
    r"|tech stack|what should i learn|learning path|roadmap|break into)\b",
    re.IGNORECASE,
)

# Countries the job APIs support (aligned with Adzuna's country coverage).
_COUNTRY_MAP: list[tuple[str, list[str]]] = [
    ("IN", ["india", "indian", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "chennai"]),
    ("US", [r"\bus\b", "usa", "united states", "america", "american", "us market", "san francisco", "new york", "bay area", "silicon valley"]),
    ("GB", [r"\buk\b", "united kingdom", "britain", "london", "manchester"]),
    ("AU", ["australia", "australian", "sydney", "melbourne"]),
    ("DE", ["germany", "german", "berlin", "munich"]),
    ("CA", ["canada", "canadian", "toronto", "vancouver"]),
    ("FR", ["france", "french", "paris", "lyon"]),
    ("SG", ["singapore"]),
    ("AE", ["uae", "united arab emirates", "dubai", "abu dhabi"]),
]

# Technology vocabulary used to detect trend queries and extract tech names.
TECHNOLOGY_VOCABULARY: list[str] = [
    "LangChain", "LangGraph", "LlamaIndex", "CrewAI", "AutoGen", "OpenAI", "Anthropic",
    "MCP", "RAG", "LLM", "LLMs", "PyTorch", "TensorFlow", "Hugging Face", "Transformers",
    "FastAPI", "Flask", "Django", "React", "Next.js", "NextJS", "Vue", "Angular",
    "Docker", "Kubernetes", "Terraform", "AWS", "Azure", "GCP", "Kafka", "Spark",
    "PostgreSQL", "Redis", "MongoDB", "ChromaDB", "Qdrant", "Pinecone", "Weaviate",
    "MLflow", "Ollama", "Groq", "CUDA", "Ray", "dbt", "Airflow", "PowerBI", "Tableau",
    "Python", "Java", "Go", "Golang", "Rust", "TypeScript", "JavaScript", "SQL",
    "RAG", "GraphQL", "gRPC", "Elasticsearch", "Snowflake", "Databricks", "Spark",
]

# Normalized lowercase forms for fast matching.
_TECH_LOWER: dict[str, str] = {t.lower(): t for t in TECHNOLOGY_VOCABULARY}


# ============================================================
# Deterministic extraction helpers (no LLM required)
# ============================================================

def _extract_role_from_query(query: str) -> Optional[str]:
    """
    Extract the job role from a user query.

    Returns None when no role is mentioned (never guesses).

    Args:
        query: User's natural language query.

    Returns:
        Job role string, or None.
    """
    query_lower = query.lower()
    for role, keywords in _ROLE_PATTERNS:
        if any(kw in query_lower for kw in keywords):
            return role
    return None


def _country_matches(query_lower: str) -> dict[str, int]:
    """
    Find all countries mentioned in a query with their match position.

    Returns:
        Dict of {country_code: first-match-index}. The earliest match in the
        query wins for each country, so "France vs Germany" orders as FR, DE.
    """
    matches: dict[str, int] = {}
    for code, keywords in _COUNTRY_MAP:
        best: int | None = None
        for kw in keywords:
            if kw.startswith("\\"):
                m = re.search(kw, query_lower)
                if m and (best is None or m.start() < best):
                    best = m.start()
            else:
                idx = query_lower.find(kw)
                if idx >= 0 and (best is None or idx < best):
                    best = idx
        if best is not None:
            matches[code] = best
    return matches


def _extract_country_from_query(query: str) -> Optional[str]:
    """
    Extract the first-mentioned country code from a user query.

    Args:
        query: User's natural language query.

    Returns:
        ISO country code string, or None.
    """
    query_lower = query.lower()
    matches = _country_matches(query_lower)
    if not matches:
        return None
    # The country mentioned earliest in the query is the primary one.
    return min(matches, key=matches.get)


def _extract_technologies_from_query(query: str) -> list[str]:
    """
    Extract technology names mentioned in a user query.

    Uses the TECHNOLOGY_VOCABULARY so we never hallucinate a tech name.

    Args:
        query: User's natural language query.

    Returns:
        List of normalized technology names (deduplicated).
    """
    query_lower = query.lower()
    found: list[str] = []
    seen: set[str] = set()
    # Match longest tech names first so overlapping names resolve deterministically.
    for tech in sorted(TECHNOLOGY_VOCABULARY, key=len, reverse=True):
        key = tech.lower()
        if tech in seen:
            continue
        # Word-boundary match so "RAG" does not match "average".
        if re.search(rf"\b{re.escape(key)}\b", query_lower):
            seen.add(tech)
            found.append(tech)
    return found


def _extract_country_names(query: str) -> list[str]:
    """Return all country codes mentioned in a query, in mention order."""
    query_lower = query.lower()
    matches = _country_matches(query_lower)
    return sorted(matches, key=matches.get)


def _last_analytical_context(conversation_context: list[dict]) -> dict:
    """
    Find the most recent analytical question in the conversation history.

    Follow-up questions like "what about salaries in India?" or "am I
    missing anything?" carry no role by themselves — they refer back to
    an earlier question. We scan the history (newest first) for the last
    analytical query and inherit its intent/role/country/technologies.

    Args:
        conversation_context: List of {"role", "content"} prior turns.

    Returns:
        Dict with inherited intent, role, country, countries, technologies,
        or empty values when no analytical predecessor exists.
    """
    if not conversation_context:
        return {"intent": None, "role": None, "country": None,
                "countries": [], "technologies": []}

    for turn in reversed(conversation_context):
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "user":
            continue
        prev_query = turn.get("content", "")
        if not prev_query:
            continue
        prev = classify_query(prev_query)
        if is_analytical(prev.intent.value):
            return {
                "intent": prev.intent.value,
                "role": prev.role,
                "country": prev.country,
                "countries": prev.countries,
                "technologies": prev.technologies,
            }
    return {"intent": None, "role": None, "country": None,
            "countries": [], "technologies": []}


# ============================================================
# Classification
# ============================================================

@dataclass
class ClassificationResult:
    """Result of classifying a user query."""

    intent: QueryIntent
    confidence: float = 0.5
    role: Optional[str] = None
    country: Optional[str] = None
    technologies: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    detail: str = ""


def classify_query(
    query: str,
    conversation_context: Optional[list[dict]] = None,
) -> ClassificationResult:
    """
    Deterministically classify a user query into a QueryIntent.

    Order of checks matters (most specific -> least specific):
      1. Greeting / thanks / capability (fast templates)
      2. Resume analysis
      3. Country comparison
      4. Salary
      5. Tech trend
      6. Job market (any role mention or generic job language)
      7. Default to general chat

    When no signal is found (would default to general chat) but the
    conversation context contains a recent analytical question, the
    ambiguous follow-up ("am I missing anything?", "what about salaries?")
    inherits the previous analytical intent and its extracted facts.

    Args:
        query: The user's raw question.
        conversation_context: Optional prior turns for follow-up resolution.

    Returns:
        ClassificationResult with intent and extracted metadata.
    """
    q = query.strip().lower()

    role = _extract_role_from_query(query)
    country = _extract_country_from_query(query)
    technologies = _extract_technologies_from_query(query)
    countries = _extract_country_names(query)

    # Follow-up questions ("what about salaries in India?") omit the role that
    # was discussed earlier. Fill any missing fact from the most recent
    # analytical turn so direct matches (salary/trend/job_market) stay correct.
    inherited = _last_analytical_context(conversation_context)
    role_had_own_signal = bool(role)
    role = role or inherited["role"]
    country = country or inherited["country"]
    countries = countries or inherited["countries"]
    # Technologies are ONLY inherited when this query has no role of its own.
    # A query that names its own role (e.g. "what technology should I learn
    # for a Cloud Engineer?") is a brand-new, self-sufficient question, not a
    # continuation of an earlier tech-trend thread — inheriting a leftover
    # technology name from an unrelated previous question (e.g. "should I
    # learn Python?") caused a spurious, unrelated Technology Trends section
    # to appear in an otherwise-correct job_market report. A true ambiguous
    # follow-up ("what about MCP too?") has no role of its own, so it still
    # inherits normally.
    if not role_had_own_signal:
        technologies = technologies or inherited["technologies"]

    # --- 1. Greetings / thanks / capability questions -------------------
    if _GREETING_RE.match(q):
        return ClassificationResult(
            intent=QueryIntent.GENERAL_CHAT, confidence=0.98,
            role=role, country=country, technologies=technologies,
            countries=countries, detail="greeting",
        )
    if _THANKS_RE.search(q) and len(q) < 40:
        return ClassificationResult(
            intent=QueryIntent.GENERAL_CHAT, confidence=0.95,
            role=role, country=country, technologies=technologies,
            countries=countries, detail="thanks",
        )
    if _CAPABILITY_RE.search(q):
        return ClassificationResult(
            intent=QueryIntent.GENERAL_CHAT, confidence=0.9,
            role=role, country=country, technologies=technologies,
            countries=countries, detail="capability",
        )

    # --- 2. Resume analysis --------------------------------------------
    if _RESUME_RE.search(q):
        return ClassificationResult(
            intent=QueryIntent.RESUME_ANALYSIS, confidence=0.9,
            role=role, country=country, technologies=technologies,
            countries=countries, detail="resume request",
        )

    # --- 3. Country comparison -----------------------------------------
    if _COMPARISON_RE.search(q) and len(countries) >= 2:
        return ClassificationResult(
            intent=QueryIntent.COUNTRY_COMPARISON, confidence=0.9,
            role=role, country=country, technologies=technologies,
            countries=countries, detail=f"comparison across {len(countries)} countries",
        )

    # --- 4. Salary ------------------------------------------------------
    if _SALARY_RE.search(q) and (role or _JOB_MARKET_RE.search(q) or country):
        return ClassificationResult(
            intent=QueryIntent.SALARY, confidence=0.85,
            role=role, country=country, technologies=technologies,
            countries=countries, detail="salary question",
        )

    # --- 5. Tech trend --------------------------------------------------
    # Require an explicit technology mention, OR generic trend language with
    # no concrete role attached. Without this guard, "What skills are in
    # demand for AI Engineers in India?" was misrouted to tech_trend just
    # because "demand for" is trend vocabulary AND "in demand" is job-market
    # vocabulary — even though a specific role was named and no technology
    # was. A named role is a strong job_market signal that should win.
    if _TREND_RE.search(q) and (technologies or (not role and _JOB_MARKET_RE.search(q))):
        return ClassificationResult(
            intent=QueryIntent.TECH_TREND, confidence=0.8,
            role=role, country=country, technologies=technologies,
            countries=countries, detail="trend question",
        )

    # --- 6. Job market --------------------------------------------------
    if role or _JOB_MARKET_RE.search(q):
        intent = QueryIntent.COUNTRY_COMPARISON if (countries and _COMPARISON_RE.search(q)) else QueryIntent.JOB_MARKET
        return ClassificationResult(
            intent=intent, confidence=0.85,
            role=role, country=country, technologies=technologies,
            countries=countries, detail=f"role={role or 'generic'}",
        )

    # --- 7. Default: conversational, OR inherit from conversation context --
    # An ambiguous follow-up that would be "general chat" on its own can
    # inherit the intent of the most recent analytical question. This is how
    # "am I missing anything?" or "what about salaries?" keep the conversation
    # on track instead of bouncing to the chat responder.
    if inherited["intent"]:
        return ClassificationResult(
            intent=QueryIntent(inherited["intent"]), confidence=0.55,
            role=role or inherited["role"],
            country=country or inherited["country"],
            technologies=technologies or inherited["technologies"],
            countries=countries or inherited["countries"],
            detail="inherited analytical context from conversation history",
        )

    # --- 8. Truly conversational ----------------------------------------
    return ClassificationResult(
        intent=QueryIntent.GENERAL_CHAT, confidence=0.5,
        role=role, country=country, technologies=technologies,
        countries=countries, detail="no job-market signal",
    )


# ============================================================
# LangGraph node function
# ============================================================

async def run_query_classifier(state: GraphState) -> dict:
    """
    Classifier agent node function.

    Reads the user query and writes the intent + extracted metadata
    into state so the graph can route correctly.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with query_intent and extracted facts.
    """
    user_query = state["user_query"]
    conversation_history = state.get("conversation_history") or []
    result = classify_query(user_query, conversation_context=conversation_history)

    logger.info(
        "Query classified",
        extra={
            "intent": result.intent.value,
            "confidence": result.confidence,
            "role": result.role,
            "country": result.country,
            "technologies": result.technologies,
        },
    )

    route = "analytical pipeline" if is_analytical(result.intent.value) else "direct chat"
    return {
        "query_intent": result.intent.value,
        "intent_confidence": result.confidence,
        "role": state.get("role") or result.role,
        "country_filter": state.get("country_filter") or result.country,
        "countries": state.get("countries") or result.countries,
        "technologies_to_analyze": state.get("technologies_to_analyze") or result.technologies,
        "completed_agents": state.get("completed_agents", []) + ["query_classifier"],
        "messages": [
            AIMessage(
                content=(
                    f"🧭 **Intent detected:** `{result.intent.value}` "
                    f"(confidence {result.confidence:.0%}) — routed to **{route}**."
                )
            )
        ],
    }
