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

CONVERSATION CONTEXT
--------------------
A message may borrow facts (role, country, technologies) from earlier
turns only when it is *elliptical*: once the entities we recognize
(roles, countries, technologies) and domain words (salary, trend, skills,
jobs) are removed, nothing of its own is left. "What about Germany?" and
"what about salaries?" are elliptical; "top 5 restaurants in Ahmedabad"
and "what's the weather in Bangalore?" are not, so they never inherit an
earlier role and never trigger a market report.

RESPONSIBILITIES
----------------
- classify_query()        -> ClassificationResult (deterministic)
- run_query_classifier()  -> LangGraph node function
- _extract_role_from_query()       -> job role mention (or None)
- _extract_country_from_query()    -> ISO country code (or None)
- _extract_technologies_from_query() -> tech names mentioned (or [])
- _extract_unsupported_countries() -> countries our job sources can't serve
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from typing import Optional

from langchain_core.messages import AIMessage

from agents.state import GraphState
from tools.job_search_tool import SUPPORTED_COUNTRIES

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
    r"^\s*(hi+|hello|hey|hlo|yo|namaste|hola|howdy|good (morning|afternoon|evening)"
    r"|what'?s up|sup)\b[\s,!.:;-]*",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"\b(thanks|thank you|thankyou|thx|tyvm|appreciate(d)?|grateful)\b[\s,!.:;-]*",
    re.IGNORECASE,
)

_CAPABILITY_RE = re.compile(
    r"\b(what can you do|what do you do|how (can|do) you help|your (features|capabilities|abilities)"
    r"|who are you|what are you|what is your purpose|how (does|do) (this|the|you) work"
    r"|show me what you can do|what can i ask|capabilit(y|ies))\b",
    re.IGNORECASE,
)

# "help" / "help me" ONLY counts as a capability question when it's the
# entire message — "help me compare salaries" is an ordinary request.
_BARE_HELP_RE = re.compile(r"^\s*help( me)?[\s!.?]*$", re.IGNORECASE)

# "cv" is a resume only when it isn't "CV engineer" (computer vision).
_RESUME_RE = re.compile(
    r"\b(resume|résumé|resume analysis|analy[sz]e my resume|review my resume"
    r"|cv(?!\s+(engineer|developer|scientist|engineering))|my skills|my profile"
    r"|skill gap|gap analysis|missing skills|am i ready|am i a good fit|(am i|i am|i'm) (a )?(good )?fit)\b",
    re.IGNORECASE,
)

# Bare "pay" is not here: "pay off my loan" is not a salary question.
_SALARY_RE = re.compile(
    r"\b(salary|salaries|paid|pays|pay\s+(scale|range|band|package|rate)s?|compensation"
    r"|earn|earns|earning|earnings|income|ctc|lpa|remuneration"
    r"|how much.*(make|earn|paid|pay)|average.*(package|lpa|k))\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|vs\.?|versus|difference between|both countries)\b",
    re.IGNORECASE,
)

_TREND_RE = re.compile(
    r"\b(trend|trends|trending|popular|popularity|should i learn|worth learning|hot(tech| skill)?"
    r"|future|growing|rise|emerging|top technologies|what.?s the next big|demand for)\b",
    re.IGNORECASE,
)

# Generic job-market language (used when no explicit role is mentioned).
_JOB_MARKET_RE = re.compile(
    r"\b(job|jobs|job market|posting|postings|vacanc\w*|hiring|recruit\w*|openings?"
    r"|skills required|required skills|in demand|market demand|career path|interview"
    r"|tech stack|what should i learn|learning path|roadmap|break into)\b",
    re.IGNORECASE,
)

# Elliptical follow-up openers: "what about…", "and for…", "tell me more".
_FOLLOW_UP_CUE_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|so|and|also|then|now|but)?[\s,]*(?:what|how)\s+about\b"
    r"|^\s*(?:and|also|but)\b"
    r"|\b(?:same|similar)\s+(?:for|in|but|thing)\b"
    r"|\b(?:what|anything)\s+else\b"
    r"|\b(?:am i|what am i|anything i'?m)\s+missing\b"
    r"|\btell me more\b|\bmore (?:details?|info|on (?:it|this|that))\b"
    r"|\b(?:elaborate|go on|continue|break (?:it|that|this) down)\b|\binstead\b",
    re.IGNORECASE,
)

# Words that carry no topic of their own. A message whose remaining words
# are all in here (after entities and domain words are removed) is
# elliptical and may inherit context.
_FILLER_WORDS: frozenset[str] = frozenset(
    "the a an too also then it its this that them those these there here anything "
    "something else more please me us i im i'm my we our you your now so and or but "
    "for in of about on at to with what what's whats how is are was be being ok okay "
    "same thing things one ones instead again should learn know need needs focus improve "
    "add prioritize study do does can could would will get have has missing am tell "
    "details detail info just only like as vs versus country countries market markets "
    "role roles job jobs skill skills position positions case any which where who "
    "good better best top most required needed".split()
)

# Concrete job roles the system knows well (with their common spellings).
_ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("AI Engineer", ["ai engineer", "ai/ml engineer", "ai ml engineer", "artificial intelligence engineer",
                     "genai engineer", "genai", "gen ai", "generative ai engineer", "llm engineer",
                     "ai job", "ai job market", "ai market", "ai role",
                     "ai career", "ai position", "ai hiring"]),
    ("Data Scientist", ["data scientist"]),
    ("ML Engineer", ["ml engineer", "machine learning engineer", "mlops engineer"]),
    ("Data Engineer", ["data engineer"]),
    ("Data Analyst", ["data analyst", "business analyst"]),
    ("Backend Engineer", ["backend engineer", "backend developer", "back end developer", "back end engineer"]),
    ("Software Engineer", ["software engineer", "software developer", "sde"]),
    ("Full Stack Engineer", ["full stack", "fullstack", "full-stack"]),
    ("DevOps Engineer", ["devops", "sre", "site reliability engineer", "platform engineer"]),
    ("Frontend Engineer", ["frontend engineer", "front end engineer", "frontend developer", "front end developer"]),
    ("Cloud Engineer", ["cloud engineer", "cloud architect", "solutions architect"]),
    ("Security Engineer", ["security engineer", "cybersecurity", "cyber security"]),
    ("QA Engineer", ["qa engineer", "test engineer", "sdet"]),
    ("Python Developer", ["python developer"]),
    ("Java Developer", ["java developer"]),
    ("Mobile Developer", ["mobile developer", "ios developer", "android developer", "flutter developer"]),
    ("Product Manager", ["product manager", "technical program manager"]),
    ("Data Architect", ["data architect", "database architect"]),
    ("Tech Lead", ["tech lead", "engineering manager", "lead engineer"]),
]
# Whole-word match with optional plural, so "sde" never matches inside a word.
_ROLE_REGEXES: list[tuple[str, re.Pattern]] = [
    (role, re.compile(r"\b(?:" + "|".join(map(re.escape, kws)) + r")s?\b", re.IGNORECASE))
    for role, kws in _ROLE_PATTERNS
]

# Any other "<something> developer/engineer/…" is taken as the role the
# user named, so "blockchain developers" searches for Blockchain Developer
# instead of falling back to a generic default.
_ROLE_SUFFIXES = r"engineer|developer|architect|scientist|analyst|programmer|administrator|tester|designer"
_GENERIC_ROLE_RE = re.compile(
    rf"\b((?:[A-Za-z][\w+#./-]*\s+){{1,3}})({_ROLE_SUFFIXES})s?\b", re.IGNORECASE
)
_ROLE_QUALIFIER_STOP: frozenset[str] = _FILLER_WORDS | frozenset(
    "become becoming hire hiring want wants work working as from like new great "
    "does did can could should would will is are skilled successful".split()
)

# Countries the job APIs support (aligned with Adzuna's country coverage),
# matched as whole words so "comparison" never matches "paris".
_COUNTRY_MAP: list[tuple[str, list[str]]] = [
    ("IN", ["india", "indian", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune",
            "chennai", "ahmedabad", "kolkata", "noida", "gurgaon", "gurugram"]),
    ("US", ["usa", "u.s.a", "united states", "america", "american", "san francisco", "new york",
            "bay area", "silicon valley", "seattle", "austin", "boston", "chicago", "california",
            "texas", "new jersey"]),
    ("GB", ["uk", "u.k", "united kingdom", "britain", "great britain", "england", "london",
            "manchester", "edinburgh"]),
    ("AU", ["australia", "australian", "sydney", "melbourne"]),
    ("DE", ["germany", "german", "berlin", "munich", "hamburg", "frankfurt"]),
    ("CA", ["canada", "canadian", "toronto", "vancouver", "montreal"]),
    ("FR", ["france", "french", "paris", "lyon"]),
    ("SG", ["singapore"]),
    ("AE", ["uae", "united arab emirates", "dubai", "abu dhabi"]),
]
_COUNTRY_REGEXES: list[tuple[str, re.Pattern]] = [
    (code, re.compile(r"\b(?:" + "|".join(map(re.escape, kws)) + r")\b", re.IGNORECASE))
    for code, kws in _COUNTRY_MAP
]
# "US" is also the pronoun "us" ("tell us"), so lowercase "us" counts only
# after a word that introduces a place; the uppercase form always counts.
_US_UPPER_RE = re.compile(r"\bU\.?S\.?(?![A-Za-z])")
_US_LOWER_RE = re.compile(r"\b(?:in|the|to|from|vs\.?|versus|and|across|within|for)\s+(us)\b")

# Common names pycountry doesn't carry.
_COUNTRY_NAME_ALIASES = {"russia": "Russia", "turkey": "Turkey", "czech republic": "Czechia"}
# A country name counts only right after a word that introduces a place, so
# "Jersey" in "New Jersey" or "Jordan" as a first name is not a country.
_PLACE_PREFIX = r"(?:in|to|from|for|across|vs\.?|versus|and|about|of|within|at|with|or)\s+(?:the\s+)?"

# Technology vocabulary used to detect trend queries and extract tech names.
TECHNOLOGY_VOCABULARY: list[str] = [
    "LangChain", "LangGraph", "LlamaIndex", "CrewAI", "AutoGen", "OpenAI", "Anthropic",
    "MCP", "RAG", "LLM", "LLMs", "PyTorch", "TensorFlow", "Hugging Face", "Transformers",
    "FastAPI", "Flask", "Django", "React", "Next.js", "NextJS", "Vue", "Angular",
    "Docker", "Kubernetes", "Terraform", "AWS", "Azure", "GCP", "Kafka", "Spark",
    "PostgreSQL", "Redis", "MongoDB", "ChromaDB", "Qdrant", "Pinecone", "Weaviate",
    "MLflow", "Ollama", "Groq", "CUDA", "Ray", "dbt", "Airflow", "PowerBI", "Tableau",
    "Python", "Java", "Go", "Golang", "Rust", "TypeScript", "JavaScript", "SQL",
    "GraphQL", "gRPC", "Elasticsearch", "Snowflake", "Databricks",
]
# Names that are also everyday English words only count when written as a
# name ("Go", not "go for it").
_CASE_SENSITIVE_TECH: frozenset[str] = frozenset({"Go", "Ray"})
_TECH_REGEXES: list[tuple[str, re.Pattern]] = [
    (
        tech,
        re.compile(
            rf"\b{re.escape(tech)}\b",
            0 if tech in _CASE_SENSITIVE_TECH else re.IGNORECASE,
        ),
    )
    # Longest names first so overlapping names resolve deterministically.
    for tech in sorted(TECHNOLOGY_VOCABULARY, key=len, reverse=True)
]


# ============================================================
# Deterministic extraction helpers (no LLM required)
# ============================================================

def _extract_role_from_query(query: str) -> Optional[str]:
    """
    Extract the job role from a user query.

    Known roles (with their common spellings) win; otherwise any
    "<qualifier> developer/engineer/…" phrase is used as the role, in the
    user's own words. Returns None when no role is mentioned (never guesses).

    Args:
        query: User's natural language query.

    Returns:
        Job role string, or None.
    """
    for role, pattern in _ROLE_REGEXES:
        if pattern.search(query):
            return role

    for match in _GENERIC_ROLE_RE.finditer(query):
        # Keep the words directly before the suffix, stopping at the first
        # filler ("skills for a react developer" -> "react").
        qualifier: list[str] = []
        for word in reversed(match.group(1).split()):
            if word.lower() in _ROLE_QUALIFIER_STOP:
                break
            qualifier.insert(0, word)
        if qualifier:
            words = [*qualifier, match.group(2)]
            return " ".join(w[0].upper() + w[1:] if w.islower() else w for w in words)
    return None


def _country_matches(query: str) -> dict[str, int]:
    """
    Find all supported countries mentioned in a query with their position.

    Returns:
        Dict of {country_code: first-match-index}. The earliest match in the
        query wins for each country, so "France vs Germany" orders as FR, DE.
    """
    matches: dict[str, int] = {}
    for code, pattern in _COUNTRY_REGEXES:
        m = pattern.search(query)
        if m:
            matches[code] = m.start()

    us_positions = [m.start() for m in _US_UPPER_RE.finditer(query)]
    us_positions += [m.start(1) for m in _US_LOWER_RE.finditer(query.lower())]
    if us_positions:
        matches["US"] = min([*us_positions, matches.get("US", len(query))])
    return matches


def _extract_country_from_query(query: str) -> Optional[str]:
    """
    Extract the first-mentioned supported country code from a user query.

    Args:
        query: User's natural language query.

    Returns:
        ISO country code string, or None.
    """
    matches = _country_matches(query)
    if not matches:
        return None
    # The country mentioned earliest in the query is the primary one.
    return min(matches, key=matches.get)


def _extract_country_names(query: str) -> list[str]:
    """Return all supported country codes mentioned in a query, in mention order."""
    matches = _country_matches(query)
    return sorted(matches, key=matches.get)


@lru_cache(maxsize=1)
def _unsupported_country_regex() -> tuple[re.Pattern, dict[str, str]]:
    """Regex matching every ISO country our job sources can't serve."""
    import pycountry

    names: dict[str, str] = dict(_COUNTRY_NAME_ALIASES)
    for country in pycountry.countries:
        if country.alpha_2 in SUPPORTED_COUNTRIES:
            continue
        display = getattr(country, "common_name", None) or country.name
        for attr in ("name", "common_name", "official_name"):
            value = getattr(country, attr, None)
            if value:
                names[value.lower()] = display
    alternatives = "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
    return re.compile(rf"\b{_PLACE_PREFIX}({alternatives})\b", re.IGNORECASE), names


def _extract_unsupported_countries(query: str) -> list[str]:
    """
    Countries the user named that the job sources don't cover (e.g. Japan).

    Detected so the system says so honestly instead of silently answering
    with another country's data.
    """
    pattern, names = _unsupported_country_regex()
    found: list[str] = []
    for m in pattern.finditer(query):
        display = names[m.group(1).lower()]
        if display not in found:
            found.append(display)
    return found


def _extract_technologies_from_query(query: str) -> list[str]:
    """
    Extract technology names mentioned in a user query.

    Uses the TECHNOLOGY_VOCABULARY so we never hallucinate a tech name.

    Args:
        query: User's natural language query.

    Returns:
        List of normalized technology names (deduplicated).
    """
    return [tech for tech, pattern in _TECH_REGEXES if pattern.search(query)]


def _is_elliptical(query: str) -> bool:
    """
    True if nothing of the query's own is left once recognized entities,
    domain words and follow-up openers are removed ("what about Germany?").
    """
    text = query.lower()
    for _, pattern in (*_ROLE_REGEXES, *_COUNTRY_REGEXES, *_TECH_REGEXES):
        text = pattern.sub(" ", text)
    for pattern in (_GENERIC_ROLE_RE, _US_LOWER_RE, _unsupported_country_regex()[0],
                    _FOLLOW_UP_CUE_RE, _SALARY_RE, _TREND_RE, _JOB_MARKET_RE, _COMPARISON_RE):
        text = pattern.sub(" ", text)
    words = re.findall(r"[a-z][a-z']*", text)
    return all(w in _FILLER_WORDS for w in words)


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
    unsupported_countries: list[str] = field(default_factory=list)
    detail: str = ""


_NO_CONTEXT: dict = {"intent": None, "role": None, "country": None,
                     "countries": [], "technologies": []}


def _resolve_context(conversation_context: Optional[list[dict]]) -> dict:
    """
    Replay the conversation's user turns in order and return the facts of
    the latest analytical one, so a chain of follow-ups ("AI jobs in
    India" -> "what about salaries?" -> "and Germany?") keeps its thread.

    Args:
        conversation_context: List of {"role", "content"} prior turns.

    Returns:
        Dict with inherited intent, role, country, countries, technologies.
    """
    context = _NO_CONTEXT
    for turn in conversation_context or []:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        content = turn.get("content") or ""
        if not content.strip():
            continue
        result = _classify(content, context)
        if is_analytical(result.intent.value):
            context = {
                "intent": result.intent.value,
                "role": result.role,
                "country": result.country,
                "countries": result.countries,
                "technologies": result.technologies,
            }
    return context


def _without_pleasantry(query: str) -> Optional[str]:
    """
    If the query contains a greeting or thanks, return the rest of it
    (possibly ""); None if there is no pleasantry.
    """
    greeting = _GREETING_RE.match(query)
    if greeting:
        return query[greeting.end():].strip()
    thanks = _THANKS_RE.search(query)
    if thanks:
        return (query[:thanks.start()] + " " + query[thanks.end():]).strip(" ,!.")
    return None


def _classify(query: str, inherited: dict) -> ClassificationResult:
    """Classify one message given the facts inherited from earlier turns."""
    q = query.strip()

    # --- 1. Greetings / thanks: answer the rest of the message if it asks
    # something analytical ("Hi, top skills for Python devs in Germany?").
    rest = _without_pleasantry(q)
    if rest is not None:
        if rest:
            inner = _classify(rest, inherited)
            if is_analytical(inner.intent.value) or len(rest.split()) > 2:
                return inner
        return ClassificationResult(
            intent=QueryIntent.GENERAL_CHAT, confidence=0.95, detail="greeting or thanks",
        )

    # --- 2. Capability questions ------------------------------------------
    if _CAPABILITY_RE.search(q) or _BARE_HELP_RE.match(q):
        return ClassificationResult(
            intent=QueryIntent.GENERAL_CHAT, confidence=0.9, detail="capability",
        )

    role = _extract_role_from_query(q)
    countries = _extract_country_names(q)
    country = countries[0] if countries else None
    technologies = _extract_technologies_from_query(q)
    unsupported = _extract_unsupported_countries(q)
    elliptical = bool(inherited["intent"]) and _is_elliptical(q)

    def result(intent: QueryIntent, confidence: float, detail: str) -> ClassificationResult:
        # Fill only what the message left out, and only when it is a
        # continuation of the earlier question. Naming a new (even
        # unsupported) country replaces the old one.
        fill = elliptical
        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            role=role or (inherited["role"] if fill else None),
            country=country or (inherited["country"] if fill and not unsupported else None),
            countries=countries or (inherited["countries"] if fill and not unsupported else []),
            technologies=technologies or (
                inherited["technologies"] if fill and not role else []
            ),
            unsupported_countries=unsupported,
            detail=detail,
        )

    # --- 3. Resume analysis -----------------------------------------------
    if _RESUME_RE.search(q):
        return result(QueryIntent.RESUME_ANALYSIS, 0.9, "resume request")

    # --- 4. Country comparison (needs two markets; "compare with the US"
    # after an India question counts the earlier country) ----------------
    if _COMPARISON_RE.search(q):
        markets = list(countries)
        if len(markets) == 1 and inherited["country"] and inherited["country"] not in markets:
            markets.insert(0, inherited["country"])
        if len(markets) >= 2:
            res = result(QueryIntent.COUNTRY_COMPARISON, 0.9,
                         f"comparison across {len(markets)} countries")
            res.countries, res.country = markets, markets[0]
            res.role = role or inherited["role"]
            return res
        # "Compare Python vs Rust" compares technologies, not markets.
        if len(technologies) >= 2:
            return result(QueryIntent.TECH_TREND, 0.85, "technology comparison")

    # --- 5. Salary --------------------------------------------------------
    if _SALARY_RE.search(q) and (role or _JOB_MARKET_RE.search(q) or country or elliptical):
        return result(QueryIntent.SALARY, 0.85, "salary question")

    # --- 6. Tech trend ----------------------------------------------------
    # A named role is a stronger job_market signal than generic trend words
    # ("skills in demand for AI Engineers" is a job_market question).
    if _TREND_RE.search(q) and (technologies or (not role and _JOB_MARKET_RE.search(q))):
        return result(QueryIntent.TECH_TREND, 0.8, "trend question")

    # --- 7. Job market ----------------------------------------------------
    if role or _JOB_MARKET_RE.search(q):
        return result(QueryIntent.JOB_MARKET, 0.85, f"role={role or 'generic'}")

    # --- 8. Elliptical follow-up of the previous analytical question -----
    named_something = bool(countries or technologies or unsupported)
    if elliptical and (named_something or _FOLLOW_UP_CUE_RE.search(q)):
        return result(QueryIntent(inherited["intent"]), 0.6,
                      "follow-up of the previous analytical question")

    # --- 9. Conversational ------------------------------------------------
    return ClassificationResult(
        intent=QueryIntent.GENERAL_CHAT, confidence=0.5,
        role=role, country=country, countries=countries, technologies=technologies,
        unsupported_countries=unsupported, detail="no job-market signal",
    )


def classify_query(
    query: str,
    conversation_context: Optional[list[dict]] = None,
    resume_attached: bool = False,
) -> ClassificationResult:
    """
    Deterministically classify a user query into a QueryIntent.

    Order of checks matters (most specific -> least specific): pleasantries,
    capability questions, resume, country comparison, salary, tech trend,
    job market, elliptical follow-up, then general chat.

    Args:
        query: The user's raw question.
        conversation_context: Optional prior turns for follow-up resolution.
        resume_attached: True when the user attached a valid resume to this
            very message — a non-analytical text ("now tell me", "hi") then
            means "analyze this resume".

    Returns:
        ClassificationResult with intent and extracted metadata.
    """
    inherited = _resolve_context(conversation_context)
    result = _classify(query, inherited)
    if resume_attached and not is_analytical(result.intent.value):
        # "now tell me" + resume after "AI Engineers in India?" means: score
        # it against AI Engineer roles in India.
        fill = not query.strip() or _is_elliptical(query)
        return replace(
            result,
            intent=QueryIntent.RESUME_ANALYSIS,
            confidence=0.9,
            role=result.role or (inherited["role"] if fill else None),
            country=result.country or (inherited["country"] if fill else None),
            countries=result.countries or (inherited["countries"] if fill else []),
            detail="resume attached to this message",
        )
    return result


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
    result = classify_query(
        state["user_query"],
        conversation_context=state.get("conversation_history") or [],
        resume_attached=bool(state.get("resume_attached")),
    )

    logger.info(
        "Query classified",
        extra={
            "intent": result.intent.value,
            "confidence": result.confidence,
            "role": result.role,
            "country": result.country,
            "technologies": result.technologies,
            "unsupported_countries": result.unsupported_countries,
        },
    )

    route = "analytical pipeline" if is_analytical(result.intent.value) else "direct chat"
    return {
        "query_intent": result.intent.value,
        "intent_confidence": result.confidence,
        "role": state.get("role") or result.role,
        "country_filter": state.get("country_filter") or result.country,
        "countries": state.get("countries") or result.countries,
        "unsupported_countries": result.unsupported_countries,
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
