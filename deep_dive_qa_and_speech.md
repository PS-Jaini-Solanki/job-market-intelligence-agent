# Deep-Dive Q&A — Job Market Intelligence Agent

---

## Q1: How do MCP servers get used — hardwired or dynamically?

There are **two tiers** of MCP usage:

### Tier 1: Hardwired MCP usage (direct in agent code, via the registry)
```
MarketCollection  → jobs MCP search_jobs         (collect live postings)
ComparisonAgent   → jobs MCP compare_countries    (side-by-side country comparison)
TrendAnalysis     → github MCP get_repo_stats / get_trending
ResumeAnalysis    → filesystem MCP read_file      (resume parsing)
ReportGeneration  → filesystem MCP save_report    (persist report to disk)
ChatResponder     → search MCP web_search         (ground off-topic answers)
```

Every one of these calls goes through the `MCPRegistry` singleton and records an `MCPCallRecord` in state, so the Chainlit UI can display **which MCP server + tool each agent called**, with success status and a one-line summary.

### Tier 2: Planner-directed MCP usage (dynamic)
The `reddit` MCP is registered in `MCPRegistry` and available at any time. When the **Planner** runs, its LLM output includes a list of `required_mcp_servers` per task:

```python
# From agents/planner.py — PlannerOutput tasks include:
task = AgentTask(
    task_type=TaskType.COLLECT_JOBS,
    parameters=params,
    required_mcp_servers=[MCPServerType.REDDIT],   # ← Planner tells which MCP to use
)
```

Any agent can then call the registry:
```python
result = await registry.call_tool("reddit", "search_posts", {"query": "..."})
```

> **In short:** the `jobs`, `github`, `filesystem`, and `search` MCPs are hardwired to the agents that need them. `reddit` stays in the registry as an available capability that the LLM Planner can opt into at runtime.

---

## Q2: Where is it written that some agents always run and others are conditional?

All of this lives in **[`agents/graph.py`](file:///c:/Users/JainiSolanki/L2_Project/agents/graph.py)**. Here's exactly how it works:

### Entry — every query passes through the QueryClassifier
```python
graph.add_edge(START, "query_classifier")        # deterministic, no LLM

# Then a binary route:
graph.add_conditional_edges(
    "query_classifier",
    route_after_classifier,                       # ← routing function
    {
        "planner": "planner",                     # analytical intents
        "chat_responder": "chat_responder",       # conversational intents
    },
)
```

### "Always run" agents (analytical path) = plain `add_edge()`
```python
graph.add_edge("market_collection", "skill_extraction")  # always after market
graph.add_edge("recommendation", "final_answer")         # always
graph.add_edge("final_answer", "report_generation")      # always — report is a by-product
graph.add_edge("report_generation", END)                 # always last
```

### "Conditional" agents = `add_conditional_edges()` + routing functions
```python
# After planner — stop early only if the plan is invalid/empty
graph.add_conditional_edges(
    "planner",
    route_after_planner,                        # ← routing function
    {"market_collection": "market_collection", "chat_responder": "chat_responder"}
)

# After skill_extraction — should we analyze trends?
graph.add_conditional_edges(
    "skill_extraction",
    should_analyze_trends,                      # ← routing function
    {"trend_analysis": "trend_analysis", "resume_analysis": "resume_analysis"}
)

# After trend_analysis — resume, country comparison, or straight to recommendations?
graph.add_conditional_edges(
    "trend_analysis",
    should_analyze_resume,                      # ← routing function
    {"resume_analysis": "resume_analysis",
     "comparison_agent": "comparison_agent",
     "recommendation": "recommendation"}
)

# After resume_analysis — is a compare_countries task still pending?
graph.add_conditional_edges(
    "resume_analysis",
    should_run_country_comparison,              # ← routing function
    {"comparison_agent": "comparison_agent", "recommendation": "recommendation"}
)

# comparison_agent → recommendation → final_answer → report_generation are always-run edges
```

### The routing functions read from `GraphState`:
```python
def route_after_classifier(state) -> Literal["planner", "chat_responder"]:
    if is_analytical(state.get("query_intent")):
        return "planner"
    return "chat_responder"                     # ← off-topic questions never hit the pipeline

def should_analyze_trends(state) -> Literal["trend_analysis", "resume_analysis"]:
    plan = state.get("execution_plan", {})
    tasks = plan.get("tasks", [])
    technologies = state.get("technologies_to_analyze", [])
    has_trend_task = any(t.get("task_type") == "analyze_trends" for t in tasks)
    if has_trend_task or technologies:          # ← condition checked here
        return "trend_analysis"
    return "resume_analysis"

def should_analyze_resume(state) -> Literal["resume_analysis", "comparison_agent", "recommendation"]:
    if state.get("resume_path"):
        return "resume_analysis"
    # No resume provided? Don't let the comparison task get skipped:
    if _has_compare_countries_task(state):
        return "comparison_agent"
    return "recommendation"

def should_run_country_comparison(state) -> Literal["comparison_agent", "recommendation"]:
    plan = state.get("execution_plan", {})
    tasks = plan.get("tasks", [])
    # The planner emits task_type == "compare_countries" (lowercase, plural).
    if any(t.get("task_type") == "compare_countries" for t in tasks):
        return "comparison_agent"               # ← multi-country query detected
    return "recommendation"
```

> **Key design point:** the `trend_analysis` branch re-checks for a pending `compare_countries` task through `should_analyze_resume`, so a comparison query with **no resume uploaded** still reaches the ComparisonAgent — the comparison step is never silently skipped just because it follows a trends analysis.

> **Key insight:** The routing functions return **string names of the next node**. LangGraph uses that string to pick which edge to follow. The conditions are checked *at runtime* on the actual state.

---

## Q3: Where are reports actually stored?

There are **two levels** of report output — one in-memory in state, one on disk:

### Level 1: In-memory (always happens)
[`report_generation.py`](file:///c:/Users/JainiSolanki/L2_Project/agents/report_generation.py) builds the report using `ReportBuilder` from [`services/report_service.py`](file:///c:/Users/JainiSolanki/L2_Project/services/report_service.py) and stores it in `GraphState`:
```python
return {
    "final_report": report.model_dump(),     # ← stored in LangGraph state
    "executive_summary": executive_summary,  # ← stored in LangGraph state
}
```
This is then displayed in the **Chainlit UI** (`chainlit_ui/app.py`).

### Level 2: On disk (via filesystem MCP `save_report`)
`report_generation.py` now calls the filesystem MCP **through the registry**:

```python
registry = get_registry()
result = await registry.call_tool(
    "filesystem", "save_report",
    {"filename": f"{sanitized}.md", "content": markdown},
)
mcp_calls.append(MCPCallRecord(agent="report_generation", server="filesystem",
                               tool="save_report", success=...))
```

The `save_report` tool writes to:
```
./reports/<sanitized_filename>
```

So the disk path is:
```
c:\Users\JainiSolanki\L2_Project\reports\<report_filename>.md
```

> **Important:** The `reports/` directory is git-ignored — generated reports are runtime artifacts, not source.

---

## Q4: What is used for resume parsing?

Resume parsing lives in [`tools/resume_parser_tool.py`](file:///c:/Users/JainiSolanki/L2_Project/tools/resume_parser_tool.py). Three separate libraries handle three formats:

| Format | Library | Function |
|---|---|---|
| `.pdf` | **`pypdf`** (`PdfReader`) | `extract_text_from_pdf()` — iterates pages, calls `page.extract_text()` |
| `.docx` / `.doc` | **`python-docx`** (`Document`) | `extract_text_from_docx()` — reads paragraphs AND table cells (skills in tables!) |
| `.txt` / `.text` | Python built-in `Path.read_text()` | `extract_text_from_txt()` — tries UTF-8 → latin-1 → cp1252 encoding fallback |

The **ResumeAnalysis** agent accesses these through the filesystem MCP `read_file` tool via the registry — the agent never reads the filesystem directly:

```python
registry = get_registry()
result = await registry.call_tool("filesystem", "read_file", {"file_path": str(path)})
```

After text extraction, `_clean_extracted_text()` strips null bytes, collapses whitespace, and removes control characters.

Then [`tools/skill_extractor_tool.py`](file:///c:/Users/JainiSolanki/L2_Project/tools/skill_extractor_tool.py) runs **keyword matching** (not LLM) on the cleaned text to extract skills.

`estimate_experience_years()` uses **regex patterns** like `"5+ years experience"` to detect seniority.

---

## Q5: Technical Speech — Proving Deep Project Understanding

---

# 🎤 Technical Presentation Speech

**"Job Market Intelligence Agent — A Multi-Agent LLM System with MCP Orchestration"**

---

Good [morning/afternoon]. I'm going to walk you through a production-grade multi-agent AI system I built that answers questions like *"What skills do AI Engineers need in India right now?"* or *"How does my resume compare to what the market actually wants?"* — and backs every answer with real, sourced data, not LLM hallucinations.

---

### The Problem I Set Out to Solve

Traditional career advice is either generic — from LLMs that make up statistics — or outdated — from surveys published once a year. I wanted a system that could crawl the actual job market in real time, extract signals from it, and give grounded, evidence-based career intelligence. That meant I couldn't rely on an LLM to *know* things. The LLM's only job is to *format and reason* — the data has to come from verified sources.

There was also a second problem: the model used to give the **same generic AI-Engineer report for every question** — even "what's the weather?" — and the MCP servers were invisible. So I added a **query classifier** front door and made every MCP call **traceable and visible** in the UI.

---

### The Architecture — Two Layers

My system has two distinct layers working together.

**Layer 1 is the MCP Layer** — the Model Context Protocol layer. MCP is a protocol that standardizes how AI agents communicate with external tools. Think of it like USB — you plug any MCP server in and any agent can use it without custom integration code. I built five MCP servers:

- The **Jobs MCP** wraps the Adzuna and Arbeitnow job APIs. It collects live postings (`search_jobs`) and runs cross-country comparisons (`compare_countries`) — volume, salaries, and top skills per market.
- The **GitHub MCP** wraps the GitHub API to fetch real repository statistics — stars, forks, last release date — as technology adoption signals.
- The **Search MCP** wraps the Tavily search API, which is purpose-built for AI agents. It returns clean, structured web results about salary reports, tech news, and market data.
- The **Filesystem MCP** provides path-traversal-protected access to read uploaded resume files and persist generated reports to disk.
- The **Reddit MCP** is an optional server that uses PRAW to query developer communities — subreddits like r/LocalLLaMA and r/cscareerquestions — to gauge community sentiment about technologies.

All five are registered in a **central MCPRegistry singleton**. The registry provides two call patterns: `call_tool(server_name, tool_name, args)` for direct routing, and `call_tool_auto_route(tool_name, args)` which finds the right server automatically. This means no agent needs to know *where* a tool lives — just *what* tool it needs. And every call is recorded as an `MCPCallRecord` in the shared state, so the UI can show the demo audience exactly which MCP server was called for which task.

**Layer 2 is the Agent Layer** — a LangGraph `StateGraph` with **eleven specialized agent nodes** sharing a typed `GraphState`. This is the orchestration engine.

---

### The Orchestration — How LangGraph Wires It Together

LangGraph uses a directed graph model. Every agent is a **node**, every transition is an **edge**, and they all share a typed dictionary called `GraphState` — a shared whiteboard where each agent reads upstream data and writes its own output.

The graph has a **classifier front door**:

**Step 0 — QueryClassifier.** The very first node, always. This is a **deterministic** Python classifier — no LLM, so it's fast and predictable. It detects the query intent: is this a job-market question, a tech-trend question, a salary question, a country comparison, a resume request — or just someone saying *"hi"* or asking about the weather? If the intent is **conversational**, the graph routes to the ChatResponder and **the heavy pipeline never runs**. If it's **analytical**, it goes to the Planner. This is what fixed the "same report for every question" problem — a weather question now gets a weather answer, not an AI-Engineer market report.

**Step 0b — ChatResponder.** Answers conversational and off-topic queries directly — greetings, thank-yous, "what can you do", and even real questions like the weather, where it optionally grounds itself with the **Search MCP** web tool and cites its sources.

For analytical queries, the pipeline runs:

**Step 1 — Planner.** It receives the raw user query and calls the LLM with a system prompt that describes all available task types. The LLM returns a structured JSON plan — a list of `AgentTask` objects, each with a `task_type`, parameters, and a list of `required_mcp_servers`. I also do lightweight keyword extraction in Python — no LLM needed — to pull out the target role, country code, and whether the query is about trends, comparisons, or resume gaps. Critically, the classifier **never guesses a role** — if the query doesn't mention one, the planner falls back to a sensible default like "Software Engineer" for analytical queries only.

**Step 2 — MarketCollection.** Uses the **Jobs MCP** `search_jobs` tool to hit Adzuna and Arbeitnow in parallel, and stores raw `JobPosting` objects in state. No analysis here, just collection. This agent intentionally does one thing. It also handles multi-country queries: the classifier records every country the user mentions into `state["countries"]` (in the order they mentioned them), and MarketCollection issues **one `search_jobs` MCP call per country** — merging all postings into a single collection while splitting the configured result budget across markets (`per_country = max(settings_max // len(countries), 1)`).

**Step 3 — SkillExtraction.** Iterates over every job description and runs a keyword matcher — not an LLM — to extract skills. It uses a curated vocabulary of technologies, tools, and frameworks — and importantly, it **only counts skills that are in that vocabulary**, so junk tags from job boards (like German non-skill words) can never pollute the statistics. It computes frequency statistics for each skill across all postings, giving me evidence strings like *"Python: 143/200 postings, 71.5% demand"*. It also computes salary statistics from postings that disclosed compensation.

**Steps 4 through 7 are conditional.** This is where the graph branches. After SkillExtraction, LangGraph calls a routing function called `should_analyze_trends`. This function reads the execution plan from state — if the Planner included an `analyze_trends` task, or if `technologies_to_analyze` is populated, the graph routes to TrendAnalysis. Otherwise it skips it.

**TrendAnalysis** uses the GitHub MCP — `get_repo_stats` or `get_trending` — to fetch real stats for every technology in scope. I maintain a mapping of technology names to their canonical GitHub repositories. I then compute a **popularity score** that combines log-scaled GitHub stars — capped at 50 points — with job demand percentage — also capped at 50 points — into a single comparable number.

**ResumeAnalysis** reads the uploaded file through the **Filesystem MCP** `read_file` tool, then uses `pypdf` for PDFs, `python-docx` for Word documents, with encoding fallback for plain text. After extracting raw text, I run the same skill keyword matcher to get the candidate's skill set. I compare that against the top market skills from SkillExtraction, classifying gaps as HIGH, MEDIUM, or LOW priority based on demand percentage. The output is a **Market Fit Score from 0 to 100**.

**ComparisonAgent** — if the query asked for a multi-country comparison, the graph routes here. It calls the **Jobs MCP** `compare_countries` tool and produces a side-by-side table: posting volume, median salary, and top skills per country, plus cross-market insights like shared and country-unique skills. It runs even when the user did **not** upload a resume — the graph's `should_analyze_resume` routing function routes a pending comparison straight to this agent instead of dropping it.

**Step 8 — Recommendation.** Gathers all evidence from state — skill stats, trend rankings, resume gaps, comparison data — and passes it to the LLM with a strict system prompt that says: *"Use only this data, do not invent statistics."* This is hallucination prevention by design. The LLM formats, it doesn't fabricate. It also skips any skill the user already has, so recommendations only show genuine gaps.

**Step 9 — FinalAnswer.** A rule-based composer (no LLM) reads the state and writes a **concise, question-aligned `direct_answer`** — a heading plus a few bullet points that answer exactly what was asked. Because it's deterministic, it always runs even if the LLM is down, and the user never gets a wall of tables in the chat.

**Step 10 — ReportGeneration.** The `ReportBuilder` service assembles a structured Markdown report with sections for skills, salary, trends, country comparison, and resume analysis. Each data point includes an evidence string with source attribution. The report is stored in state, **saved to disk** through the **Filesystem MCP** `save_report` tool, and shown in the Chainlit UI inside a collapsible `<details>` block — the record is kept, but it never drowns out the direct answer.

---

### Key Design Decisions I Made

**Separation of collection, extraction, and analysis.** Each agent does exactly one thing. This makes the system testable in isolation and easy to debug — if skill data is wrong, I look at SkillExtraction, not anywhere else.

**A classifier front door.** Instead of every question triggering the full pipeline, a deterministic classifier decides between a direct conversational answer and the analytical pipeline. This is the difference between a system that *feels* smart and one that *is* smart about when to spend compute.

**Hallucination prevention as a first-class concern.** I have a `data_limitations` section in every report that honestly says what we don't know. Every statistic has source attribution. The LLM only sees serialized real data — it cannot invent numbers it wasn't given.

**Traceable MCP usage.** Every MCP call is recorded with agent, server, tool, arguments, and outcome, then rendered in the UI — so the demo shows exactly *which MCP was called for which task*, and a failed call is visible instead of silent.

**Data correctness on the edges.** Small things that would silently corrupt output are handled explicitly: Adzuna salaries are labeled with the **local currency per country board** (INR for India, EUR for France, not a blanket USD), and date parsing accepts ISO-8601 with `Z`, plain `YYYY-MM-DD`, *and* the epoch timestamps Arbeitnow emits — so "no data" is never caused by a parser rejecting a valid value.

**Graceful degradation.** The Reddit MCP is optional — if credentials aren't configured, it returns a clean error instead of crashing. The Planner has a rule-based fallback if the LLM is unavailable. The `should_analyze_trends` gate means we never call GitHub if it's not needed — saving API quota. And the classifier is fully deterministic, so routing works even with zero API calls.

**MCP as a standard interface.** By wrapping all external integrations as MCP servers with a uniform `call_tool()` interface, I can add new data sources just by writing a new `MCPServer` subclass and registering it. The agents don't need to change.

---

### What This Demonstrates

This system demonstrates that I understand:
- How to design a **multi-agent pipeline** with a classifier front door, conditional routing, and shared state using LangGraph
- How to apply the **MCP protocol** to standardize external tool access
- How to make **MCP usage visible and traceable** per agent task
- How to **prevent LLM hallucinations** through grounded data pipelines and source attribution
- How to **separate concerns** at both the agent level and the service level
- How to build **graceful degradation** into a system that depends on multiple external APIs
- The difference between **LangChain tools** — like `analyze_github_trends` decorated with `@tool` — and **MCP servers** — which are protocol-compliant wrappers around groups of tools

Thank you.

---

> [!TIP]
> **For interviews**: Highlight the classifier front door and MCP traceability — they show you understand two real-world problems with LLM agents: wasteful/irrelevant computation, and black-box tool usage. Interviewers love when you proactively address failure modes.

> [!NOTE]
> **For demos**: Start with an off-topic question like *"what's the weather in Bangalore?"* — it shows the classifier routing to a direct answer **with** a recorded Search-MCP call (no report). Then ask *"What skills are required for AI Engineers in India?"* — it shows the full pipeline: Planner → Jobs MCP → Skills → Trends → Recommendation → Final Answer → Report, with each MCP call visible in the UI. Follow up with *"what about salaries?"* — the classifier inherits the role and country from the conversation, so you get an answer with zero re-typing. Close with *"Compare AI job markets in India vs USA"* to show the country comparison agent.

