# System Architecture

## The Core Concept
The Job Market Intelligence Agent is not a simple chatbot. It is a **Multi-Agent System**. Instead of sending a single prompt to an LLM, we use a team of specialized AI agents that collaborate to solve complex data analysis problems.

## High-Level Workflow

1. **User Request**: The user asks a question via the **Chainlit UI** (e.g., "What skills do I need for AI Engineering in India?").
2. **Planner Agent**: The "manager". It analyzes the request and breaks it down into a list of tasks.
3. **Data Collection Agents**: 
   - **Market Collection Agent**: Calls external APIs (Adzuna, Arbeitnow) to fetch real job postings.
   - **Trend Analysis Agent**: Uses the **GitHub MCP Server** to fetch repository stats for technologies.
4. **Analysis Agents**:
   - **Skill Extraction Agent**: Parses job descriptions to count exactly how many times skills are mentioned.
   - **Resume Analysis Agent**: If a resume is uploaded, reads it via the **Filesystem MCP Server** and compares it to the market data.
5. **Synthesis Agents**:
   - **Recommendation Agent**: Suggests what the user should learn based on gaps.
   - **Report Generation Agent**: Formats all the collected data into a Markdown report.

## Key Technologies

### LangGraph
We use LangGraph to wire the agents together. The agents don't talk directly to each other; instead, they all read from and write to a shared `GraphState` object (like a shared whiteboard).

### Model Context Protocol (MCP)
Instead of hardcoding API integrations directly into our agents, we wrap them in MCP Servers. This standardizes how the AI interacts with the outside world (Filesystem, GitHub, Web Search).

### FastAPI & PostgreSQL
While Chainlit provides the UI, the heavy lifting and data storage are handled by a robust FastAPI backend connected to a PostgreSQL database (running in Docker), ensuring all queries and statistics are persisted.
