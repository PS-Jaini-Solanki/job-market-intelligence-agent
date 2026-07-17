# Agentic AI and LangGraph Explained

## What is Agentic AI?
Traditional AI applications operate as a single LLM call: you ask a question, and it gives you an answer.

**Agentic AI** systems break down complex problems into multiple steps, utilizing different "agents" (specialized LLM prompts with specific tools) to solve parts of the problem.

## How our Multi-Agent System Works
We use **LangGraph** to orchestrate our agents. LangGraph represents the workflow as a state machine (a graph):

1. **State**: We have a shared `GraphState` object (like a whiteboard).
2. **Nodes**: Our agents are nodes in the graph. They read the state, perform an action, and update the state.
3. **Edges**: Logic that determines which node runs next.

## Our Agents
- **Planner**: The "brain". It reads the user query and creates an execution plan (a checklist of tasks).
- **Market Collection**: Uses job search APIs to gather raw job postings.
- **Skill Extraction**: Parses the job postings to find and count required skills.
- **Trend Analysis**: Looks at GitHub to see if those skills are actually growing in popularity.
- **Resume Analysis**: Compares a user's uploaded resume to the market data to find gaps.
- **Recommendation**: Looks at the gaps and market demand to suggest what to learn next.
- **Report Generation**: The "writer". It takes all the gathered data and formats it into a professional report.
