# 👋 Job Market Intelligence Agent

A **multi-agent AI system** that analyzes the tech job market in real time using **LangGraph** + **MCP** (Model Context Protocol).

## What I can do

| Capability | Example |
|---|---|
| 🔍 **Job market analysis** | "What skills are required for AI Engineers in India?" |
| 📈 **Technology trends** | "Should I learn LangGraph or CrewAI?" |
| 🌍 **Country comparison** | "Compare AI job markets in India vs USA" |
| 💰 **Salary insights** | "What's the average salary of a Data Scientist in the UK?" |
| 📄 **Resume analysis** | Upload your resume → get a market fit score |

## How it works

Every question passes through a **Query Classifier** that decides between:

1. **Analytical questions** → routed through the multi-agent pipeline:
   `Planner → Market Collection (jobs MCP) → Skill Extraction → Trend Analysis (github MCP) → Resume Analysis (filesystem MCP) → Recommendation → Final Answer → Report (filesystem MCP)`
2. **Conversational questions** → answered directly by the **ChatResponder** (optionally grounded with the `search` MCP)

## Try it

- Ask "hi" — a quick greeting, no heavy pipeline
- Ask "what can you do?" — capability overview
- Ask a job-market question — watch the full agent + MCP pipeline run live
