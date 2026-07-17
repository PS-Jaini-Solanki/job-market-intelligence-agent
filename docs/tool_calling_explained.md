# Tool Calling vs Hallucination

## The Hallucination Problem
Large Language Models (LLMs) are great at text generation, but they are terrible at factual recall. If you ask an LLM, "What percentage of AI jobs require Python?", it will confidently invent a plausible-sounding number like "82%". This is called a **hallucination**.

## The Tool Calling Solution
In this project, we implement a strict rule: **The LLM is not allowed to state market facts from its own memory.**

Instead, we use **Tool Calling**:
1. We give the LLM tools (e.g., `search_jobs`, `parse_salaries`).
2. The LLM decides which tool to call based on the user's question.
3. The tool executes Python code to fetch *real data* from an external API (like Adzuna or GitHub).
4. The tool returns the factual data back to the LLM.
5. The LLM interprets and summarizes the data.

## Evidence-Based Reporting
Every statistic in our generated reports comes directly from the tools. For example, our `SkillStatistic` model generates a string like:
`"Python: 184 / 220 postings (83.6%) — Source: adzuna, arbeitnow"`

This proves to the user that the data is real and traceable.
