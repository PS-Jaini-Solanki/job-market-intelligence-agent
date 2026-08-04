# Job Market Intelligence Agent

An advanced, production-quality Agentic AI system that analyzes the technology job market using multi-agent orchestration, MCP (Model Context Protocol), and real-time tool calling.

## Features
- **Multi-Agent Architecture**: Built with LangGraph, utilizing specialized agents (Planner, Market Collection, Skill Extraction, Trend Analysis, Resume Analysis, Recommendation, Final Answer, Report Generation).
- **Zero Hallucination Guarantee**: All market statistics are derived from real-time job posting data via tool calling, never from LLM memory.
- **MCP Integration**: Uses the Model Context Protocol to standardize communication with tools (GitHub, Filesystem, Search, Reddit).
- **Resume Gap Analysis**: Compares your uploaded resume against current market requirements to identify missing skills.
- **Technology Trend Scoring**: Combines GitHub activity metrics with job market demand to produce an objective popularity score.
- **Real-Time UI**: Built with Chainlit to visualize agent reasoning, tool calls, and execution steps in real-time.

## Tech Stack
- **AI/Agents**: LangChain, LangGraph, Groq (Llama 3.3 70b)
- **Backend**: Python 3.12, FastAPI, PostgreSQL, SQLAlchemy, Alembic
- **Frontend**: Chainlit
- **Infrastructure**: Docker, Docker Compose

## Quick Start

### 1. Prerequisites
- Docker Desktop installed
- Python 3.12+ installed
- Free API Keys: Groq, GitHub Token, Tavily, Adzuna (optional)

### 2. Setup
Clone the repo and set up the environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

### 4. Start Infrastructure
Start the PostgreSQL database via Docker:
```bash
docker compose up -d
```

### 5. Run Database Migrations
Initialize the database schema:
```bash
alembic upgrade head
# (Or if Alembic isn't configured yet, the Docker init.sql will create the tables automatically on first boot).
```

### 6. Run the Application
Start the Chainlit UI:
```bash
chainlit run chainlit_ui/app.py -w
```
Navigate to `http://localhost:8001` in your browser.

## Project Structure
- `/agents` - LangGraph multi-agent implementation
- `/api` - FastAPI backend routes
- `/chainlit_ui` - Frontend UI
- `/database` - PostgreSQL ORM models and repositories
- `/mcp_servers` - Model Context Protocol servers
- `/models` - Pydantic schemas
- `/tools` - Anti-hallucination data gathering tools
- `/docs` - Beginner-friendly explanations of architecture
