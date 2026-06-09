# Travel Concierge Orchestration Test Agent

This is a cleaned local copy of `sample-agent`. It keeps the ADK multi-agent
orchestration and removes deployment-specific code so you can test the agent
locally with OpenAI and Langfuse.

## What Is Kept

- `travel_concierge/`: the root agent, sub-agents, prompts, tools, schemas, and
  profile fixtures used by the orchestration.
- `tests/`: local unit and programmatic examples for exercising agent behavior.
- `eval/`: ADK trajectory tests and JSON cases for checking orchestration paths.
- `example.env`: OpenAI/Langfuse/local scenario configuration.

## What Was Removed

- Agent Engine deployment scripts and deployment dependency group.
- Google Cloud/Vertex AI setup requirements.
- Arize/Phoenix eval scripts and dependencies.
- External Airbnb MCP demo.
- Generated lock/virtual environment content.

## Workflow

1. `travel_concierge/__init__.py` loads `.env` and creates the shared OpenAI
   model through ADK `LiteLlm`.
2. `travel_concierge/agent.py` instruments ADK with Langfuse, opens an
   OpenInference session, and constructs `root_agent`.
3. `root_agent` receives every user turn and decides which specialist should
   handle it:
   - `inspiration_agent` for destination ideas and activities.
   - `planning_agent` for flights, hotels, seats, rooms, and itinerary creation.
   - `booking_agent` for reservation/payment simulation.
   - `pre_trip_agent` for before-trip updates and packing.
   - `in_trip_agent` for trip monitoring and day-of travel help.
   - `post_trip_agent` for feedback and preference capture.
4. The root agent uses ADK sub-agent transfer. Specialist agents can call
   `AgentTool` wrappers around smaller agents, so the workflow can fan out into
   nested agent calls.
5. Session state is the memory layer. Before each run,
   `_load_precreated_itinerary` loads the selected profile JSON into ADK state.
   Tools like `memorize` write back into the same state.
6. Tests and evals inspect whether agents/tools can be constructed and whether
   expected handoffs/tool calls happen for sample conversations.
7. If Langfuse keys are present, ADK spans are exported to Langfuse. If they are
   absent, the agent still runs locally with tracing disabled.

## Setup

```powershell
cd faraaz-codes-2
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .
pip install pytest pytest-asyncio "google-adk[eval]>=1.18.0,<2.0.0"
copy example.env .env
```

If you use `uv`, you can install the project and test dependencies with:

```powershell
uv sync --group dev
```

Fill in at least:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=openai/gpt-5.2
OPENAI_BASE_URL=https://openaiqc.gep.com/summerintern/openai/v1
```

For Langfuse tracing, also set:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Run Locally

CLI:

```powershell
adk run travel_concierge
```

Web UI:

```powershell
adk web --port=8000
```

Programmatic API example:

```powershell
adk api_server travel_concierge
python tests\programmatic_example.py
```

## Test

Unit tests:

```powershell
pytest tests
```

ADK trajectory evals:

```powershell
pytest eval
```

The pre-trip search tool is now a deterministic local stub named
`google_search_grounding`. It keeps the same orchestration path without needing
Google Search grounding credentials.
