"""Run the Travel Concierge ADK root agent for Opik evaluation tasks."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from travel_concierge.agent import root_agent
from travel_concierge.tools import memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCENARIO_PROFILES: dict[str, str] = {
    "empty_itinerary": "travel_concierge/profiles/itinerary_empty_default.json",
    "seattle_itinerary": "travel_concierge/profiles/itinerary_seattle_example.json",
}


def _resolve_scenario_path(scenario: str | None) -> Path:
    if not scenario:
        return PROJECT_ROOT / SCENARIO_PROFILES["empty_itinerary"]

    if scenario in SCENARIO_PROFILES:
        return PROJECT_ROOT / SCENARIO_PROFILES[scenario]

    candidate = Path(scenario)
    if candidate.is_file():
        return candidate

    return PROJECT_ROOT / scenario


def _set_scenario(scenario: str | None) -> None:
    memory.SAMPLE_SCENARIO_PATH = str(_resolve_scenario_path(scenario))


async def _run_agent_async(user_message: str, *, scenario: str | None = None) -> str:
    _set_scenario(scenario)

    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=f"opik_eval_{uuid.uuid4().hex[:8]}",
    )

    message = Content(role="user", parts=[Part(text=user_message)])
    response_parts: list[str] = []

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event and event.content and event.content.parts:
            for part in event.content.parts:
                if part and getattr(part, "text", None):
                    response_parts.append(part.text)

    return "\n".join(response_parts).strip()


def run_travel_concierge_agent(
    user_message: str,
    *,
    scenario: str | None = None,
) -> str:
    """Synchronously invoke the root agent and return the final text response."""
    return asyncio.run(_run_agent_async(user_message, scenario=scenario))
