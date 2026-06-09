"""
Self-contained Travel Concierge agent harness with Langfuse tracing.

This file is for the current `faraaz-codes` folder, which does not contain the
complete ADK `travel_concierge` agent package. It provides a small but complete
local agentic workflow:

1. Route the user message to a specialist agent.
2. Execute deterministic travel tools for that specialist.
3. Ask OpenAI to synthesize the final user-facing answer.
4. Send the run, tool calls, and eval scores to Langfuse.

Usage:
    python langfuse_eval_2.py run "Inspire me about beach destinations in Asia"
    python langfuse_eval_2.py interactive
    python langfuse_eval_2.py eval --dataset data/langfuse_eval_2_dataset.json

Use --dry-run to test routing/evaluation without calling OpenAI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "langfuse_eval_2_dataset.json"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

SYSTEM_PROMPT = """You are a concise travel concierge.
Use the provided routing and tool results as ground truth.
Return a specific, actionable answer in 2-5 bullet points or a short paragraph.
Do not invent bookings, prices, or official requirements beyond the tool data.
"""


@dataclass
class AgentRun:
    user_input: str
    selected_agent: str
    tool_calls: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    response: str

    @property
    def tool_names(self) -> list[str]:
        return [call["name"] for call in self.tool_calls]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def choose_agent(user_input: str) -> str:
    """Choose the specialist agent for a user message."""

    text = user_input.lower()

    if _contains_any(text, ["feedback", "review", "post-trip", "post trip"]):
        return "post_trip_agent"
    if _contains_any(text, ["monitor", "delay", "gate", "during trip", "in trip"]):
        return "in_trip_agent"
    if _contains_any(text, ["visa", "pack", "packing", "advisory", "pre-trip", "update"]):
        return "pre_trip_agent"
    if _contains_any(text, ["pay", "payment", "reserve", "reservation", "book it"]):
        return "booking_agent"
    if _contains_any(text, ["flight", "hotel", "itinerary", "plan", "seat", "room"]):
        return "planning_agent"

    return "inspiration_agent"


def place_agent(query: str) -> dict[str, Any]:
    text = query.lower()
    if _contains_any(text, ["asia", "beach", "tropical"]):
        places = [
            "Bali, Indonesia",
            "Phuket, Thailand",
            "Maldives",
        ]
    elif _contains_any(text, ["europe", "culture", "museum"]):
        places = [
            "Lisbon, Portugal",
            "Florence, Italy",
            "Prague, Czech Republic",
        ]
    elif _contains_any(text, ["america", "americas", "nature"]):
        places = [
            "Machu Picchu, Peru",
            "Banff National Park, Canada",
            "Costa Rica",
        ]
    else:
        places = [
            "Kyoto, Japan",
            "Barcelona, Spain",
            "Queenstown, New Zealand",
        ]

    return {
        "agent": "place_agent",
        "query": query,
        "places": places,
        "reason": "Matched broad destination preferences from the user request.",
    }


def poi_agent(destination: str) -> dict[str, Any]:
    return {
        "agent": "poi_agent",
        "destination": destination,
        "activities": [
            "Take a guided local food or culture walk.",
            "Visit the highest-rated historic or natural landmark.",
            "Reserve one flexible half-day for a slower neighborhood itinerary.",
        ],
    }


def flight_search_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "flight_search_agent",
        "query": query,
        "options": [
            {"airline": "Example Air", "stops": 0, "note": "fastest mock option"},
            {"airline": "Budget Jet", "stops": 1, "note": "cheaper mock option"},
        ],
    }


def hotel_search_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "hotel_search_agent",
        "query": query,
        "options": [
            {"name": "Central Stay", "type": "city hotel"},
            {"name": "Quiet Suites", "type": "apartment-style hotel"},
        ],
    }


def itinerary_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "itinerary_agent",
        "query": query,
        "outline": [
            "Day 1: arrival, check-in, easy local dinner",
            "Day 2: main attraction plus neighborhood exploration",
            "Day 3: flexible activity and departure buffer",
        ],
    }


def create_reservation(query: str) -> dict[str, Any]:
    return {
        "agent": "create_reservation",
        "query": query,
        "status": "mock_reservation_ready_for_confirmation",
    }


def payment_choice(query: str) -> dict[str, Any]:
    return {
        "agent": "payment_choice",
        "query": query,
        "available_methods": ["credit_card", "company_card", "wallet"],
    }


def process_payment(query: str) -> dict[str, Any]:
    return {
        "agent": "process_payment",
        "query": query,
        "status": "mock_payment_not_charged",
    }


def google_search_grounding(query: str) -> dict[str, Any]:
    return {
        "agent": "google_search_grounding",
        "query": query,
        "summary": (
            "Local test stub: verify official visa, medical, weather, and travel "
            "advisory sources before departure."
        ),
    }


def what_to_pack_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "what_to_pack_agent",
        "query": query,
        "items": ["comfortable shoes", "light jacket", "charger", "travel documents"],
    }


def trip_monitor_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "trip_monitor_agent",
        "query": query,
        "checks": [
            "No mock flight disruption found.",
            "Leave extra transit time for airport or station transfers.",
        ],
    }


def day_of_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "day_of_agent",
        "query": query,
        "guidance": "Use the fastest route now, but keep a 30-minute buffer.",
    }


def post_trip_agent(query: str) -> dict[str, Any]:
    return {
        "agent": "post_trip_agent",
        "query": query,
        "questions": [
            "What part of the trip would you repeat?",
            "What should be avoided next time?",
            "Should these preferences be remembered?",
        ],
    }


def memorize(query: str) -> dict[str, Any]:
    return {
        "agent": "memorize",
        "query": query,
        "status": "mock_preference_saved",
    }


TOOL_REGISTRY = {
    "place_agent": place_agent,
    "poi_agent": poi_agent,
    "flight_search_agent": flight_search_agent,
    "hotel_search_agent": hotel_search_agent,
    "itinerary_agent": itinerary_agent,
    "create_reservation": create_reservation,
    "payment_choice": payment_choice,
    "process_payment": process_payment,
    "google_search_grounding": google_search_grounding,
    "what_to_pack_agent": what_to_pack_agent,
    "trip_monitor_agent": trip_monitor_agent,
    "day_of_agent": day_of_agent,
    "post_trip_agent": post_trip_agent,
    "memorize": memorize,
}


def plan_tools(user_input: str, selected_agent: str) -> list[str]:
    """Return the tool sequence for the selected specialist."""

    text = user_input.lower()
    if selected_agent == "inspiration_agent":
        if _contains_any(text, ["activity", "activities", "things to do", "what can i do"]):
            return ["poi_agent"]
        return ["place_agent"]
    if selected_agent == "planning_agent":
        return ["flight_search_agent", "hotel_search_agent", "itinerary_agent"]
    if selected_agent == "booking_agent":
        return ["create_reservation", "payment_choice", "process_payment"]
    if selected_agent == "pre_trip_agent":
        return ["google_search_grounding", "what_to_pack_agent"]
    if selected_agent == "in_trip_agent":
        return ["trip_monitor_agent", "day_of_agent"]
    if selected_agent == "post_trip_agent":
        return ["post_trip_agent", "memorize"]
    return ["place_agent"]


def execute_tools(user_input: str, selected_agent: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tool_calls = [
        {
            "name": "transfer_to_agent",
            "args": {"agent_name": selected_agent},
        }
    ]
    tool_outputs: list[dict[str, Any]] = []

    for tool_name in plan_tools(user_input, selected_agent):
        tool_calls.append({"name": tool_name, "args": {"query": user_input}})
        tool_outputs.append(TOOL_REGISTRY[tool_name](user_input))

    return tool_calls, tool_outputs


def get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai: pip install openai") from exc

    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


def synthesize_response(
    user_input: str,
    selected_agent: str,
    tool_outputs: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> str:
    if dry_run:
        tool_summary = "; ".join(output["agent"] for output in tool_outputs)
        tool_details = json.dumps(tool_outputs)
        return (
            f"Dry run routed to {selected_agent}. "
            f"Executed tools: {tool_summary}. "
            f"Tool details: {tool_details}. "
            "Set OpenAI and Langfuse keys to generate and trace the final answer."
        )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_input": user_input,
                        "selected_agent": selected_agent,
                        "tool_outputs": tool_outputs,
                    },
                    indent=2,
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def run_agent(user_input: str, *, dry_run: bool = False) -> AgentRun:
    selected_agent = choose_agent(user_input)
    tool_calls, tool_outputs = execute_tools(user_input, selected_agent)
    response = synthesize_response(user_input, selected_agent, tool_outputs, dry_run=dry_run)
    return AgentRun(
        user_input=user_input,
        selected_agent=selected_agent,
        tool_calls=tool_calls,
        tool_outputs=tool_outputs,
        response=response,
    )


def init_langfuse(disabled: bool = False) -> Any | None:
    if disabled:
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("Langfuse keys not set; run will continue without Langfuse tracing.")
        return None

    try:
        from langfuse import get_client
        from openinference.instrumentation.openai import OpenAIInstrumentor
    except ImportError as exc:
        print(f"Langfuse tracing disabled; missing package: {exc}")
        return None

    try:
        OpenAIInstrumentor().instrument()
    except Exception as exc:  # Instrumentation should not block local tests.
        print(f"OpenAI instrumentation warning: {exc}")

    langfuse = get_client()
    try:
        if not langfuse.auth_check():
            print("Langfuse auth failed; check LANGFUSE_* variables.")
            return None
    except Exception as exc:
        print(f"Langfuse auth check failed: {exc}")
        return None

    return langfuse


def observation_context(
    langfuse: Any | None,
    *,
    name: str,
    input_data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Any:
    if langfuse and hasattr(langfuse, "start_as_current_observation"):
        return langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
            metadata=metadata or {},
        )
    return nullcontext(None)


def safe_update(observation: Any | None, *, output: Any) -> None:
    if not observation or not hasattr(observation, "update"):
        return
    try:
        observation.update(output=output)
    except Exception as exc:
        print(f"Langfuse update warning: {exc}")


def safe_score(
    observation: Any | None,
    *,
    name: str,
    value: float | bool,
    comment: str,
    data_type: str,
) -> None:
    if not observation or not hasattr(observation, "score_trace"):
        return
    try:
        observation.score_trace(
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
        )
    except TypeError:
        try:
            observation.score_trace(name=name, value=value, comment=comment)
        except Exception as exc:
            print(f"Langfuse score warning: {exc}")
    except Exception as exc:
        print(f"Langfuse score warning: {exc}")


def score_route(expected_agent: str | None, actual_agent: str) -> tuple[float, str]:
    if not expected_agent:
        return 1.0, "No expected agent supplied."
    if expected_agent == actual_agent:
        return 1.0, f"Routed to expected agent {actual_agent}."
    return 0.0, f"Expected {expected_agent}, got {actual_agent}."


def score_tools(expected_tools: list[str], actual_tools: list[str]) -> tuple[float, str]:
    if not expected_tools:
        return 1.0, "No expected tools supplied."
    hits = [tool for tool in expected_tools if tool in actual_tools]
    return len(hits) / len(expected_tools), f"Expected {expected_tools}; got {actual_tools}."


def score_keywords(expected_keywords: list[str], response: str) -> tuple[float, str]:
    if not expected_keywords:
        return 1.0, "No expected keywords supplied."
    response_lower = response.lower()
    hits = [word for word in expected_keywords if word.lower() in response_lower]
    return len(hits) / len(expected_keywords), f"Matched keywords: {hits}."


def llm_judge(user_input: str, response: str, *, dry_run: bool = False) -> tuple[float, str]:
    if dry_run:
        return 1.0, "Skipped LLM judge in dry-run mode."

    prompt = f"""Return valid JSON only.
Score this travel assistant answer from 0.0 to 1.0 for usefulness.

User request:
{user_input}

Assistant answer:
{response}

JSON schema: {{"score": <float>, "reasoning": "<one sentence>"}}"""

    client = get_openai_client()
    raw = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = raw.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        return float(parsed.get("score", 0.0)), str(parsed.get("reasoning", ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0, f"Could not parse judge response: {content[:200]}"


def first_text_from_adk_turn(turn: dict[str, Any]) -> str:
    return turn["user_content"]["parts"][0]["text"]


def expected_tools_from_adk_turn(turn: dict[str, Any]) -> list[str]:
    return [
        item.get("name", "")
        for item in turn.get("intermediate_data", {}).get("tool_uses", [])
        if item.get("name")
    ]


def load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "cases" in payload:
        return payload["cases"]

    if "eval_cases" in payload:
        converted: list[dict[str, Any]] = []
        for case_index, eval_case in enumerate(payload["eval_cases"]):
            for turn_index, turn in enumerate(eval_case.get("conversation", [])):
                converted.append(
                    {
                        "name": f"adk_case_{case_index}_turn_{turn_index}",
                        "input": first_text_from_adk_turn(turn),
                        "expected_agent": None,
                        "expected_tools": expected_tools_from_adk_turn(turn),
                        "expected_keywords": [],
                    }
                )
        return converted

    raise ValueError("Dataset must contain either `cases` or `eval_cases`.")


def evaluate_run(
    run: AgentRun,
    case: dict[str, Any],
    observation: Any | None,
    *,
    dry_run: bool,
    skip_llm_judge: bool,
) -> dict[str, Any]:
    route_value, route_comment = score_route(
        case.get("expected_agent"),
        run.selected_agent,
    )
    tool_value, tool_comment = score_tools(
        case.get("expected_tools", []),
        run.tool_names,
    )
    keyword_value, keyword_comment = score_keywords(
        case.get("expected_keywords", []),
        run.response,
    )
    not_empty = bool(run.response.strip())

    safe_score(
        observation,
        name="route_accuracy",
        value=route_value,
        data_type="NUMERIC",
        comment=route_comment,
    )
    safe_score(
        observation,
        name="tool_accuracy",
        value=tool_value,
        data_type="NUMERIC",
        comment=tool_comment,
    )
    safe_score(
        observation,
        name="keyword_coverage",
        value=keyword_value,
        data_type="NUMERIC",
        comment=keyword_comment,
    )
    safe_score(
        observation,
        name="response_not_empty",
        value=not_empty,
        data_type="BOOLEAN",
        comment="Response contains text." if not_empty else "Response is empty.",
    )

    judge_value: float | None = None
    judge_comment = "Skipped."
    if not skip_llm_judge:
        try:
            judge_value, judge_comment = llm_judge(run.user_input, run.response, dry_run=dry_run)
            safe_score(
                observation,
                name="llm_helpfulness",
                value=judge_value,
                data_type="NUMERIC",
                comment=judge_comment,
            )
        except Exception as exc:
            judge_comment = f"LLM judge failed: {exc}"

    return {
        "name": case.get("name", "custom_input"),
        "route_accuracy": route_value,
        "tool_accuracy": tool_value,
        "keyword_coverage": keyword_value,
        "response_not_empty": not_empty,
        "llm_helpfulness": judge_value,
        "llm_comment": judge_comment,
    }


def run_with_tracing(
    user_input: str,
    *,
    langfuse: Any | None,
    case: dict[str, Any] | None = None,
    dry_run: bool = False,
    skip_llm_judge: bool = False,
) -> tuple[AgentRun, dict[str, Any]]:
    case = case or {
        "name": "custom_input",
        "input": user_input,
        "expected_agent": None,
        "expected_tools": [],
        "expected_keywords": [],
    }

    with observation_context(
        langfuse,
        name="travel_concierge_agent_run",
        input_data={"input": user_input},
        metadata={"case_name": case.get("name", "custom_input")},
    ) as observation:
        run = run_agent(user_input, dry_run=dry_run)
        safe_update(
            observation,
            output={
                "selected_agent": run.selected_agent,
                "tool_calls": run.tool_calls,
                "tool_outputs": run.tool_outputs,
                "response": run.response,
            },
        )
        scores = evaluate_run(
            run,
            case,
            observation,
            dry_run=dry_run,
            skip_llm_judge=skip_llm_judge,
        )

    return run, scores


def print_run(run: AgentRun, scores: dict[str, Any] | None = None) -> None:
    print("\nSelected agent:", run.selected_agent)
    print("Tool calls:", ", ".join(run.tool_names))
    print("\nResponse:\n", run.response)
    if scores:
        print("\nScores:")
        for key, value in scores.items():
            if key in {"name", "llm_comment"}:
                continue
            print(f"  {key}: {value}")


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No results.")
        return

    numeric_keys = ["route_accuracy", "tool_accuracy", "keyword_coverage"]
    print("\nEvaluation summary")
    print("-" * 72)
    for result in results:
        print(
            f"{result['name']:<28} "
            f"route={result['route_accuracy']:.2f} "
            f"tools={result['tool_accuracy']:.2f} "
            f"keywords={result['keyword_coverage']:.2f} "
            f"non_empty={result['response_not_empty']}"
        )
    print("-" * 72)
    for key in numeric_keys:
        avg = sum(float(result[key]) for result in results) / len(results)
        print(f"Average {key}: {avg:.2f}")


def command_run(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    run, scores = run_with_tracing(
        args.input,
        langfuse=langfuse,
        dry_run=args.dry_run,
        skip_llm_judge=args.skip_llm_judge,
    )
    print_run(run, scores)
    flush_langfuse(langfuse)


def command_interactive(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    print("Enter a travel request. Type `exit` to stop.")
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        run, scores = run_with_tracing(
            user_input,
            langfuse=langfuse,
            dry_run=args.dry_run,
            skip_llm_judge=args.skip_llm_judge,
        )
        print_run(run, scores)
    flush_langfuse(langfuse)


def command_eval(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    cases = load_cases(Path(args.dataset))
    results: list[dict[str, Any]] = []

    for case in cases:
        user_input = case["input"]
        print(f"\nRunning case: {case.get('name', user_input[:40])}")
        run, scores = run_with_tracing(
            user_input,
            langfuse=langfuse,
            case=case,
            dry_run=args.dry_run,
            skip_llm_judge=args.skip_llm_judge,
        )
        print_run(run, scores)
        results.append(scores)
        time.sleep(args.delay)

    print_summary(results)
    flush_langfuse(langfuse)


def flush_langfuse(langfuse: Any | None) -> None:
    if langfuse and hasattr(langfuse, "flush"):
        try:
            langfuse.flush()
        except Exception as exc:
            print(f"Langfuse flush warning: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small travel-concierge agent with Langfuse evals."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call OpenAI; use deterministic responses.",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Do not initialize Langfuse.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the optional OpenAI LLM-as-judge score.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one custom input.")
    run_parser.add_argument("input", help="Custom travel request.")
    run_parser.set_defaults(func=command_run)

    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Run multiple custom inputs from the terminal.",
    )
    interactive_parser.set_defaults(func=command_interactive)

    eval_parser = subparsers.add_parser("eval", help="Run dataset evaluation.")
    eval_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to dataset JSON.",
    )
    eval_parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between cases for API rate limits.",
    )
    eval_parser.set_defaults(func=command_eval)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Use --dry-run or set it in .env.")
        return 1

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
