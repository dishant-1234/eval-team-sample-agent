"""
Travel Concierge – Langfuse Evaluations (OpenAI edition)
=========================================================
Three evaluator types:
  1. CODE EVAL   – Deterministic Python checks sent to Langfuse via SDK.
  2. LLM-JUDGE   – Custom prompt-based judge using OpenAI, sent to Langfuse.
  3. PREBUILT    – Langfuse-managed templates (configured once in the UI).

Run top-to-bottom locally or in a notebook.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 0 – Install dependencies                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
# Uncomment and run once:
#
# !pip install -q langfuse openai openinference-instrumentation-openai python-dotenv


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 1 – Environment variables                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
from dotenv import load_dotenv

load_dotenv()

# Fill in your keys here or put them in a .env file
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-YOUR_PUBLIC_KEY")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-YOUR_SECRET_KEY")
os.environ.setdefault("LANGFUSE_BASE_URL",   "https://cloud.langfuse.com")
os.environ.setdefault("OPENAI_API_KEY",      "sk-YOUR_OPENAI_KEY")
os.environ.setdefault("OPENAI_MODEL",        "gpt-5.2")

MODEL = os.environ["OPENAI_MODEL"]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2 – Langfuse + OpenAI instrumentation                     ║
# ╚══════════════════════════════════════════════════════════════════╝
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.openai import OpenAIInstrumentor

# Instrument BEFORE creating the OpenAI client so all calls are captured
OpenAIInstrumentor().instrument()

langfuse = get_client()
assert langfuse.auth_check(), (
    "❌ Langfuse auth failed. Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY."
)
print("✅ Langfuse connected.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 3 – Simple OpenAI agent                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
import json
from openai import AsyncOpenAI

_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# Optional tools – extend or replace with your own
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_destinations",
            "description": "Search for travel destinations matching criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Destination or activity keywords"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
]

SYSTEM_PROMPT = "You are a helpful travel concierge. Be specific and actionable."


async def run_agent_turn(
    messages: list[dict],
    use_tools: bool = True,
) -> tuple[str, list[str]]:
    """
    Run one turn of the agent. Returns (response_text, tool_names_called).
    Tool calls are auto-resolved with stub responses so the model can continue.
    """
    tool_calls_made: list[str] = []
    kwargs = {"model": MODEL, "messages": messages}
    if use_tools:
        kwargs["tools"] = TOOLS

    response = await _client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    # Handle tool calls (stub executor – replace with real logic if needed)
    while msg.tool_calls:
        messages.append(msg)  # assistant message with tool_calls
        for tc in msg.tool_calls:
            tool_calls_made.append(tc.function.name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({"result": f"[stub result for {tc.function.name}]"}),
            })
        # Follow-up completion after tool results
        response = await _client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

    return (msg.content or "").strip(), tool_calls_made


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 4 – Test-case loader and conversation runner              ║
# ╚══════════════════════════════════════════════════════════════════╝
import pathlib
import uuid
import asyncio
from typing import Any

DATA_DIR = pathlib.Path(__file__).parent / "data"


def load_test_cases(filename: str) -> list[dict]:
    with open(DATA_DIR / filename) as f:
        data = json.load(f)
    return data.get("eval_cases", [])


async def run_agent_conversation(eval_case: dict) -> tuple[list[str], list[str]]:
    """
    Run every turn in eval_case through the OpenAI agent.

    Returns
    -------
    responses  : list of final text responses (one per turn)
    tool_calls : flat list of tool names called across all turns
    """
    system_prompt = eval_case.get("system_prompt", SYSTEM_PROMPT)
    history: list[dict] = [{"role": "system", "content": system_prompt}]
    responses: list[str] = []
    all_tool_calls: list[str] = []

    for turn in eval_case["conversation"]:
        # Compatible with the Gemini-style test-case format
        user_text = turn["user_content"]["parts"][0]["text"]
        history.append({"role": "user", "content": user_text})

        response_text, tool_calls = await run_agent_turn(list(history))
        all_tool_calls.extend(tool_calls)

        history.append({"role": "assistant", "content": response_text})
        responses.append(response_text)

    return responses, all_tool_calls


def expected_tools_from_case(eval_case: dict) -> list[str]:
    tools: list[str] = []
    for turn in eval_case["conversation"]:
        for tool_use in turn.get("intermediate_data", {}).get("tool_uses", []):
            if tool_use.get("name"):
                tools.append(tool_use["name"])
    return tools


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5 – CODE evaluators (deterministic)                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def eval_tool_accuracy(expected: list[str], actual: list[str]) -> tuple[float, str]:
    """Ratio of expected tools that were actually called."""
    if not expected:
        return 1.0, "No tools expected."
    overlap = set(expected) & set(actual)
    score = len(overlap) / len(expected)
    return round(score, 3), f"Expected {expected} | Got {actual} | Hit {len(overlap)}/{len(expected)}"


def eval_response_not_empty(responses: list[str]) -> tuple[bool, str]:
    """All turns must return a non-empty text response."""
    empties = [i for i, r in enumerate(responses) if not r.strip()]
    if empties:
        return False, f"Empty responses at turn(s): {empties}"
    return True, f"All {len(responses)} turn(s) returned text."


def eval_min_response_length(responses: list[str], min_words: int = 15) -> tuple[float, str]:
    """Fraction of responses meeting the minimum word count."""
    passing = [r for r in responses if len(r.split()) >= min_words]
    score = len(passing) / max(len(responses), 1)
    return round(score, 3), f"{len(passing)}/{len(responses)} responses ≥ {min_words} words"


def eval_correct_agent_handoff(expected: list[str], actual: list[str]) -> tuple[float, str]:
    """Check that transfer/handoff tool calls used the right names."""
    exp_transfers = [t for t in expected if "transfer" in t or "agent" in t.lower()]
    act_transfers = [t for t in actual  if "transfer" in t or "agent" in t.lower()]
    if not exp_transfers:
        return 1.0, "No agent handoffs expected."
    overlap = set(exp_transfers) & set(act_transfers)
    return round(len(overlap) / len(exp_transfers), 3), (
        f"Expected: {exp_transfers} | Actual: {act_transfers}"
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 6 – LLM-as-a-judge (OpenAI)                              ║
# ╚══════════════════════════════════════════════════════════════════╝
import re
from openai import OpenAI

_judge_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def _call_judge(prompt: str) -> tuple[float, str]:
    """Call OpenAI synchronously for judging; parse JSON score."""
    resp = _judge_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
        return float(parsed["score"]), parsed.get("reasoning", "")
    except Exception:
        return 0.5, "Could not parse judge response."


def llm_judge_helpfulness(user_query: str, agent_response: str) -> tuple[float, str]:
    """Judge whether the travel concierge response is genuinely helpful."""
    prompt = f"""You are an expert evaluator for a travel concierge AI assistant.

USER QUERY:
{user_query}

AGENT RESPONSE:
{agent_response}

Score the response from 0.0 to 1.0 on TRAVEL HELPFULNESS:
  1.0 = Excellent: specific, actionable, clearly addresses travel need
  0.7 = Good: helpful with minor gaps
  0.4 = Adequate: partially helpful, missing key info
  0.1 = Poor: vague, off-topic, or incorrect

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""
    return _call_judge(prompt)


def llm_judge_handoff_quality(
    user_query: str, actual_tools: list[str], expected_tools: list[str]
) -> tuple[float, str]:
    """Judge whether the agent routed to the correct sub-agents/tools."""
    prompt = f"""You are evaluating whether a travel concierge AI called the correct tools.

USER QUERY: {user_query}
EXPECTED TOOLS: {expected_tools}
ACTUAL TOOLS CALLED: {actual_tools}

Rate routing quality 0.0–1.0:
  1.0 = All expected tools called correctly
  0.5 = Partially correct
  0.0 = Wrong tools entirely

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""
    return _call_judge(prompt)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 7 – Main evaluation loop                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
import time

EVAL_DATASETS = {
    "inspire": "inspire.test.json",
    "pretrip": "pretrip.test.json",
    "intrip":  "intrip.test.json",
}

results: list[dict[str, Any]] = []


async def run_all_evals():
    for dataset_name, filename in EVAL_DATASETS.items():
        try:
            cases = load_test_cases(filename)
        except FileNotFoundError:
            print(f"⚠️  {filename} not found, skipping.")
            continue

        print(f"\n{'═'*60}")
        print(f"  Dataset: {dataset_name}  ({len(cases)} case(s))")
        print(f"{'═'*60}")

        for case_idx, case in enumerate(cases):
            case_label = f"{dataset_name}_case_{case_idx}"
            print(f"\n▶ Running {case_label}…")

            with langfuse.start_as_current_observation(
                as_type="span",
                name="travel_concierge_eval",
                input={"dataset": dataset_name, "case_index": case_idx},
                metadata={"eval_type": dataset_name, "eval_id": case.get("eval_id", "")},
            ) as root_obs:

                with propagate_attributes(
                    tags=["evaluation", "travel-concierge", dataset_name],
                    metadata={"eval_dataset": dataset_name, "eval_case": case_idx},
                ):
                    try:
                        responses, actual_tools = await run_agent_conversation(case)
                    except Exception as exc:
                        print(f"  ❌ Agent error: {exc}")
                        root_obs.score_trace(
                            name="run_success", value=False,
                            data_type="BOOLEAN", comment=str(exc),
                        )
                        continue

                root_obs.update(output={"responses": responses, "tool_calls": actual_tools})
                expected_tools = expected_tools_from_case(case)

                # ── Code evaluators ──────────────────────────────────
                tool_score, tool_comment = eval_tool_accuracy(expected_tools, actual_tools)
                root_obs.score_trace(name="tool_usage_accuracy", value=tool_score,
                                     data_type="NUMERIC", comment=tool_comment)

                not_empty, empty_comment = eval_response_not_empty(responses)
                root_obs.score_trace(name="response_not_empty", value=not_empty,
                                     data_type="BOOLEAN", comment=empty_comment)

                len_score, len_comment = eval_min_response_length(responses)
                root_obs.score_trace(name="response_min_length", value=len_score,
                                     data_type="NUMERIC", comment=len_comment)

                handoff_score, handoff_comment = eval_correct_agent_handoff(
                    expected_tools, actual_tools
                )
                root_obs.score_trace(name="agent_handoff_accuracy", value=handoff_score,
                                     data_type="NUMERIC", comment=handoff_comment)

                # ── LLM-as-a-judge (per turn) ────────────────────────
                helpfulness_scores: list[float] = []

                for turn_idx, (turn, response) in enumerate(
                    zip(case["conversation"], responses)
                ):
                    user_text = turn["user_content"]["parts"][0]["text"]
                    h_score, h_reason = llm_judge_helpfulness(user_text, response)
                    helpfulness_scores.append(h_score)
                    root_obs.score_trace(
                        name=f"helpfulness_turn_{turn_idx + 1}",
                        value=h_score, data_type="NUMERIC", comment=h_reason,
                    )
                    time.sleep(0.3)  # stay inside rate limits

                if helpfulness_scores:
                    avg_helpfulness = round(sum(helpfulness_scores) / len(helpfulness_scores), 3)
                    root_obs.score_trace(name="avg_helpfulness", value=avg_helpfulness,
                                         data_type="NUMERIC",
                                         comment=f"Average over {len(helpfulness_scores)} turn(s)")
                else:
                    avg_helpfulness = None

                user_text_first = case["conversation"][0]["user_content"]["parts"][0]["text"]
                hq_score, hq_reason = llm_judge_handoff_quality(
                    user_text_first, actual_tools, expected_tools
                )
                root_obs.score_trace(name="handoff_quality_llm", value=hq_score,
                                     data_type="NUMERIC", comment=hq_reason)

                results.append({
                    "case": case_label,
                    "tool_accuracy": tool_score,
                    "avg_helpfulness": avg_helpfulness,
                    "handoff_quality": hq_score,
                    "response_not_empty": not_empty,
                })
                print(
                    f"  ✅ tool_acc={tool_score:.2f} | "
                    f"helpfulness={avg_helpfulness:.2f if avg_helpfulness is not None else 'N/A'} | "
                    f"handoff={hq_score:.2f}"
                )

    langfuse.flush()
    print("\n✅ All scores flushed to Langfuse.")


asyncio.run(run_all_evals())


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 8 – Results summary                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
def print_summary(results: list[dict]) -> None:
    if not results:
        print("No results collected.")
        return

    print(f"\n{'═' * 65}")
    print(f"  EVALUATION SUMMARY  ({len(results)} case(s))")
    print(f"{'═' * 65}")
    print(f"  {'Case':<30} {'ToolAcc':>8} {'Helpful':>8} {'Handoff':>8}")
    print(f"  {'-' * 60}")

    for r in results:
        helpful = f"{r['avg_helpfulness']:.2f}" if r["avg_helpfulness"] else "  N/A"
        print(
            f"  {r['case']:<30} "
            f"{r['tool_accuracy']:>8.2f} "
            f"{helpful:>8} "
            f"{r['handoff_quality']:>8.2f}"
        )

    tool_avg = sum(r["tool_accuracy"] for r in results) / len(results)
    h_vals   = [r["avg_helpfulness"] for r in results if r["avg_helpfulness"]]
    h_avg    = sum(h_vals) / len(h_vals) if h_vals else 0
    ho_avg   = sum(r["handoff_quality"] for r in results) / len(results)

    print(f"  {'-' * 60}")
    print(f"  {'AVERAGE':<30} {tool_avg:>8.2f} {h_avg:>8.2f} {ho_avg:>8.2f}")
    print(f"{'═' * 65}")
    print(f"\n  📊 Full results → {os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')}")


print_summary(results)
