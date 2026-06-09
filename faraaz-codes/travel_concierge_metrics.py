"""Langfuse metric definitions for Travel Concierge experiments.

Metrics are not stored on the dataset. Define them here and pass them to
``langfuse.run_experiment()`` at evaluation time.

Usage:
    from travel_concierge_metrics import build_metrics

    result = langfuse.run_experiment(
        name="travel-concierge-dataset",
        data=items,
        task=task,
        evaluators=build_metrics(mode="dataset"),
    )
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from langfuse import Evaluation

JudgeModel = str | None


def format_output_for_scoring(
    *,
    user_input: str,
    agent_output: str,
    expected_output: str | None = None,
) -> str:
    """Pack fields so LLM judges see full context."""
    parts = [
        f"USER INPUT:\n{user_input}",
        f"AGENT RESPONSE:\n{agent_output}",
    ]
    if expected_output:
        parts.append(f"EXPECTED BEHAVIOR:\n{expected_output}")
    return "\n\n".join(parts)


def _extract_task_fields(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
) -> tuple[str, str, str | None, dict[str, Any]]:
    user_input = input if isinstance(input, str) else str(input)
    metadata = metadata or {}

    if isinstance(output, dict):
        agent_output = str(output.get("response", output))
        metadata = {**metadata, **output}
    else:
        agent_output = str(output)

    expected = (
        expected_output
        if expected_output is None or isinstance(expected_output, str)
        else str(expected_output)
    )
    return user_input, agent_output, expected, metadata


def _call_json_judge(prompt: str, *, model: str | None = None) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.2"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"score": 0.0, "reasoning": f"Could not parse judge output: {content[:200]}"}


def response_not_empty(*, input, output, expected_output=None, metadata=None, **kwargs):
    _, agent_output, _, _ = _extract_task_fields(
        input=input,
        output=output,
        expected_output=expected_output,
        metadata=metadata,
    )
    value = bool(agent_output.strip())
    return Evaluation(
        name="response_not_empty",
        value=value,
        data_type="BOOLEAN",
        comment="Response contains text." if value else "Response is empty.",
    )


def sub_agent_routing(*, input, output, expected_output=None, metadata=None, **kwargs):
    _, _, _, metadata = _extract_task_fields(
        input=input,
        output=output,
        expected_output=expected_output,
        metadata=metadata,
    )
    expected_agent = metadata.get("expected_sub_agent")
    actual_agent = metadata.get("selected_agent")
    if not expected_agent:
        return Evaluation(
            name="sub_agent_routing",
            value=1.0,
            comment="No expected sub-agent supplied.",
        )
    score = 1.0 if actual_agent == expected_agent else 0.0
    return Evaluation(
        name="sub_agent_routing",
        value=score,
        comment=f"Expected {expected_agent}, got {actual_agent}.",
    )


def build_answer_relevance(judge_model: JudgeModel = None) -> Callable[..., Evaluation]:
    def answer_relevance(*, input, output, expected_output=None, metadata=None, **kwargs):
        user_input, agent_output, _, _ = _extract_task_fields(
            input=input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )
        prompt = f"""You are evaluating answer relevance for a travel concierge.

Return JSON only:
{{"score": <float 0-1>, "reasoning": "<one sentence>"}}

Score how directly and usefully the agent response addresses the user request.
1.0 = fully relevant and actionable
0.5 = partially relevant
0.0 = off-topic or unhelpful

USER INPUT:
{user_input}

AGENT RESPONSE:
{agent_output}
"""
        parsed = _call_json_judge(prompt, model=judge_model)
        return Evaluation(
            name="answer_relevance",
            value=float(parsed.get("score", 0.0)),
            comment=str(parsed.get("reasoning", "")),
        )

    return answer_relevance


def build_agent_task_completion(judge_model: JudgeModel = None) -> Callable[..., Evaluation]:
    def agent_task_completion(*, input, output, expected_output=None, metadata=None, **kwargs):
        user_input, agent_output, _, metadata = _extract_task_fields(
            input=input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )
        prompt = f"""You are evaluating whether a travel concierge completed the user's task.

Return JSON only:
{{"score": <float 0-1>, "reasoning": "<one sentence>"}}

Score whether the agent made meaningful progress on the travel workflow implied by the request.
1.0 = task clearly progressed or completed appropriately
0.5 = partial progress
0.0 = no meaningful progress

USER INPUT:
{user_input}

AGENT RESPONSE:
{agent_output}

SELECTED AGENT:
{metadata.get("selected_agent", "unknown")}

TOOL CALLS:
{metadata.get("tool_names", [])}
"""
        parsed = _call_json_judge(prompt, model=judge_model)
        return Evaluation(
            name="agent_task_completion",
            value=float(parsed.get("score", 0.0)),
            comment=str(parsed.get("reasoning", "")),
        )

    return agent_task_completion


def build_hallucination(judge_model: JudgeModel = None) -> Callable[..., Evaluation]:
    def hallucination(*, input, output, expected_output=None, metadata=None, **kwargs):
        user_input, agent_output, _, metadata = _extract_task_fields(
            input=input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )
        prompt = f"""You are evaluating hallucination risk in a travel concierge response.

Return JSON only:
{{"score": <float 0-1>, "reasoning": "<one sentence>"}}

Use this scale:
1.0 = no harmful hallucination; response stays within plausible travel assistance
0.5 = minor unsupported claims
0.0 = clearly fabricated bookings, impossible travel, or unsafe false confirmations

USER INPUT:
{user_input}

AGENT RESPONSE:
{agent_output}

TOOL OUTPUTS:
{metadata.get("tool_outputs", [])}
"""
        parsed = _call_json_judge(prompt, model=judge_model)
        return Evaluation(
            name="hallucination",
            value=float(parsed.get("score", 0.0)),
            comment=str(parsed.get("reasoning", "")),
        )

    return hallucination


def build_expected_behavior_match(judge_model: JudgeModel = None) -> Callable[..., Evaluation]:
    def expected_behavior_match(*, input, output, expected_output=None, metadata=None, **kwargs):
        user_input, agent_output, expected, _ = _extract_task_fields(
            input=input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )
        if not expected:
            return Evaluation(
                name="expected_behavior_match",
                value=1.0,
                comment="No expected behavior supplied.",
            )

        prompt = f"""You are evaluating a travel concierge agent against expected behavior.

Return JSON only:
{{"score": <float 0-1>, "reasoning": "<one sentence>"}}

Scoring guide:
1.0 = fully satisfies expected behavior
0.5 = partially satisfies it
0.0 = clearly fails or uses the wrong workflow

{format_output_for_scoring(
    user_input=user_input,
    agent_output=agent_output,
    expected_output=expected,
)}
"""
        parsed = _call_json_judge(prompt, model=judge_model)
        return Evaluation(
            name="expected_behavior_match",
            value=float(parsed.get("score", 0.0)),
            comment=str(parsed.get("reasoning", "")),
        )

    return expected_behavior_match


def build_assertion_pass_rate(judge_model: JudgeModel = None) -> Callable[..., Evaluation]:
    def assertion_pass_rate(*, input, output, expected_output=None, metadata=None, **kwargs):
        user_input, agent_output, _, metadata = _extract_task_fields(
            input=input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )
        assertions = metadata.get("assertions") or []
        if not assertions and isinstance(output, dict):
            assertions = output.get("assertions") or []
        if not assertions:
            return Evaluation(
                name="assertion_pass_rate",
                value=1.0,
                comment="No assertions supplied.",
            )

        passed = 0
        comments: list[str] = []
        for assertion in assertions:
            prompt = f"""You are checking one assertion about a travel concierge answer.

Return JSON only:
{{"pass": <true|false>, "reasoning": "<one sentence>"}}

USER INPUT:
{user_input}

AGENT RESPONSE:
{agent_output}

ASSERTION:
{assertion}
"""
            parsed = _call_json_judge(prompt, model=judge_model)
            if parsed.get("pass") is True:
                passed += 1
            comments.append(f"{assertion}: {parsed.get('reasoning', '')}")

        score = passed / len(assertions)
        return Evaluation(
            name="assertion_pass_rate",
            value=round(score, 3),
            comment=" | ".join(comments),
            metadata={"passed": passed, "total": len(assertions)},
        )

    return assertion_pass_rate


def build_code_metrics(*, mode: str = "dataset") -> list[Callable[..., Evaluation]]:
    metrics = [response_not_empty]
    if mode == "dataset":
        metrics.append(sub_agent_routing)
    return metrics


def build_llm_metrics(
    *,
    mode: str = "dataset",
    judge_model: JudgeModel = None,
    include_custom: bool = True,
) -> list[Callable[..., Evaluation]]:
    metrics = [
        build_answer_relevance(judge_model),
        build_agent_task_completion(judge_model),
        build_hallucination(judge_model),
    ]
    if include_custom and mode == "dataset":
        metrics.append(build_expected_behavior_match(judge_model))
    if mode == "suite":
        metrics.append(build_assertion_pass_rate(judge_model))
    return metrics


def build_metrics(
    *,
    mode: str = "dataset",
    include_llm: bool = True,
    include_custom: bool = True,
    judge_model: JudgeModel = None,
) -> list[Callable[..., Evaluation]]:
    """Return evaluators for Langfuse experiments.

    mode:
      - ``dataset``: use travel_concierge_dataset.json style cases
      - ``suite``: use travel_concierge_test_suite.json assertion cases
      - ``manual``: lightweight metrics for ad-hoc inputs
    """
    metrics = build_code_metrics(mode=mode)
    if include_llm:
        metrics.extend(
            build_llm_metrics(
                mode=mode,
                judge_model=judge_model,
                include_custom=include_custom,
            )
        )
    return metrics
