"""Metric definitions for Travel Concierge dataset experiments.

Important: Opik metrics are NOT stored on the dataset.
You create/import them here, then pass them to opik.evaluate() at experiment time.

Usage:
    from travel_concierge_metrics import build_metrics

    evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=build_metrics(),
        ...
    )
"""

from __future__ import annotations

from typing import Optional, Union

from opik.evaluation.metrics import (
    AgentTaskCompletionJudge,
    AnswerRelevance,
    GEval,
    Hallucination,
)
from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.models import LiteLLMChatModel
from opik.evaluation.models.base_model import OpikBaseModel

JudgeModel = Union[str, OpikBaseModel, LiteLLMChatModel, None]


def build_builtin_metrics(*, judge_model: JudgeModel = None) -> list[BaseMetric]:
    """Built-in LLM-as-judge metrics from the Opik SDK."""
    return [
        AnswerRelevance(
            name="answer_relevance",
            model=judge_model,
            require_context=False,
        ),
        AgentTaskCompletionJudge(model=judge_model),
        Hallucination(
            name="hallucination",
            model=judge_model,
        ),
    ]


def build_custom_metrics(*, judge_model: JudgeModel = None) -> list[BaseMetric]:
    """Project-specific G-Eval metric using expected_output from the dataset."""
    return [
        GEval(
            name="expected_behavior_match",
            model=judge_model,
            task_introduction=(
                "You are evaluating a travel concierge agent. "
                "The text to score includes the user input, the agent response, "
                "and the expected behavior description."
            ),
            evaluation_criteria=(
                "Score 1.0 if the agent response satisfies the expected behavior. "
                "Score 0.5 if it partially satisfies it. "
                "Score 0.0 if it clearly fails, ignores the request, "
                "or does the wrong travel workflow."
            ),
        ),
    ]


def build_metrics(
    *,
    include_custom: bool = True,
    judge_model: JudgeModel = None,
) -> list[BaseMetric]:
    """Return all metrics to attach to a dataset experiment."""
    metrics = build_builtin_metrics(judge_model=judge_model)
    if include_custom:
        metrics.extend(build_custom_metrics(judge_model=judge_model))
    return metrics


def format_output_for_scoring(
    *,
    user_input: str,
    agent_output: str,
    expected_output: str | None = None,
) -> str:
    """Pack fields so GEval and task-completion judges see full context."""
    parts = [
        f"USER INPUT:\n{user_input}",
        f"AGENT RESPONSE:\n{agent_output}",
    ]
    if expected_output:
        parts.append(f"EXPECTED BEHAVIOR:\n{expected_output}")
    return "\n\n".join(parts)
