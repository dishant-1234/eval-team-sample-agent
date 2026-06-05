"""Run a dataset experiment with metrics against the Travel Concierge agent.

Usage:
    python opik/run_dataset_eval.py
    python opik/run_dataset_eval.py --experiment-name ds-baseline-v1 --no-custom-metrics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from adk_task import run_travel_concierge_agent
from opik_llm_config import build_judge_model, configure_company_openai_env
from travel_concierge_metrics import build_metrics, format_output_for_scoring

DEFAULT_DATASET_NAME = "travel-concierge-eval-dataset"
DEFAULT_PROJECT_NAME = "demo"


def build_task():
    def evaluation_task(dataset_item: dict) -> dict:
        user_input = dataset_item["input"]
        expected_output = dataset_item.get("expected_output")
        scenario = dataset_item.get("scenario")

        agent_output = run_travel_concierge_agent(
            user_input,
            scenario=scenario,
        )

        packed_output = format_output_for_scoring(
            user_input=user_input,
            agent_output=agent_output,
            expected_output=expected_output,
        )

        context = dataset_item.get("context")
        return {
            "input": user_input,
            "output": packed_output,
            "context": [context] if context else None,
            "expected_output": expected_output,
        }

    return evaluation_task


def main() -> None:
    load_dotenv()
    configure_company_openai_env()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for LLM-judge metrics.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run Opik dataset evaluation with metrics."
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("OPIK_DATASET_NAME", DEFAULT_DATASET_NAME),
    )
    parser.add_argument(
        "--project-name",
        default=os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT_NAME),
    )
    parser.add_argument(
        "--experiment-name",
        default="travel-concierge-dataset-v1",
    )
    parser.add_argument(
        "--no-custom-metrics",
        action="store_true",
        help="Use only built-in Opik metrics",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("OPIK_EVAL_WORKERS", "1")),
        help="Parallel workers (keep low for custom OpenAI rate limits)",
    )
    args = parser.parse_args()

    try:
        import opik
        from opik.evaluation import evaluate

        judge_model = build_judge_model()
        metrics = build_metrics(
            include_custom=not args.no_custom_metrics,
            judge_model=judge_model,
        )
        metric_names = [metric.name for metric in metrics]

        client = opik.Opik()
        dataset = client.get_dataset(
            name=args.dataset_name,
            project_name=args.project_name,
        )

        result = evaluate(
            dataset=dataset,
            task=build_task(),
            scoring_metrics=metrics,
            experiment_name=args.experiment_name,
            project_name=args.project_name,
            task_threads=args.workers,
            experiment_config={
                "agent": "travel-concierge",
                "metrics": metric_names,
                "judge_model": judge_model.model_name,
                "openai_api_base": os.getenv("OPENAI_API_BASE"),
            },
        )

        print("Dataset experiment complete.")
        print(f"  Experiment: {args.experiment_name}")
        print(f"  Judge:      {judge_model.model_name}")
        print(f"  Metrics:    {', '.join(metric_names)}")
        scores = result.aggregate_evaluation_scores()
        for metric_name, stats in scores.aggregated_scores.items():
            print(f"  {metric_name}: {stats}")

    except Exception as exc:  # noqa: BLE001
        print(f"Dataset evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
