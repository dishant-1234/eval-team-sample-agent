"""
Travel Concierge evaluation runner for Langfuse.

Supports:
  - manual: one or many custom questions
  - suite: data/travel_concierge_test_suite.json
  - dataset: data/travel_concierge_dataset.json
  - all: suite then dataset

Examples:
    python langfuse_eval_3.py manual "Inspire me about the Maldives"
    python langfuse_eval_3.py manual "Need ideas for Europe" "Plan a London trip"
    python langfuse_eval_3.py manual --file my_questions.txt
    python langfuse_eval_3.py suite
    python langfuse_eval_3.py dataset
    python langfuse_eval_3.py all
    python langfuse_eval_3.py suite --dry-run --skip-llm-judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_SUITE = DATA_DIR / "travel_concierge_test_suite.json"
DEFAULT_DATASET = DATA_DIR / "travel_concierge_dataset.json"

from langfuse_eval_2 import (  # noqa: E402
    AgentRun,
    flush_langfuse,
    init_langfuse,
    run_agent,
)
from travel_concierge_metrics import build_metrics  # noqa: E402


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _task_output_from_run(run: AgentRun) -> dict[str, Any]:
    return {
        "response": run.response,
        "selected_agent": run.selected_agent,
        "tool_names": run.tool_names,
        "tool_calls": run.tool_calls,
        "tool_outputs": run.tool_outputs,
    }


def travel_concierge_task(*, item: dict[str, Any], dry_run: bool = False, **kwargs) -> dict[str, Any]:
    """Langfuse experiment task function."""
    user_input = item["input"]
    run = run_agent(user_input, dry_run=dry_run)
    output = _task_output_from_run(run)
    item_metadata = item.get("metadata", {})
    output.update(
        {
            "assertions": item_metadata.get("assertions", []),
            "expected_sub_agent": item_metadata.get("expected_sub_agent"),
        }
    )
    return output


def load_manual_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.file:
        path = Path(args.file)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [
                    {
                        "input": row if isinstance(row, str) else row["input"],
                        "expected_output": None if isinstance(row, str) else row.get("expected_output"),
                        "metadata": {
                            "name": f"manual_{index}",
                            "source": "manual_file",
                        },
                    }
                    for index, row in enumerate(payload)
                ]
            raise ValueError("Manual JSON file must be a list of strings or objects.")

        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [
            {
                "input": line,
                "expected_output": None,
                "metadata": {"name": f"manual_{index}", "source": "manual_file"},
            }
            for index, line in enumerate(lines)
        ]

    if not args.inputs:
        raise ValueError("Provide one or more inputs, or use --file.")

    return [
        {
            "input": question,
            "expected_output": None,
            "metadata": {"name": f"manual_{index}", "source": "manual_cli"},
        }
        for index, question in enumerate(args.inputs)
    ]


def load_suite_items(path: Path, *, runs_per_item: int = 1) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = payload.get("global_execution_policy", {})
    repeat = runs_per_item or policy.get("runs_per_item", 1)
    items: list[dict[str, Any]] = []

    for item_index, item in enumerate(payload.get("items", [])):
        question = item["data"]["question"]
        assertions = item.get("assertions", [])
        for run_index in range(repeat):
            items.append(
                {
                    "input": question,
                    "expected_output": None,
                    "metadata": {
                        "name": f"suite_{item_index}_run_{run_index + 1}",
                        "source": "test_suite",
                        "assertions": assertions,
                        "assertion_count": len(assertions),
                        "suite_name": payload.get("name", "travel-concierge-test-suite"),
                        "run_index": run_index + 1,
                        "pass_threshold": policy.get("pass_threshold"),
                    },
                }
            )
    return items


def load_dataset_items(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Dataset JSON must be a list of objects.")

    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        items.append(
            {
                "input": row["input"],
                "expected_output": row.get("expected_output"),
                "metadata": {
                    "name": f"dataset_{index}",
                    "source": "dataset",
                    "expected_sub_agent": row.get("expected_sub_agent"),
                    "phase": row.get("phase"),
                    "scenario": row.get("scenario"),
                    "tags": row.get("tags"),
                    "demo_purpose": row.get("demo_purpose"),
                },
            }
        )
    return items


def average_metric(results: Any, metric_name: str) -> float | None:
    values: list[float] = []
    for item_result in results.item_results:
        for evaluation in item_result.evaluations:
            if evaluation.name == metric_name and isinstance(evaluation.value, (int, float)):
                values.append(float(evaluation.value))
    if not values:
        return None
    return sum(values) / len(values)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def print_experiment_summary(result: Any, *, label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"{label}")
    print(f"{'=' * 72}")
    _safe_print(result.format())

    metric_names = sorted(
        {
            evaluation.name
            for item_result in result.item_results
            for evaluation in item_result.evaluations
        }
    )
    if metric_names:
        print("\nMetric averages:")
        for metric_name in metric_names:
            avg = average_metric(result, metric_name)
            if avg is not None:
                print(f"  {metric_name}: {avg:.3f}")

    if getattr(result, "dataset_run_url", None):
        print(f"\nLangfuse run URL:\n  {result.dataset_run_url}")
    elif os.getenv("LANGFUSE_BASE_URL"):
        print(f"\nOpen Langfuse:\n  {os.getenv('LANGFUSE_BASE_URL')}")


def run_experiment(
    *,
    langfuse: Any,
    name: str,
    items: list[dict[str, Any]],
    mode: str,
    dry_run: bool,
    skip_llm_judge: bool,
    run_name: str | None = None,
) -> Any:
    def task(*, item, **kwargs):
        return travel_concierge_task(item=item, dry_run=dry_run, **kwargs)

    evaluators = build_metrics(
        mode=mode,
        include_llm=not skip_llm_judge,
        include_custom=mode == "dataset",
    )

    return langfuse.run_experiment(
        name=name,
        run_name=run_name or f"{name}-{_timestamp_slug()}",
        description=f"Travel concierge {mode} evaluation via langfuse_eval_3.py",
        data=items,
        task=task,
        evaluators=evaluators,
        metadata={
            "runner": "langfuse_eval_3.py",
            "mode": mode,
            "dry_run": str(dry_run),
            "skip_llm_judge": str(skip_llm_judge),
        },
    )


def command_manual(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    if not langfuse:
        print("Langfuse is required for experiment runs. Set LANGFUSE_* keys in .env.")
        sys.exit(1)

    items = load_manual_inputs(args)
    result = run_experiment(
        langfuse=langfuse,
        name="travel-concierge-manual",
        items=items,
        mode="manual",
        dry_run=args.dry_run,
        skip_llm_judge=args.skip_llm_judge,
    )
    print_experiment_summary(result, label="Manual evaluation complete")
    flush_langfuse(langfuse)


def command_suite(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    if not langfuse:
        print("Langfuse is required for experiment runs. Set LANGFUSE_* keys in .env.")
        sys.exit(1)

    items = load_suite_items(Path(args.suite), runs_per_item=args.runs)
    result = run_experiment(
        langfuse=langfuse,
        name="travel-concierge-test-suite",
        items=items,
        mode="suite",
        dry_run=args.dry_run,
        skip_llm_judge=args.skip_llm_judge,
        run_name=args.run_name,
    )
    print_experiment_summary(result, label="Test suite evaluation complete")
    flush_langfuse(langfuse)


def command_dataset(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    if not langfuse:
        print("Langfuse is required for experiment runs. Set LANGFUSE_* keys in .env.")
        sys.exit(1)

    items = load_dataset_items(Path(args.dataset))
    result = run_experiment(
        langfuse=langfuse,
        name="travel-concierge-dataset",
        items=items,
        mode="dataset",
        dry_run=args.dry_run,
        skip_llm_judge=args.skip_llm_judge,
        run_name=args.run_name,
    )
    print_experiment_summary(result, label="Dataset evaluation complete")
    flush_langfuse(langfuse)


def command_all(args: argparse.Namespace) -> None:
    langfuse = init_langfuse(disabled=args.no_langfuse)
    if not langfuse:
        print("Langfuse is required for experiment runs. Set LANGFUSE_* keys in .env.")
        sys.exit(1)

    suite_items = load_suite_items(Path(args.suite), runs_per_item=args.runs)
    dataset_items = load_dataset_items(Path(args.dataset))

    suite_result = run_experiment(
        langfuse=langfuse,
        name="travel-concierge-test-suite",
        items=suite_items,
        mode="suite",
        dry_run=args.dry_run,
        skip_llm_judge=args.skip_llm_judge,
        run_name=args.run_name or f"test-suite-{_timestamp_slug()}",
    )
    print_experiment_summary(suite_result, label="Test suite evaluation complete")

    time.sleep(args.delay)

    dataset_result = run_experiment(
        langfuse=langfuse,
        name="travel-concierge-dataset",
        items=dataset_items,
        mode="dataset",
        dry_run=args.dry_run,
        skip_llm_judge=args.skip_llm_judge,
        run_name=args.run_name or f"dataset-{_timestamp_slug()}",
    )
    print_experiment_summary(dataset_result, label="Dataset evaluation complete")
    flush_langfuse(langfuse)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI response synthesis; still run code metrics.",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Disable Langfuse initialization.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Run only deterministic/code metrics.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between suite and dataset when using `all`.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional fixed Langfuse experiment run name.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Travel Concierge evaluations with Langfuse metrics."
    )
    _add_shared_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    manual_parser = subparsers.add_parser(
        "manual",
        help="Evaluate one or more custom questions.",
    )
    manual_parser.add_argument(
        "inputs",
        nargs="*",
        help="One or more manual travel questions.",
    )
    manual_parser.add_argument(
        "--file",
        help="Text file with one question per line, or JSON list.",
    )
    _add_shared_args(manual_parser)
    manual_parser.set_defaults(func=command_manual)

    suite_parser = subparsers.add_parser(
        "suite",
        help="Run data/travel_concierge_test_suite.json.",
    )
    suite_parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE),
        help="Path to test suite JSON.",
    )
    suite_parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Override runs_per_item from the suite JSON. 0 keeps the file default.",
    )
    _add_shared_args(suite_parser)
    suite_parser.set_defaults(func=command_suite)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Run data/travel_concierge_dataset.json.",
    )
    dataset_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to dataset JSON.",
    )
    _add_shared_args(dataset_parser)
    dataset_parser.set_defaults(func=command_dataset)

    all_parser = subparsers.add_parser(
        "all",
        help="Run test suite, then dataset.",
    )
    all_parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE),
        help="Path to test suite JSON.",
    )
    all_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to dataset JSON.",
    )
    all_parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Override runs_per_item from the suite JSON. 0 keeps the file default.",
    )
    _add_shared_args(all_parser)
    all_parser.set_defaults(func=command_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Use --dry-run or configure .env.")
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
