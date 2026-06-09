"""
Travel Concierge evaluation runner for Langfuse.

Fixes included:
- Keeps long suite assertions out of Langfuse item metadata so they are not dropped.
- Normalizes dataset fields like tags.
- Avoids duplicate CLI option registration.
- Makes summary printing and flushing more robust.
- Preserves compatibility with the existing langfuse_eval_2.py and travel_concierge_metrics.py modules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
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


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _remove_none_values(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _coerce_list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Support comma-separated tags as well as a single tag string.
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _compact_preview(value: Any, *, max_chars: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _validate_input_text(text: Any, *, field_name: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{field_name} must be a string.")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned


def _task_output_from_run(run: AgentRun) -> dict[str, Any]:
    return {
        "response": run.response,
        "selected_agent": run.selected_agent,
        "tool_names": run.tool_names,
        "tool_calls": run.tool_calls,
        "tool_outputs": run.tool_outputs,
    }


def _extract_item_fields(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}

    assertions = item.get("assertions")
    if assertions is None:
        assertions = metadata.get("assertions")
    assertions_list = _coerce_list_of_strings(assertions)

    expected_sub_agent = item.get("expected_sub_agent")
    if expected_sub_agent is None:
        expected_sub_agent = metadata.get("expected_sub_agent")

    phase = item.get("phase")
    if phase is None:
        phase = metadata.get("phase")

    scenario = item.get("scenario")
    if scenario is None:
        scenario = metadata.get("scenario")

    tags = item.get("tags")
    if tags is None:
        tags = metadata.get("tags")
    tags_list = _coerce_list_of_strings(tags)

    return {
        "assertions": assertions_list,
        "assertion_count": len(assertions_list),
        "assertions_preview": assertions_list[:2],
        "expected_sub_agent": expected_sub_agent,
        "phase": phase,
        "scenario": scenario,
        "tags": tags_list,
    }


def travel_concierge_task(*, item: dict[str, Any], dry_run: bool = False, **kwargs) -> dict[str, Any]:
    """
    Langfuse experiment task.

    Important:
    - Reads long assertions from the item root if present.
    - Returns only a small preview so we don't create another oversized metadata payload.
    """
    user_input = _validate_input_text(item.get("input"), field_name="item.input")
    run = run_agent(user_input, dry_run=dry_run)

    output = _task_output_from_run(run)
    output.update(_extract_item_fields(item))

    # Keep the return payload compact and JSON-safe.
    return _remove_none_values(output)


def _build_manual_item(question: str, index: int, *, source: str) -> dict[str, Any]:
    question = _validate_input_text(question, field_name="manual input")
    return {
        "input": question,
        "expected_output": None,
        "metadata": {
            "name": f"manual_{index}",
            "source": source,
        },
    }


def load_manual_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.file:
        path = Path(args.file)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Manual JSON file must be a list of strings or objects.")
            items: list[dict[str, Any]] = []
            for index, row in enumerate(payload):
                if isinstance(row, str):
                    items.append(_build_manual_item(row, index, source="manual_file"))
                elif isinstance(row, dict):
                    items.append(
                        {
                            "input": _validate_input_text(row.get("input"), field_name=f"manual[{index}].input"),
                            "expected_output": row.get("expected_output"),
                            "metadata": {
                                "name": row.get("name", f"manual_{index}"),
                                "source": "manual_file",
                                "tags": _coerce_list_of_strings(row.get("tags")),
                            },
                        }
                    )
                else:
                    raise ValueError("Manual JSON list must contain strings or objects.")
            return items

        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [_build_manual_item(line, index, source="manual_file") for index, line in enumerate(lines)]

    if not args.inputs:
        raise ValueError("Provide one or more inputs, or use --file.")

    return [_build_manual_item(question, index, source="manual_cli") for index, question in enumerate(args.inputs)]


def load_suite_items(path: Path, *, runs_per_item: int = 1) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Suite JSON must be an object.")

    policy = payload.get("global_execution_policy", {}) or {}
    repeat = runs_per_item or policy.get("runs_per_item", 1)
    suite_name = payload.get("name", "travel-concierge-test-suite")
    items: list[dict[str, Any]] = []

    for item_index, item in enumerate(payload.get("items", [])):
        data = item.get("data") or {}
        question = _validate_input_text(data.get("question"), field_name=f"suite[{item_index}].data.question")
        assertions = _coerce_list_of_strings(item.get("assertions"))

        for run_index in range(repeat):
            items.append(
                {
                    "input": question,
                    "expected_output": None,
                    # Keep the long assertions OUT of metadata so Langfuse does not drop them.
                    # They remain available to the task/evaluators at the item root.
                    "assertions": assertions,
                    "expected_sub_agent": data.get("expected_sub_agent"),
                    "metadata": {
                        "name": f"suite_{item_index}_run_{run_index + 1}",
                        "source": "test_suite",
                        "suite_name": suite_name,
                        "run_index": run_index + 1,
                        "pass_threshold": policy.get("pass_threshold"),
                        "assertion_count": len(assertions),
                        "assertions_preview": _compact_preview(assertions[:2]),
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
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index} must be an object.")

        question = _validate_input_text(row.get("input"), field_name=f"dataset[{index}].input")
        tags = _coerce_list_of_strings(row.get("tags"))

        items.append(
            {
                "input": question,
                "expected_output": row.get("expected_output"),
                "expected_sub_agent": row.get("expected_sub_agent"),
                "phase": row.get("phase"),
                "scenario": row.get("scenario"),
                "tags": tags,
                "metadata": {
                    "name": f"dataset_{index}",
                    "source": "dataset",
                    "expected_sub_agent": row.get("expected_sub_agent"),
                    "phase": row.get("phase"),
                    "scenario": row.get("scenario"),
                    "tags_count": len(tags),
                    "tags_preview": _compact_preview(tags),
                    "demo_purpose": row.get("demo_purpose"),
                },
            }
        )
    return items


def average_metric(results: Any, metric_name: str) -> float | None:
    values: list[float] = []
    for item_result in getattr(results, "item_results", []):
        for evaluation in getattr(item_result, "evaluations", []):
            if evaluation.name == metric_name and isinstance(evaluation.value, (int, float)):
                values.append(float(evaluation.value))
    if not values:
        return None
    return sum(values) / len(values)


def print_experiment_summary(result: Any, *, label: str) -> None:
    print(f"\n{'=' * 72}")
    print(label)
    print(f"{'=' * 72}")

    formatted = None
    if hasattr(result, "format"):
        try:
            formatted = result.format()
        except Exception as exc:  # pragma: no cover
            formatted = f"<unable to format Langfuse result: {exc}>"
    if formatted is None:
        formatted = repr(result)
    _safe_print(formatted)

    metric_names = sorted(
        {
            evaluation.name
            for item_result in getattr(result, "item_results", [])
            for evaluation in getattr(item_result, "evaluations", [])
        }
    )
    if metric_names:
        print("\nMetric averages:")
        for metric_name in metric_names:
            avg = average_metric(result, metric_name)
            if avg is not None:
                print(f"  {metric_name}: {avg:.3f}")

    dataset_run_url = getattr(result, "dataset_run_url", None)
    if dataset_run_url:
        print(f"\nLangfuse run URL:\n  {dataset_run_url}")
    else:
        base_url = os.getenv("LANGFUSE_BASE_URL")
        if base_url:
            print(f"\nOpen Langfuse:\n  {base_url}")


# def run_experiment(
#     *,
#     langfuse: Any,
#     name: str,
#     items: list[dict[str, Any]],
#     mode: str,
#     dry_run: bool,
#     skip_llm_judge: bool,
#     run_name: str | None = None,
# ) -> Any:
#     def task(*, item, **kwargs):
#         return travel_concierge_task(item=item, dry_run=dry_run, **kwargs)

#     evaluators = build_metrics(
#         mode=mode,
#         include_llm=not skip_llm_judge,
#         include_custom=mode == "dataset",
#     )

#     return langfuse.run_experiment(
#         name=name,
#         run_name=run_name or f"{name}-{_timestamp_slug()}",
#         description=f"Travel concierge {mode} evaluation via langfuse_eval_3.py",
#         data=items,
#         task=task,
#         evaluators=evaluators,
#         metadata={
#             "runner": "langfuse_eval_3.py",
#             "mode": mode,
#             "dry_run": dry_run,
#             "skip_llm_judge": skip_llm_judge,
#             "item_count": len(items),
#         },
#     )

import asyncio
import time

def run_experiment(
    *,
    langfuse: Any,
    name: str,
    items: list[dict[str, Any]],
    mode: str,
    dry_run: bool,
    skip_llm_judge: bool,
    run_name: str | None = None,
):
    def task(item, **kwargs):
        return travel_concierge_task(item=item, dry_run=dry_run, **kwargs)

    evaluators = build_metrics(
        mode=mode,
        include_llm=not skip_llm_judge,
        include_custom=True,
    )

    print(f"\n🚀 Starting experiment: {name}")
    print(f"📦 Total items: {len(items)}\n")

    results = []

    for i, item in enumerate(items):
        start = time.time()

        print(f"▶ Running item {i+1}/{len(items)} ...", flush=True)

        try:
            # HARD TIMEOUT PROTECTION
            result = asyncio.run(
                asyncio.wait_for(
                    asyncio.to_thread(
                        langfuse.run_experiment,
                        name=name,
                        run_name=f"{run_name}-{i}",
                        data=[item],   # run 1 item at a time (prevents full freeze)
                        task=task,
                        evaluators=evaluators,
                        metadata={"index": i, "mode": mode},
                    ),
                    timeout=120,  # 2 min max per item
                )
            )

            results.append(result)

            print(f"✅ Done item {i+1} in {time.time() - start:.1f}s")

        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT on item {i+1}")
            continue

        except Exception as e:
            print(f"❌ ERROR on item {i+1}: {repr(e)}")
            continue

    print("\n🏁 Experiment complete")

    # IMPORTANT: force flush
    flush_langfuse(langfuse)

    return results

def _require_langfuse(disabled: bool) -> Any:
    langfuse = init_langfuse(disabled=disabled)
    if not langfuse:
        print("Langfuse is required for experiment runs. Set LANGFUSE_* keys in .env.")
        sys.exit(1)
    return langfuse


def command_manual(args: argparse.Namespace) -> None:
    langfuse = _require_langfuse(args.no_langfuse)
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
    langfuse = _require_langfuse(args.no_langfuse)
    items = load_suite_items(Path(args.suite).expanduser(), runs_per_item=args.runs)
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
    langfuse = _require_langfuse(args.no_langfuse)
    items = load_dataset_items(Path(args.dataset).expanduser())
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
    langfuse = _require_langfuse(args.no_langfuse)

    suite_items = load_suite_items(Path(args.suite).expanduser(), runs_per_item=args.runs)
    dataset_items = load_dataset_items(Path(args.dataset).expanduser())

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
    shared = argparse.ArgumentParser(add_help=False)
    _add_shared_args(shared)

    parser = argparse.ArgumentParser(
        description="Run Travel Concierge evaluations with Langfuse metrics.",
        parents=[shared],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    manual_parser = subparsers.add_parser(
        "manual",
        parents=[shared],
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
    manual_parser.set_defaults(func=command_manual)

    suite_parser = subparsers.add_parser(
        "suite",
        parents=[shared],
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
    suite_parser.set_defaults(func=command_suite)

    dataset_parser = subparsers.add_parser(
        "dataset",
        parents=[shared],
        help="Run data/travel_concierge_dataset.json.",
    )
    dataset_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to dataset JSON.",
    )
    dataset_parser.set_defaults(func=command_dataset)

    all_parser = subparsers.add_parser(
        "all",
        parents=[shared],
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
