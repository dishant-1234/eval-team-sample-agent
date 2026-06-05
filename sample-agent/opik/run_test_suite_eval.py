"""Run the uploaded Travel Concierge test suite against the ADK agent.

Usage:
    python opik/run_test_suite_eval.py
    python opik/run_test_suite_eval.py --experiment-name ts-baseline-v1
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
from opik_llm_config import configure_company_openai_env, judge_model_name

DEFAULT_SUITE_NAME = "travel-concierge-test-suite"
DEFAULT_PROJECT_NAME = "demo"


def build_task():
    def evaluation_task(item: dict) -> dict:
        question = item.get("question") or item.get("input", "")
        if not question:
            raise ValueError(f"Test item missing 'question': {item}")

        agent_output = run_travel_concierge_agent(question)

        return {
            "input": question,
            "output": agent_output,
        }

    return evaluation_task


def main() -> None:
    load_dotenv()
    configure_company_openai_env()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for the LLM judge.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run an Opik test suite against the Travel Concierge agent."
    )
    parser.add_argument(
        "--suite-name",
        default=os.getenv("OPIK_TEST_SUITE_NAME", DEFAULT_SUITE_NAME),
    )
    parser.add_argument(
        "--project-name",
        default=os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT_NAME),
    )
    parser.add_argument(
        "--experiment-name",
        default="travel-concierge-test-suite-v1",
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

        client = opik.Opik()
        suite = client.get_test_suite(
            name=args.suite_name,
            project_name=args.project_name,
        )

        result = opik.run_tests(
            test_suite=suite,
            task=build_task(),
            experiment_name=args.experiment_name,
            experiment_config={
                "agent": "travel-concierge",
                "judge_model": judge_model_name(),
                "openai_api_base": os.getenv("OPENAI_API_BASE"),
            },
            worker_threads=args.workers,
            model=judge_model_name(),
        )

        print("Test suite run complete.")
        print(f"  Experiment: {args.experiment_name}")
        print(f"  Judge:      {judge_model_name()}")
        pass_rate = (
            f"{result.pass_rate:.0%}" if result.pass_rate is not None else "N/A"
        )
        print(f"  Pass rate:  {pass_rate}")
        print(f"  Passed:     {result.items_passed}/{result.items_total}")

    except Exception as exc:  # noqa: BLE001
        print(f"Test suite evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
