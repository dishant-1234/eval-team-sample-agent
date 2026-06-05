"""Upload a Travel Concierge test suite from CSV into Opik.

CSV format (one row per test item):
    question,assertion_1,assertion_2,assertion_3

The script maps each row to the Opik TestSuite item shape:
    {"data": {"question": "..."}, "assertions": ["...", "..."]}

Usage:
    python opik/upload_test_suite.py
    python opik/upload_test_suite.py --csv opik/travel_concierge_test_suite.csv --replace
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_CSV = Path(__file__).with_name("travel_concierge_test_suite.csv")
DEFAULT_SUITE_NAME = "travel-concierge-test-suite"
DEFAULT_PROJECT_NAME = "demo"
ASSERTION_PREFIX = "assertion_"


def _parse_assertion_columns(row: dict[str, str]) -> list[str]:
    """Collect assertion_N columns in numeric order, skipping blanks."""
    numbered: list[tuple[int, str]] = []
    for key, value in row.items():
        if not key.startswith(ASSERTION_PREFIX):
            continue
        suffix = key[len(ASSERTION_PREFIX) :]
        if not suffix.isdigit():
            continue
        text = (value or "").strip()
        if text:
            numbered.append((int(suffix), text))

    if numbered:
        return [text for _, text in sorted(numbered)]

    # Optional fallback: a single pipe-separated "assertions" column.
    raw = (row.get("assertions") or "").strip()
    if raw:
        return [part.strip() for part in raw.split("|") if part.strip()]

    return []


def load_items_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read CSV rows and convert them to Opik TestSuite insert payloads."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    items: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError(
                "CSV must include a 'question' column. "
                f"Found columns: {reader.fieldnames}"
            )

        for index, row in enumerate(reader, start=1):
            question = (row.get("question") or "").strip()
            if not question:
                raise ValueError(f"Row {index} has an empty question.")

            assertions = _parse_assertion_columns(row)
            if not assertions:
                raise ValueError(
                    f"Row {index} has no assertions. "
                    f"Add assertion_1/assertion_2/... columns."
                )

            item: dict[str, Any] = {
                "data": {"question": question},
                "assertions": assertions,
            }

            description = (row.get("description") or "").strip()
            if description:
                item["description"] = description

            items.append(item)

    if not items:
        raise ValueError(f"No items found in {csv_path}")

    return items


def upload_test_suite(
    *,
    csv_path: Path,
    suite_name: str,
    project_name: str,
    replace: bool,
    runs_per_item: int,
    pass_threshold: int,
    global_assertions: list[str],
) -> None:
    import opik

    items = load_items_from_csv(csv_path)
    client = opik.Opik()

    suite = client.get_or_create_test_suite(
        name=suite_name,
        description="Travel Concierge regression test suite uploaded from CSV",
        project_name=project_name,
        global_assertions=global_assertions or None,
        global_execution_policy={
            "runs_per_item": runs_per_item,
            "pass_threshold": pass_threshold,
        },
    )

    if replace and suite.items_count:
        print(f"Clearing {suite.items_count} existing item(s) from '{suite_name}'...")
        suite.clear()

    print(f"Uploading {len(items)} item(s) to '{suite_name}' in project '{project_name}'...")
    suite.insert(items)

    print("Done.")
    print(f"  Suite ID:   {suite.id}")
    print(f"  Item count: {suite.items_count}")
    print(f"  Version:    {suite.get_current_version_name()}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload an Opik test suite from a CSV file."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to CSV file (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--suite-name",
        default=os.getenv("OPIK_TEST_SUITE_NAME", DEFAULT_SUITE_NAME),
        help="Opik test suite name",
    )
    parser.add_argument(
        "--project-name",
        default=os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT_NAME),
        help="Opik project name",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing suite items before uploading",
    )
    parser.add_argument(
        "--runs-per-item",
        type=int,
        default=int(os.getenv("OPIK_RUNS_PER_ITEM", "2")),
        help="Suite-level runs_per_item execution policy",
    )
    parser.add_argument(
        "--pass-threshold",
        type=int,
        default=int(os.getenv("OPIK_PASS_THRESHOLD", "2")),
        help="Suite-level pass_threshold execution policy",
    )
    args = parser.parse_args()

    try:
        upload_test_suite(
            csv_path=args.csv.resolve(),
            suite_name=args.suite_name,
            project_name=args.project_name,
            replace=args.replace,
            runs_per_item=args.runs_per_item,
            pass_threshold=args.pass_threshold,
            global_assertions=[],
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
