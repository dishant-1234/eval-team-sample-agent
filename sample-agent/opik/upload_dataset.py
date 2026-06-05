"""Upload a Travel Concierge evaluation dataset from CSV into Opik.

CSV format (one row per dataset item):
    input,expected_output,context,phase,expected_sub_agent,scenario

Required columns:
    input            - user message sent to the agent
    expected_output  - reference description of correct behavior (ground truth)

Optional columns become extra dataset metadata fields used by metrics or filters.

Usage:
    python opik/upload_dataset.py
    python opik/upload_dataset.py --csv opik/travel_concierge_dataset.csv --replace
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_CSV = Path(__file__).with_name("travel_concierge_dataset.csv")
DEFAULT_DATASET_NAME = "travel-concierge-eval-dataset"
DEFAULT_PROJECT_NAME = "travel-concierge"

REQUIRED_COLUMNS = ("input", "expected_output")
OPTIONAL_COLUMNS = ("context", "phase", "expected_sub_agent", "scenario")


def load_items_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read CSV rows and convert them to Opik dataset insert payloads."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    items: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        for index, row in enumerate(reader, start=1):
            item: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                text = (value or "").strip()
                if text:
                    item[key] = text

            if "input" not in item:
                raise ValueError(f"Row {index} has an empty input.")
            if "expected_output" not in item:
                raise ValueError(f"Row {index} has an empty expected_output.")

            items.append(item)

    if not items:
        raise ValueError(f"No items found in {csv_path}")

    return items


def upload_dataset(
    *,
    csv_path: Path,
    dataset_name: str,
    project_name: str,
    description: str,
    replace: bool,
) -> None:
    import opik

    items = load_items_from_csv(csv_path)
    client = opik.Opik()

    dataset = client.get_or_create_dataset(
        name=dataset_name,
        description=description,
        project_name=project_name,
    )

    if replace and dataset.dataset_items_count:
        print(
            f"Clearing {dataset.dataset_items_count} existing item(s) "
            f"from '{dataset_name}'..."
        )
        dataset.clear()

    print(
        f"Uploading {len(items)} item(s) to dataset '{dataset_name}' "
        f"in project '{project_name}'..."
    )
    dataset.insert(items)

    print("Done.")
    print(f"  Dataset ID:  {dataset.id}")
    print(f"  Item count:  {dataset.dataset_items_count}")
    print(f"  Version:     {dataset.get_current_version_name()}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload an Opik evaluation dataset from a CSV file."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to CSV file (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("OPIK_DATASET_NAME", DEFAULT_DATASET_NAME),
        help="Opik dataset name",
    )
    parser.add_argument(
        "--project-name",
        default=os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT_NAME),
        help="Opik project name",
    )
    parser.add_argument(
        "--description",
        default="Travel Concierge evaluation dataset uploaded from CSV",
        help="Dataset description",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing dataset items before uploading",
    )
    args = parser.parse_args()

    try:
        upload_dataset(
            csv_path=args.csv.resolve(),
            dataset_name=args.dataset_name,
            project_name=args.project_name,
            description=args.description,
            replace=args.replace,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
