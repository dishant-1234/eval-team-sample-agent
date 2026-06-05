"""Generate Excel workbook for Opik dataset and test suite manual upload reference."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

dataset_df = pd.read_csv(ROOT / "travel_concierge_dataset.csv")
test_items_df = pd.read_csv(ROOT / "travel_concierge_test_suite_items.csv")
test_globals_df = pd.read_csv(ROOT / "travel_concierge_test_suite_globals.csv")

with open(ROOT / "travel_concierge_test_suite.json", encoding="utf-8") as f:
    suite = json.load(f)

suite_meta_df = pd.DataFrame(
    [
        {"field": "name", "value": suite["name"]},
        {"field": "project_name", "value": suite["project_name"]},
        {"field": "description", "value": suite["description"]},
        {"field": "global_execution_policy", "value": json.dumps(suite["global_execution_policy"])},
    ]
)
suite_assertions_df = pd.DataFrame({"global_assertion": suite["global_assertions"]})

output = ROOT / "travel_concierge_opik_eval.xlsx"
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    dataset_df.to_excel(writer, sheet_name="Dataset", index=False)
    test_items_df.to_excel(writer, sheet_name="TestSuite_Items", index=False)
    test_globals_df.to_excel(writer, sheet_name="TestSuite_Globals", index=False)
    suite_meta_df.to_excel(writer, sheet_name="TestSuite_Meta", index=False)
    suite_assertions_df.to_excel(writer, sheet_name="TestSuite_Assertions", index=False)

print(f"Wrote {output}")
