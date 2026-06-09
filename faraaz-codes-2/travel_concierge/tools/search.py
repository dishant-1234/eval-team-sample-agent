# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local travel information tool used by the pre-trip agent.

The original sample used Google Search grounding. For orchestration tests we keep
the same tool name but return deterministic information, so runs do not require
Google AI/search credentials.
"""


def google_search_grounding(query: str) -> dict[str, str]:
    """Return a deterministic travel-info stub for local orchestration tests."""

    normalized_query = query.replace("_", " ").strip()
    return {
        "source": "local_stub",
        "query": query,
        "summary": (
            f"Use standard current travel guidance for {normalized_query}. "
            "Confirm official requirements before travel."
        ),
    }
