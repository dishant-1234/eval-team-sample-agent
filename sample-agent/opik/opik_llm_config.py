"""Opik LLM-judge configuration for the company OpenAI deployment.

The Travel Concierge agent already uses a custom OpenAI base URL via LiteLLM.
Opik judges (metrics + test-suite assertions) also go through LiteLLM, so we
point them at the same deployment before any evaluation run.
"""

from __future__ import annotations

import os

from opik.evaluation.models import LiteLLMChatModel

DEFAULT_OPENAI_API_BASE = "https://openaiqc.gep.com/summerintern/openai/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.2"


def configure_company_openai_env() -> None:
    """Ensure LiteLLM-backed Opik judges use the company OpenAI endpoint."""
    api_base = os.getenv("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE)
    os.environ["OPENAI_API_BASE"] = api_base

    judge_model = os.getenv("OPIK_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    os.environ["OPIK_JUDGE_MODEL"] = judge_model


def judge_model_name() -> str:
    configure_company_openai_env()
    return os.getenv("OPIK_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def build_judge_model() -> LiteLLMChatModel:
    """Return a LiteLLM judge model wired to the company deployment."""
    configure_company_openai_env()
    return LiteLLMChatModel(
        model_name=judge_model_name(),
        api_base=os.getenv("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
