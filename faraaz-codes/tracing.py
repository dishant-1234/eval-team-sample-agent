# tracing.py  ── OpenAI + Langfuse version
# Replaces Google ADK instrumentation with OpenAI instrumentation.
# Drop this over the original travel_concierge/tracing.py.

import os
import warnings
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.openai import OpenAIInstrumentor

load_dotenv()


def instrument_openai_with_langfuse():
    """Instrument OpenAI calls with Langfuse via OpenTelemetry."""

    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not pub_key:
        warnings.warn("LANGFUSE_PUBLIC_KEY is not set – tracing disabled.", stacklevel=2)
        return None
    if not sec_key:
        warnings.warn("LANGFUSE_SECRET_KEY is not set – tracing disabled.", stacklevel=2)
        return None

    langfuse = get_client()

    if not langfuse.auth_check():
        warnings.warn(
            "Langfuse authentication failed – check your keys and LANGFUSE_BASE_URL.",
            stacklevel=2,
        )
        return None

    # Instrument every OpenAI model call / tool call automatically
    OpenAIInstrumentor().instrument()

    print(
        f"[Langfuse] Tracing active → "
        f"{os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')}"
    )
    return langfuse
