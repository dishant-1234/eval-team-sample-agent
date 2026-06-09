import os
import warnings
from base64 import b64encode

from dotenv import load_dotenv
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()


def instrument_adk_with_langfuse() -> trace.Tracer | None:
    """Instrument ADK orchestration spans and export them to Langfuse."""

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key:
        warnings.warn("LANGFUSE_PUBLIC_KEY is not set; tracing disabled.", stacklevel=2)
        return None
    if not secret_key:
        warnings.warn("LANGFUSE_SECRET_KEY is not set; tracing disabled.", stacklevel=2)
        return None

    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    endpoint = f"{base_url}/api/public/otel/v1/traces"
    auth_header = b64encode(f"{public_key}:{secret_key}".encode()).decode()

    tracer_provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": os.getenv(
                    "LANGFUSE_PROJECT_NAME",
                    "adk-travel-concierge-local",
                )
            }
        )
    )
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                headers={"Authorization": f"Basic {auth_header}"},
            )
        )
    )

    trace.set_tracer_provider(tracer_provider)

    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider.get_tracer(__name__)
