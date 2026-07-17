#!/usr/bin/env python3
"""
otel_setup.py

Behavioral Attestation Phase 3: shared OpenTelemetry tracer setup for AI
workflow scripts (retrain runs, etc), so a workflow run is traceable
end-to-end rather than just isolated log lines.

Exports spans via OTLP gRPC to the local Tempo instance (see
/etc/tempo/config.yml; ports locked to localhost via UFW).
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

OTLP_ENDPOINT = "localhost:4317"


def get_tracer(service_name):
    """
    Returns an OpenTelemetry tracer that exports spans to the local Tempo
    instance. Uses SimpleSpanProcessor (synchronous export per span) rather
    than the batching default, since these are short one-shot scripts, not
    long-running services -- a batch processor could exit before flushing.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)
