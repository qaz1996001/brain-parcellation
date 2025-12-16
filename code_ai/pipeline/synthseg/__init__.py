"""
SynthSeg pipeline package.

This package holds the modernized SynthSeg execution pipelines that reuse
shared Template Method abstractions defined in ``pipelinecore.core``.
"""

from typing import TYPE_CHECKING, Any

from .models import (
    SynthsegCasePlan,
    SynthsegJobConfig,
    SynthsegJobResult,
    SynthsegPreparedBatch,
)

if TYPE_CHECKING:
    from .context import build_runtime_components
    from .pipelines import SynthsegFiveClassPipeline, SynthsegPipeline

__all__ = [
    "SynthsegCasePlan",
    "SynthsegJobConfig",
    "SynthsegJobResult",
    "SynthsegPreparedBatch",
    "SynthsegPipeline",
    "SynthsegFiveClassPipeline",
    "build_runtime_components",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple re-export helper
    if name in {"SynthsegPipeline", "SynthsegFiveClassPipeline"}:
        from .pipelines import SynthsegFiveClassPipeline, SynthsegPipeline

        return {"SynthsegPipeline": SynthsegPipeline,
                "SynthsegFiveClassPipeline": SynthsegFiveClassPipeline}[name]
    if name == "build_runtime_components":
        from .context import build_runtime_components

        return build_runtime_components
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
