"""
SynthSeg pipeline package.

This package holds the modernized SynthSeg execution pipelines that reuse
shared Template Method abstractions defined in ``code_ai.pipeline.core``.
"""

from .models import (
    SynthsegCasePlan,
    SynthsegJobConfig,
    SynthsegJobResult,
    SynthsegPreparedBatch,
)
from .pipelines import SynthsegPipeline, SynthsegFiveClassPipeline

__all__ = [
    "SynthsegCasePlan",
    "SynthsegJobConfig",
    "SynthsegJobResult",
    "SynthsegPreparedBatch",
    "SynthsegPipeline",
    "SynthsegFiveClassPipeline",
]
