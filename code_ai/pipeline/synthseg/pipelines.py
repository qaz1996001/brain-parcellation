from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Callable

from code_ai.pipeline.core import (
    BasePipeline,
    GpuResourceManager,
    PipelineContext,
    TensorflowPipelineMixin,
)

from . import workflow, workflow5
from .models import (
    SynthsegJobConfig,
    SynthsegJobResult,
)

WorkflowFn = Callable[[Namespace], None]


def _to_namespace(config: SynthsegJobConfig) -> Namespace:
    flags = config.flags.enable_all() if config.run_all else config.flags
    return Namespace(
        input=str(config.input_path),
        input_name=config.input_name,
        output=str(config.output_path) if config.output_path else None,
        template=str(config.template_path) if config.template_path else None,
        template_name=config.template_name,
        all=config.run_all,
        wm_file=flags.generate_wm,
        cmb=flags.generate_cmb,
        cmb_file=config.output_names.cmb,
        dwi=flags.generate_dwi,
        dwi_file=config.output_names.dwi,
        wmh=flags.generate_wmh,
        wmh_file=config.output_names.wmh,
        depth_number=config.depth_number,
    )


class _BaseLegacySynthsegPipeline(
    TensorflowPipelineMixin,
    BasePipeline[SynthsegJobConfig, Namespace, SynthsegJobResult, SynthsegJobResult],
):
    """
    Adapter pipeline that reuses the legacy SynthSeg workflows via the Template Method skeleton.
    """

    def __init__(
        self,
        context: PipelineContext,
        gpu_manager: GpuResourceManager,
        runner: WorkflowFn,
    ) -> None:
        TensorflowPipelineMixin.__init__(self, gpu_manager)
        BasePipeline.__init__(self, context)
        self._runner = runner

    def prepare(self, payload: SynthsegJobConfig) -> Namespace:
        return _to_namespace(payload)

    def run_inference(self, prepared: Namespace) -> SynthsegJobResult:
        self._runner(prepared)
        result = SynthsegJobResult()
        result.record_success(Path(prepared.input))
        return result

    def postprocess(self, inference_result: SynthsegJobResult) -> SynthsegJobResult:
        return inference_result


class SynthsegPipeline(_BaseLegacySynthsegPipeline):
    """Full SynthSeg workflow supporting WM/CMB/DWI/WMH outputs."""

    def __init__(
        self,
        context: PipelineContext,
        gpu_manager: GpuResourceManager,
        runner: WorkflowFn = workflow.run_workflow,
    ) -> None:
        super().__init__(context=context, gpu_manager=gpu_manager, runner=runner)


class SynthsegFiveClassPipeline(_BaseLegacySynthsegPipeline):
    """Specialized pipeline for 5-class SynthSeg outputs."""

    def __init__(
        self,
        context: PipelineContext,
        gpu_manager: GpuResourceManager,
        runner: WorkflowFn = workflow5.run_workflow,
    ) -> None:
        super().__init__(context=context, gpu_manager=gpu_manager, runner=runner)
