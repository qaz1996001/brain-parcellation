from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context import build_runtime_components
from .models import SynthsegJobConfig, SynthsegOutputFlags
from .pipelines import SynthsegFiveClassPipeline, SynthsegPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SynthSeg pipeline runner powered by pipelinecore.")
    parser.add_argument("--input", required=True, help="Input file or directory containing NIfTI files.")
    parser.add_argument("--input-name", help="Substring filter applied to discovered NIfTI files.")
    parser.add_argument("--output", help="Optional output directory. Defaults to pipeline-managed directory.")
    parser.add_argument("--template", help="Template file or directory used for CMB/DWI workflows.")
    parser.add_argument("--template-name", help="Substring filter applied to template files.")
    parser.add_argument("--run-all", action="store_true", help="Enable all derivative outputs.")
    parser.add_argument("--enable-wm", action="store_true", default=True, help="Generate WM/david outputs.")
    parser.add_argument("--enable-cmb", action="store_true", help="Generate CMB outputs.")
    parser.add_argument("--enable-dwi", action="store_true", help="Generate DWI outputs.")
    parser.add_argument("--enable-wmh", action="store_true", help="Generate WMH outputs.")
    parser.add_argument("--depth-number", type=int, default=5, help="White matter parcellation depth.")
    parser.add_argument("--pipeline-id", default="synthseg", help="Pipeline identifier used for context/logging.")
    parser.add_argument("--runtime-root", default=str(Path("./synthseg_runs").resolve()), help="Root directory for pipeline runtime artifacts.")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index reserved for this pipeline.")
    parser.add_argument("--five-class", action="store_true", help="Run the lightweight 5-class pipeline.")
    return parser.parse_args()


def _to_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def main() -> None:
    args = parse_args()
    flags = SynthsegOutputFlags(
        generate_wm=args.enable_wm,
        generate_cmb=args.enable_cmb,
        generate_dwi=args.enable_dwi,
        generate_wmh=args.enable_wmh,
    )
    job_config = SynthsegJobConfig(
        input_path=Path(args.input),
        input_name=args.input_name,
        output_path=_to_path(args.output),
        template_path=_to_path(args.template),
        template_name=args.template_name,
        run_all=args.run_all,
        flags=flags,
        depth_number=args.depth_number,
    )
    context, gpu_manager = build_runtime_components(
        pipeline_id=args.pipeline_id,
        root_dir=Path(args.runtime_root),
        gpu_index=args.gpu_index,
    )
    pipeline_cls = SynthsegFiveClassPipeline if args.five_class else SynthsegPipeline
    pipeline = pipeline_cls(context=context, gpu_manager=gpu_manager)
    result = pipeline.execute(job_config)
    print(
        json.dumps(
            {
                "completed_cases": [str(path) for path in result.completed_cases],
                "failed_cases": [str(path) for path in result.failed_cases],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

