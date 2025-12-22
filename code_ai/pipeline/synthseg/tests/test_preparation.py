from dataclasses import dataclass
from pathlib import Path

from code_ai.pipeline.synthseg.models import SynthsegJobConfig, SynthsegOutputFlags
from code_ai.pipeline.synthseg.preparation import build_prepared_batch


@dataclass(frozen=True)
class DummyPaths:
    process_dir: Path
    output_dir: Path
    log_dir: Path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def test_build_prepared_batch_single_case(tmp_path):
    input_dir = tmp_path / "inputs"
    input_file = _touch(input_dir / "case.nii.gz")
    config = SynthsegJobConfig(
        input_path=input_file,
        input_name=None,
        output_path=None,
        template_path=None,
        template_name=None,
        run_all=False,
        flags=SynthsegOutputFlags(
            generate_wm=True,
            generate_cmb=True,
            generate_dwi=False,
            generate_wmh=True,
        ),
    )
    paths = DummyPaths(
        process_dir=tmp_path / "proc",
        output_dir=tmp_path / "out",
        log_dir=tmp_path / "log",
    )

    batch = build_prepared_batch(config, paths)

    assert len(batch.cases) == 1
    case = batch.cases[0]
    assert case.resample_file.parent == paths.output_dir
    assert case.cmb_file is not None
    assert case.cmb_file.name.endswith("_CMB.nii.gz")
    assert case.wmh_file is not None
    assert case.wmh_file.name.endswith("_WMH_PVS.nii.gz")
    assert case.dwi_file is None

