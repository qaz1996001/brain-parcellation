import argparse
import os
import pathlib
from typing import Callable, List, Optional
from code_ai.utils import study_id_pattern
from code_ai.utils.inference import InferenceEnum, Task
from code_ai.pipeline.upload import upload_json, platform_json
from code_ai.pipeline.dicomseg import dicom_seg_cmb_file


def get_study_id(file_name: str) -> Optional[str]:
    result = study_id_pattern.match(file_name)
    if result is not None:
        return result.groups()[0]
    return ""


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resource", "models"
)


def pipeline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ID",
        type=str,
        default="10516407_20231215_MR_21210200091",
        help="目前執行的case的patient_id or study id",
    )

    parser.add_argument(
        "--Inputs",
        type=str,
        nargs="+",
        default=[
            "/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/SWAN.nii.gz",
            "/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/T1BRAVO_AXI.nii.gz",
        ],
        help="用於輸入的檔案",
    )
    parser.add_argument(
        "--Output_folder",
        type=str,
        default="/mnt/d/wsl_ubuntu/pipeline/sean/example_output/",
        help="用於輸出結果的資料夾",
    )
    parser.add_argument(
        "--InputsDicomDir",
        type=str,
        default="/mnt/e/rename_dicom_202505051/10516407_20231215_MR_21210200091/SWAN",
        help="用於輸入的檔案",
    )
    return parser


class PipelineConfig:
    base_path = pathlib.Path(__file__).parent.parent.parent.absolute()
    python3 = os.getenv("PYTHON3")
    conda = "conda"
    conda_env = "tf_2_14"
    chuan_root_data_key = ["Aneurysm", "WMH", "Infarct"]

    def __init__(self, script_name, data_key):
        self.script_name = script_name
        self.data_key = data_key

    def generate_cmd(
        self,
        study_id: str,
        task: Task,
        input_dicom_dir: Optional[str] = None,
        input_dicom_dirs: Optional[List[str]] = None,
    ) -> str:
        input_paths = [str(path) for path in task.input_path_list]
        dicom_args = self._collect_dicom_args(input_dicom_dir, input_dicom_dirs)
        builder = self._resolve_command_builder()
        return builder(study_id, task, input_paths, dicom_args)

    def _collect_dicom_args(
        self,
        input_dicom_dir: Optional[str],
        input_dicom_dirs: Optional[List[str]],
    ) -> List[str]:
        if input_dicom_dirs:
            return [dicom_dir for dicom_dir in input_dicom_dirs if dicom_dir]
        if input_dicom_dir:
            return [input_dicom_dir]
        return []

    def _resolve_command_builder(
        self,
    ) -> Callable[[str, Task, List[str], List[str]], str]:
        if self.data_key in self.chuan_root_data_key:
            return self._build_chuan_command
        return self._build_python_command

    def _build_chuan_command(
        self,
        study_id: str,
        task: Task,
        input_paths: List[str],
        dicom_args: List[str],
    ) -> str:
        path_root = pathlib.Path(os.getenv("PATH_ROOT"))
        chuan_root = path_root.parent.joinpath("chuan")
        chuan_code = chuan_root.joinpath("code")

        command_parts = [
            f"cd {chuan_code}",
            "&&",
            f"bash {chuan_code}/{self.script_name}",
            study_id,
            *input_paths,
            *dicom_args,
            task.output_path,
        ]
        return " ".join(command_parts)

    def _build_python_command(
        self,
        study_id: str,
        task: Task,
        input_paths: List[str],
        dicom_args: List[str],
    ) -> str:
        output_path = os.path.dirname(task.output_path)
        command_parts = [
            f"export PYTHONPATH={self.base_path}",
            "&&",
            f"{self.python3} code_ai/pipeline/{self.script_name}",
            f"--ID {study_id}",
            "--Inputs",
            *input_paths,
            f"--Output_folder {output_path}",
        ]

        if dicom_args:
            command_parts.extend(["--InputsDicomDir", dicom_args[0]])

        return " ".join(command_parts)


pipelines = {
    # InferenceEnum.Aneurysm: PipelineConfig('pipeline_aneurysm_tensorflow.py', 'Aneurysm'),
    InferenceEnum.Aneurysm: PipelineConfig("pipeline_aneurysm.sh", "Aneurysm"),
    # InferenceEnum.Area: PipelineConfig('pipeline_synthseg_tensorflow.py', 'Area'),
    InferenceEnum.CMB: PipelineConfig("pipeline_cmb_tensorflow.py", "CMB"),
    # InferenceEnum.DWI: PipelineConfig('pipeline_synthseg_dwi_tensorflow.py', 'DWI'),
    # InferenceEnum.WMH_PVS: PipelineConfig('pipeline_synthseg_wmh_tensorflow.py', 'WMH_PVS'),
    InferenceEnum.WMH: PipelineConfig("pipeline_wmh.sh", "WMH"),
    InferenceEnum.Infarct: PipelineConfig("pipeline_infarct.sh", "Infarct"),
}
