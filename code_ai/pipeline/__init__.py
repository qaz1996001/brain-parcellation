import argparse
import os
import pathlib
from typing import Optional
from code_ai.utils import study_id_pattern
from code_ai.utils.inference import InferenceEnum, Task
from code_ai.pipeline.upload import upload_json,platform_json
from code_ai.pipeline.dicomseg import dicom_seg_cmb_file


def get_study_id(file_name: str) -> Optional[str]:
    result = study_id_pattern.match(file_name)
    if result is not None:
        return result.groups()[0]
    return ""


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource', 'models')


def pipeline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default='10516407_20231215_MR_21210200091',
                        help='目前執行的case的patient_id or study id')

    parser.add_argument('--Inputs', type=str, nargs='+',
                        default=['/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/SWAN.nii.gz',
                                 '/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/T1BRAVO_AXI.nii.gz'],
                        help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--InputsDicomDir', type=str,
                        default='/mnt/e/rename_dicom_202505051/10516407_20231215_MR_21210200091/SWAN',
                        help='用於輸入的檔案')
    return parser


class PipelineConfig:
    base_path = pathlib.Path(__file__).parent.parent.parent.absolute()
    python3 = os.getenv("PYTHON3")
    conda = "conda"
    conda_env = "tf_2_14"

    def __init__(self, script_name, data_key):
        self.script_name = script_name
        self.data_key = data_key


    def generate_cmd(self, study_id: str, task: Task, input_dicom_dir: Optional[str] = None):
        input_path_list = [str(x) for x in task.input_path_list]
        output_path = os.path.dirname(task.output_path)
        PATH_ROOT = pathlib.Path(os.getenv('PATH_ROOT'))
        chuan_root = PATH_ROOT.parent.joinpath('chuan')
        chuan_code = chuan_root.joinpath('code')
        # PATH_ROOT = / mnt / e / pipeline / sean
        if self.data_key == 'Aneurysm':
            # pipeline_aneurysm_tensorflow.py [-h] [--ID ID]
            #                                        [--Inputs INPUTS [INPUTS ...]]
            #                                        [--DicomDir DICOMDIR [DICOMDIR ...]]
            #                                        [--Output_folder OUTPUT_FOLDE
            # if input_dicom_dir is None:
            #     return (f'cd {str(chuan_code)}  && '
            #             f'{self.python3} {self.script_name} '
            #             f'--ID {study_id} '
            #             f'--Inputs {" ".join(input_path_list)} '
            #             f'--Output_folder {task.output_path} ')
            # else:
            #     return (f'cd {str(chuan_code)}  && '
            #             f'{self.python3} {self.script_name} '
            #             f'--ID {study_id} '
            #             f'--Inputs {" ".join(input_path_list)} '
            #             f'--Output_folder {task.output_path} '
            #             f'--DicomDir {input_dicom_dir} '
            #             )
            if input_dicom_dir is None:
                return (f'cd {str(chuan_code)}  && '
                        f'{self.conda} run -n {self.conda_env} python {self.script_name} '
                        f'--ID {study_id} '
                        f'--Inputs {" ".join(input_path_list)} '
                        f'--Output_folder {task.output_path} ')
            else:
                return (f'cd {str(chuan_code)}  && '
                        f'{self.conda} run -n {self.conda_env} python {self.script_name} '
                        f'--ID {study_id} '
                        f'--Inputs {" ".join(input_path_list)} '
                        f'--Output_folder {task.output_path} '
                        f'--DicomDir {input_dicom_dir} '
                        )
        else:
            if input_dicom_dir is None:
                return (f'export PYTHONPATH={self.base_path} && '
                        f'{self.python3} code_ai/pipeline/{self.script_name} '
                        f'--ID {study_id} '
                        f'--Inputs {" ".join(input_path_list)} '
                        f'--Output_folder {output_path} ')
            else:
                return (f'export PYTHONPATH={self.base_path} && '
                        f'{self.python3} code_ai/pipeline/{self.script_name} '
                        f'--ID {study_id} '
                        f'--Inputs {" ".join(input_path_list)} '
                        f'--Output_folder {output_path} '
                        f'--InputsDicomDir {input_dicom_dir} '
                        )


pipelines = {
    # InferenceEnum.Aneurysm: PipelineConfig('pipeline_aneurysm_tensorflow.py', 'Aneurysm'),
    InferenceEnum.Aneurysm: PipelineConfig('script.py', 'Aneurysm'),
    # InferenceEnum.Area: PipelineConfig('pipeline_synthseg_tensorflow.py', 'Area'),
    InferenceEnum.CMB: PipelineConfig('pipeline_cmb_tensorflow.py', 'CMB'),
    # InferenceEnum.DWI: PipelineConfig('pipeline_synthseg_dwi_tensorflow.py', 'DWI'),
    # InferenceEnum.WMH_PVS: PipelineConfig('pipeline_synthseg_wmh_tensorflow.py', 'WMH_PVS'),
    # InferenceEnum.WMH: PipelineConfig('pipeline_wmh_tensorflow.py', 'WMH'),
    # InferenceEnum.Infarct: PipelineConfig('pipeline_infarct_tensorflow.py', 'Infarct'),
    }