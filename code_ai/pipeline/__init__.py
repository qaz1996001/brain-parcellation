import os
import pathlib
import re
import subprocess

from code_ai import PYTHON3
from code_ai.utils_inference import InferenceEnum, Task

study_id_pattern = re.compile('.*(_[0-9]{8,11}_[0-9]{8}_(MR|CT|PR|CR)_E?[0-9]{8,14})+.*', re.IGNORECASE)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource', 'models')


class PipelineConfig:
    base_path = pathlib.Path(__file__).parent.parent.parent.absolute()
    python3   = os.getenv("PYTHON3")

    def __init__(self, script_name, data_key):
        self.script_name = script_name
        self.data_key = data_key

    def generate_cmd(self, study_id: str,task:Task):
        input_path_list = [str(x) for x in task.input_path_list]
        output_path = os.path.dirname(task.output_path)

        return (f'export PYTHONPATH={self.base_path} && '
                f'{self.python3} code_ai/pipeline/{self.script_name} '
                f'--ID {study_id} '
                f'--Inputs {" ".join(input_path_list)} '
                f'--Output_folder {output_path} ')


pipelines = {
    InferenceEnum.Area: PipelineConfig('pipeline_synthseg_tensorflow.py', 'Area'),
    InferenceEnum.CMB: PipelineConfig('pipeline_cmb_tensorflow.py', 'CMB'),
    InferenceEnum.DWI: PipelineConfig('pipeline_synthseg_dwi_tensorflow.py', 'DWI'),
    InferenceEnum.WMH_PVS: PipelineConfig('pipeline_synthseg_wmh_tensorflow.py', 'WMH_PVS'),

    InferenceEnum.Infarct: PipelineConfig('pipeline_infarct_tensorflow.py', 'Infarct'),
    }


def dicom_seg_multi_file(ID:str,
                         InputsDicomDir:str,
                         nii_path_str:str,
                         path_output:str):
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/create_dicomseg_multi_file_claude.py '
               '--ID {} '
               '--InputsDicom {} '
               '--InputsNifti {} '
               '--OutputDicomSegFolder {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                                   PYTHON3,
                                                   ID,
                                                   InputsDicomDir,
                                                   nii_path_str,
                                                   path_output)
               )

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


def upload_dicom_seg(input_dicom_seg_folder:str,input_nifti:str):
    input_dicom_seg_folder_path  = pathlib.Path(input_dicom_seg_folder)
    input_nifti_path = pathlib.Path(input_nifti)

    dicom_seg_base_name = input_nifti_path.name.split('.')[0]
    file_list = sorted(input_dicom_seg_folder_path.rglob(dicom_seg_base_name + '*.dcm'))
    file_str_list = list(map(lambda x: str(x), file_list))
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/upload_dicom_seg.py '
               '--Input {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                    PYTHON3,
                                    ' '.join(file_str_list)
                                    )
               )
    print('upload_dicom_seg',cmd_str)

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


def upload_json():
    pass