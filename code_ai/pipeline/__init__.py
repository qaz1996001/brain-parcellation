import os
import pathlib
import re

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
