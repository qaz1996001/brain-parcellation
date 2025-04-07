import os
import re

study_id_pattern = re.compile('.*(_[0-9]{8,11}_[0-9]{8}_(MR|CT|PR|CR)_E?[0-9]{8,14})+.*', re.IGNORECASE)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource', 'models')

path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
path_processModel = os.path.join(path_process, 'Deep_Infarct')  # 前處理dicom路徑(test case)