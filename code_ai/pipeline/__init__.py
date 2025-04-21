import os
import re

study_id_pattern = re.compile('.*(_[0-9]{8,11}_[0-9]{8}_(MR|CT|PR|CR)_E?[0-9]{8,14})+.*', re.IGNORECASE)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource', 'models')

path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)

# pipeline_cmb_tensorflow.py
# path_code     = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
# path_process  = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
# path_processModel = os.path.join(path_process, 'Deep_CMB')  # 前處理dicom路徑(test case)
# path_json     = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
# path_log      = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
# path_synthseg = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'

# pipeline_synthseg5class_tensorflow.py
# path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
# path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
# path_processModel = os.path.join(path_process, 'Deep_synthseg')  # 前處理dicom路徑(test case)
# path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
# path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
# gpu_n = 0  # 使用哪一顆gpu


# pipeline_synthseg_dwi_tensorflow.py
# path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
# path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
# path_processModel = os.path.join(path_process, 'Deep_synthseg')  # 前處理dicom路徑(test case)
# path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
# path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
# gpu_n = 0  # 使用哪一顆gpu

# pipeline_synthseg_tensorflow.py
# path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
# path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
# path_processModel = os.path.join(path_process, 'Deep_synthseg')  # 前處理dicom路徑(test case)
# path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
# path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
# gpu_n = 0  # 使用哪一顆gpu

# pipeline_synthseg_tensorflow.py
# path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
# path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
# path_processModel = os.path.join(path_process, 'Deep_synthseg')  # 前處理dicom路徑(test case)
# path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
# path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
# gpu_n = 0  # 使用哪一顆gpu

# D:\00_Chen\Task04_git\code_ai\pipeline\inacrct\__init__.py
# path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
# path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
# path_processModel = os.path.join(path_process, 'Deep_Infarct')  # 前處理dicom路徑(test case)
# gpu_n = 0  # 使用哪一顆gpu
# path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
# path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾

