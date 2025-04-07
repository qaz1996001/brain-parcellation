import os

from code_ai.pipeline import MODEL_DIR

DATASET_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.json')
CUATOM_MODEL = os.path.join(MODEL_DIR, 'Unet-0194-0.17124-0.14731-0.86344.h5')


path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
path_processModel = os.path.join(path_process, 'Deep_Infarct')  # 前處理dicom路徑(test case)
gpu_n = 0  # 使用哪一顆gpu
path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
