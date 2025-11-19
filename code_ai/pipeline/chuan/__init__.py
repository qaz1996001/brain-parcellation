import os

from code_ai.pipeline import MODEL_DIR

DATASET_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.json')
CUATOM_MODEL_INFARCT = os.path.join(MODEL_DIR, 'Unet-0194-0.17124-0.14731-0.86344.h5')
CUATOM_MODEL_WMH = os.path.join(MODEL_DIR, 'Unet-0084-0.16622-0.19441-0.87395.h5')
CUATOM_MODEL_ANEURYSM = os.path.join(MODEL_DIR)

gpu_n = 0  # 使用哪一顆gpu
