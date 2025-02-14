from code_ai.utils_synthseg import SynthSeg,TemplateProcessor
from code_ai.utils_parcellation import CMBProcess,DWIProcess,run_wmh,run_with_WhiteMatterParcellation
from code_ai.utils_resample import resample_one,resampleSynthSEG2original
from code_ai.celery_app import app

# RabbitMQ 鎖配置
RABBITMQ_URL = "amqp://guest:guest@localhost:5672//"
LOCK_NAME = "synthseg_task_lock"


# http://localhost:3000/cmb_classify
CMB_INFERENCE_URL = 'http://localhost:3000'
# http://127.0.0.1:3000/synthseg_classify
SYNTHSEG_INFERENCE_URL = 'http://localhost:3000'
TIME_OUT    = 360
COUNTDOWN   = 120
MAX_RETRIES = 10