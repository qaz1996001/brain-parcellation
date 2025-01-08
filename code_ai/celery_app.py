from celery import Celery, Task
from celery.worker.consumer import Consumer
from .utils_synthseg import SynthSeg

app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             include=['code_ai.task.task_synthseg',
                      'code_ai.task.task_01',
                      ],
             backend='rpc://'
             )

app.config_from_object('code_ai.celery_config')
app.conf.task_default_retry_delay = 5  # Seconds delay between retries
app.conf.task_max_retries = 5  # Maximum number of retries

app.conf.task_routes = {
    'code_ai.task.task_synthseg.synthseg_task': {'queue': 'synthseg_queue'},  # 將synthseg_task指派到專屬隊列
    'code_ai.task.task_synthseg.resample_task': {'queue': 'default'},         # 默認處理其他任務
    'code_ai.task.task_synthseg.resample_to_original_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.process_synthseg_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.save_file_tasks': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.log_error_task': {'queue': 'default'},
    'code_ai.task.task_synthseg.cmb_save_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.dwi_save_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_dicom2nii.read_dicom_file': {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.get_output_study': {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.rename_dicom_file': {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.copy_dicom_file': {'queue': 'dicom2nii_queue'},
}

app.conf.task_queues = {
    'synthseg_queue': {'routing_key': 'synthseg_queue'},  # 專屬synthseg_task
    'default': {'routing_key': 'default'},               # 默認隊列
}

# app.conf.task_queues.update( {
#     'synthseg_queue': {'routing_key': 'synthseg_queue'},  # 專屬synthseg_task
#     'default': {'routing_key': 'default'},               # 默認隊列
# })


# app.conf.task_routes.update({
#     'code_ai.task.task_synthseg.synthseg_task': {'queue': 'synthseg_queue'},  # 將synthseg_task指派到專屬隊列
#     'code_ai.task.task_synthseg.resample_task': {'queue': 'default'},         # 默認處理其他任務
#     'code_ai.task.task_synthseg.resample_to_original_task': {'queue': 'default'},
# })


# 在启动Celery worker时注册任务上下文

@app.on_after_configure.connect
def setup_global_context(sender, **kwargs):
    sender.conf.CELERY_CONTEXT = {}
    print('setup_global_context',sender,type(sender))
#     import tensorflow as tf
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
#     sender.conf.CELERY_CONTEXT = {}
#     model = SynthSeg()
#     sender.conf.CELERY_CONTEXT['synth_seg'] = model

# @app.task
# def use_model(x):
#     synth_seg = app.conf.CELERY_CONTEXT['synth_seg']
#
#


from celery.signals import worker_ready
from celery.concurrency.solo import TaskPool

@worker_ready.connect
def configure_environment(sender, **kwargs):

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
            # )
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        except RuntimeError as e:
            print(e)

    print('configure_environment', sender, type(sender),sender.hostname)
    if isinstance(sender,Consumer) and isinstance(sender.pool,TaskPool) and sender.hostname == 'worker1@DESKTOP-TPJC1AV':
        print('sender.pool',sender.pool)
        print('sender.app.conf.CELERY_CONTEXT')
        print(sender.app.conf.CELERY_CONTEXT)
        model = sender.app.conf.CELERY_CONTEXT.get('synth_seg')
        if model is None:
            model = SynthSeg()
            sender.app.conf.CELERY_CONTEXT['synth_seg'] = model
            print('CELERY_CONTEXT',sender.app.conf.CELERY_CONTEXT)
