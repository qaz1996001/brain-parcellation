import os

from celery import Celery, Task
from celery.worker.consumer import Consumer
from .utils_synthseg import SynthSeg

app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             include=['code_ai.task.task_CMB',
                      'code_ai.task.task_dicom2nii',
                      'code_ai.task.task_infarct',
                      'code_ai.task.task_synthseg',
                      'code_ai.task.task_WMH'
                      ],
             # backend='rpc://'
             backend='redis://localhost:10079/1'
             )

app.config_from_object('code_ai.celery_config')
# app.conf.task_default_retry_delay = 5  # Seconds delay between retries
# app.conf.task_max_retries = 5  # Maximum number of retries

app.conf.task_routes = {
    'code_ai.task.task_synthseg.celery_workflow': {'queue': 'default'},
    'code_ai.task.task_synthseg.resample_task': {'queue': 'default'},  # 默認處理其他任務
    'code_ai.task.task_synthseg.log_error_task': {'queue': 'default'},
    'code_ai.task.task_synthseg.post_process_synthseg_task': {'queue': 'default'},

    'code_ai.task.task_synthseg.synthseg_task': {'queue': 'synthseg_queue'},  # 將synthseg_task指派到專屬隊列
    'code_ai.task.task_synthseg.process_synthseg_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.save_file_tasks': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.cmb_save_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.dwi_save_task': {'queue': 'synthseg_queue'},

    'code_ai.task.task_CMB.inference_cmb': {'queue': 'dicom2nii_queue'},

    'code_ai.task.task_dicom2nii.celery_workflow':  {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.dicom_2_nii_file': {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.process_dir_next': {'queue': 'dicom2nii_queue'},


    'code_ai.task.task_synthseg.inference_synthseg': {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_synthseg.resample_to_original_task': {'queue': 'dicom2nii_queue'},

    'code_ai.task.task_dicom2nii.process_dir': {'queue': 'dicom_rename_queue'},
    'code_ai.task.task_dicom2nii.process_instances': {'queue': 'dicom_rename_queue'},

}

app.conf.task_queues = {
    'synthseg_queue': {'routing_key': 'synthseg_queue'},  # 專屬synthseg_task
    'default': {'routing_key': 'default'},               # 默認隊列
    'dicom2nii_queue': {'routing_key': 'dicom2nii_queue'},    #  專屬dicom2nii_queue
    'dicom_rename_queue': {'routing_key': 'dicom_rename_queue'},  # 專屬dicom2nii_queue
}

# 在启动Celery worker时注册任务上下文
@app.on_after_configure.connect
def setup_global_context(sender, **kwargs):
    sender.conf.CELERY_CONTEXT = {}
    print('setup_global_context',sender,type(sender))


from celery.signals import worker_ready
from celery.concurrency.solo import TaskPool


@worker_ready.connect
def configure_environment(sender, **kwargs):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    print('configure_environment', sender, type(sender),sender.hostname)
    if isinstance(sender,Consumer) and isinstance(sender.pool,TaskPool) and sender.hostname.startswith('worker1'):
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        if gpus:
            try:
                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(device=gpu, enable=True)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            except RuntimeError as e:
                print(e)
        # # 動態調整GPU記憶體用量
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # # 如果使用Keras的話，就設定TensorFlow Session
        # tf.compat.v1.keras.backend.set_session(sess)
        # gpu = sender.app.conf.CELERY_CONTEXT.get('gpu')
        # if gpu is None:
        #     sender.app.conf.CELERY_CONTEXT['gpu'] = gpus[0]
        #     print('CELERY_CONTEXT',sender.app.conf.CELERY_CONTEXT)


# @worker_ready.connect
# def configure_environment(sender, **kwargs):
#     import tensorflow as tf
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     print(gpus)
#     if gpus:
#         try:
#             # tf.config.experimental.set_virtual_device_configuration(
#             #     gpus[0],
#             #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
#             # )
#             tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
#         except RuntimeError as e:
#             print(e)
#
#     print('configure_environment', sender, type(sender),sender.hostname)
#     if isinstance(sender,Consumer) and isinstance(sender.pool,TaskPool) and sender.hostname == 'worker1@DESKTOP-TPJC1AV':
#         print('sender.pool',sender.pool)
#         print('sender.app.conf.CELERY_CONTEXT')
#         print(sender.app.conf.CELERY_CONTEXT)
#         model = sender.app.conf.CELERY_CONTEXT.get('synth_seg')
#         if model is None:
#             model = SynthSeg()
#             sender.app.conf.CELERY_CONTEXT['synth_seg'] = model
#             print('CELERY_CONTEXT',sender.app.conf.CELERY_CONTEXT)

