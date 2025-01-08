from celery import Celery

app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             include=['code_ai.task.task_synthseg',
                      'code_ai.task.task_01',
                      ],
             backend='rpc://'
             )

app.config_from_object('code_ai.celery_config')
app.conf.task_default_retry_delay = 10  # Seconds delay between retries
app.conf.task_max_retries = 5  # Maximum number of retries

app.conf.task_routes = {

    'code_ai.task.task_synthseg.synthseg_task': {'queue': 'synthseg_queue'},  # 將synthseg_task指派到專屬隊列
    'code_ai.task.task_synthseg.resample_task': {'queue': 'default'},         # 默認處理其他任務
    'code_ai.task.task_synthseg.resample_to_original_task': {'queue': 'default'},
    'code_ai.task.task_synthseg.process_synthseg_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.save_file_tasks': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.log_error_task': {'queue': 'default'},
    'code_ai.task.task_synthseg.cmb_save_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.dwi_save_task': {'queue': 'synthseg_queue'},
    'code_ai.task.task_synthseg.wmh_save_task': {'queue': 'synthseg_queue'},
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




from celery.signals import worker_ready


@worker_ready.connect
def configure_environment(sender, **kwargs):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
            )
        except RuntimeError as e:
            print(e)
    # from code_ai.utils_synthseg import set_gpu
    # set_gpu('0')
