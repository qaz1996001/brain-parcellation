import uuid
from datetime import datetime
from typing import Dict, List

from celery import Celery, Task
from celery.app.task import Context
from celery.worker.consumer import Consumer
from celery.signals import task_success, task_received
from pydantic import BaseModel

from sqlalchemy.orm import Session
from .model import TaskModel, SessionLocal

app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             include=['code_ai.task.task_CMB',
                      'code_ai.task.task_dicom2nii',
                      'code_ai.task.task_infarct',
                      'code_ai.task.task_synthseg',
                      'code_ai.task.task_WMH',
                      'code_ai.task.task_inference',
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

    'code_ai.task.task_inference.task_inference': {'queue': 'default'},

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
    from .model import engine
    sender.app.conf.CELERY_CONTEXT['engine'] = engine
    sender.app.conf.CELERY_CONTEXT['SessionLocal'] = SessionLocal






class ArgsModel(BaseModel):
    args : List[any] = []
    model_config = {
        "arbitrary_types_allowed": True
    }


@task_received.connect
def task_received_handler(sender=None, request=None, **kwargs):
    if isinstance(sender, Consumer):
        session = SessionLocal()
        with session:
            if isinstance(request.id,str):
                uid = uuid.UUID(request.id)
            else:
                uid = request.id
            task_query = session.query(TaskModel).filter(TaskModel.task_id == uid)
            task: TaskModel = task_query.first()
            if task:
                task.error_massage = str(sender)
            else:
                task = TaskModel(task_id=request.id,
                                 name=request.task_name,
                                 args=request.argsrepr,
                                 status='PENDING')
                session.add(task)
                session.commit()
                session.refresh(task)


# task_received_handler sender ['app', 'controller', 'init_callback', 'hostname', 'pid', 'pool', 'timer', 'strategies', 'conninfo', 'connection_errors', 'channel_errors', '_restart_state', '_does_info', '_limit_order', 'on_task_request', 'on_task_message', 'amqheartbeat_rate', 'disable_rate_limits', 'initial_prefetch_count', 'prefetch_multiplier', '_maximum_prefetch_restored', 'task_buckets', 'hub', 'amqheartbeat', 'loop', '_pending_operations', 'steps', 'blueprint', 'connection', 'event_dispatcher', 'heart', 'task_consumer', 'qos', 'gossip', '_mutex', 'restart_count', 'first_connection_attempt', '__module__', '__doc__', 'Strategies', 'Blueprint', '__init__', 'call_soon', 'perform_pending_operations', 'bucket_for_task', 'reset_rate_limits', '_update_prefetch_count', '_update_qos_eventually', '_limit_move_to_pool', '_schedule_bucket_request', '_limit_task', '_limit_post_eta', 'start', '_get_connection_retry_type', 'on_connection_error_before_connected', 'on_connection_error_after_connected', 'register_with_event_loop', 'shutdown', 'stop', 'on_ready', 'loop_args', 'on_decode_error', 'on_close', 'connect', 'connection_for_read', 'connection_for_write', 'ensure_connected', '_flush_events', 'on_send_event_buffered', 'add_task_queue', 'cancel_task_queue', 'apply_eta_task', '_message_report', 'on_unknown_message', 'on_unknown_task', 'on_invalid_task', 'update_strategies', 'create_task_handler', '_restore_prefetch_count_after_connection_restart', 'max_prefetch_count', '_new_prefetch_count', '__repr__', '__dict__', '__weakref__', '__new__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']




@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    session :Session = SessionLocal()
    print(f'sender {sender}')
    with session:
        task_query = session.query(TaskModel).filter(TaskModel.task_id == sender.request.id)
        task:TaskModel = task_query.first()
        if task is not None:

            task.status = 'SUCCESS'
            task.end_time_at = datetime.now()
            session.commit()
        else:
            if isinstance(sender.request, Context):
                if sender.request.args is not None:
                    args_model = ArgsModel(args = [sender.request.args])
                    task = TaskModel(task_id=sender.request.id,
                                     name=sender.name,
                                     args=args_model.model_dump_json(),
                                     status='SUCCESS',
                                     end_time_at = datetime.now())
                else:
                    task = TaskModel(task_id=sender.request.id,
                                     name=sender.name,
                                     args="",
                                     status='SUCCESS',
                                     end_time_at=datetime.now())
            session.add(task)
            session.commit()
            session.refresh(task)

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

