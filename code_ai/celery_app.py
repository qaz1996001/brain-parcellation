import uuid
from datetime import datetime
from typing import Dict, List

import psycopg2
import sqlalchemy
from celery import Celery, Task
from celery.app.task import Context
from celery.worker.consumer import Consumer
from celery.signals import task_success, task_received
from celery.signals import worker_ready

from pydantic import BaseModel


from sqlalchemy.orm import Session, Query
from .model import TaskModel

app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             include=['code_ai.task.task_CMB',
                      'code_ai.task.task_dicom2nii',
                      'code_ai.task.task_infarct',
                      'code_ai.task.task_inference',
                      'code_ai.task.task_synthseg',
                      'code_ai.task.task_WMH',
                      'code_ai.task.workflow',
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


    'code_ai.task.task_CMB.inference_cmb': {'queue': 'default'},
    'code_ai.task.task_infarct.inference_infarct': {'queue': 'default'},
    'code_ai.task.task_inference.task_inference': {'queue': 'default'},
    'code_ai.task.task_synthseg.inference_synthseg': {'queue': 'default'},
    'code_ai.task.task_WMH.inference_wmh': {'queue': 'default'},


    'code_ai.task.task_dicom2nii.celery_workflow':  {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.dicom_2_nii_file': {'queue': 'dicom2nii_queue'},
    'code_ai.task.task_dicom2nii.process_dir_next': {'queue': 'dicom2nii_queue'},

    'code_ai.task.task_synthseg.resample_to_original_task': {'queue': 'dicom2nii_queue'},

    'code_ai.task.task_dicom2nii.process_dir': {'queue': 'dicom_rename_queue'},
    'code_ai.task.task_dicom2nii.process_instances': {'queue': 'dicom_rename_queue'},

    'code_ai.task.workflow.celery_workflow': {'queue': 'default'},

}

app.conf.task_queues = {
    'synthseg_queue': {'routing_key': 'synthseg_queue'},  # 專屬synthseg_task
    'default': {'routing_key': 'default'},               # 默認隊列
    'dicom2nii_queue': {'routing_key': 'dicom2nii_queue'},    #  專屬dicom2nii_queue
    'dicom_rename_queue': {'routing_key': 'dicom_rename_queue'},  # 專屬dicom2nii_queue
}


# app.conf.broker_transport_options = {'visibility_timeout': 2*60*60} # 2 hours

@app.on_after_configure.connect
def setup_global_context(sender, **kwargs):
    sender.conf.CELERY_CONTEXT = {}
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    # 用create_engine對這個URL_DATABASE建立一個引擎
    engine = create_engine('postgresql+psycopg2://postgres_n:postgres_p@127.0.0.1:15433/db_name',
                           pool_recycle=3600, pool_size=10, max_overflow=10)
    # 使用sessionmaker來與資料庫建立一個對話，記得要bind=engine，這才會讓專案和資料庫連結
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, )
    sender.conf.CELERY_CONTEXT.update({"SessionLocal":SessionLocal})
    sender.conf.CELERY_CONTEXT.update({"db_engine":engine})



class ArgsModel(BaseModel):
    args : List[any] = []
    model_config = {
        "arbitrary_types_allowed": True
    }


@task_received.connect
def task_received_handler(sender=None, request=None, **kwargs):
    SessionLocal = sender.app.conf.CELERY_CONTEXT.get('SessionLocal')
    if isinstance(sender, Consumer):
        session = SessionLocal()
        with session:
            try:
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
            except Exception as e:
                session.rollback()
            finally:
                sender.app.conf.CELERY_CONTEXT.update({"SessionLocal": SessionLocal})



@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    SessionLocal = sender.app.conf.CELERY_CONTEXT.get('SessionLocal')
    session :Session = SessionLocal()
    print(f'sender {sender}')
    with session:
        try:
            task_query :Query = session.query(TaskModel).filter(TaskModel.task_id == sender.request.id)
            task: TaskModel = task_query.first()
            if task is not None:

                task.status = 'SUCCESS'
                task.end_time_at = datetime.now()
                session.commit()
            else:
                if isinstance(sender.request, Context):
                    if sender.request.args is not None:
                        args_model = ArgsModel(args=[sender.request.args])
                        task = TaskModel(task_id=sender.request.id,
                                         name=sender.name,
                                         args=args_model.model_dump_json(),
                                         status='SUCCESS',
                                         end_time_at=datetime.now())
                    else:
                        task = TaskModel(task_id=sender.request.id,
                                         name=sender.name,
                                         args="",
                                         status='SUCCESS',
                                         end_time_at=datetime.now())
                session.add(task)
                session.commit()
                session.refresh(task)
        except Exception as e:
            session.rollback()
        except sqlalchemy.exc.DatabaseError:
            session.rollback()
        except psycopg2.DatabaseError:
            session.rollback()
        finally:
            sender.app.conf.CELERY_CONTEXT.update({"SessionLocal": SessionLocal})

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

