from celery import Celery, group, chain,chord

import pathlib
import nibabel as nib
import numpy as np
import gc
import os
import traceback
from . import SynthSeg
from . import CMBProcess,DWIProcess,run_wmh,run_with_WhiteMatterParcellation
from . import resample_one,resampleSynthSEG2original
from . import app


from celery import shared_task
from kombu import Connection, Exchange, Queue, Message, Producer
from kombu.exceptions import NotBoundError

# RabbitMQ 鎖配置
RABBITMQ_URL = "amqp://guest:guest@localhost:5672//"

LOCK_NAME = "synthseg_task_lock"



def acquire_lock():
    """
    Attempt to acquire a RabbitMQ lock. Returns True if successful, otherwise False.
    """
    try:
        with Connection(RABBITMQ_URL) as conn:
            with conn.channel() as channel:
                # Define exchange and queue
                lock_exchange = Exchange(LOCK_NAME, type="direct", durable=True)
                lock_queue = Queue(LOCK_NAME, exchange=lock_exchange, routing_key=LOCK_NAME, durable=True)

                # Declare exchange and queue
                lock_exchange(channel).declare()
                lock_queue(channel).declare()

                bound_queue = lock_queue(channel)

                # Check if lock already exists
                if bound_queue.get() is None:  # If the queue is empty
                    # Use a producer to put a message into the queue
                    producer = Producer(channel)
                    producer.publish(
                        "lock",
                        exchange=lock_exchange,
                        routing_key=LOCK_NAME,
                        declare=[bound_queue],
                        serializer="json"
                    )
                    return True
                else:
                    return False  # Lock already exists
    except NotBoundError as e:
        print(f"Error in acquiring lock: {e}")
        return False


def release_lock():
    try:
        with Connection(RABBITMQ_URL) as conn:
            with conn.channel() as channel:
                # Define exchange and queue
                lock_exchange = Exchange(LOCK_NAME, type="direct", durable=True)
                lock_queue = Queue(LOCK_NAME, exchange=lock_exchange, routing_key=LOCK_NAME, durable=True)

                # Declare exchange and queue
                lock_exchange(channel).declare()
                lock_queue(channel).declare()

                bound_queue = lock_queue(channel)
                bound_queue.purge(channel)
    except NotBoundError as e:
        print(f"Error in releasing lock: {e}")



@app.task(acks_late=True)
@shared_task
def resample_task(file, resample_file):
    if not resample_file.parent.exists():
        resample_file.parent.mkdir(parents=True, exist_ok=True)
    resample_one(str(file), str(resample_file))
    return str(resample_file)


# @app.task(bind=True,rate_limit='1/m')
@app.task(bind=True,rate_limit='30/s',priority=50)
def synthseg_task(self, resample_file, synthseg_file, synthseg33_file):
    """
        限制同時只能執行一個的任務。
        """
    if acquire_lock():
        try:
            synth_seg = app.conf.CELERY_CONTEXT['synth_seg']
            synth_seg.run(path_images=str(resample_file), path_segmentations=str(synthseg_file),
                          path_segmentations33=str(synthseg33_file))
            gc.collect()
            release_lock()
            print('return release_lock', 'release_lock')
            return synthseg_file, synthseg33_file
        finally:
            print('finally','release_lock')
            release_lock()
    else:
        self.retry(countdown=30, max_retries=10)  # 重試任務



@app.task
def process_synthseg_task(synthseg_file_tuple, depth_number, david_file,wm_file):
    synthseg_file = synthseg_file_tuple[0]
    synthseg33_file = synthseg_file_tuple[1]
    synthseg_nii = nib.load(synthseg_file)
    synthseg33_nii = nib.load(synthseg33_file)

    synthseg_array = np.array(synthseg_nii.dataobj)
    synthseg33_array = np.array(synthseg33_nii.dataobj)

    seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
        synthseg_array, synthseg33_array, depth_number)
    out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, david_file)
    out_nib = nib.Nifti1Image(synthseg_array_wm, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, wm_file)
    gc.collect()
    return synthseg_file,david_file


@app.task
def resample_to_original_task(intpu_tuple,raw_file, resample_image_file, resample_seg_file):
    print('resample_to_original_task')
    print('raw_file', raw_file)
    print('resample_image_file', resample_image_file)
    print('resample_seg_file', resample_seg_file)
    return resampleSynthSEG2original(raw_file, resample_image_file, resample_seg_file)

@app.task
@shared_task
def log_error_task(file, error_msg):
    print(f"Error processing {file}: {error_msg}")
    return False

@app.task
@shared_task
def wm_save_task(seg_array, affine, header, wm_file):
    """保存白质分区文件"""
    out_nib = nib.Nifti1Image(seg_array, affine, header)
    nib.save(out_nib, wm_file)
    return f"Saved WM file: {wm_file}"

@app.task
@shared_task
def cmb_save_task(seg_array, affine, header, cmb_file):
    """保存CMB文件"""
    cmb_array = CMBProcess.run(seg_array)
    out_nib = nib.Nifti1Image(cmb_array, affine, header)
    nib.save(out_nib, cmb_file)
    return f"Saved CMB file: {cmb_file}"

@app.task
@shared_task
def dwi_save_task(seg_array, affine, header, dwi_file):
    """保存DWI文件"""
    dwi_array = DWIProcess.run(seg_array)
    out_nib = nib.Nifti1Image(dwi_array, affine, header)
    nib.save(out_nib, dwi_file)
    return f"Saved DWI file: {dwi_file}"

@app.task
def wmh_save_task(synthseg_array, synthseg_array_wm, affine, header, depth_number, wmh_file):
    """保存WMH文件"""
    wmh_array = run_wmh(synthseg_array, synthseg_array_wm, depth_number)
    out_nib = nib.Nifti1Image(wmh_array, affine, header)
    nib.save(out_nib, wmh_file)
    return f"Saved WMH file: {wmh_file}"


@app.task
def clear_tensorflow_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    return True

@app.task
def save_file_tasks(synthseg_david_tuple, intput_args, index):
    print('synthseg_david_tuple',synthseg_david_tuple)
    print('index',index)
    tasks = []
    synthseg_file = intput_args.synthseg_file_list[index]
    synthseg_nii = nib.load(synthseg_file)
    affine = synthseg_nii.affine
    header = synthseg_nii.header

    david_file = intput_args.david_file_list[index]
    david_nii = nib.load(david_file)
    seg_array = np.array(david_nii.dataobj)

    wm_file  = intput_args.wm_file_list[index]
    synthseg_wm_nii = nib.load(wm_file)
    synthseg_array_wm = np.array(synthseg_wm_nii.dataobj)

    file          = intput_args.intput_file_list[index]
    resample_file = intput_args.resample_file_list[index]
    # 添加 WM 保存任务
    if intput_args.wm_file:
        wm_file = intput_args.wm_file_list[index]
        tasks.append((wm_save_task.s(seg_array, affine, header, wm_file) |
                     resample_to_original_task.s(raw_file=file,
                                                 resample_image_file=resample_file,
                                                 resample_seg_file=wm_file))
                     )

    # 添加 CMB 保存任务
    if intput_args.cmb:
        cmb_file = intput_args.cmb_file_list[index]
        tasks.append((cmb_save_task.s(seg_array, affine, header,  cmb_file) |
                      resample_to_original_task.s(raw_file=file,
                                                  resample_image_file=resample_file,
                                                  resample_seg_file=cmb_file)
                      ))

    # 添加 DWI 保存任务
    if intput_args.dwi:
        dwi_file = intput_args.dwi_file_list[index]
        tasks.append((dwi_save_task.s(seg_array, affine, header, dwi_file) |
                     resample_to_original_task.s(raw_file=file,
                                                 resample_image_file=resample_file,
                                                 resample_seg_file=dwi_file)))

    # 添加 WMH 保存任务
    if intput_args.wmh:
        wmh_file = intput_args.wmh_file_list[index]
        tasks.append(
            (wmh_save_task.s(
                synthseg_array=synthseg_nii.get_fdata(),
                synthseg_array_wm=synthseg_array_wm,
                affine=affine,
                header=header,
                depth_number=intput_args.depth_number,
                wmh_file=wmh_file,)|
             resample_to_original_task.s(raw_file=file,
                                         resample_image_file=resample_file,
                                         resample_seg_file=wmh_file)
             ))

    # 组合任务并执行
    # return group(tasks).apply_async()
    return group([group(tasks),clear_tensorflow_backend.s()]).apply_async()


def build_celery_workflow(args, file_list):
    workflows = []
    depth_number = args.depth_number or 5

    for i, file in enumerate(file_list):
        try:
            resample_file = args.resample_file_list[i]
            synthseg_file = args.synthseg_file_list[i]
            synthseg33_file = args.synthseg33_file_list[i]
            david_file = args.david_file_list[i]
            wm_file = args.wm_file_list[i]

            workflow = chain(
                resample_task.s(file, resample_file),
                synthseg_task.s(synthseg_file, synthseg33_file),
                process_synthseg_task.s(depth_number=depth_number,david_file=david_file,wm_file=wm_file),
                save_file_tasks.s(intput_args=args, index=i),
            )
            workflows.append(workflow)
        except Exception as e:
            log_error_task.s(file, str(e))

    return group(workflows)

