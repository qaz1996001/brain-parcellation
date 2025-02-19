import shutil
import subprocess
from typing import List
import pathlib
import nibabel as nib
import numpy as np
import gc
from celery import Celery, group, chain,chord

from celery import shared_task
from kombu import Connection, Exchange, Queue, Message, Producer
from kombu.exceptions import NotBoundError
from . import TemplateProcessor, task_CMB
from . import CMBProcess,DWIProcess,run_wmh,run_with_WhiteMatterParcellation
from . import resample_one,resampleSynthSEG2original
from . import app,RABBITMQ_URL,LOCK_NAME,SYNTHSEG_INFERENCE_URL,TIME_OUT,COUNTDOWN,MAX_RETRIES

from code_ai.utils_inference import check_study_id,check_study_mapping_inference, generate_output_files
from code_ai.utils_inference import get_synthseg_args_file,replace_suffix
from code_ai.utils_inference import Analysis,InferenceEnum,Dataset,Task,Result
import bentoml



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
@app.task(bind=True,rate_limit='30/s',acks_late=True,priority=50)
def synthseg_task(self, resample_file, synthseg_file, synthseg33_file):
    """
        限制同時只能執行一個的任務。
        """
    # try:
    #     import bentoml
    #     if acquire_lock():
    #         with bentoml.SyncHTTPClient(SYNTHSEG_INFERENCE_URL, timeout=TIME_OUT) as client:
    #             client.synthseg_classify(path_images=str(resample_file),
    #                                      path_segmentations=str(synthseg_file),
    #                                      path_segmentations33=str(synthseg33_file))
    #
    #         release_lock()
    #     return synthseg_file, synthseg33_file
    # except :
    #     release_lock()
    #     self.retry(countdown=COUNTDOWN//2, max_retries=MAX_RETRIES)  # 重試任務
    # finally:
    #     print('finally', 'release_lock')
    #     release_lock()
    try:
        with bentoml.SyncHTTPClient(SYNTHSEG_INFERENCE_URL,
                                    timeout=TIME_OUT) as client:
                client.synthseg_classify(path_images=str(resample_file),
                                         path_segmentations=str(synthseg_file),
                                         path_segmentations33=str(synthseg33_file))
        return synthseg_file, synthseg33_file
    except :
        self.retry(countdown=COUNTDOWN//2, max_retries=MAX_RETRIES)  # 重試任務




@app.task
def process_synthseg_task(synthseg_file_tuple, depth_number, david_file, wm_file):

    synthseg_file = synthseg_file_tuple[0]
    synthseg33_file = synthseg_file_tuple[1]
    synthseg_nii = nib.load(synthseg_file)
    synthseg33_nii = nib.load(synthseg33_file)

    synthseg_array = np.array(synthseg_nii.dataobj)
    synthseg33_array = np.array(synthseg33_nii.dataobj)
    seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
        synthseg_array, synthseg33_array, depth_number)
    print('200')
    out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, david_file)
    out_nib = nib.Nifti1Image(synthseg_array_wm, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, wm_file)
    print('300')
    gc.collect()
    return synthseg_file,david_file

@app.task
def post_process_synthseg_task(args,intput_args, ):
    print('post_process_synthseg_task',args)
    print('intput_args', intput_args)
    if intput_args.cmb:
        original_cmb_file_list: List[pathlib.Path] = list(map(lambda x:x.parent.joinpath(f"synthseg_{x.name.replace('resample', 'original')}"),intput_args.cmb_file_list))
        if original_cmb_file_list[0].name .startswith('synthseg_SWAN'):
            swan_file = original_cmb_file_list[0]
            t1_file = original_cmb_file_list[1]
        else:
            swan_file = original_cmb_file_list[1]
            t1_file = original_cmb_file_list[0]

        template_basename: pathlib.Path = replace_suffix(t1_file.name, '')
        synthseg_basename: str = replace_suffix(swan_file.name, '')
        template_coregistration_file_name = swan_file.parent.joinpath(f'{synthseg_basename}_from_{template_basename}')
        cmd_str = TemplateProcessor.flirt_cmd_base.format(t1_file,swan_file,template_coregistration_file_name)
        # apply_cmd_str = TemplateProcessor.flirt_cmd_apply.format(t1_file, swan_file, template_coregistration_file_name)
        process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    gc.collect()
    return intput_args



@app.task(bind=True,rate_limit='5/m',priority=20,acks_late=True)
def resample_to_original_task(self,original_file, resample_image_file, resample_seg_file):
    original_seg_file = resampleSynthSEG2original(original_file, resample_image_file, resample_seg_file)
    outpput_raw_file = resample_seg_file.parent.joinpath(original_file.name)
    print('original_file', original_file)
    print('outpput_raw_file', outpput_raw_file)
    if str(original_file) == str(outpput_raw_file):
        return original_seg_file
    else:
        with open(original_file,mode='rb') as raw_file_f:
            with open(outpput_raw_file,mode='wb') as outpput_raw_file_f:
                shutil.copyfileobj(raw_file_f, outpput_raw_file_f)
        return original_seg_file



@app.task(acks_late=True)
def save_file_tasks(synthseg_file,david_file,wm_file,
                    depth_number,
                    save_mode,save_file_path):
    synthseg_nii = nib.load(synthseg_file)
    david_nii = nib.load(david_file)
    seg_array = np.array(david_nii.dataobj)
    affine = synthseg_nii.affine
    header = synthseg_nii.header
    match save_mode:
        case InferenceEnum.CMB:
            result_array = CMBProcess.run(seg_array)

        case InferenceEnum.WMH_PVS:
            synthseg_nii = nib.load(synthseg_file)
            synthseg_wm_nii = nib.load(wm_file)
            synthseg_array = np.array(synthseg_nii.dataobj)
            synthseg_array_wm = np.array(synthseg_wm_nii.dataobj)
            affine = synthseg_nii.affine
            header = synthseg_nii.header
            result_array = run_wmh(synthseg_array, synthseg_array_wm, depth_number)
        case InferenceEnum.DWI:
            result_array = DWIProcess.run(seg_array)
        case _ :
            result_array = None
    if result_array is not None:
        out_nib = nib.Nifti1Image(result_array, affine, header)
        nib.save(out_nib, save_file_path)



def build_save_file_tasks(intput_args, index):
    synthseg_file = intput_args.synthseg_file_list[index]
    david_file = intput_args.david_file_list[index]
    file = intput_args.intput_file_list[index]
    resample_file = intput_args.resample_file_list[index]
    depth_number = intput_args.depth_number or 5
    wm_file: pathlib.Path = intput_args.wm_file_list[index]
    job_list = []

    # 添加 CMB 保存任务
    if intput_args.cmb:
        cmb_file = intput_args.cmb_file_list[index]
        # save_file_tasks(synthseg_file,david_file,save_mode,save_file_path)
        temp_task = chain(save_file_tasks.si(synthseg_file=synthseg_file,david_file=david_file, wm_file=wm_file,
                                             depth_number=depth_number,
                                             save_mode=InferenceEnum.CMB, save_file_path=cmb_file),
                          resample_to_original_task.si(original_file=file,
                                                       resample_image_file=resample_file,
                                                       resample_seg_file=cmb_file))
        job = temp_task
        job_list.append(job)

    # 添加 DWI 保存任务
    if intput_args.dwi:
        dwi_file = intput_args.dwi_file_list[index]
        temp_task = chain(save_file_tasks.si(synthseg_file=synthseg_file,david_file=david_file, wm_file=wm_file,
                                             depth_number=depth_number,
                                             save_mode=InferenceEnum.DWI, save_file_path=dwi_file),
                          resample_to_original_task.si(original_file=file,
                                                       resample_image_file=resample_file,
                                                       resample_seg_file=dwi_file))
        job = temp_task
        job_list.append(job)

    # 添加 WMH 保存任务
    if intput_args.wmh:
        wmh_file = intput_args.wmh_file_list[index]
        temp_task = chain(save_file_tasks.si(synthseg_file=synthseg_file,david_file=david_file, wm_file=wm_file,
                                             depth_number=depth_number,
                                             save_mode=InferenceEnum.WMH_PVS,save_file_path=wmh_file),
                          resample_to_original_task.si(original_file=file,
                                                       resample_image_file=resample_file,
                                                       resample_seg_file=wmh_file))
        job = temp_task
        job_list.append(job)
    return group(job_list)




# @app.task
# def celery_workflow(args, file_list):
#     workflows = []
#     depth_number = args.depth_number or 5
#     print('args',args)
#
#     for i, file in enumerate(file_list):
#         try:
#             resample_file   :pathlib.Path = args.resample_file_list[i]
#             synthseg_file   :pathlib.Path = args.synthseg_file_list[i]
#             synthseg33_file :pathlib.Path = args.synthseg33_file_list[i]
#             david_file      :pathlib.Path = args.david_file_list[i]
#             wm_file         :pathlib.Path = args.wm_file_list[i]
#             if all([synthseg_file.exists(),synthseg33_file.exists(),david_file.exists(),wm_file.exists()]):
#                 continue
#
#             workflow = chain(resample_task.s(file, resample_file),
#                              synthseg_task.s(synthseg_file, synthseg33_file),
#                              process_synthseg_task.s(depth_number=depth_number,
#                                                      david_file=david_file,
#                                                      wm_file=wm_file),
#                              save_file_tasks.s(intput_args=args, index=i),
#                              post_process_synthseg_task.s(intput_args=args))
#             workflows.append(workflow)
#         except Exception as e:
#             pass
#     job = group(workflows).delay()
#     return job

@app.task(acks_late=True)
def celery_workflow(inference_name, file_dict):
    args, file_list = get_synthseg_args_file(inference_name, file_dict)
    print(f'task_synthseg celery_workflow inference_name {inference_name}')
    print(f'task_synthseg celery_workflow file_dict {file_dict}')
    print(f'task_synthseg celery_workflow args {args}')
    print(f'task_synthseg celery_workflow file_list {file_list}')
    output_path_list = file_dict['output_path_list']
    depth_number = args.depth_number or 5
    job_list = []
    for i, file in enumerate(file_list):
        resample_file   :pathlib.Path = args.resample_file_list[i]
        synthseg_file   :pathlib.Path = args.synthseg_file_list[i]
        synthseg33_file :pathlib.Path = args.synthseg33_file_list[i]
        david_file      :pathlib.Path = args.david_file_list[i]
        wm_file         :pathlib.Path = args.wm_file_list[i]
        output_path_file:pathlib.Path = pathlib.Path(output_path_list[0])
        if all([synthseg_file.exists(),synthseg33_file.exists(),david_file.exists(),wm_file.exists(),output_path_file.exists()]):
            continue

        temp_task = chain(resample_task.s(file, resample_file),
                          synthseg_task.s(synthseg_file, synthseg33_file),
                          process_synthseg_task.s(depth_number=depth_number,
                                                  david_file=david_file,
                                                  wm_file=wm_file),
                          build_save_file_tasks(intput_args=args, index=i),
                          post_process_synthseg_task.s(intput_args=args))
        # job = temp_task.apply_async()
        job = temp_task
        job_list.append(job)
    print('task_synthseg job_list',job_list)
    return chain(*job_list)


def build_synthseg(inference_name, file_dict):
    args, file_list = get_synthseg_args_file(inference_name, file_dict)
    output_path_list = file_dict['output_path_list']
    depth_number = args.depth_number or 5
    job_list = []
    for i, file in enumerate(file_list):
        resample_file   :pathlib.Path = args.resample_file_list[i]
        synthseg_file   :pathlib.Path = args.synthseg_file_list[i]
        synthseg33_file :pathlib.Path = args.synthseg33_file_list[i]
        david_file      :pathlib.Path = args.david_file_list[i]
        wm_file         :pathlib.Path = args.wm_file_list[i]
        output_path_file:pathlib.Path = pathlib.Path(output_path_list[0])
        if all([synthseg_file.exists(),synthseg33_file.exists(),david_file.exists(),wm_file.exists(),output_path_file.exists()]):
            continue
        temp_task = chain(resample_task.s(file, resample_file),
                          synthseg_task.s(synthseg_file, synthseg33_file),
                          process_synthseg_task.s(depth_number=depth_number,
                                                  david_file=david_file,
                                                  wm_file=wm_file),
                          build_save_file_tasks(intput_args=args, index=i),
                          post_process_synthseg_task.s(intput_args=args))
        # job = temp_task.apply_async()
        job = temp_task
        job_list.append(job)
    return chain(*job_list)