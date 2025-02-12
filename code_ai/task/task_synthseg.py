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
import tensorflow as tf
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
@app.task(bind=True,rate_limit='30/s',priority=50)
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
        template_coregistration_file_name= swan_file.parent.joinpath(f'{synthseg_basename}_from_{template_basename}')
        cmd_str = TemplateProcessor.flirt_cmd_base.format(t1_file,swan_file,template_coregistration_file_name)
        # apply_cmd_str = TemplateProcessor.flirt_cmd_apply.format(t1_file, swan_file, template_coregistration_file_name)
        process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    return intput_args



@app.task(bind=True,rate_limit='5/m',priority=20)
def resample_to_original_task(self, raw_file, resample_image_file, resample_seg_file):
    original_seg_file = resampleSynthSEG2original(raw_file, resample_image_file, resample_seg_file)
    outpput_raw_file = resample_seg_file.parent.joinpath(raw_file.name)
    print('raw_file', raw_file)
    print('outpput_raw_file', outpput_raw_file)

    if str(raw_file) == str(outpput_raw_file):
        return original_seg_file
    else:
        with open(raw_file,mode='rb') as raw_file_f:
            with open(outpput_raw_file,mode='wb') as outpput_raw_file_f:
                shutil.copyfileobj(raw_file_f, outpput_raw_file_f)
        return original_seg_file


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
    print('wm_save_task')
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
def save_file_tasks(synthseg_david_tuple, intput_args, index):
# def save_file_tasks(intput_args, index):
    print('intput_args', intput_args)
    tasks = []
    synthseg_file = intput_args.synthseg_file_list[index]
    synthseg_nii = nib.load(synthseg_file)

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
        tasks.append(wm_save_task.s(seg_array, synthseg_nii.affine, synthseg_nii.header, wm_file))
        tasks.append(resample_to_original_task.s(raw_file=file,
                                                 resample_image_file=resample_file,
                                                 resample_seg_file=wm_file))

    # 添加 CMB 保存任务
    if intput_args.cmb:
        cmb_file = intput_args.cmb_file_list[index]
        tasks.append(cmb_save_task.s(seg_array, synthseg_nii.affine, synthseg_nii.header, cmb_file))
        tasks.append(resample_to_original_task.s(raw_file=file,
                                                 resample_image_file=resample_file,
                                                 resample_seg_file=cmb_file))

    # 添加 DWI 保存任务
    if intput_args.dwi:
        dwi_file = intput_args.dwi_file_list[index]
        tasks.append(dwi_save_task.s(seg_array, synthseg_nii.affine, synthseg_nii.header, dwi_file))
        tasks.append(resample_to_original_task.s(raw_file=file,
                                                 resample_image_file=resample_file,
                                                 resample_seg_file=dwi_file))

    # 添加 WMH 保存任务
    if intput_args.wmh:
        wmh_file = intput_args.wmh_file_list[index]
        tasks.append(
            wmh_save_task.s(
                synthseg_array=synthseg_nii.get_fdata(),
                synthseg_array_wm=synthseg_array_wm,
                affine=synthseg_nii.affine,
                header=synthseg_nii.header,
                depth_number=intput_args.depth_number,
                wmh_file=wmh_file,
            )
        )
        tasks.append(resample_to_original_task.s(raw_file=file,
                                                 resample_image_file=resample_file,
                                                 resample_seg_file=wmh_file))
    # 组合任务并执行
    job = group(tasks).apply()
    return job


@shared_task(acks_late=True)
def inference_synthseg(intput_args,
                       output_inference:pathlib.Path):
    print('intput_args', intput_args)
    output_inference_path = pathlib.Path(output_inference)
    print('output_inference_path', output_inference_path)
    study_list = [intput_args[1]]
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    analyses = {}
    for mapping_inference in mapping_inference_list:
        study_id = mapping_inference.keys()
        model_dict_values = mapping_inference.values()
        base_output_path = str(output_inference_path.joinpath(*study_id))
        print('base_output_path',base_output_path)
        for task_dict in model_dict_values:
            tasks = {}
            for model_name, input_paths in task_dict.items():
                task_output_files = generate_output_files(input_paths, model_name, base_output_path)
                tasks[model_name] = Task(
                    intput_path_list = input_paths,
                    output_path      = base_output_path,
                    output_path_list = task_output_files,
                    # result=Result(output_file=task_output_files)
                )
        analyses[str(*study_id)] = Analysis(**tasks)
    workflows = []
    dataset = Dataset(analyses=analyses)
    mapping_inference_data = dataset.model_dump()
    miss_inference = {InferenceEnum.Aneurysm,
                      InferenceEnum.WMH,
                      InferenceEnum.Infarct,
                      InferenceEnum.SynthSeg,
                      InferenceEnum.CMBSynthSeg}

    for study_id, mapping_inference in mapping_inference_data['analyses'].items():
        for inference_name, file_dict in mapping_inference.items():
            if inference_name in miss_inference:
                continue
            if file_dict is None:
                continue
            args, file_list = get_synthseg_args_file(inference_name, file_dict)
            workflows.append(celery_workflow.s(args, file_list))
    job = group(workflows).apply(link=task_CMB.inference_cmb.s(dataset.model_dump_json()))
    return job,mapping_inference_data


@app.task
def celery_workflow(args, file_list):
    workflows = []
    depth_number = args.depth_number or 5
    print('args',args)

    for i, file in enumerate(file_list):
        try:
            resample_file   :pathlib.Path = args.resample_file_list[i]
            synthseg_file   :pathlib.Path = args.synthseg_file_list[i]
            synthseg33_file :pathlib.Path = args.synthseg33_file_list[i]
            david_file      :pathlib.Path = args.david_file_list[i]
            wm_file         :pathlib.Path = args.wm_file_list[i]
            if all([synthseg_file.exists(),synthseg33_file.exists(),david_file.exists(),wm_file.exists()]):
                continue

            workflow = chain(resample_task.s(file, resample_file),
                             synthseg_task.s(synthseg_file, synthseg33_file),
                             process_synthseg_task.s(depth_number=depth_number,
                                                     david_file=david_file,
                                                     wm_file=wm_file),
                             save_file_tasks.s(intput_args=args, index=i),
                             post_process_synthseg_task.s(intput_args=args))
            workflows.append(workflow)
        except Exception as e:
            log_error_task.s(file, str(e))
    job = group(workflows).delay()
    return job