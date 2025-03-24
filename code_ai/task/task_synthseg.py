import os.path
import shutil
import subprocess
from typing import List, Dict
import pathlib
import nibabel as nib
import numpy as np
import gc

import bentoml
from funboost import boost, BrokerEnum, BoosterParams, Booster

from code_ai.task import TemplateProcessor
from code_ai.task import CMBProcess, DWIProcess, run_wmh, run_with_WhiteMatterParcellation
from code_ai.task import resample_one, resampleSynthSEG2original
from code_ai.task import app, RABBITMQ_URL, LOCK_NAME, SYNTHSEG_INFERENCE_URL, TIME_OUT, COUNTDOWN, MAX_RETRIES

from code_ai.utils_inference import get_synthseg_args_file, replace_suffix
from code_ai.utils_inference import InferenceEnum

from code_ai.task.schema import intput_params

# 定義 Funboost 任務
@Booster('resample_task_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
# def resample_task(file, resample_file):
def resample_task(func_params  : Dict[str,any]):
    #
    task_params = intput_params.ResampleTaskParams.model_validate(func_params)
    file = task_params.file
    resample_file = task_params.resample_file
    if not resample_file.parent.exists():
        resample_file.parent.mkdir(parents=True, exist_ok=True)
    if not resample_file.exists():
        resample_one(str(file), str(resample_file))
    return str(resample_file)


@Booster('synthseg_task_queue', broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
def synthseg_task(func_params  : Dict[str,any]):
    task_params     = intput_params.SynthsegTaskParams.model_validate(func_params)
    resample_file   = task_params.resample_file
    synthseg_file   = task_params.synthseg_file
    synthseg33_file = task_params.synthseg33_file

    try:
        if all([resample_file.exists(), synthseg_file.exists(), synthseg33_file.exists()]):
            pass
        else:
            print('SYNTHSEG_INFERENCE_URL',SYNTHSEG_INFERENCE_URL)
            with bentoml.SyncHTTPClient(SYNTHSEG_INFERENCE_URL, timeout=TIME_OUT) as client:
                client.synthseg_classify(path_images=str(resample_file),
                                         path_segmentations=str(synthseg_file),
                                         path_segmentations33=str(synthseg33_file))
        return synthseg_file, synthseg33_file
    except Exception as e:
        print(e)
        raise e  # Funboost 會自動處理重試

@Booster('process_synthseg_task_queue', broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
# def process_synthseg_task(synthseg_file_tuple, depth_number, david_file, wm_file):
def process_synthseg_task(func_params  : Dict[str,any]):
    task_params     = intput_params.ProcessSynthsegTaskParams.model_validate(func_params)
    synthseg_file   = task_params.synthseg_file
    synthseg33_file = task_params.synthseg33_file
    david_file      = task_params.david_file
    wm_file         = task_params.wm_file
    depth_number    = task_params.depth_number

    if os.path.exists(wm_file) and os.path.exists(david_file):
        return synthseg_file, david_file
    else:
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
        return synthseg_file, david_file

@Booster('post_process_synthseg_task_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
def post_process_synthseg_task(args, intput_args):
    if intput_args.cmb:
        original_cmb_file_list: List[pathlib.Path] = list(map(lambda x: x.parent.joinpath(f"synthseg_{x.name.replace('resample', 'original')}"), intput_args.cmb_file_list))
        if original_cmb_file_list[0].name.startswith('synthseg_SWAN'):
            swan_file = original_cmb_file_list[0]
            t1_file = original_cmb_file_list[1]
        else:
            swan_file = original_cmb_file_list[1]
            t1_file = original_cmb_file_list[0]

        template_basename: pathlib.Path = replace_suffix(t1_file.name, '')
        synthseg_basename: str = replace_suffix(swan_file.name, '')
        template_coregistration_file_name = swan_file.parent.joinpath(f'{synthseg_basename}_from_{template_basename}')
        cmd_str = TemplateProcessor.flirt_cmd_base.format(t1_file, swan_file, template_coregistration_file_name)
        process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    gc.collect()
    return intput_args

@Booster('resample_to_original_task_queue', broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
def resample_to_original_task(original_file, resample_image_file, resample_seg_file):
    original_seg_file = resampleSynthSEG2original(original_file, resample_image_file, resample_seg_file)
    outpput_raw_file = resample_seg_file.parent.joinpath(original_file.name)
    if str(original_file) == str(outpput_raw_file):
        return original_seg_file
    else:
        with open(original_file, mode='rb') as raw_file_f:
            with open(outpput_raw_file, mode='wb') as outpput_raw_file_f:
                shutil.copyfileobj(raw_file_f, outpput_raw_file_f)
        return original_seg_file

@Booster('save_file_tasks_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
def save_file_tasks(synthseg_file, david_file, wm_file, depth_number, save_mode, save_file_path):
    synthseg_nii = nib.load(synthseg_file)
    david_nii = nib.load(david_file)
    seg_array = np.array(david_nii.dataobj)
    affine = synthseg_nii.affine
    header = synthseg_nii.header
    match save_mode:
        case InferenceEnum.CMB:
            result_array = CMBProcess.run(seg_array)
        case InferenceEnum.WMH_PVS:
            synthseg_wm_nii = nib.load(wm_file)
            synthseg_array_wm = np.array(synthseg_wm_nii.dataobj)
            result_array = run_wmh(np.array(synthseg_nii.dataobj), synthseg_array_wm, depth_number)
        case InferenceEnum.DWI:
            result_array = DWIProcess.run(seg_array)
        case _:
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

    if intput_args.wm_file:
        job_list.append(resample_to_original_task(original_file=file, resample_image_file=resample_file, resample_seg_file=david_file))
        job_list.append(resample_to_original_task(original_file=file, resample_image_file=resample_file, resample_seg_file=wm_file))

    if intput_args.cmb:
        cmb_file = intput_args.cmb_file_list[index]
        job_list.append(save_file_tasks(synthseg_file=synthseg_file, david_file=david_file, wm_file=wm_file,
                                       depth_number=depth_number, save_mode=InferenceEnum.CMB, save_file_path=cmb_file))
        job_list.append(resample_to_original_task(original_file=file, resample_image_file=resample_file, resample_seg_file=cmb_file))

    if intput_args.dwi:
        dwi_file = intput_args.dwi_file_list[index]
        job_list.append(save_file_tasks(synthseg_file=synthseg_file, david_file=david_file, wm_file=wm_file,
                                       depth_number=depth_number, save_mode=InferenceEnum.DWI, save_file_path=dwi_file))
        job_list.append(resample_to_original_task(original_file=file, resample_image_file=resample_file, resample_seg_file=dwi_file))

    if intput_args.wmh:
        wmh_file = intput_args.wmh_file_list[index]
        job_list.append(save_file_tasks(synthseg_file=synthseg_file, david_file=david_file, wm_file=wm_file,
                                       depth_number=depth_number, save_mode=InferenceEnum.WMH_PVS, save_file_path=wmh_file))
        job_list.append(resample_to_original_task(original_file=file, resample_image_file=resample_file, resample_seg_file=wmh_file))

    return job_list


def funboost_workflow(inference_name, file_dict):
    args, file_list = get_synthseg_args_file(inference_name, file_dict)
    output_path_list = file_dict['output_path_list']
    depth_number = args.depth_number or 5
    job_list = []
    for i, file in enumerate(file_list):
        resample_file: pathlib.Path = args.resample_file_list[i]
        synthseg_file: pathlib.Path = args.synthseg_file_list[i]
        synthseg33_file: pathlib.Path = args.synthseg33_file_list[i]
        david_file: pathlib.Path = args.david_file_list[i]
        wm_file: pathlib.Path = args.wm_file_list[i]
        output_path_file: pathlib.Path = pathlib.Path(output_path_list[0])
        if all([synthseg_file.exists(), synthseg33_file.exists(), david_file.exists(), wm_file.exists(), output_path_file.exists()]):
            continue

        resample_task(file, resample_file)
        synthseg_task(resample_file, synthseg_file, synthseg33_file)
        process_synthseg_task((synthseg_file, synthseg33_file), depth_number, david_file, wm_file)
        for job in build_save_file_tasks(args, i):
            job()
        post_process_synthseg_task(args, args)

    return job_list


def build_synthseg(inference_name, file_dict):
    args, file_list = get_synthseg_args_file(inference_name, file_dict)
    output_path_list = file_dict['output_path_list']
    depth_number = args.depth_number or 5
    job_list = []
    for i, file in enumerate(file_list):
        resample_file: pathlib.Path = args.resample_file_list[i]
        synthseg_file: pathlib.Path = args.synthseg_file_list[i]
        synthseg33_file: pathlib.Path = args.synthseg33_file_list[i]
        david_file: pathlib.Path = args.david_file_list[i]
        wm_file: pathlib.Path = args.wm_file_list[i]
        output_path_file: pathlib.Path = pathlib.Path(output_path_list[0])
        if all([synthseg_file.exists(), synthseg33_file.exists(), david_file.exists(), wm_file.exists(), output_path_file.exists()]):
            continue

        resample_task(file, resample_file)
        synthseg_task(resample_file, synthseg_file, synthseg33_file)
        process_synthseg_task((synthseg_file, synthseg33_file), depth_number, david_file, wm_file)
        for job in build_save_file_tasks(args, i):
            job()
        post_process_synthseg_task(args, args)

    return job_list