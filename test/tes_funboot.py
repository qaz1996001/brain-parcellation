from functools import partial
from pathlib import Path
import time
from typing import Callable


def test_resample_task():
    from code_ai.task.task_synthseg import resample_task
    from code_ai.task.schema.intput_params import ResampleTaskParams
    resample_task_params = ResampleTaskParams(
        file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
        resample_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz')
    )
    print('resample_task_params', resample_task_params.get_str_dict())
    resample_task.push(resample_task_params.get_str_dict())


def test_synthseg_task():
    from code_ai.task.task_synthseg import synthseg_task
    from code_ai.task.schema.intput_params import  SynthsegTaskParams
    task_params = SynthsegTaskParams(
        file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
        resample_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz'),
        synthseg_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg.nii.gz'),
        synthseg33_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg33.nii.gz'),
    )
    synthseg_task.push(task_params.get_str_dict())



def test_process_synthseg_task():
    from code_ai.task.task_synthseg import process_synthseg_task
    from code_ai.task.schema.intput_params import ProcessSynthsegTaskParams
    task_params = ProcessSynthsegTaskParams(
        file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
        resample_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz'),
        synthseg_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg.nii.gz'),
        synthseg33_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg33.nii.gz'),
        david_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_david.nii.gz'),
        wm_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_wm.nii.gz'),

    )
    process_synthseg_task.push(task_params.get_str_dict())



def test_resample_to_original_task():
    from code_ai.task.task_synthseg import resample_to_original_task
    from code_ai.task.schema.intput_params import ResampleToOriginalTaskParams
    task_params = ResampleToOriginalTaskParams(
        original_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
        resample_image_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz'),
        resample_seg_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_wm.nii.gz')
    )
    resample_to_original_task.push(task_params.get_str_dict())



def test_save_file_tasks():
    from code_ai.task.task_synthseg import save_file_tasks
    from code_ai.task.schema.intput_params import SaveFileTaskParams
    task_params = SaveFileTaskParams(
        file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
        resample_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz'),
        synthseg_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg.nii.gz'),
        synthseg33_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg33.nii.gz'),
        david_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_david.nii.gz'),
        wm_file=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_wm.nii.gz'),
        save_mode='CMB',
        save_file_path=Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_WMH_PVS.nii.gz'),
    )
    save_file_tasks.push(task_params.get_str_dict())



def call_chain(status_and_result: dict,func:Callable, *args, **kwargs):
    return func(status_and_result['msg_dict']['func_params'])


def test_post_process_synthseg_task():
    from code_ai.task.schema.intput_params import PostProcessSynthsegTaskParams
    from code_ai.task.task_synthseg import save_file_tasks, post_process_synthseg_task,resample_task,synthseg_task,process_synthseg_task
    from code_ai.task.schema.intput_params import SaveFileTaskParams
    # task_params = SaveFileTask(
    #     file=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
    #     resample_file=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz'),
    #     synthseg_file=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg.nii.gz'),
    #     synthseg33_file=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg33.nii.gz'),
    #     david_file=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_david.nii.gz'),
    #     wm_file=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_wm.nii.gz'),
    #     save_mode='CMB',
    #     save_file_path=Path(
    #         '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_CMB.nii.gz'),
    # )
    # t1_task = save_file_tasks.push(task_params.get_str_dict())
    task_params = SaveFileTaskParams(
        file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN.nii.gz'),
        resample_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN_resample.nii.gz'),
        synthseg_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN_resample_synthseg.nii.gz'),
        synthseg33_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN_resample_synthseg33.nii.gz'),
        david_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN_resample_david.nii.gz'),
        wm_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN_resample_wm.nii.gz'),
        save_mode='CMB',
        save_file_path=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/SWAN_resample_CMB.nii.gz'),
    )
    resample_swan_task = resample_task.push(task_params.get_str_dict())
    synthseg_swan_task = None
    # temp = partial(call_chain,func = synthseg_task.push)
    # resample_swan_task.set_callback(temp)
    # swan_task = save_file_tasks.push(task_params.get_str_dict())
    while True :
        time.sleep(2)
        print('resample_swan_task id',resample_swan_task.task_id)
        if resample_swan_task.is_success():
            if synthseg_swan_task is None:
                print(resample_swan_task.status_and_result)
                synthseg_swan_task = synthseg_task.push(task_params.get_str_dict())
            else:
                print('synthseg_swan_task id', synthseg_swan_task.task_id)
                if synthseg_swan_task.is_success():
                    print(synthseg_swan_task.status_and_result)
                    return 'ok'


if __name__ == '__main__':
    test_post_process_synthseg_task()


    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)