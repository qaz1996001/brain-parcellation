from functools import partial
from pathlib import Path
import time
from typing import Callable, Dict

from funboost import Booster, BrokerEnum


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
    from code_ai.task.schema.intput_params import SynthsegTaskParams
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
    from code_ai.task.task_synthseg import resample_task,synthseg_task
    from code_ai.task.schema.intput_params import SaveFileTaskParams
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


## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def test_dicom2nii():
    from code_ai.task.task_dicom2nii import dicom_to_nii,process_dir
    from code_ai.task.schema.intput_params import Dicom2NiiParams
    task_params = Dicom2NiiParams(
        sub_dir=Path(
            '/mnt/e/raw_dicom/02695350_21210300104'),
        output_dicom_path=Path(
            '/mnt/e/rename_dicom_0327'),
        output_nifti_path=Path(
            '/mnt/e/rename_nifti_0327'),
        )
    print('task_params.get_str_dict()',task_params.get_str_dict())
    task = dicom_to_nii.push(task_params.get_str_dict())


def test_task_inference():
    from code_ai.task.workflow import ResampleHandler
    from code_ai.task.workflow import SynthSegHandler
    from code_ai.task.workflow import ProcessSynthSegHandler
    from code_ai.task.workflow import PostProcessSynthSegHandler
    from code_ai.task.workflow import ResampleToOriginalHandler
    from code_ai.task.schema.intput_params import SaveFileTaskParams

    task_params = SaveFileTaskParams(
        file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/example_input/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz'),
        resample_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample.nii.gz'),
        synthseg_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg.nii.gz'),
        synthseg33_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg33.nii.gz'),
        david_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_david.nii.gz'),
        wm_file=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_wm.nii.gz'),
        save_mode='CMB',
        save_file_path=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_CMB.nii.gz'),
    )
    # -----------------------------------------------------------------------------
    # 以下示例展示如何構建整個任務流程鏈
    # -----------------------------------------------------------------------------


    # 例如，我們目前只構造一個簡單的鏈，只包含 ResampleHandler
    # 構造責任鏈：ResampleHandler -> (後續可以 set_next 其他 Handler)
    resample_handler              = ResampleHandler()
    synthseg_handler              = SynthSegHandler()
    process_synthseg_handler      = ProcessSynthSegHandler()
    post_process_synthseg_handler = PostProcessSynthSegHandler()
    resample_to_original_handler  = ResampleToOriginalHandler()
    #
    resample_handler.set_next(synthseg_handler)
    synthseg_handler.set_next(process_synthseg_handler)
    process_synthseg_handler.set_next(post_process_synthseg_handler)
    post_process_synthseg_handler.set_next(resample_to_original_handler)

    # 執行責任鏈
    final_result = resample_handler.handle(task_params.get_str_dict())
    print("Final result from chain:", final_result)


if __name__ == '__main__':
    test_task_inference()
    # test_post_process_synthseg_task()
    # test_dicom2nii()
    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)
    # export PYTHONPATH=$(pwd) && python code_ai/pipeline/dicom_to_nii.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_0328 --output_nifti /mnt/e/rename_nifti_0328