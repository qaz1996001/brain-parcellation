import pathlib
from celery import Celery, chain, group, shared_task, chord
from . import app
from code_ai.task.task_inference import build_task_inference
from code_ai.task.task_dicom2nii import build_dicom2nii


@app.task(rate_limit='4/s', acks_late=True)
def celery_workflow(input_dicom_str, output_dicom_str, output_nifti_str):
    if isinstance(input_dicom_str,str):
        input_dicom_path: pathlib.Path = pathlib.Path(input_dicom_str)
    else:
        input_dicom_path = input_dicom_str
    if isinstance(output_dicom_str,str):
        output_dicom_path: pathlib.Path = pathlib.Path(output_dicom_str)
    else:
        output_dicom_path = output_dicom_str
    if isinstance(output_nifti_str,str):
        output_nifti_path: pathlib.Path = pathlib.Path(output_nifti_str)
    else:
        output_nifti_path = output_nifti_str

    job_list = []
    is_dir_flag = all(list(map(lambda x: x.is_dir(), input_dicom_path.iterdir())))
    print('is_dir_flag', is_dir_flag)
    if is_dir_flag:
        for sub_dir in list(input_dicom_path.iterdir()):
            dicom2nii_chain = build_dicom2nii(sub_dir=sub_dir,
                                              output_dicom_path=output_dicom_path,
                                              output_nifti_path=output_nifti_path)

            task_inference_chain = build_task_inference(output_inference=output_nifti_str)
            workflow = chain(chain(dicom2nii_chain),
                             chain(task_inference_chain)
                             )
            print('workflow', workflow)
            result = workflow.apply_async()
            job_list.append(result)

    else:
        for sub_dir in list(input_dicom_path.iterdir()):
            if sub_dir.is_dir():
                for sub_dir in list(input_dicom_path.iterdir()):
                    dicom2nii_chain = build_dicom2nii(sub_dir=sub_dir,
                                                      output_dicom_path=output_dicom_path,
                                                      output_nifti_path=output_nifti_path)

                    task_inference_chain = build_task_inference(output_inference=output_nifti_str)
                    workflow = chain(chain(dicom2nii_chain),
                                     chain(task_inference_chain)
                                     )
                    print('workflow', workflow)
                    result = workflow.apply_async()
                    job_list.append(result)

    return job_list



@app.task(rate_limit='4/s', acks_late=True)
def task_inference_workflow(output_nifti_str,output_inference_str):

    if isinstance(output_nifti_str,str):
        output_nifti_path: pathlib.Path = pathlib.Path(output_nifti_str)
    else:
        output_nifti_path = output_nifti_str

    if isinstance(output_nifti_str,str):
        output_inference_path: pathlib.Path = pathlib.Path(output_inference_str)
    else:
        output_inference_path = output_inference_str

    job_list = []
    is_dir_flag = all(list(map(lambda x: x.is_dir(), output_nifti_path.iterdir())))
    print('is_dir_flag', is_dir_flag)
    if is_dir_flag:
        for sub_dir in list(output_nifti_path.iterdir()):

            task_inference_chain = build_task_inference(output_inference=output_inference_path,
                                                        sub_dir=sub_dir)
            workflow = chain(
                             chain(task_inference_chain)
                             )
            print('workflow', workflow)
            result = workflow.apply_async()
            job_list.append(result)

    else:
        for sub_dir in list(output_nifti_path.iterdir()):
            if sub_dir.is_dir():
                for sub_dir in list(output_nifti_path.iterdir()):
                    task_inference_chain = build_task_inference(output_inference=output_inference_path,
                                                                sub_dir=sub_dir)
                    workflow = chain(
                        chain(task_inference_chain)
                    )
                    print('workflow', workflow)
                    result = workflow.apply_async()
                    job_list.append(result)

    return job_list