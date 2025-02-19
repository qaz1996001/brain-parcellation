import pathlib

from celery import group, chain, shared_task

from . import app, task_CMB, task_synthseg, task_WMH, task_infarct
from .task_synthseg import build_synthseg
from ..utils_inference import Dataset, Analysis, check_study_mapping_inference, generate_output_files, Task, \
    get_synthseg_args_file, InferenceEnum


miss_inference_name = {InferenceEnum.WMH_PVS,
                       InferenceEnum.DWI,
                       }

#
# @app.task(rate_limit='30/s',acks_late=True,)
# def task_inference(intput_args, output_inference: pathlib.Path):
#     print('task_inference intput_args', intput_args)
#     print('task_inference output_inference', output_inference)
#     study_list = [intput_args[1]]
#     output_inference_path = pathlib.Path(output_inference)
#     mapping_inference_list = list(map(check_study_mapping_inference, study_list))
#     analyses = {}
#     for mapping_inference in mapping_inference_list:
#         study_id = mapping_inference.keys()
#         model_dict_values = mapping_inference.values()
#         base_output_path = str(output_inference_path.joinpath(*study_id))
#         print('base_output_path', base_output_path)
#         for task_dict in model_dict_values:
#             tasks = {}
#             for model_name, input_paths in task_dict.items():
#                 task_output_files = generate_output_files(input_paths, model_name, base_output_path)
#                 tasks[model_name] = Task(
#                     intput_path_list=input_paths,
#                     output_path=base_output_path,
#                     output_path_list=task_output_files,
#                     # result=Result(output_file=task_output_files)
#                 )
#         analyses[str(*study_id)] = Analysis(**tasks)
#     dataset = Dataset(analyses=analyses)
#     mapping_inference_data = dataset.model_dump()
#     job_list = []
#     for study_id, mapping_inference in mapping_inference_data['analyses'].items():
#         for inference_name, file_dict in mapping_inference.items():
#             if file_dict is None:
#                 continue
#             print('inference_name {} file_dict {}'.format(inference_name, file_dict))
#
#             match inference_name:
#                 case InferenceEnum.CMB:
#                     temp_task = chain(task_synthseg.celery_workflow.s(inference_name, file_dict))
#                     # temp_task.link(task_CMB.inference_cmb.s(dataset.model_dump_json()))
#                     job = temp_task.apply_async(link = task_CMB.inference_cmb.s(dataset.model_dump_json()))
#                     job_list.append(job)
#                 case InferenceEnum.WMH:
#                     temp_task = chain(task_synthseg.celery_workflow.s(InferenceEnum.WMH_PVS, mapping_inference[InferenceEnum.WMH_PVS]))
#                     # temp_task.link(task_WMH.inference_wmh.s(dataset.model_dump_json()))
#                     job = temp_task.apply_async(link = task_WMH.inference_wmh.s(dataset.model_dump_json()))
#                     job_list.append(job)
#
#                 case InferenceEnum.Infarct:
#                     temp_task = chain(task_synthseg.celery_workflow.s(InferenceEnum.DWI, mapping_inference[InferenceEnum.DWI]))
#                     # temp_task.link(task_infarct.inference_infarct.s(dataset.model_dump_json()))
#                     job = temp_task.apply_async(link = task_infarct.inference_infarct.s(dataset.model_dump_json()))
#                     job_list.append(job)
#                 case InferenceEnum.Area:
#                     cmb_dict = file_dict.get(InferenceEnum.CMB)
#                     area_dict = file_dict.get(InferenceEnum.Area)
#                     if (cmb_dict is not None) and (area_dict is not None):
#                         continue
#                     else:
#                         temp_task = chain(task_synthseg.celery_workflow.s(inference_name, file_dict))
#                         job = temp_task.apply_async()
#                         job_list.append(job)
#                 # case _:
#                 #     temp_task = chain(task_synthseg.celery_workflow.s(inference_name, file_dict))
#
#     return job_list



# @app.task(rate_limit='30/s',acks_late=True,)
@shared_task
def task_inference(intput_args, output_inference: pathlib.Path):
    # intput_args = (<GroupResult: b59006e6-7998-40a4-ba4f-94114bf14fd1 []>, PosixPath('/mnt/e/rename_nifti_0219/00647793_20190307_MR_20803070001'))
    print('task_inference intput_args', intput_args)
    print('task_inference output_inference', output_inference)
    study_list = [intput_args[1]]
    output_inference_path = pathlib.Path(output_inference)
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    analyses = {}
    for mapping_inference in mapping_inference_list:
        study_id = mapping_inference.keys()
        model_dict_values = mapping_inference.values()
        base_output_path = str(output_inference_path.joinpath(*study_id))
        print('base_output_path', base_output_path)
        for task_dict in model_dict_values:
            tasks = {}
            for model_name, input_paths in task_dict.items():
                task_output_files = generate_output_files(input_paths, model_name, base_output_path)
                tasks[model_name] = Task(
                    intput_path_list=input_paths,
                    output_path=base_output_path,
                    output_path_list=task_output_files,
                )
        analyses[str(*study_id)] = Analysis(**tasks)
    dataset = Dataset(analyses=analyses)
    mapping_inference_data = dataset.model_dump()
    job_list = []
    for study_id, mapping_inference in mapping_inference_data['analyses'].items():
        for inference_name, file_dict in mapping_inference.items():
            if file_dict is None:
                continue
            match inference_name:
                case InferenceEnum.CMB:
                    temp_task = chain( build_synthseg(inference_name, file_dict),
                                      task_CMB.inference_cmb.si(intput_args=dataset.model_dump_json()))
                    job_list.append(temp_task)
                case InferenceEnum.WMH:
                    temp_task = chain(build_synthseg(InferenceEnum.WMH_PVS, mapping_inference[InferenceEnum.WMH_PVS]),
                                      task_WMH.inference_wmh.si(intput_args=dataset.model_dump_json()))
                    job_list.append(temp_task)
                case InferenceEnum.Infarct:
                    task_chain = chain(build_synthseg(InferenceEnum.DWI, mapping_inference[InferenceEnum.DWI]),
                                       task_infarct.inference_infarct.si(intput_args=dataset.model_dump_json()))
                    job_list.append(task_chain)
                case InferenceEnum.Area:
                    cmb_dict = file_dict.get(InferenceEnum.CMB)
                    area_dict = file_dict.get(InferenceEnum.Area)
                    if (cmb_dict is not None) and (area_dict is not None):
                        continue
                    else:
                        temp_task = build_synthseg(inference_name, file_dict)
                        job = temp_task
                        job_list.append(job)
                case InferenceEnum.Aneurysm:
                    temp_task = build_synthseg(inference_name, file_dict)
                    job = temp_task
                    job_list.append(job)
                # case _:
                #     temp_task = chain(task_synthseg.celery_workflow.s(inference_name, file_dict))
    print('task_inference job_list',job_list)
    # return chain(*job_list)
    return group(job_list).apply_async()
    # return group(job_list)()


def build_task_inference(output_inference: pathlib.Path):
    return task_inference.s(output_inference)
