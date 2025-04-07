import pathlib
from typing import Dict
from funboost import BrokerEnum, ConcurrentModeEnum, Booster
from code_ai.task.schema import intput_params
from code_ai.utils_inference import Dataset, Analysis, check_study_mapping_inference, generate_output_files, Task, \
    InferenceEnum


# def task_inference(intput_args, output_inference: pathlib.Path):
#     # intput_args = (<GroupResult: b59006e6-7998-40a4-ba4f-94114bf14fd1 []>, PosixPath('/mnt/e/rename_nifti_0219/00647793_20190307_MR_20803070001'))
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
#                 )
#         analyses[str(*study_id)] = Analysis(**tasks)
#     dataset = Dataset(analyses=analyses)
#     mapping_inference_data = dataset.model_dump()
#     job_list = []
#     for study_id, mapping_inference in mapping_inference_data['analyses'].items():
#         for inference_name, file_dict in mapping_inference.items():
#
#
#             if file_dict is None:
#                 continue
#             match inference_name:
#                 case InferenceEnum.CMB:
#                     temp_task = chain( task_synthseg.build_synthseg(inference_name, file_dict),
#                                       task_CMB.inference_cmb.si(intput_args=dataset.model_dump_json()))
#                     job_list.append(temp_task)
#                 case InferenceEnum.WMH:
#                     temp_task = chain(task_synthseg.build_synthseg(InferenceEnum.WMH_PVS, mapping_inference[InferenceEnum.WMH_PVS]),
#                                       task_WMH.inference_wmh.si(intput_args=dataset.model_dump_json()))
#                     job_list.append(temp_task)
#                 case InferenceEnum.Infarct:
#                     task_chain = chain(task_synthseg.build_synthseg(InferenceEnum.DWI, mapping_inference[InferenceEnum.DWI]),
#                                        task_infarct.inference_infarct.si(intput_args=dataset.model_dump_json()))
#                     job_list.append(task_chain)
#                 case InferenceEnum.Area:
#                     cmb_dict = file_dict.get(InferenceEnum.CMB)
#                     area_dict = file_dict.get(InferenceEnum.Area)
#                     if (cmb_dict is not None) and (area_dict is not None):
#                         continue
#                     else:
#                         temp_task = task_synthseg.build_synthseg(inference_name, file_dict)
#                         job = temp_task
#                         job_list.append(job)
#                 case InferenceEnum.Aneurysm:
#                     temp_task = task_synthseg.build_synthseg(inference_name, file_dict)
#                     job = temp_task
#                     job_list.append(job)
#                 # case _:
#                 #     temp_task = chain(task_synthseg.celery_workflow.s(inference_name, file_dict))
#     print('task_inference job_list',job_list)
#     # return chain(*job_list)
#     return group(job_list).apply_async()

@Booster('task_inference_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         concurrent_num=10,
         is_send_consumer_hearbeat_to_redis=True)
def task_inference(func_params  : Dict[str,any]):
    task_params = intput_params.TaskInferenceParams.model_validate(func_params,
                                                                   strict=False)
    input_study_nifti_path  = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    print('input_study_nifti_path ', input_study_nifti_path)
    print('output_study_nifti_path', output_study_nifti_path)
    study_list = [input_study_nifti_path]
    # output_inference_path = pathlib.Path(output_inference)
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    analyses = {}
    for mapping_inference in mapping_inference_list:
        study_id = mapping_inference.keys()
        model_dict_values = mapping_inference.values()
        base_output_path = str(output_study_nifti_path.joinpath(*study_id))
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

    print('dataset', dataset)

    mapping_inference_data = dataset.model_dump()
    for study_id, mapping_inference in mapping_inference_data['analyses'].items():
        for inference_name, file_dict in mapping_inference.items():

            if file_dict is None:
                continue
            match inference_name:
                case InferenceEnum.CMB:
                    temp_task = chain( task_synthseg.build_synthseg(inference_name, file_dict),
                                      task_CMB.inference_cmb.si(intput_args=dataset.model_dump_json()))
                    job_list.append(temp_task)
                case InferenceEnum.WMH:
                    temp_task = chain(task_synthseg.build_synthseg(InferenceEnum.WMH_PVS, mapping_inference[InferenceEnum.WMH_PVS]),
                                      task_WMH.inference_wmh.si(intput_args=dataset.model_dump_json()))
                    job_list.append(temp_task)
                case InferenceEnum.Infarct:
                    task_chain = chain(task_synthseg.build_synthseg(InferenceEnum.DWI, mapping_inference[InferenceEnum.DWI]),
                                       task_infarct.inference_infarct.si(intput_args=dataset.model_dump_json()))
                    job_list.append(task_chain)
                case InferenceEnum.Area:
                    cmb_dict = file_dict.get(InferenceEnum.CMB)
                    area_dict = file_dict.get(InferenceEnum.Area)
                    if (cmb_dict is not None) and (area_dict is not None):
                        continue
                    else:
                        temp_task = task_synthseg.build_synthseg(inference_name, file_dict)
                case InferenceEnum.Aneurysm:
                    temp_task = task_synthseg.build_synthseg(inference_name, file_dict)

                # case _:
                #     temp_task = chain(task_synthseg.celery_workflow.s(inference_name, file_dict))
    return chain(*job_list)
