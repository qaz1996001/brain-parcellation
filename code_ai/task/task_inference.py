import pathlib
from typing import Dict
from funboost import BrokerEnum, ConcurrentModeEnum, Booster
from code_ai.task.schema import intput_params
from code_ai.utils_inference import Dataset, Analysis, check_study_mapping_inference, generate_output_files, Task, \
    InferenceEnum
from code_ai.task.task_infarct import inference_infarct
from code_ai.task.task_CMB import inference_cmb
from code_ai.task.workflow import ResampleHandler
from code_ai.task.workflow import SynthSegHandler
from code_ai.task.workflow import ProcessSynthSegHandler
from code_ai.task.workflow import SaveFileTaskHandler
from code_ai.task.workflow import PostProcessSynthSegHandler
from code_ai.task.workflow import ResampleToOriginalHandler


@Booster('call_handler_inference_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=2,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         concurrent_num=4,
         is_send_consumer_hearbeat_to_redis=True,
         is_using_rpc_mode=True)
def call_handler_inference(func_params  : Dict[str,any]):
    resample_handler = ResampleHandler()
    synthseg_handler = SynthSegHandler()
    process_synthseg_handler = ProcessSynthSegHandler()
    save_file_tasks_handler  =  SaveFileTaskHandler()
    # post_process_synthseg_handler = PostProcessSynthSegHandler()
    resample_to_original_handler = ResampleToOriginalHandler()
    resample_handler.set_next(synthseg_handler)
    synthseg_handler.set_next(process_synthseg_handler)
    process_synthseg_handler.set_next(save_file_tasks_handler)
    save_file_tasks_handler.set_next(resample_to_original_handler)
    # post_process_synthseg_handler.set_next(resample_to_original_handler)

    resample_handler.handle(func_params)
    return func_params


@Booster('task_inference_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=2,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         concurrent_num=5,
         is_send_consumer_hearbeat_to_redis=True,
         is_using_rpc_mode=True)
def task_inference(func_params  : Dict[str,any]):
    task_params = intput_params.TaskInferenceParams.model_validate(func_params,
                                                                   strict=False)
    input_study_nifti_path  = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    #study_list = [input_study_nifti_path]
    study_list = sorted(input_study_nifti_path.iterdir())
    print('study_list',study_list)
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    call_handler_inference_task = []
    print('mapping_inference_list', mapping_inference_list)

    for mapping_inference in mapping_inference_list:
        study_id = mapping_inference.keys()
        model_dict_values = mapping_inference.values()
        base_output_path = output_study_nifti_path.joinpath(*study_id)
        base_output_path_str = str(base_output_path)
        print('base_output_path', base_output_path_str)
        task_list = []
        for task_dict in model_dict_values:
            tasks = {}
            for model_name, input_paths in task_dict.items():
                task_output_files = generate_output_files(input_paths, model_name, base_output_path_str)
                task =Task(intput_path_list=input_paths,
                           output_path=base_output_path_str,
                           output_path_list=task_output_files,)
                tasks_dump =  {model_name:task.model_dump()}
                tasks[model_name] = task
                # synthseg
                task_params_list = ResampleHandler.generate_save_file_params(tasks_dump,)
                for task_params in task_params_list:
                    call_handler_inference_task.append(call_handler_inference.push(task_params.get_str_dict()))
                    task_list.append((model_name,task_params))
                # synthseg

        # analyses[str(*study_id)] = Analysis(**tasks)
        task_result_list = [temp_task.get() for temp_task in call_handler_inference_task]
        cmb_list = list(filter(lambda x:x[0] == InferenceEnum.CMB,task_list))
        if len(cmb_list) > 0:
            cmd_save_file_path_list = list(map(lambda x:x[1].save_file_path,cmb_list))

            post_process_synthseg_params = intput_params.PostProcessSynthsegTaskParams(save_mode=cmb_list[0][1].save_mode,
                                                                                       cmb_file_list=cmd_save_file_path_list)
            post_process_synthseg_handler = PostProcessSynthSegHandler()
            post_process_synthseg_handler.handle(post_process_synthseg_params.model_dump())
        analyses = Analysis(study_id=base_output_path.name,
                            **tasks)
        print('analyses',analyses)
        infarct_result = inference_infarct.push(analyses.model_dump())
        print('infarct_result',infarct_result)
        cmb_result = inference_cmb.push(analyses.model_dump())
        print('cmb_result', cmb_result)