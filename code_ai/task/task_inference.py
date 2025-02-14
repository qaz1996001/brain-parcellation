# from .task_synthseg import inference_synthseg
# from .task_CMB import inference_cmb
# from .task_infarct import
# from task_WMH import
import pathlib

from celery import group

from . import app, task_CMB, task_synthseg
from ..utils_inference import Dataset, Analysis, check_study_mapping_inference, generate_output_files, Task, \
    get_synthseg_args_file, InferenceEnum


@app.task(acks_late=True)
def task_inference(intput_args,output_inference:pathlib.Path):
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
    dataset   = Dataset(analyses=analyses)
    mapping_inference_data = dataset.model_dump()
    for study_id, mapping_inference in mapping_inference_data['analyses'].items():
        for inference_name, file_dict in mapping_inference.items():
            if file_dict is None:
                continue
            print('inference_name {} file_dict {}'.format(inference_name, file_dict))
            match inference_name:
                case InferenceEnum.CMB:
                    args, file_list = get_synthseg_args_file(inference_name, file_dict)
                    temp_task = (task_synthseg.celery_workflow.s(args, file_list) | task_CMB.inference_cmb.s(dataset.model_dump_json()))
                    workflows.append(temp_task)
    job = group(workflows).delay()
    return mapping_inference_data


