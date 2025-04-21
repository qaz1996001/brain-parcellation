from pathlib import Path


def call_cmb(status_and_result: dict):
    from code_ai.task.task_CMB import inference_cmb
    import code_ai.utils_inference as utils_inference
    cmb_file_list = status_and_result['msg_dict']['func_params'].get('cmb_file_list',None)
    if (cmb_file_list is not None) and (len(cmb_file_list) > 0):
        cmb_file = cmb_file_list[0]
        study_path = Path(cmb_file).parent
        analyses = utils_inference.build_analysis(study_path =study_path)
        cmb_result = inference_cmb.push(analyses.model_dump())


def test_cmb():
    from code_ai.task.task_synthseg import post_process_synthseg_task
    from code_ai.task.workflow import ResampleHandler

    from code_ai.task.schema import intput_params
    import code_ai.utils_inference as utils_inference
    from code_ai.utils_inference import Analysis,Task,InferenceEnum
    #study_path = Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/
    study_path = Path('/mnt/d/wsl_ubuntu/pipeline/sean/example_input')
    output_study_nifti_path = study_path
    study_path_list = sorted(study_path.iterdir())
    for study_folder_path in study_path_list:
        analysis = utils_inference.build_analysis(study_path=study_folder_path)
        if analysis.CMB is not None:
            tasks_dump = {'CMB': analysis.CMB.model_dump()}
            task_params_list = ResampleHandler.generate_save_file_params(tasks_dump, )
            cmd_save_file_path_list = list(map(lambda x: x.save_file_path, task_params_list))
            if len(cmd_save_file_path_list) > 0:
                func_params = intput_params.PostProcessSynthsegTaskParams(
                    save_mode='CMB',
                    cmb_file_list=cmd_save_file_path_list)
                task_id = post_process_synthseg_task.push(func_params.get_str_dict())
                task_id.set_callback(call_cmb)


if __name__ == '__main__':
    test_cmb()
