import argparse
import os.path
import pathlib

if __name__ == '__main__':
    from code_ai.task.task_dicom2nii import dicom_to_nii, dicom_rename
    from code_ai.task.task_pipeline import task_pipeline_inference
    from code_ai.task.schema.intput_params import Dicom2NiiParams

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', type=str,
                        help='input_dicom  /mnt/e/raw_dicom ')
    parser.add_argument('--output_dicom', type=str,
                        help='output_dicom /mnt/e/rename_dicom_0407 ')
    parser.add_argument('--output_nifti', type=str,
                        help='output_nifti /mnt/e/rename_nifti_0407')


    args = parser.parse_args()
    result_list = []
    if all((args.input_dicom, args.output_dicom, args.output_nifti)):
        input_dicom  = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)

        input_dicom_list = sorted(input_dicom.iterdir())
        input_dicom_list = list(filter(lambda x: x.is_dir(),input_dicom_list))
        if len(input_dicom_list) == 0:
            input_dicom_list = [Dicom2NiiParams(sub_dir=input_dicom,
                                                output_dicom_path=output_dicom_path,
                                                output_nifti_path=output_nifti_path, )]
        for input_dicom_path in input_dicom_list:
            task_params = Dicom2NiiParams(
                sub_dir=input_dicom_path,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path,)
            task = dicom_to_nii.push(task_params.get_str_dict())
            result_list.append(task)
    elif all((args.input_dicom, args.output_dicom)):
        input_dicom = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)

        input_dicom_list = sorted(input_dicom.iterdir())
        input_dicom_list = list(filter(lambda x: x.is_dir(), input_dicom_list))
        if len(input_dicom_list) == 0:
            input_dicom_list = [Dicom2NiiParams(sub_dir=input_dicom,
                                                output_dicom_path=output_dicom_path,
                                                output_nifti_path=None, )]
        for input_dicom_path in input_dicom_list:
            task_params = Dicom2NiiParams(
                sub_dir=input_dicom_path,
                output_dicom_path=output_dicom_path,
                output_nifti_path=None, )
            task = dicom_to_nii.push(task_params.get_str_dict())
            result_list.append(task)
    elif all((args.output_dicom, args.output_nifti)):
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)

        input_dicom_list = sorted(output_dicom_path.iterdir())
        input_dicom_list = list(filter(lambda x: x.is_dir(), input_dicom_list))
        if len(input_dicom_list) == 0:
            input_dicom_list = [Dicom2NiiParams(sub_dir=None,
                                                output_dicom_path=output_dicom_path,
                                                output_nifti_path=output_nifti_path, )]


        for input_dicom_path in input_dicom_list:
            task_params = Dicom2NiiParams(
                sub_dir=None,
                output_dicom_path=input_dicom_path,
                output_nifti_path=output_nifti_path,)
            task = dicom_to_nii.push(task_params.get_str_dict())
            result_list.append(task)
    else:
        raise ValueError(f'input_dicom {args.input_dicom} or {args.output_dicom}')
    for async_result in result_list:
        async_result.set_timeout(3600)

    result_list = [async_result.result for async_result in result_list]
    print('result_list',result_list)
    if len(result_list) > 0:
        study_id_set = set()
        output_nifti_path = pathlib.Path(args.output_nifti)
        dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path_list = list(map(lambda x:output_nifti_path.joinpath(os.path.basename(x)),
                                          result_list))
        for nifti_study_path in output_nifti_path_list:
            dicom_study_path = dicom_path.joinpath(nifti_study_path.name)
            if dicom_study_path.name in study_id_set:
                continue
            else:
                study_id_set.add(dicom_study_path.name)
                task_pipeline_result = task_pipeline_inference.push({'nifti_study_path':str(nifti_study_path),
                                                                     'dicom_study_path':str(dicom_study_path),
                                                                     })
