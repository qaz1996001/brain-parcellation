from pathlib import Path


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
        synthseg_file='/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg.nii.gz',
        synthseg33_file='/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/12292196_20200223_MR_20902230007/T1FLAIR_AXI_resample_synthseg33.nii.gz',
    )
    print('task_params', task_params.get_str_dict())
    synthseg_task.push(task_params.get_str_dict())


if __name__ == '__main__':
    test_synthseg_task()

#     D:\wsl_ubuntu\pipeline\sean\process\Deep_synthseg\12292196_20200223_MR_20902230007

# 12292196_20200223_MR_20902230007_T1FLAIR_AXI_resample_david.nii