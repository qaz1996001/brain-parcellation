import pathlib

import pandas as pd

if __name__ == '__main__':
    from code_ai.task.task_pipeline import task_subprocess_inference
    df = pd.read_csv('/data/10TB/sean/monai_3D_latent/SPADEAutoencoderKL_more_dataset/code/dataset_t1_t2.csv')
    print(df.columns)
    data_list = []
    data_list.extend(df['t1_path'].to_list())
    data_list.extend(df['t2_path'].to_list())
    PYTHONPATH = "/data/10TB/sean/brain-parcellation"
    PYTHON3 = "/home/seanho/anaconda3/envs/tf_2_14/bin/python3"
    for data in data_list[1:]:
        Inputs = pathlib.Path(data.replace('/data/a_dataset','/data/10TB/sean/a_dataset'))
        ID     = Inputs.parent.name
        Output_folder = Inputs.parent.parent

        func_params = {'cmd_str':  "export PYTHONPATH={PYTHONPATH} && {PYTHON3}"
                                   " code_ai/pipeline/pipeline_synthseg_tensorflow.py"
                                   " --ID {ID} --Inputs {Inputs}"
                                   " --Output_folder {Output_folder}"
                                   " --InputsDicomDir {InputsDicomDir}".format(PYTHONPATH=PYTHONPATH,
                                                                               PYTHON3 = PYTHON3,
                                                                               ID = ID,
                                                                               Inputs=str(Inputs),
                                                                               Output_folder = str(Output_folder),
                                                                               InputsDicomDir=str(Output_folder)
                                                                               )}
        result = task_subprocess_inference.push(func_params)
        print(result)