import gc
import os
import re
import sys
import pathlib
import traceback
import argparse
from typing import List, Optional, Dict, Any
import orjson


def run_model(study_key:str, model_key:str, file_list:List,output_path:pathlib.Path ):
    output_study_path = output_path.joinpath(study_key)
    if output_study_path.exists():
        pass
    else:
        output_study_path.mkdir(parents=True,exist_ok=True)
    match model_key:
        case 'Area':
            print('Area', file_list)
        case 'DWI':
            print(file_list)
        case 'WMH':
            print(file_list)
        case 'CMB':
            print(file_list)
        case 'Infarct':
            print(file_list)
        case _:
            print('******')
    print('**************')


def main(intput_json :str = r'/mnt/c/Users/tmu3090/Desktop/Task/dicom2nii/src/mapping.json',
         output_path :pathlib.Path = pathlib.Path('/mnt/e/PC_3090/data/output/PSCL_MRI'),
         ):
    # C:\Users\tmu3090\Desktop\Task\dicom2nii\src\mapping.json
    # E:\PC_3090\data\output\PSCL_MRI\08292236_20160707_MR_E42557741501
    with open(intput_json) as f:
        study_list : List[Dict[str, Any]] = orjson.loads(f.read())
    for study_dict in study_list:
        for study_key, study_model_dict in study_dict.items():
            for model_key, file_list in study_model_dict.items():
                run_model(study_key, model_key, file_list,output_path)
        break


if __name__ == '__main__':
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name BRAVO.nii
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name SWAN.nii --template_name BRAVO.nii --CMB True
    # args = parse_arguments()
    main()
