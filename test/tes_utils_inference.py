import argparse
import pathlib
import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
import code_ai.utils_inference as utils_inference



if __name__ == '__main__':
    input_path = pathlib.Path(r'/mnt/e/rename_nifti/02794664_20151006_MR_241005003')
    data = utils_inference.check_study_mapping_inference(input_path)
    print(data)

