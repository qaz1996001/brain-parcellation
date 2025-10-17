import argparse
import dataclasses
import os
import pathlib
from typing import Union, List
import numpy as np
import nibabel as nib

def main():
    file_path = pathlib.Path('/mnt/e/pipeline/test_data/01901124_20250617_MR_21404020048/Image_reslice/MIP_Pitch_pred.nii.gz')
    pred_nii = nib.load(file_path)
    pred_np  = np.array(pred_nii.dataobj)
    print('pred_np shape', pred_np.shape)
    print('np.unique(pred_np)',np.unique(pred_np))
    medians = {label: np.median(np.where(pred_np == label)[2]) * 3 - 3
               for label in [1, 2, 3]
               if np.any(pred_np == label)}
    print(medians)
    z_index = np.where(pred_np == 3)
    print('z_index',z_index)



if __name__ == '__main__':
    main()