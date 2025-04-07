import argparse
import pathlib
import subprocess
from typing import List
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from code_ai.utils_synthseg import SynthSeg

    parser = argparse.ArgumentParser()
    parser.add_argument('--resample_file', type=str, required=True,
                        help='resample_file')
    parser.add_argument('--synthseg_file', type=str, required=True,
                        help='synthseg_file')
    parser.add_argument('--synthseg33_file', type=str, required=True,
                        help='synthseg33_file')

    args = parser.parse_args()
    print('args',args)
    resample_file = args.resample_file
    synthseg_file = args.synthseg_file
    synthseg33_file = args.synthseg33_file
    synth_seg = SynthSeg()
    synth_seg.run(path_images          = str(resample_file),
                  path_segmentations   = str(synthseg_file),
                  path_segmentations33 = str(synthseg33_file))