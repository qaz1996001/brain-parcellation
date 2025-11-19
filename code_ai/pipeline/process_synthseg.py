import argparse
import gc
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from code_ai.task import run_with_WhiteMatterParcellation

    parser = argparse.ArgumentParser()
    parser.add_argument('--synthseg_file', type=str, required=True,
                        help='synthseg_file')
    parser.add_argument('--synthseg33_file', type=str, required=True,
                        help='synthseg33_file')
    parser.add_argument('--david_file', type=str, required=True,
                        help='david_file')
    parser.add_argument('--wm_file', type=str, required=True,
                        help='wm_file')
    parser.add_argument('--depth_number', type=int, default=5,
                        help='depth_number')
    args = parser.parse_args()
    synthseg_file = args.synthseg_file
    synthseg33_file = args.synthseg33_file
    david_file = args.david_file
    wm_file = args.wm_file
    depth_number = args.depth_number

    synthseg_nii = nib.load(synthseg_file)
    synthseg33_nii = nib.load(synthseg33_file)

    synthseg_array = np.array(synthseg_nii.dataobj)
    synthseg33_array = np.array(synthseg33_nii.dataobj)
    seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
        synthseg_array, synthseg33_array, depth_number)
    out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, david_file)
    out_nib = nib.Nifti1Image(synthseg_array_wm, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, wm_file)
    gc.collect()