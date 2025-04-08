#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-28 10:40
Python 3.10.13
tensorflow==2.14.0
numpy==1.26.0

@author: sean
"""
import argparse
import datetime
import itertools
import pathlib
import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy import ndimage as ndi
from skimage.morphology import ball
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops, regionprops_table

from code_ai.pipeline import MODEL_DIR


class CMBServiceTF:

    label_index_name_mapping_dict = {0: "Background",
                                     1: "CSF",
                                     301: "brainstem",
                                     102: "left cerebellum",
                                     103: "left basal ganglion",
                                     104: "left thalamus",
                                     105: "left internal capsule",
                                     106: "left external capsule",
                                     107: "left corpus callosum",
                                     108: "left DPWM",
                                     109: "left frontal",
                                     110: "left parietal",
                                     111: "left occipital",
                                     112: "left temporal",
                                     113: "left insular",
                                     114: "left cingulate",
                                     202: "right cerebellum",
                                     203: "right basal ganglion",
                                     204: "right thalamus",
                                     205: "right internal capsule",
                                     206: "right external capsule",
                                     207: "right corpus callosum",
                                     208: "right DPWM",
                                     209: "right frontal",
                                     210: "right parietal",
                                     211: "right occipital",
                                     212: "right temporal",
                                     213: "right insular",
                                     214: "right cingulate", }
    patch_size = (64, 64, 64)
    blend_mode = "gaussian"
    sigma = 1 / 8  # 0.125
    n_class = 1
    overlap = 0.5
    name_rule = 'file'

    MODEL1_PATH = os.path.join(MODEL_DIR, "2025_02_10_MP-input64-aug_rot_bc10-bz32-unet5_32-bce_dice-Adam1e3_cosine_ema")
    MODEL2_PATH =  os.path.join(MODEL_DIR, "2025_02_10_MP-norm1-input26ch2_gauss_loc-aug2-bz64-Res32x3FPN-D128x4D1-cw-Adam1E3_ema")

    def __init__(self) -> None:
        import tensorflow as tf
        self.importance_kernel = self.get_importance_kernel(self.patch_size, self.blend_mode, self.sigma)
        self.importance_map = tf.tile(tf.reshape(self.importance_kernel, shape=[1, *self.patch_size, 1]),
                                      multiples=[1, 1, 1, 1, self.n_class], )

    def cmb_classify(self, swan_path_str: str,
                     temp_path_str: str,
                     output_nii_path_str: str,
                     output_json_path_str:str) -> str:
        '''
        Classify input image to label
        '''
        swan_path = pathlib.Path(swan_path_str)
        temp_path = pathlib.Path(temp_path_str)

        image_arr, spacing, aff = self.load_volume(swan_path, dtype='float32')
        target_spacing = np.array([spacing[0], spacing[1], min(spacing)])  # isotropical z-axis only
        target_spacing = target_spacing / 2 if target_spacing[0] > 0.6 else target_spacing  # force upsampling?
        image_arr = self.resize_volume(image_arr, spacing, target_spacing, dtype='float32')
        if temp_path is not None:
            import tensorflow as tf
            model1 = tf.saved_model.load(self.MODEL1_PATH)
            model2 = tf.saved_model.load(self.MODEL2_PATH)
            seg_arr, _ = self.load_synthseg_seg(seg_path=str(temp_path),
                                                target_size=image_arr.shape)
            brain_mask = seg_arr > 0
            image_arr = self.custom_normalize_1(image_arr, brain_mask)

            output = self.sliding_window_inference(inputs=image_arr[np.newaxis, :, :, :, np.newaxis],
                                                   roi_size=self.patch_size, model=model1,
                                                   overlap=self.overlap,
                                                   n_class=self.n_class, importance_map=self.importance_map,
                                                   mask=brain_mask[np.newaxis, :, :, :, np.newaxis], )
            pred_map = output[0, :, :, :, 0].numpy() * brain_mask
            df_pred, pred_label = self.object_analysis(image_arr, pred_map, target_spacing, model2)
            self.save_nii_trio(nifti_file = swan_path, pred_label = pred_label,
                               output_path = output_nii_path_str)
            df_label = self.save_label_table(swan_path, df_pred, pred_label, seg_arr,output_json_path_str)
            gc.collect()
        return output_nii_path_str

    @classmethod
    def load_synthseg_seg(cls, seg_path='synthseg33.nii.gz', target_size=None, verbose=False):
        # Load synthseg result
        x = nib.load(seg_path)
        x = nib.as_closest_canonical(x)  # to RAS space
        spacing = list(x.header.get_zooms())
        seg_arr = x.get_fdata().astype('uint16')[::-1, ::-1, :]  # RAS to LPS orientation

        if target_size != None:
            seg_arr = cls.resize_volume(seg_arr, target_size=target_size, dtype='uint16')
            spacing = None

        return seg_arr, spacing

    @classmethod
    def load_volume(cls, path_volume, im_only=False, squeeze=True, dtype=None, LPS_coor=True):
        """
        Load volume file.
        :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
        If npz format, 1) the variable name is assumed to be 'vol_data',
        2) the volume is associated with an identity affine matrix and blank header.
        :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
        :param squeeze: (optional) whether to squeeze the volume when loading.
        :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
        The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
        :return: the volume, with corresponding affine matrix and header if im_only is False.
        """
        path_volume = str(path_volume)
        assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

        if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
            x = nib.load(path_volume)
            x = nib.as_closest_canonical(x)  # to RAS space
            if squeeze:
                volume = np.squeeze(x.get_fdata())
            else:
                volume = x.get_fdata()
            aff = x.affine
            header = x.header
            spacing = list(x.header.get_zooms())
        else:  # npz
            volume = np.load(path_volume)['vol_data']
            if squeeze:
                volume = np.squeeze(volume)
            aff = np.eye(4)
            header = nib.Nifti1Header()
            spacing = [1., 1., 1.]
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
        if LPS_coor:
            volume = volume[::-1, ::-1, :]

        if im_only:
            return volume
        else:
            return volume, spacing, aff

    @classmethod
    def resize_volume(cls, arr, spacing=None, target_spacing=None, target_size=None, dtype='float32'):
        order = 1 if 'float' in dtype else 0
        if (spacing is not None) and (target_spacing is not None):
            scale = np.array(spacing) / np.array(target_spacing)
            out_vol = ndi.zoom(arr, zoom=scale, order=order, prefilter=True, grid_mode=False)
        elif target_size is not None:
            scale = np.array(target_size) / np.array(arr.shape)
            out_vol = ndi.zoom(arr, zoom=scale, output=np.zeros(target_size, dtype=dtype), order=order, prefilter=False,
                               grid_mode=False)
        return out_vol.astype(dtype)


    @classmethod
    def custom_normalize_1(cls, volume, mask=None, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5,
                           use_positive_only=True):
        """This function linearly rescales a volume between new_min and new_max.
        :param volume: a numpy array
        :param new_min: (optional) minimum value for the rescaled image.
        :param new_max: (optional) maximum value for the rescaled image.
        :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
        where 0 = np.min
        :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
        where 100 = np.max
        :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
        :return: rescaled volume
        """
        # select intensities
        new_volume = volume.copy()
        new_volume = new_volume.astype("float32")
        if (mask is not None) and (use_positive_only):
            intensities = new_volume[mask].ravel()
            intensities = intensities[intensities > 0]
        elif mask is not None:
            intensities = new_volume[mask].ravel()
        elif use_positive_only:
            intensities = new_volume[new_volume > 0].ravel()
        else:
            intensities = new_volume.ravel()

        # define min and max intensities in original image for normalisation
        robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
        robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

        # trim values outside range
        new_volume = np.clip(new_volume, robust_min, robust_max)

        # rescale image
        if robust_min != robust_max:
            return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
        else:  # avoid dividing by zero
            return np.zeros_like(new_volume)

    @classmethod
    def get_histogram_xy(cls, vol_arr, brain_mask):
        hist, bin_edges = np.histogram(vol_arr[brain_mask].ravel(), bins=100)
        bins_mean = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(100)]
        return [bins_mean, hist]

    # @title Model1 sliding_window_inference
    # sliding_window functions (https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/nnUNet/models/sliding_window.py)
    @classmethod
    def get_window_slices(cls, image_size, roi_size, overlap, strategy):
        dim_starts = []
        for image_x, roi_x in zip(image_size, roi_size):
            interval = roi_x if roi_x == image_x else int(roi_x * (1 - overlap))
            starts = list(range(0, image_x - roi_x + 1, interval))
            if strategy == "overlap_inside" and starts[-1] + roi_x < image_x:
                starts.append(image_x - roi_x)
            dim_starts.append(starts)
        slices = [(starts + (0,), roi_size + (-1,)) for starts in itertools.product(*dim_starts)]
        batched_window_slices = [((0,) + start, (1,) + roi_size) for start, roi_size in slices]
        return batched_window_slices


    @classmethod
    def get_importance_kernel(cls, roi_size, blend_mode, sigma):
        import tensorflow as tf
        @tf.function
        def gaussian_kernel(roi_size, sigma):
            gauss = signal.windows.gaussian(roi_size[0], std=sigma * roi_size[0])
            for s in roi_size[1:]:
                gauss = np.outer(gauss, signal.windows.gaussian(s, std=sigma * s))
            gauss = np.reshape(gauss, roi_size)
            gauss = np.power(gauss, 1 / len(roi_size))
            gauss /= gauss.max()
            return tf.convert_to_tensor(gauss, dtype=tf.float32)

        if blend_mode == "constant":
            return tf.ones(roi_size, dtype=tf.float32)
        elif blend_mode == "gaussian":
            return gaussian_kernel(roi_size, sigma)
        else:
            raise ValueError(f'Invalid blend mode: {blend_mode}. Use either "constant" or "gaussian".')



    @classmethod
    def sliding_window_inference(cls, inputs, roi_size, model, overlap, n_class, importance_map,
                                 strategy="overlap_inside",
                                 mask=None, **kwargs):
        import tensorflow as tf
        @tf.function
        def run_model(x, model, importance_map, **kwargs):
            return tf.cast(model(x, **kwargs), dtype=tf.float32) * importance_map
        image_size = tuple(inputs.shape[1:-1])
        roi_size = tuple(roi_size)
        # Padding to make sure that the image size is at least roi size
        padded_image_size = tuple(max(image_size[i], roi_size[i]) for i in range(3))
        padding_size = [image_x - input_x for image_x, input_x in zip(image_size, padded_image_size)]
        paddings = [[0, 0]] + [[x // 2, x - x // 2] for x in padding_size] + [[0, 0]]
        input_padded = tf.pad(inputs, paddings)
        if mask is not None:
            mask_padded = tf.pad(tf.convert_to_tensor(mask, dtype=tf.bool), paddings)

        output_shape = (1, *padded_image_size, n_class)
        output_sum = tf.zeros(output_shape, dtype=tf.float32)
        output_weight_sum = tf.ones(output_shape, dtype=tf.float32)
        window_slices = cls.get_window_slices(padded_image_size, roi_size, overlap, strategy)

        for window_slice in window_slices:
            if (mask is None) or ((mask is not None) and (
                    tf.math.reduce_any(tf.slice(mask_padded, begin=window_slice[0], size=window_slice[1])))):
                window = tf.slice(input_padded, begin=window_slice[0], size=window_slice[1])
                pred = run_model(window, model, importance_map, **kwargs)
                padding = [
                    [start, output_size - (start + size)] for start, size, output_size in
                    zip(*window_slice, output_shape)
                ]
                padding = padding[:-1] + [[0, 0]]
                output_sum = output_sum + tf.pad(pred, padding)
                output_weight_sum = output_weight_sum + tf.pad(importance_map, padding)

        output = output_sum / output_weight_sum
        crop_slice = [slice(pad[0], pad[0] + input_x) for pad, input_x in zip(paddings, inputs.shape[:-1])]
        return output[crop_slice]

    @classmethod
    def get_watershed_label(cls, pred_map, threshold, r=4):
        pred_mask = pred_map > threshold
        # distance = ndi.distance_transform_edt(pred_mask)
        distance = pred_map
        struc = ndi.zoom(ball(r), zoom=(1, 1, 3), order=0)
        coords = peak_local_max(distance, footprint=struc, labels=pred_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=pred_mask)
        return labels

    @classmethod
    def center_crop(cls, volume, centroid, size=26):
        """ center crop a cube with desired size,
        if the target area is smaller than the desired size, it will be padded with the minimum
        input:: volume is a 3d-array
        input:: centroid is center coordinates (x, y, z)
        output:: cropped cube with desired size ex.(28, 28, 28)
        """
        w, h, d = volume.shape
        r = int(size // 2)
        x0 = int(centroid[0])
        y0 = int(centroid[1])
        z0 = int(centroid[2])
        if (r < x0 < w - r - 1) and (r < y0 < h - r - 1) and (r < z0 < d - r - 1):
            return volume[x0 - r:x0 - r + size, y0 - r:y0 - r + size, z0 - r:z0 - r + size]
        else:
            volume = np.pad(volume, ((r, r), (r, r), (r, r)), 'minimum')
            return volume[x0:x0 + size, y0:y0 + size, z0:z0 + size]

    @classmethod
    def object_analysis(cls, image_arr, pred_map, spacing, model, min_th=0.084, FP_reduction_th=0.357,
                        uncertain_th=0.5175):
        import tensorflow as tf
        @tf.function
        def gaussian_kernel(roi_size, sigma):
            gauss = signal.windows.gaussian(roi_size[0], std=sigma * roi_size[0])
            for s in roi_size[1:]:
                gauss = np.outer(gauss, signal.windows.gaussian(s, std=sigma * s))
            gauss = np.reshape(gauss, roi_size)
            gauss = np.power(gauss, 1 / len(roi_size))
            gauss /= gauss.max()
            return tf.convert_to_tensor(gauss, dtype=tf.float32)
        pred_label = label(pred_map > 0.05)
        # pred_label = get_watershed_label(pred_map, 0.05)  # separate connected labels
        gaussian = gaussian_kernel(roi_size=(26, 26, 26), sigma=1 / 8).numpy().astype('float16')
        # FP_reduction model predict object by object
        if (pred_label.max() > 0) and (FP_reduction_th > 0):
            # FP_reduction model predict object by object
            props = regionprops(pred_label)
            TP_conf = []
            for p in props:
                x0, y0, z0, x1, y1, z1 = p.bbox
                center = [(x0 + x1) // 2, (y0 + y1) // 2, (z0 + z1) // 2]
                # crop_image = center_crop(image_arr, center).astype('float16')
                crop_image = cls.center_crop(image_arr, center)
                x = np.stack([crop_image, gaussian], axis=-1)[np.newaxis, ...]
                TP_conf.append(model(x, training=False).numpy()[0][0])  # <-- model2 prediction
            TP_conf = np.stack(TP_conf).astype('float32') / 2  # covert to [0.0:1.0]
            pred_type = ['CMB' if x * 2 > uncertain_th else 'Uncertain' for x in TP_conf]
        else:
            TP_conf = np.ones((pred_label.max(),), dtype='float32')
            pred_type = 'other'
        # pred measurement
        props = regionprops_table(pred_label, pred_map, properties=('label', 'bbox', 'intensity_mean'))
        df_pred = pd.DataFrame({'ori_Pred_label': props['label'],
                                'Pred_diameter': ((props['bbox-3'] - props['bbox-0']) * spacing[0] + (
                                        props['bbox-4'] - props['bbox-1']) * spacing[1]) / 2,
                                'Pred_mean': props['intensity_mean'] / 0.6,
                                'TP_conf': TP_conf, 'CMB_prob': (props['intensity_mean'] / 0.6 + TP_conf) / 2,
                                'Pred_type': pred_type})
        # sort and false-positive filtering
        df_pred = df_pred[(df_pred['CMB_prob'] > min_th)]  # filter too small CMB_prob
        df_pred = df_pred.sort_values(by='CMB_prob', ascending=False)
        df_pred['Pred_label'] = np.arange(1, df_pred.shape[0] + 1)
        df_pred.loc[df_pred['CMB_prob'] < FP_reduction_th, 'Pred_type'] = 'other'
        # remap pred_label array
        new_pred_label = np.zeros_like(pred_label)
        for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
            new_pred_label[pred_label == k] = v
        return df_pred, new_pred_label

    @classmethod
    def save_nii_trio(cls, nifti_file, pred_label,output_path = 'Pred.nii.gz',
                       verbose=False):
        # name output folder

        ## load original image
        img = nib.load(nifti_file)
        img = nib.as_closest_canonical(img)  # to RAS space
        aff = img.affine
        hdr = img.header
        spacing = tuple(img.header.get_zooms())
        shape = tuple(img.header.get_data_shape())

        ## CMB label-map
        pred_label = cls.resize_volume(pred_label, target_size=shape, dtype='uint16')
        # build NifTi1 image
        pred_label = pred_label[::-1, ::-1, :]  # LPS to RAS orientation
        x = nib.nifti1.Nifti1Image(pred_label, affine=aff)
        label_arr = x.get_fdata().astype('uint16')
        nib.save(x, output_path)
        if verbose:
            print(
                f"size={x.header.get_data_shape()} spacing={x.header.get_zooms()}  {label_arr.dtype}:{label_arr.min()}-{label_arr.max()}")
            print("affine matrix =\n", x.affine)
    @classmethod
    def save_label_table(cls, nifti_file, df_pred, label_arr, seg_arr, output_json_path_str,):
        def get_location(lb):
            counts = np.bincount(seg_arr[label_arr == lb].ravel())
            mode = np.argmax(counts)
            return mode

        # write position
        cols = ['label#', 'class_name',
                'review_class_name', 'type', 'type_name', 'LPS_coordinates', 'pred_diameter',
                # 'cube_shape','mask_in_cube','image_size',
                'CAD_method', 'complete', 'Doctor', 'c-time', 'Reviewer', 'r-time']
        df_tsv = pd.DataFrame(columns=cols)
        for props in regionprops(label_arr):
            lb                                = props.label
            df_tsv.loc[lb, 'label#']          = lb
            df_tsv.loc[lb, 'class_name']      = df_pred.loc[df_pred['Pred_label'] == lb, 'Pred_type'].values[0]
            label_index                       = get_location(lb)
            df_tsv.loc[lb, 'type']            = f"C{label_index}"  # location
            df_tsv.loc[lb, 'type_name']       = cls.label_index_name_mapping_dict.get(label_index, '')
            df_tsv.loc[lb, 'LPS_coordinates'] = ",".join([str(x) for x in props.bbox])
            df_tsv.loc[lb, 'pred_diameter']   = df_pred.loc[df_pred['Pred_label'] == lb, 'Pred_diameter'].values[0]
            # df_tsv.loc[lb, 'cube_shape']    = ",".join([str(x) for x in props.image.shape])
            # df_tsv.loc[lb, 'mask_in_cube']  = ",".join(['1' if x > 0 else '0' for x in props.image.ravel()])
        # df_tsv['image_size']                = ",".join([str(x) for x in label_arr.shape])
        df_tsv['CAD_method'] = "AI_v3"
        df_tsv['Doctor']     = "AI"
        df_tsv['c-time']     = datetime.datetime.now().strftime('%Y/%m/%d %H:%M')

        with open(output_json_path_str, 'w') as f:
            df_tsv.to_json(path_or_buf = f,
                           orient='records',
                           index=False)

        # df_tsv.to_csv(f"{out_root}/{case_name}_label.tsv", index=False, sep='\t')
        return df_tsv




if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    # print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--swan_path_str', type=str,
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_input/12292196_20200223_MR_20902230007/SWAN.nii.gz',
                        help='用於輸入的檔案')
    parser.add_argument('--temp_path_str', type=str,
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_input/12292196_20200223_MR_20902230007/synthseg_SWAN_original_CMB_from_synthseg_T1FLAIR_AXI_original_CMB.nii.gz',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--output_nii_path_str', type=str,
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_input/12292196_20200223_MR_20902230007/Pred_CMB.nii.gz',
                        help='用於輸出的檔案')
    parser.add_argument('--output_json_path_str', type=str,
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_input/12292196_20200223_MR_20902230007/Pred_CMB.json',
                        help='用於輸出的檔案')

    args = parser.parse_args()
    cmb_pipeline = CMBServiceTF()
    cmb_pipeline.cmb_classify(swan_path_str=args.swan_path_str,
                              temp_path_str=args.temp_path_str,
                              output_nii_path_str=args.output_nii_path_str,
                              output_json_path_str=args.output_json_path_str)
