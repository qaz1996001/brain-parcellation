"""
Brain Parcellation

Example：
-------- Command Line
    # Command Line  1 is use all methods
    # The files  are output to input_path.
        python utils_parcellation.py -i input_path
    # ------------------------
    #  - forler
    #  -- A01_synthseg.nii.gz
    #  -- A02_synthseg.nii.gz
    #  -- A03_synthseg.nii.gz
    # ------------------------
    #  - forler
    #  -- A01_synthseg.nii.gz
    #  -- A01_synthseg_david.nii.gz
    #  -- A01_synthseg_CMB.nii.gz
    #  -- A01_synthseg_WMH.nii.gz
    #  -- A01_synthseg_DWI.nii.gz
    #  -- A02_synthseg.nii.gz
    #  -- A02_synthseg_david.nii.gz
    #  -- A02_synthseg_CMB.nii.gz
    #  -- A02_synthseg_WMH.nii.gz
    #  -- A02_synthseg_DWI.nii.gz
    #  -- A03_synthseg.nii.gz
    #  -- A03_synthseg_david.nii.gz
    #  -- A03_synthseg_CMB.nii.gz
    #  -- A03_synthseg_WMH.nii.gz
    #  -- A03_synthseg_DWI.nii.gz
    # ------------------------

    # Command Line  2 is use all methods
    # The files are output to output_path.
        python utils_parcellation.py -i input_path -o output_path

    # Command Line  3 is use optional methods
    # This command line is use CMB methods
        # The files  are output to input_path.
        #         python utils_parcellation.py -i input_path --all False --cmb True
        # The files are output to output_path.
            python utils_parcellation.py -i input_path -o output_path --all False --cmb True

    # Command Line  4 are match input_path with file_name
        # This command line is use dwi methods
        # ------------------------
        #  -forler
        #  -- A01_resampled.nii.gz
        #  -- A01_synthseg.nii.gz
        #  -- A02_resampled.nii.gz
        #  -- A02_synthseg.nii.gz
        #  -- A03_resampled.nii.gz
        #  -- A03_synthseg.nii.gz
        # ------------------------
        # use this command then select the A01_synthseg.nii.gz,A02_synthseg.nii.gz,A03_synthseg.nii.gz
            # use dwi methods
            #   python utils_parcellation.py -i input_path --input_name synthseg --all False --dwi True
            # use all methods
            #   python utils_parcellation.py -i input_path --input_name synthseg --all True
        # --------------------------------------
            # The files are output to output_path.
                python utils_parcellation.py -i input_path -o output_path --input_name synthseg --all False --dwi True

-------- SHH_Seg
    import utils_parcellation
    import nibabel as nib
    synthseg_nii = nib.load(file_path)
    synthseg_array = synthseg_nii.get_fdata()
    out_array = utils_parcellation.run(synthseg_array=synthseg_array)

-------- SHH_Seg for wmh
    import utils_parcellation
    import nibabel as nib
    synthseg_nii = nib.load(file_path)
    synthseg_array = synthseg_nii.get_fdata()
    out_array = utils_parcellation.run_wmh(synthseg_array

    =synthseg_array)

-------- SHH_Seg for cmb
    import utils_parcellation
    import nibabel as nib
    synthseg_nii = nib.load(file_path)
    synthseg_array = synthseg_nii.get_fdata()
    out_array = utils_parcellation.run_cmb(synthseg_array=synthseg_array)

-------- SHH_Seg for dwi
    import utils_parcellation
    import nibabel as nib
    synthseg_nii = nib.load(file_path)
    synthseg_array = synthseg_nii.get_fdata()
    out_array = utils_parcellation.run_dwi(synthseg_array=synthseg_array)

"""

import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import argparse

from skimage import measure
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure, \
    distance_transform_edt
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    except RuntimeError as e:
        print(e)


def pairwise_distance(a, b):
    """
    Compute pairwise distance between two sets of points.

    Args:
        a: First set of points.
        b: Second set of points.

    Returns:
        tf.Tensor: Pairwise distance matrix.
    """
    return tf.reduce_sum(tf.pow(tf.abs(a - b), 2), axis=-1)


def loss_distance(index_sub, label_index_sub, decimal_places=8):
    """
    Compute loss distance between two sets of points.

    Args:
        index_sub: Subset of points.
        label_index_sub: Subset of label points.
        decimal_places (int, optional): Number of decimal places.

    Returns:
        tf.Tensor: Loss distance.
    """
    XA = tf.constant(index_sub, dtype=tf.float64)
    XB = tf.constant(label_index_sub, dtype=tf.float64)
    # 假設 index_sub 和 label_index_sub 為 numpy 或 TensorFlow 張量
    # 在維度 1 上增加一個維度
    XA = tf.expand_dims(XA, axis=1)
    # 在維度 0 上增加一個維度
    XB = tf.expand_dims(XB, axis=0)
    # 廣播機制：將 XA 和 XB 的大小擴展為相同的形狀
    XA = tf.tile(XA, multiples=[1, XB.shape[1], 1])
    XB = tf.tile(XB, multiples=[XA.shape[0], 1, 1])
    # 計算點之間的整數距離
    distances = pairwise_distance(XA, XB)
    decimals = 5
    distances_float = tf.round(distances * 10 ** decimals) / (10 ** decimals)
    loss_min = tf.reduce_min(distances_float, axis=1)
    return loss_min



class WhiteMatterParcellation:
    synseg_label_freesurfer_GM_mapping = {
        'left_hemi': {
            1003: 1001,
            1032: 1001,
            1012: 1001,
            1014: 1001,
            1018: 1001,
            1019: 1001,
            1020: 1001,
            1024: 1001,
            1027: 1001,
            1028: 1001,
            1017: 1001,
            1008: 1006,
            1022: 1006,
            1025: 1006,
            1029: 1006,
            1031: 1006,
            1005: 1004,
            1011: 1004,
            1013: 1004,
            1021: 1004,
            1001: 1005,
            1006: 1005,
            1007: 1005,
            1009: 1005,
            1015: 1005,
            1016: 1005,
            1030: 1005,
            1033: 1005,
            1034: 1005,
            1002: 1003,
            1010: 1003,
            1023: 1003,
            1026: 1003,
            1035: 1007,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
            17: 1005,
            18: None, },
        'right_hemi': {
            2003: 2001,
            2012: 2001,
            2014: 2001,
            2018: 2001,
            2019: 2001,
            2020: 2001,
            2024: 2001,
            2027: 2001,
            2028: 2001,
            2032: 2001,
            2017: 2001,
            2008: 2006,
            2022: 2006,
            2025: 2006,
            2029: 2006,
            2031: 2006,
            2005: 2004,
            2011: 2004,
            2013: 2004,
            2021: 2004,
            2001: 2005,
            2006: 2005,
            2007: 2005,
            2009: 2005,
            2015: 2005,
            2016: 2005,
            2030: 2005,
            2033: 2005,
            2034: 2005,
            2002: 2003,
            2010: 2003,
            2023: 2003,
            2026: 2003,
            2035: 2007,
            49: 49,
            50: 50,
            51: 51,
            52: 52,
            53: 2005,
            54: None
        },
    }
    white_matter_mapping = {
        'left_hemi': {1001: 3001,
                      1003: 3003,
                      1004: 3004,
                      1005: 3005,
                      1006: 3006,
                      1007: 3007,
                      12: 3007,
                      13: 3007
                      },
        'right_hemi': {2001: 4001,
                       2003: 4003,
                       2004: 4004,
                       2005: 4005,
                       2006: 4006,
                       2007: 4007,
                       51: 4007,
                       52: 4007
                       }
    }
    cerebral_white_matter = {
        'left_hemi': 2,
        'right_hemi': 41,
    }

    synseg_label_david_mapping = {
        'left_hemi': {
            1003: 112,
            1032: 112,
            1012: 112,
            1014: 112,
            1018: 112,
            1019: 112,
            1020: 112,
            1024: 112,
            1027: 112,
            1028: 112,
            1017: 117,
            1008: 113,
            1022: 113,
            1025: 113,
            1029: 113,
            1031: 113,
            1005: 114,
            1011: 114,
            1013: 114,
            1021: 114,
            1001: 115,
            1006: 115,
            1007: 115,
            1009: 115,
            1015: 115,
            1016: 115,
            1030: 115,
            1033: 115,
            1034: 115,
            1002: 116,
            1010: 116,
            1023: 116,
            1026: 116,
            1035: 118,
            10: 108,
            11: 109,
            12: 110,
            13: 111,
            17: 115,
            18: 115,

        },
        'right_hemi': {
            2003: 212,
            2012: 212,
            2014: 212,
            2018: 212,
            2019: 212,
            2020: 212,
            2024: 212,
            2027: 212,
            2028: 212,
            2032: 212,
            2017: 217,
            2008: 213,
            2022: 213,
            2025: 213,
            2029: 213,
            2031: 213,
            2005: 214,
            2011: 214,
            2013: 214,
            2021: 214,
            2001: 215,
            2006: 215,
            2007: 215,
            2009: 215,
            2015: 215,
            2016: 215,
            2030: 215,
            2033: 215,
            2034: 215,
            2002: 216,
            2010: 216,
            2023: 216,
            2026: 216,
            2035: 218,
            49: 208,
            50: 209,
            51: 210,
            52: 211,
            53: 215,
            54: 215,

        }
    }

    white_matter_david_mapping = {
        'left_hemi': {3001: 119,
                      3006: 120,
                      3004: 121,
                      3005: 122,
                      3003: 123,
                      3007: 124,
                      3008: 125,
                      3009: 126,
                      3010: 127,
                      3011: 128,
                      3031: 129,
                      3036: 130,
                      3034: 131,
                      3035: 132,
                      },
        'right_hemi': {4001: 219,
                       4006: 220,
                       4004: 221,
                       4005: 222,
                       4003: 223,
                       4007: 224,
                       4008: 225,
                       4009: 226,
                       4010: 227,
                       4011: 228,
                       4031: 229,
                       4036: 230,
                       4034: 231,
                       4035: 232,
                       }
    }

    gray_matter_david_mapping = {
        'left_hemi': {1001: 112,
                      1006: 113,
                      1004: 114,
                      1005: 115,
                      1003: 116,
                      1007: 118,
                      10: 108,
                      11: 109,
                      12: 110,
                      13: 111,
                      17: 115,
                      18: 115,
                      3: 101,
                      4: 102,
                      5: 102,
                      7: 104,
                      8: 105,
                      26: 106,
                      28: 103,
                      },
        'right_hemi': {2001: 212,
                       2006: 213,
                       2004: 214,
                       2005: 215,
                       2003: 216,
                       2007: 218,
                       49: 208,
                       50: 209,
                       51: 210,
                       52: 211,
                       53: 215,
                       54: 215,
                       42: 201,
                       43: 202,
                       44: 202,
                       46: 204,
                       47: 205,
                       58: 206,
                       60: 203,
                       }
    }

    decimal_places = 8
    scaling_factor = 10 ** decimal_places

    re_white_matter_mapping = {
        'left_hemi': {1001: 3001,
                      1004: 3004,
                      1005: 3005,
                      1006: 3006,
                      3001: 3001,
                      3004: 3004,
                      3005: 3005,
                      3006: 3006,
                      },
        'right_hemi': {2001: 4001,
                       2004: 4004,
                       2005: 4005,
                       2006: 4006,
                       4001: 4001,
                       4004: 4004,
                       4005: 4005,
                       4006: 4006,
                       }
    }

    @classmethod
    def label_to_david_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.gray_matter_david_mapping:
            # run every synseg label to freesurfer label
            for key in cls.gray_matter_david_mapping[hemi]:
                if cls.gray_matter_david_mapping[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.gray_matter_david_mapping[hemi][key]
                # left hemi and right hemi
        for hemi in cls.white_matter_david_mapping:
            # run every synseg label to freesurfer label
            for key in cls.white_matter_david_mapping[hemi]:
                if cls.white_matter_david_mapping[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.white_matter_david_mapping[hemi][key]
        return new_label_array

    @classmethod
    def synseg_label_to_freesurfer_GM(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = label_array.copy()
        # left hemi and right hemi
        for hemi in cls.synseg_label_freesurfer_GM_mapping:
            # run every synseg label to freesurfer label
            for key in cls.synseg_label_freesurfer_GM_mapping[hemi]:
                if cls.synseg_label_freesurfer_GM_mapping[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.synseg_label_freesurfer_GM_mapping[hemi][key]
        return new_label_array

    @classmethod
    def white_matter_parcellation(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = label_array.copy()

        for k in cls.cerebral_white_matter:
            # 2 or 41
            index = np.argwhere(label_array == cls.cerebral_white_matter[k])
            white_matter_mapping_keys = list(cls.white_matter_mapping[k].keys())
            np_loss = np.zeros((index.shape[0], len(white_matter_mapping_keys)))
            np_loss[:, :] = 999999
            for i in range(len(white_matter_mapping_keys)):
                # 3001 ... 4007
                label_index = np.argwhere(label_array == white_matter_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
                # Z 軸切片
                for j in np.unique(label_index[:, 2]):
                    index_sub = index[index[:, 2] == j]
                    index_sub_arg = np.argwhere(index[:, 2] == j)
                    label_index_sub = label_index[label_index[:, 2] == j]
                    if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                        loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                        loss_list.append(loss_min.numpy())
                        index_sub_arg_list.append(index_sub_arg)
                    else:
                        continue
                if len(index_sub_arg_list) > 0:
                    index_sub_arg = np.concatenate(index_sub_arg_list)
                    np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
                else:
                    continue
            new_label = np_loss.argmin(axis=1)
            # 指定分類
            for i in np.unique(new_label):
                select_index = index[np.argwhere(new_label == i)].squeeze()
                new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                    cls.white_matter_mapping[k][white_matter_mapping_keys[i]]
        return new_label_array

    @classmethod
    def re_run(cls, synthseg_array_wm, synthseg_array_cc, synthseg_array_ec):
        out_array = synthseg_array_wm.copy()
        for k in CorpusCallosumParcellation.cc_white_matter:
            cc_target = CorpusCallosumParcellation.cc_target[k]
            base_mask = (out_array == CorpusCallosumParcellation.cc_white_matter[k])
            target_mask = np.isin(synthseg_array_cc, cc_target)
            diff_mask = np.logical_and(base_mask,
                                       np.logical_not(target_mask))
            out_array[diff_mask] = WhiteMatterParcellation.cerebral_white_matter[k]
        for k in ECICParcellation.ec_ic_white_matter:
            ec_ic_target_list = list(set(ECICParcellation.ec_ic_parcellation_mapping[k].values()))
            base_mask = (out_array == ECICParcellation.ec_ic_white_matter[k])
            target_mask = np.isin(synthseg_array_ec, ec_ic_target_list)
            diff_mask = np.logical_and(base_mask,
                                       np.logical_not(target_mask))
            out_array[diff_mask] = WhiteMatterParcellation.cerebral_white_matter[k]
        return out_array

    @classmethod
    def re_white_matter_parcellation(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = label_array.copy()
        for k in cls.cerebral_white_matter:
            # 2 or 41
            index = np.argwhere(label_array == cls.cerebral_white_matter[k])
            white_matter_mapping_keys = list(cls.re_white_matter_mapping[k].keys())
            np_loss = np.zeros((index.shape[0], len(white_matter_mapping_keys)))
            np_loss[:, :] = 999999
            for i in range(len(white_matter_mapping_keys)):
                # 3001 ... 4007
                label_index = np.argwhere(label_array == white_matter_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
                # Z 軸切片
                for j in np.unique(label_index[:, 2]):
                    index_sub = index[index[:, 2] == j]
                    index_sub_arg = np.argwhere(index[:, 2] == j)
                    label_index_sub = label_index[label_index[:, 2] == j]
                    if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                        loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                        loss_list.append(loss_min.numpy())
                        index_sub_arg_list.append(index_sub_arg)
                    else:
                        continue
                if len(index_sub_arg_list) > 0:
                    index_sub_arg = np.concatenate(index_sub_arg_list)
                    np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
                else:
                    continue
            new_label = np_loss.argmin(axis=1)
            # 指定分類
            for i in np.unique(new_label):
                select_index = index[np.argwhere(new_label == i)].squeeze()
                if select_index.ndim == 2:
                    new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                        cls.re_white_matter_mapping[k][white_matter_mapping_keys[i]]
                else:
                    new_label_array[select_index[0], select_index[1], select_index[2]] = \
                        cls.re_white_matter_mapping[k][white_matter_mapping_keys[i]]
        return new_label_array

    @classmethod
    def run(cls, synthseg_array):
        synthseg_freesurfer_array = cls.synseg_label_to_freesurfer_GM(synthseg_array)
        synthseg_freesurfer_array_wm = cls.white_matter_parcellation(synthseg_freesurfer_array)
        return synthseg_freesurfer_array_wm


class WhiteMatterParcellation2(WhiteMatterParcellation):

    synthseg_left_label = np.array([2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28,
                                    1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
                                    1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
                                    1030, 1031, 1032, 1033, 1034, 1035, ])
    synthseg_right_label = np.array([41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
                                     2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                     2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
                                     2030, 2031, 2032, 2033, 2034, 2035, ])

    synthseg33_left_label = np.array([2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28])
    synthseg33_right_label = np.array([41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])
    @classmethod
    def hemi_revise(cls, synthseg_array, synthseg33_array):
        synthseg33_array_right_mask = np.isin(synthseg33_array, cls.synthseg33_right_label)
        synthseg33_array_left_mask  = np.isin(synthseg33_array, cls.synthseg33_left_label)
        synthseg_array_right_mask   = np.isin(synthseg_array, cls.synthseg_right_label)
        synthseg_array_left_mask    = np.isin(synthseg_array, cls.synthseg_left_label)
        temp_array = synthseg_array.copy()

        temp_array[(synthseg33_array_left_mask & synthseg_array_right_mask)] = 100
        temp_array[(synthseg33_array_right_mask & synthseg_array_left_mask)] = 200
        ## 2023-10-25 debug by 16041934_20230913 case
        # temp_array[(synthseg33_array_right_mask & synthseg_array_left_mask)] = 100
        # temp_array[(synthseg33_array_left_mask & synthseg_array_right_mask)] = 200
        kk = 3
        while np.logical_or(np.any(temp_array == 100), np.any(temp_array == 200)):
            kk += 1
            window_size = kk
            half_window = window_size // 2
            temp_array_index = np.argwhere(temp_array == 100)
            for row in temp_array_index:
                center_x = row[0]
                center_y = row[1]
                center_z = row[2]
                neighborhood = temp_array[center_x - half_window:center_x + half_window + 1,
                               center_y - half_window:center_y + half_window + 1,
                               center_z - half_window:center_z + half_window + 1, ]
                neighborhood = np.where(neighborhood < 1000, np.nan, neighborhood)
                neighborhood = np.where(neighborhood > 2000, np.nan, neighborhood)
                unique, counts = np.unique(neighborhood, return_counts=True, equal_nan=False)
                if len(unique) > 0:
                    if np.isnan(unique[counts.argmax()]):
                        pass
                    else:
                        temp_array[center_x, center_y, center_z] = unique[counts.argmax()]
            temp_array_index = np.argwhere(temp_array == 200)
            for row in temp_array_index:
                center_x = row[0]
                center_y = row[1]
                center_z = row[2]
                neighborhood = temp_array[center_x - half_window:center_x + half_window + 1,
                               center_y - half_window:center_y + half_window + 1,
                               center_z - half_window:center_z + half_window + 1, ]
                neighborhood = np.where(neighborhood < 2000, np.nan, neighborhood)
                neighborhood = np.where(neighborhood > 3000, np.nan, neighborhood)
                unique, counts = np.unique(neighborhood, return_counts=True, equal_nan=False)
                if len(unique) > 0:
                    if np.isnan(unique[counts.argmax()]):
                        pass
                    else:
                        temp_array[center_x, center_y, center_z] = unique[counts.argmax()]
        return temp_array

    @classmethod
    def run(cls, synthseg_array, *args, **kwargs):
        synthseg33_array = kwargs.get('synthseg33_array')
        synthseg_array_revise = cls.hemi_revise(synthseg_array=synthseg_array,
                                                synthseg33_array=synthseg33_array)
        synthseg_freesurfer_array = cls.synseg_label_to_freesurfer_GM(synthseg_array_revise)
        synthseg_freesurfer_array_wm = cls.white_matter_parcellation(synthseg_freesurfer_array)

        return synthseg_freesurfer_array_wm


class CorpusCallosumParcellation:
    cc_prerequisite = {'left_hemi': {'start': 1003,
                                     'end': None
                                     },
                       'right_hemi': {'start': 2003,
                                      'end': None
                                      }
                       }
    lateral_ventricle = {
        'left_hemi': 4,
        'right_hemi': 43,
    }
    cc_white_matter = {
        'left_hemi': 3003,
        'right_hemi': 4003,
    }
    cc_target = {
        'left_hemi': 3010,
        'right_hemi': 4010,
    }

    @classmethod
    def get_cluster_min_max(cls, prerequisite_cluster):
        prerequisite_cluster_unique = np.unique(prerequisite_cluster)
        cluster_x_list = []
        cluster_y_list = []
        for i in prerequisite_cluster_unique:
            if i == 0:
                continue
            else:
                y_max = np.argwhere(prerequisite_cluster == i)[:, 0].max()
                y_min = np.argwhere(prerequisite_cluster == i)[:, 0].min()
                x_max = np.argwhere(prerequisite_cluster == i)[:, 1].max()
                x_min = np.argwhere(prerequisite_cluster == i)[:, 1].min()
                cluster_x_list.append(x_min)
                cluster_x_list.append(x_max)
                cluster_y_list.append(y_min)
                cluster_y_list.append(y_max)
        return sorted(cluster_x_list), sorted(cluster_y_list)

    @classmethod
    def get_prerequisite_cluster(cls, measure_np):
        prerequisite_cluster = measure.label(measure_np, connectivity=2)
        prerequisite_cluster_unique = np.unique(prerequisite_cluster)
        prerequisite_cluster_index_count = np.sum(prerequisite_cluster != 0)
        prerequisite_cluster_list = []
        if prerequisite_cluster_unique.shape[0] > 2:
            for i in prerequisite_cluster_unique:
                if i == 0:
                    prerequisite_cluster_list.append([0, 0])
                else:
                    prerequisite_cluster_i_count = np.sum(prerequisite_cluster == i)
                    prerequisite_cluster_list.append(
                        [i, (prerequisite_cluster_i_count / prerequisite_cluster_index_count)])
            prerequisite_df = pd.DataFrame(prerequisite_cluster_list)
            prerequisite_df = prerequisite_df.sort_values(by=1)
            del_cluster = prerequisite_df.iloc()[:-2, 0].to_numpy()
            select_cluster = prerequisite_df.iloc()[-2:, 0].to_numpy()
            prerequisite_cluster = np.where(np.isin(prerequisite_cluster, del_cluster), 0, prerequisite_cluster)
            return prerequisite_cluster, select_cluster
        else:
            return prerequisite_cluster, prerequisite_cluster_unique

    @classmethod
    def cc_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array)
        for k in cls.cc_white_matter:
            index = np.argwhere(label_array == cls.cc_white_matter[k])
            prerequisite_start_index = np.argwhere(label_array == cls.cc_prerequisite[k]['start'])
            index_z_unique = np.unique(index[:, 2])
            prerequisite_index_z_unique = np.unique(prerequisite_start_index[:, 2])
            z_aixs_intersect = np.intersect1d(index_z_unique, prerequisite_index_z_unique)
            z_aixs_intersect = z_aixs_intersect[::-1]
            # Z 軸切片
            start_status = False
            for j in z_aixs_intersect:
                measure_np = np.zeros(label_array.shape[:2])
                index_sub = index[index[:, 2] == j]
                prerequisite_index_sub = prerequisite_start_index[prerequisite_start_index[:, 2] == j]
                measure_np[prerequisite_index_sub[:, 0], prerequisite_index_sub[:, 1]] = j
                prerequisite_cluster, prerequisite_cluster_unique = cls.get_prerequisite_cluster(measure_np)

                if prerequisite_cluster_unique.shape[0] >= 2:
                    cluster_x_list, cluster_y_list = cls.get_cluster_min_max(prerequisite_cluster)
                    if start_status:
                        if 'left_hemi' == k:
                            select_index = index_sub[
                                (index_sub[:, 1] >= cluster_x_list[1]) &
                                (index_sub[:, 0] >= cluster_y_list[1]) &
                                (index_sub[:, 0] <= cluster_y_list[-2])]
                        else:
                            select_index = index_sub[
                                (index_sub[:, 1] <= cluster_x_list[-2]) &
                                (index_sub[:, 0] >= cluster_y_list[1]) &
                                (index_sub[:, 0] <= cluster_y_list[-2])]
                        new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.cc_target[k]
                    else:
                        index_x_max = np.max(index_sub[:, 1])
                        prerequisite_index_x_max = np.max(prerequisite_index_sub[:, 1])
                        prerequisite_index_x_max = int(prerequisite_index_x_max * 0.97)
                        if index_x_max >= prerequisite_index_x_max:
                            start_status = True
                            if 'left_hemi' == k:
                                select_index = index_sub[
                                    (index_sub[:, 1] >= cluster_x_list[1]) &
                                    (index_sub[:, 0] >= cluster_y_list[1]) &
                                    (index_sub[:, 0] <= cluster_y_list[-2])]
                            else:
                                select_index = index_sub[
                                    (index_sub[:, 1] <= cluster_x_list[-2]) &
                                    (index_sub[:, 0] >= cluster_y_list[1]) &
                                    (index_sub[:, 0] <= cluster_y_list[-2])]
                            # if 'left_hemi' == k:
                            #     select_index = index_sub[
                            #         (index_sub[:, 1] >= cluster_x_list[1]) &
                            #         (index_sub[:, 0] >= cluster_y_list[1]) &
                            #         (index_sub[:, 0] <= cluster_y_list[-2])]
                            # else:
                            #     select_index = index_sub[
                            #         (index_sub[:, 1] <= cluster_x_list[-2]) & (index_sub[:, 0] >= cluster_y_list[1]) & (
                            #                 index_sub[:, 0] <= cluster_y_list[-2])]
                            new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.cc_target[
                                k]
        return new_label_array

    @classmethod
    def _get_cc_x_y_axis_index(cls, cc_index):
        z_axis = np.unique(cc_index[:, 2])
        norm_vector = []
        for i in z_axis:
            vector_x = cc_index[cc_index[:, 2] == i][:, 1]
            vector_y = cc_index[cc_index[:, 2] == i][:, 0]
            if vector_x.shape[0] > 1:
                norm_vector.append([i, vector_x.min(),
                                    vector_y.max(),
                                    vector_y.min(), ])
        norm_vector = np.array(norm_vector)
        norm_vector = norm_vector[::-1]
        return norm_vector

    @classmethod
    def _get_df_lateral_ventricle(cls, left_lateral_ventricle_index, right_lateral_ventricle_index):
        lateral_ventricle_vector = []
        left_lateral_ventricle_z_axis = np.unique(left_lateral_ventricle_index[:, 2])
        right_lateral_ventricle_z_axis = np.unique(right_lateral_ventricle_index[:, 2])
        z_axis_intersect = np.intersect1d(left_lateral_ventricle_z_axis, right_lateral_ventricle_z_axis)
        for i in z_axis_intersect:
            left_vector_x = left_lateral_ventricle_index[left_lateral_ventricle_index[:, 2] == i][:, 1]
            right_vector_x = right_lateral_ventricle_index[right_lateral_ventricle_index[:, 2] == i][:, 1]
            # lateral_ventricle_vector.append([i,
            #                                  np.percentile(left_vector_x, 10, axis=0).astype(int),
            #                                  np.percentile(right_vector_x, 90, axis=0).astype(int)])
            lateral_ventricle_vector.append([i,
                                             np.percentile(left_vector_x, 20, axis=0).astype(int),
                                             np.percentile(right_vector_x, 80, axis=0).astype(int)])

        intersect_lateral_ventricle_array = np.array(lateral_ventricle_vector)
        df_lateral_ventricle = pd.DataFrame(intersect_lateral_ventricle_array, columns=['lateral_ventricle_z',
                                                                                        'left_lateral_ventricle_x',
                                                                                        'right_lateral_ventricle_x'])
        df_lateral_ventricle.index = df_lateral_ventricle['lateral_ventricle_z']
        return df_lateral_ventricle

    @classmethod
    def _get_df_cc(cls, left_cc_index, right_cc_index) -> pd.DataFrame:
        left_cc_axis_vector = cls._get_cc_x_y_axis_index(left_cc_index)
        right_cc_axis_vector = cls._get_cc_x_y_axis_index(right_cc_index)
        z_axis_intersect = np.intersect1d(left_cc_axis_vector[:, 0], right_cc_axis_vector[:, 0])[::-1]
        z_axis_intersect = z_axis_intersect.reshape(-1, 1)
        intersect_cc_array = np.concatenate([z_axis_intersect,
                                             left_cc_axis_vector[
                                                 np.isin(left_cc_axis_vector[:, 0], z_axis_intersect)][:, 1:],
                                             right_cc_axis_vector[
                                                 np.isin(right_cc_axis_vector[:, 0], z_axis_intersect)][:, 1:], ],
                                            axis=1)
        df_cc = pd.DataFrame(intersect_cc_array, columns=['cc_z',
                                                          'left_cc_x', 'left_cc_y_max', 'left_cc_y_min',
                                                          'right_cc_x', 'right_cc_y_max', 'right_cc_y_min'])
        df_cc.index = df_cc['cc_z']
        return df_cc

    @classmethod
    def cc_adapt(cls, cc_array, label_array):
        new_label_array = np.zeros_like(label_array)
        left_cc_index = np.argwhere(cc_array == cls.cc_target['left_hemi'])
        right_cc_index = np.argwhere(cc_array == cls.cc_target['right_hemi'])
        left_lateral_ventricle_index = np.argwhere(label_array == cls.lateral_ventricle['left_hemi'])
        right_lateral_ventricle_index = np.argwhere(label_array == cls.lateral_ventricle['right_hemi'])

        df_cc = cls._get_df_cc(left_cc_index=left_cc_index, right_cc_index=right_cc_index)
        df_lateral_ventricle = cls._get_df_lateral_ventricle(left_lateral_ventricle_index=left_lateral_ventricle_index,
                                                             right_lateral_ventricle_index=right_lateral_ventricle_index
                                                             )
        df = df_cc.join(df_lateral_ventricle, how='inner')
        df = df.drop(columns=['lateral_ventricle_z'])

        left_index = np.argwhere(label_array == cls.cc_white_matter['left_hemi'])
        right_index = np.argwhere(label_array == cls.cc_white_matter['right_hemi'])
        for row in df.index:
            cc_z = df.loc()[row, 'cc_z']
            left_lateral_ventricle_x = df.loc()[row, 'left_lateral_ventricle_x']
            right_lateral_ventricle_x = df.loc()[row, 'right_lateral_ventricle_x']

            left_cc_y_max = df.loc()[row, 'left_cc_y_max']
            left_cc_y_min = df.loc()[row, 'left_cc_y_min']
            right_cc_y_max = df.loc()[row, 'right_cc_y_max']
            right_cc_y_min = df.loc()[row, 'right_cc_y_min']

            select_left_index = left_index[(left_index[:, 2] == cc_z) & (left_index[:, 1] > left_lateral_ventricle_x) &
                                           (left_index[:, 0] >= left_cc_y_min) & (left_index[:, 0] <= left_cc_y_max)
                                           ]
            select_right_index = right_index[
                (right_index[:, 2] == cc_z) & (right_index[:, 1] < right_lateral_ventricle_x) &
                (right_index[:, 0] >= right_cc_y_min) & (right_index[:, 0] <= right_cc_y_max)
                ]
            new_label_array[select_left_index[:, 0],
            select_left_index[:, 1],
            select_left_index[:, 2]] = cls.cc_target['left_hemi']
            new_label_array[select_right_index[:, 0],
            select_right_index[:, 1],
            select_right_index[:, 2]] = cls.cc_target['right_hemi']
        return new_label_array

    @classmethod
    def run(cls, label_array):
        label_array_translate = data_translate(label_array)
        FLIP, label_array_translate = left_right_translate(label_array_translate)
        cc_array_translate = cls.cc_parcellation(label_array_translate)
        cc_array_translate = cls.cc_adapt(cc_array_translate, label_array_translate)
        cc_array = inverse_left_right_translate(FLIP, cc_array_translate)
        out_label_array = inverse_data_translate(cc_array)
        return out_label_array


class ECICParcellation:
    ec_ic_parcellation_mapping = {
        'left_hemi': {1007: 3008,
                      11: 3009,
                      10: 3009,
                      13: 3009,
                      26: 3009,
                      },
        'right_hemi': {2007: 4008,
                       49: 4009,
                       50: 4009,
                       52: 4009,
                       58: 4009,
                       }
    }
    ec_ic_prerequisite = {'left_hemi': {'start': 12,
                                        'end': 13
                                        },
                          'right_hemi': {'start': 51,
                                         'end': 52
                                         }
                          }

    ec_ic_white_matter = {
        'left_hemi': 3007,
        'right_hemi': 4007,
    }

    decimal_places = 8
    scaling_factor = 10 ** decimal_places

    @classmethod
    def ec_ic_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array, dtype=np.int64)
        for k in cls.ec_ic_white_matter:
            # 取出putamen開始與停止層
            prerequisite_start_index = np.argwhere(label_array == cls.ec_ic_prerequisite[k]['start'])
            prerequisite_end_index = np.argwhere(label_array == cls.ec_ic_prerequisite[k]['end'])
            prerequisite_end_index = prerequisite_end_index[:, 2].min()
            unique_prerequisite_index = np.unique(prerequisite_start_index[:, 2])

            # 計算距離的各群
            ec_ic_parcellation_mapping_keys = list(cls.ec_ic_parcellation_mapping[k].keys())

            # 被分類的
            index = np.argwhere(label_array == cls.ec_ic_white_matter[k])
            unique_label_index = np.unique(index[:, 2])

            # 被分類的Z軸索引與 putamen
            z_aixs_intersect = np.intersect1d(unique_prerequisite_index, unique_label_index)
            z_aixs_intersect = z_aixs_intersect[z_aixs_intersect >= prerequisite_end_index]
            index = index[np.isin(index[:, 2], z_aixs_intersect)]

            np_loss = np.zeros((index.shape[0], len(ec_ic_parcellation_mapping_keys)))
            np_loss[:, :] = 999999
            for i in range(len(ec_ic_parcellation_mapping_keys)):
                label_index = np.argwhere(label_array == ec_ic_parcellation_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
                for j in np.unique(label_index[:, 2]):
                    index_sub = index[index[:, 2] == j]
                    index_sub_arg = np.argwhere(index[:, 2] == j)
                    label_index_sub = label_index[label_index[:, 2] == j]
                    if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                        loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                        loss_list.append(loss_min.numpy())
                        index_sub_arg_list.append(index_sub_arg)
                    else:
                        continue
                if len(index_sub_arg_list) > 0:
                    index_sub_arg = np.concatenate(index_sub_arg_list)
                    np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
                else:
                    continue
            new_label = np_loss.argmin(axis=1)
            # 指定分類
            for i in np.unique(new_label):
                select_index = index[np.argwhere(new_label == i)].squeeze()
                new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                    cls.ec_ic_parcellation_mapping[k][ec_ic_parcellation_mapping_keys[i]]
        return new_label_array

    @classmethod
    def run(cls, label_array):
        new_label_array = cls.ec_ic_parcellation(label_array)
        return new_label_array


class BullseyeProcess:
    synthseg_wm_output = {
        'left_hemi': {4: 3010,
                      5: 3010,
                      7: 3010,
                      8: 3010,
                      10: 3010,
                      11: 3010,
                      12: 3010,
                      13: 3010,
                      18: 3010,
                      26: 3010,
                      28: 3010,

                      1001: 3010,
                      1003: 3010,
                      1004: 3010,
                      1005: 3010,
                      1006: 3010,
                      1007: 3010,

                      3001: 3010,
                      3003: 3010,
                      3004: 3010,
                      3005: 3010,
                      3006: 3010,
                      3007: 3010,
                      },
        'right_hemi': {43: 4010,
                       44: 4010,
                       46: 4010,
                       47: 4010,
                       49: 4010,
                       50: 4010,
                       51: 4010,
                       52: 4010,
                       54: 4010,
                       58: 4010,
                       60: 4010,

                       2001: 4010,
                       2003: 4010,
                       2004: 4010,
                       2005: 4010,
                       2006: 4010,
                       2007: 4010,

                       4001: 4010,
                       4003: 4010,
                       4004: 4010,
                       4005: 4010,
                       4006: 4010,
                       4007: 4010},
    }
    bullseye_parcellation = {
        'left_hemi': 3010,
        'right_hemi': 4010,
    }
    lateral_ventricle = {
        'left_hemi': 4,
        'right_hemi': 43,
    }
    depth_number = 6
    inner_size = 2
    outer_size = 5
    decimal_places = 8

    @classmethod
    def filter_labels(cls, in0, include_superlist, fixed_id=None, map_pairs_list=None):
        """filters-out labels not in the include-superset. Merges labels within superset. Transforms label-ids according
        to mappings (or fixed id)"""

        # read label file and create output
        out0 = np.zeros(in0.shape, dtype=in0.dtype)

        # for each group of labels in subset assign them the same id (either 1st in subset or fixed-id, in case given)
        for labels_list in include_superlist:
            for label in labels_list:
                value = labels_list[0]
                if fixed_id is not None: value = fixed_id[0]
                out0[in0 == label] = value

        # transform label-ids in case mapping is specified
        if map_pairs_list is not None:
            out1 = np.copy(out0)
            for map_pair in map_pairs_list:
                out1[out0 == map_pair[0]] = map_pair[1]

        # save output
        out_final = out0 if not map_pairs_list else out1

        return out_final

    @classmethod
    def norm_dist_map(cls, orig, dest):
        """compute normalized distance map given an origin and destination masks, resp."""

        dist_orig = distance_transform_edt(np.logical_not(orig.astype(np.bool_)))
        dist_dest = distance_transform_edt(np.logical_not(dest.astype(np.bool_)))

        # normalized distance (0 in origin to 1 in dest)
        ndist = dist_orig / (dist_orig + dist_dest)
        return ndist

    @classmethod
    def create_shells(cls, ndist, mask, n_shells=4):
        """creates specified number of shells given normalized distance map. When mask is given, output in mask == 0 is
        set to zero"""

        out = np.zeros(ndist.shape, dtype=np.int32)

        limits = np.linspace(0., 1., n_shells + 1)
        for i in np.arange(n_shells) + 1:
            # compute shell and assing increasing label-id
            mask2 = np.logical_and(ndist >= limits[i - 1], ndist < limits[i])
            if mask is not None:  # maskout regions outside mask
                mask2 = np.logical_and(mask2, mask)
            out[mask2] = i
        out[np.isclose(ndist, 0.)] = 0  # need to assign zero to ventricles because of >= above
        return out

    @classmethod
    def merge_labels(cls, in1, in2, intersect=False):
        """merges labels from two input labelmaps, optionally computing intersection"""
        in1 = in1.round(0).astype(int)
        in2 = in2.round(0).astype(int)

        # if not intersection, simply include labels from 'in2' into 'in1'
        if not intersect:

            out = np.zeros(in1.shape, dtype=np.int32)

            out[:] = in1[:]
            mask = in2 > 0
            out[mask] = in2[mask]  # overwrite in1 where in2 > 0

        # if intersection, create new label-set as cartesian product of the two sets
        else:
            out = np.zeros(in1.shape, dtype=np.int32)

            u1_set = np.unique(in1.ravel())
            u2_set = np.unique(in2.ravel())

            u1_set = u1_set.astype(int)
            u2_set = u2_set.astype(int)

            for u1 in u1_set:
                if u1 == 0: continue
                mask1 = in1 == u1
                for u2 in u2_set:
                    if u2 == 0: continue
                    mask2 = in2 == u2
                    mask3 = np.logical_and(mask1, mask2)
                    if not np.any(mask3): continue
                    out[mask3] = int(str(u1) + str(u2))  # new label id by concatenating [u1, u2]
        return out

    @classmethod
    def generate_wmparc(cls, incl_aux, ndist, label, incl_labels=None, verbose=False):
        """generates wmparc by propagating labels in 'label_file' down the gradient defined by distance map in
        'ndist_file'. Labels are only propagated in regions where 'incl_file' > 0 (or 'incl_file' == incl_labels[i],
        if 'incl_labels is provided).
        """
        connectivity = generate_binary_structure(3, 2)

        # create inclusion mask
        if incl_labels is None:
            incl_mask = incl_aux > 0
        else:
            incl_mask = np.zeros(incl_aux.shape, dtype=np.bool_)
            for lab in incl_labels:
                incl_mask[incl_aux == lab] = True

        # get DONE and processing masks
        DONE_mask = label > 0  # this is for using freesurfer wmparc
        proc_mask = np.logical_and(np.logical_and(ndist > 0., ndist < 1.), incl_mask)

        # setup the ouptut vol
        out = np.zeros(label.shape, dtype=label.dtype)

        # initialize labels in cortex
        out[DONE_mask] = label[DONE_mask]  # this is for using freesurfer wmparc

        # start with connectivity 1
        its_conn = 1

        # main loop
        while not np.all(DONE_mask[proc_mask]):

            if verbose:
                print('%0.1f done' % (100. * float(DONE_mask[proc_mask].sum()) / float(proc_mask.sum())))

            # loop to increase connectivity for non-reachable TO-DO points
            while True:

                # dilate the SOLVED area
                aux = binary_dilation(DONE_mask, iterate_structure(connectivity, its_conn))
                # next TO-DO: close to DONE, in the processing mask and not yet done
                TODO_mask = np.logical_and(np.logical_and(aux, proc_mask), np.logical_not(DONE_mask))

                if TODO_mask.sum() > 0:
                    break

                if verbose:
                    print('Non-reachable points. Increasing connectivity')

                its_conn += 1

            # sort TO-DO points by ndist
            Idx_TODO = np.argwhere(TODO_mask)
            Idx_ravel = np.ravel_multi_index(Idx_TODO.T, label.shape)
            I_sort = np.argsort(ndist.ravel()[Idx_ravel])

            # iterate along TO-DO points
            for idx in Idx_TODO[I_sort[::-1]]:

                max_dist = -1.

                # process each neighbor
                for off in np.argwhere(iterate_structure(connectivity, its_conn)) - its_conn:

                    try:

                        # if it is not DONE then skip
                        if not DONE_mask[idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]]:
                            continue

                        # if it is the largest distance (ie, largest gradient)
                        cur_dist = ndist[idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]]
                        if cur_dist > max_dist:
                            out[idx[0], idx[1], idx[2]] = out[idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]]
                            max_dist = cur_dist

                    except:
                        print('something wrong with neighbor at: (%d, %d, %d)' % (
                            idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]))
                        pass

                if max_dist < 0.: print("something went wrong with point: (%d, %d, %d)" % (idx[0], idx[1], idx[2]))

                # mark as solved and remove from visited
                DONE_mask[idx[0], idx[1], idx[2]] = True
        return out

    @classmethod
    def bullsey_hemi_parcellation(cls, depth_array, synthseg_array):
        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(depth_array, dtype=depth_array.dtype)
        temp_array = np.zeros_like(depth_array, dtype=depth_array.dtype)
        # 取出 bullseye 部分
        index = np.argwhere(depth_array != 0)
        temp_array[index[:, 0], index[:, 1], index[:, 2]] = synthseg_array[index[:, 0], index[:, 1], index[:, 2]]

        left_mask = (temp_array // 1000) == 3
        right_mask = (temp_array // 1000) == 4
        new_label_array[left_mask] = cls.bullseye_parcellation['left_hemi']
        new_label_array[right_mask] = cls.bullseye_parcellation['right_hemi']
        new_label_array = new_label_array + depth_array
        return new_label_array

    @classmethod
    def bullseye_depth_merge(cls, bullsey_parcellation_array, synthseg_array):
        out_array = np.zeros_like(bullsey_parcellation_array)
        for k in cls.lateral_ventricle:
            lateral_ventricle_index = np.argwhere(synthseg_array == cls.lateral_ventricle[k])
            lateral_ventricle_z = np.unique(lateral_ventricle_index[:, 2])
            z_higher = lateral_ventricle_z.max()
            z_lower = lateral_ventricle_z.min()
            bullseye_cluster = np.unique(cls.bullseye_parcellation[k])
            bullseye_cluster = bullseye_cluster[bullseye_cluster != 0]
            for i in bullseye_cluster:
                bullseye_depth_mask = (bullsey_parcellation_array > i) & (
                        bullsey_parcellation_array < (i + cls.depth_number))
                bullseye_index = np.argwhere(bullseye_depth_mask)
                bullseye_index_higher = bullseye_index[bullseye_index[:, 2] > z_higher]
                bullseye_index_lower = bullseye_index[bullseye_index[:, 2] < z_lower]

                bullseye_between_inner_mask = (bullsey_parcellation_array > (i)) & (
                        bullsey_parcellation_array <= (i + cls.inner_size))
                bullseye_between_inner_index = np.argwhere(bullseye_between_inner_mask == True)
                bullseye_between_inner_index_between = bullseye_between_inner_index[
                    (bullseye_between_inner_index[:, 2] <= z_higher) &
                    (bullseye_between_inner_index[:, 2] >= z_lower)]

                bullseye_between_outer_mask = (bullsey_parcellation_array > (i + cls.inner_size)) & (
                        bullsey_parcellation_array <= (i + cls.outer_size))
                bullseye_between_outer_index = np.argwhere(bullseye_between_outer_mask == True)
                bullseye_between_outer_index_between = bullseye_between_outer_index[
                    (bullseye_between_outer_index[:, 2] <= z_higher) &
                    (bullseye_between_outer_index[:, 2] >= z_lower)]
                out_array[bullseye_index_higher[:, 0],
                bullseye_index_higher[:, 1],
                bullseye_index_higher[:, 2]] = i + 2
                out_array[bullseye_index_lower[:, 0],
                bullseye_index_lower[:, 1],
                bullseye_index_lower[:, 2]] = i + 2
                out_array[bullseye_between_inner_index_between[:, 0],
                bullseye_between_inner_index_between[:, 1],
                bullseye_between_inner_index_between[:, 2]] = i + 1
                out_array[bullseye_between_outer_index_between[:, 0],
                bullseye_between_outer_index_between[:, 1],
                bullseye_between_outer_index_between[:, 2]] = i + 2
        return out_array

    @classmethod
    def bullseye_depth_synthseg_label(cls, bullsey_depth_merge_array, label_array, synthseg_array):
        out = np.zeros(bullsey_depth_merge_array.shape, dtype=np.int32)
        u1 = np.unique(bullsey_depth_merge_array)
        mask2 = (synthseg_array == 2)
        mask41 = (synthseg_array == 41)
        for k in u1:
            mask1 = (bullsey_depth_merge_array == k)
            out_mask = np.logical_or(np.logical_and(mask1, mask2),
                                     np.logical_and(mask1, mask41))
            if (k == 3011) or (k == 4011):
                out[out_mask] = bullsey_depth_merge_array[out_mask]
            else:
                inner_mask = np.logical_and((bullsey_depth_merge_array >= 3000), out_mask)
                out[out_mask] = label_array[out_mask]
                out[inner_mask] = label_array[inner_mask] + 30
        return out

    @classmethod
    def get_depth_wmparc_np(cls, label_array, n_shells):
        filter_labels_include_superlist = [[3001, 3007], [4001, 4007], [3004], [4004], [3005], [4005], [3006],
                                           [4006]]  # lobar labels in WM
        filter_labels_fixed_id = None
        filter_labels_map_pairs_list = [[3001, 11], [4001, 21], [3004, 12], [4004, 22], [3005, 13], [4005, 23],
                                        [3006, 14], [4006, 24]]
        filter_lobes_np = cls.filter_labels(label_array,
                                            include_superlist=filter_labels_include_superlist,
                                            fixed_id=filter_labels_fixed_id,
                                            map_pairs_list=filter_labels_map_pairs_list)

        ventricles_include_superlist = [[43, 4]]
        ventricles_fixed_id = [1]
        ventricles_map_pairs_list = None
        ventricles_np = cls.filter_labels(label_array,
                                          include_superlist=ventricles_include_superlist,
                                          fixed_id=ventricles_fixed_id, map_pairs_list=ventricles_map_pairs_list)

        cortex_include_superlist = [[1001, 2001, 1004, 2004, 1005, 2005, 1006, 2006]]  # lobar labels in cortex
        cortex_fixed_id = [1]
        cortex_map_pairs_list = None
        cortex_np = cls.filter_labels(label_array,
                                      include_superlist=cortex_include_superlist,
                                      fixed_id=cortex_fixed_id, map_pairs_list=cortex_map_pairs_list)
        bgt_include_superlist = [[10, 49, 11, 12, 50, 51, 26, 58, 13, 52]]  # basal ganglia + thalamus
        bgt_fixed_id = [5]
        bgt_map_pairs_list = None
        bgt_np = cls.filter_labels(label_array,
                                   include_superlist=bgt_include_superlist,
                                   fixed_id=bgt_fixed_id, map_pairs_list=bgt_map_pairs_list)

        ndist_np = cls.norm_dist_map(ventricles_np, cortex_np)
        generate_wmparc_incl_labels = [3003, 4003, 5001, 5002]  # the labels that need to be 'filled'
        gen_wmparc_np = cls.generate_wmparc(incl_aux=label_array, ndist=ndist_np,
                                            label=filter_lobes_np,
                                            incl_labels=generate_wmparc_incl_labels, verbose=False)
        lobe_wmparc_np = cls.merge_labels(in1=gen_wmparc_np, in2=bgt_np, intersect=False)
        depth_wmparc_np = cls.create_shells(ndist=ndist_np, mask=lobe_wmparc_np, n_shells=n_shells)
        return depth_wmparc_np

    @classmethod
    def run(cls, label_array, synthseg_array, depth_number):
        cls.depth_number = depth_number
        cls.outer_size = depth_number - 1
        depth_np = cls.get_depth_wmparc_np(label_array=label_array, n_shells=depth_number)
        bullsey_parcellation_array = cls.bullsey_hemi_parcellation(depth_array=depth_np, synthseg_array=label_array)
        bullsey_depth_merge_array = cls.bullseye_depth_merge(bullsey_parcellation_array, label_array)
        bullseye_depth_synthseg_array = cls.bullseye_depth_synthseg_label(
            bullsey_depth_merge_array=bullsey_depth_merge_array,
            label_array=label_array,
            synthseg_array=synthseg_array)
        return bullseye_depth_synthseg_array


class CMBProcess:
    # brain stem(16) 分左右 (104，204)
    # left ventral DC (103) 分 (left thalamus 104)、(to left brainstem 101)
    # ventral DC 分 thalamus and brainstem
    label_david_mapping_CMB = {
        'left_hemi': {
            112: 109,
            117: 109,
            119: 109,
            113: 110,
            120: 110,
            114: 111,
            121: 111,
            115: 112,
            122: 112,
            116: 114,
            123: 114,
            118: 113,
            124: 113,
            108: 104,
            109: 103,
            106: 103,
            110: 103,
            111: 103,
            125: 106,
            126: 105,
            127: 107,
            128: 108,
            102: 1,
            104: 102,
            105: 102,
            129: 108,
            130: 108,
            131: 108,
            132: 108,
            14: 1,
            15: 1,
            24: 1,
        },
        'right_hemi': {
            212: 209,
            217: 209,
            219: 209,
            213: 210,
            220: 210,
            214: 211,
            221: 211,
            215: 212,
            222: 212,
            216: 214,
            223: 214,
            218: 213,
            224: 213,
            208: 204,
            209: 203,
            206: 203,
            210: 203,
            211: 203,
            225: 206,
            226: 205,
            227: 207,
            228: 208,
            202: 1,
            204: 202,
            205: 202,
            229: 208,
            230: 208,
            231: 208,
            232: 208,
        }
    }
    ventral_DC = {
        'left_hemi': 103,
        'right_hemi': 203
    }
    # 2023-10-24 BRAIN_STEM不分左右改為 301
    ventral_DC_mapping = {
        'left_hemi': [104, 301],
        'right_hemi': [204, 301]
    }
    # ventral_DC_mapping = {
    #     'left_hemi': [104, 101],
    #     'right_hemi': [204, 201]
    # }
    ventral_DC_prerequisite = {
        'left_hemi': 108,
        'right_hemi': 208
    }
    PREREQUISITE_THRESHOLD = 0.5

    BRAIN_STEM = 16

    # 2023-10-24 BRAIN_STEM不分左右改為 301
    brain_stem_mapping_CMB = {
        'left_hemi': 301,
        'right_hemi': 301
    }
    # brain_stem_mapping_CMB = {
    #     'left_hemi': 101,
    #     'right_hemi': 201
    # }
    brain_stem_hemi_parcellation = {
        'left_hemi': {
            104: 101,
            105: 101,
        },
        'right_hemi': {
            204: 201,
            205: 201,
        }
    }
    decimal_places = 8

    @classmethod
    def david_label_to_CMB_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.label_david_mapping_CMB:
            # run every synseg label to freesurfer label
            for key in cls.label_david_mapping_CMB[hemi]:
                if cls.label_david_mapping_CMB[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.label_david_mapping_CMB[hemi][key]
        return new_label_array

    @classmethod
    def brain_stem_parcellation(cls, label_array):

        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        # 取出 brain stem 部分
        index = np.argwhere(label_array == cls.BRAIN_STEM)
        # 建立 誤差(距離)矩陣
        hemi_list = list(cls.brain_stem_hemi_parcellation.keys())
        np_loss = np.zeros((index.shape[0], len(hemi_list)))
        np_loss[:, :] = 999999
        for k in cls.brain_stem_hemi_parcellation:
            if k == 'left_hemi':
                i = 0
            else:
                i = 1
            hemi_parcellation_mapping_keys = list(cls.brain_stem_hemi_parcellation[k].keys())
            # brain stem  的 Z 軸
            z_aixs_intersect = np.unique(index[:, 2])

            label_index = np.argwhere(np.isin(label_array, hemi_parcellation_mapping_keys))
            loss_list = []
            index_sub_arg_list = []
            # Z 軸切片
            for j in z_aixs_intersect:
                index_sub = index[index[:, 2] == j]
                index_sub_arg = np.argwhere(index[:, 2] == j)
                label_index_sub = label_index[label_index[:, 2] == j]
                if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                    loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                    loss_list.append(loss_min.numpy())
                    index_sub_arg_list.append(index_sub_arg)
                else:
                    continue
            if len(index_sub_arg_list) > 0:
                index_sub_arg = np.concatenate(index_sub_arg_list)
                np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
            else:
                continue
        new_label = np_loss.argmin(axis=1)
        # 指定分類
        for ii in np.unique(new_label):
            select_index = index[np.argwhere(new_label == ii)].squeeze()
            new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.brain_stem_mapping_CMB[
                hemi_list[ii]]
        return new_label_array

    @classmethod
    def ventral_DC_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array)
        for k in cls.ventral_DC:
            index = np.argwhere(label_array == cls.ventral_DC[k])
            prerequisite_index = np.argwhere(label_array == cls.ventral_DC_prerequisite[k])

            index_z_unique = np.unique(index[:, 2])
            # prerequisite_index_z_unique = np.unique(prerequisite_index[:, 2])
            # z_aixs_intersect = np.intersect1d(index_z_unique, prerequisite_index_z_unique)
            z_aixs_intersect = index_z_unique
            # Z 軸切片
            for j in z_aixs_intersect:
                index_sub = index[index[:, 2] == j]
                prerequisite_index_sub = prerequisite_index[prerequisite_index[:, 2] == j]
                if prerequisite_index_sub.shape[0] > 0:
                    if prerequisite_index_sub.shape[0] / index_sub.shape[0] >= cls.PREREQUISITE_THRESHOLD:
                        new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][
                            0]
                    else:
                        new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][
                            1]
                else:
                    new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][1]
        return new_label_array

    @classmethod
    def run(cls, label_array):
        brain_stem_array = cls.brain_stem_parcellation(label_array)
        ventral_DC_array = cls.ventral_DC_parcellation(label_array)
        cmd_array = cls.david_label_to_CMB_label(label_array)
        brain_stem_mask = brain_stem_array != 0
        ventral_DC_mask = ventral_DC_array != 0

        cmd_array[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        cmd_array[ventral_DC_mask] = ventral_DC_array[ventral_DC_mask]
        return cmd_array


class DWIProcess:
    label_david_mapping_DWI = {
        'left_hemi': {

            112: 118,
            117: 118,
            119: 118,
            113: 119,
            120: 119,
            114: 120,
            121: 120,
            115: 121,
            122: 121,
            116: 123,
            123: 123,
            118: 122,
            124: 122,
            108: 108,
            109: 105,
            106: 105,
            110: 106,
            111: 107,
            125: 109,
            126: 110,
            127: 111,
            # 128: 111,
            101: 0,
            102: 1,
            104: 104,
            105: 104,
            # 103 use method
            # 129 use method
            # 130 use method
            131: 116,
            132: 117,
            # 16 use method
            14: 1,
            15: 1,
            24: 1,
        },
        'right_hemi': {
            212: 218,
            217: 218,
            219: 218,
            213: 219,
            220: 219,
            214: 220,
            221: 220,
            215: 221,
            222: 221,
            216: 223,
            223: 223,
            218: 222,
            224: 222,
            208: 208,
            209: 205,
            206: 205,
            210: 206,
            211: 207,
            225: 209,
            226: 210,
            227: 211,
            228: 211,
            201: 0,
            202: 1,
            204: 204,
            205: 204,
            # 203 use method
            # 229 use method
            # 230 use method
            231: 216,
            232: 217,
            # 16 use method
        },
    }

    ventral_DC = {
        'left_hemi': 103,
        'right_hemi': 203
    }

    # 2023-10-24 BRAIN_STEM不分左右改為 301
    ventral_DC_mapping = {
        'left_hemi': [108, 301],
        'right_hemi': [208, 301]
    }
    # ventral_DC_mapping = {
    #     'left_hemi': [108, 101],
    #     'right_hemi': [208, 201]
    # }
    ventral_DC_prerequisite = {
        'left_hemi': 108,
        'right_hemi': 208
    }
    PREREQUISITE_THRESHOLD = 0.5

    BRAIN_STEM = 16

    BRAIN_STEM_mapping = {
        'midbrain': 301,
        'pons': 302,
        'medulla': 303
    }

    BRAIN_STEM_hemi_shift = {
        'left_hemi': 50,
        'right_hemi': 150,
    }

    brain_stem_mapping_DWI = {
        # midbrain 、pons、 medulla
        'left_hemi': [101, 102, 103],
        'right_hemi': [201, 202, 203],
    }

    brain_stem_hemi = {
        'left_hemi': {
            103: 101,
            104: 101,
            105: 101,
            112: 101,
            115: 101,
        },
        'right_hemi': {
            203: 201,
            204: 201,
            205: 201,
            212: 201,
            215: 201,
        }
    }

    frontal_deep_white_matter = {
        'left_hemi': 129,
        'right_hemi': 229
    }
    frontal_deep_white_matter_parcellation = {
        'left_hemi': [112, 113, 114],
        'right_hemi': [212, 213, 214],
    }
    frontal_deep_white_matter_prerequisite = {
        'left_hemi': [102, 110],
        'right_hemi': [202, 210],
    }
    parietal_deep_white_matter = {
        'left_hemi': 130,
        'right_hemi': 230
    }
    parietal_deep_white_matter_parcellation = {
        'left_hemi': [112, 113, 115],
        'right_hemi': [212, 213, 215],
    }
    parietal_deep_white_matter_prerequisite = {
        'left_hemi': [102, 110],
        'right_hemi': [202, 210],
    }

    decimal_places = 8

    bullseye_DPWM = {
        'left_hemi': 128,
        'right_hemi': 228,
    }
    bullseye_DPWM_parcellation = {
        'left_hemi': {
            129: 129,
            130: 130,
            131: 131,
            132: 132,
        },
        'right_hemi': {
            229: 229,
            230: 230,
            231: 231,
            232: 232,
        },
    }

    @classmethod
    def david_label_to_DWI_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.label_david_mapping_DWI:
            # run every synseg label to freesurfer label
            for key in cls.label_david_mapping_DWI[hemi]:
                if cls.label_david_mapping_DWI[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.label_david_mapping_DWI[hemi][key]
        return new_label_array

    @classmethod
    def DPWM_parcellation(cls, label_array, target='frontal'):
        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        # 設定 target 、parcellation
        if 'frontal' == target:
            target_dict = cls.frontal_deep_white_matter
            parcellation_dict = cls.frontal_deep_white_matter_parcellation
            prerequisite_dict = cls.frontal_deep_white_matter_prerequisite
        elif 'parietal' == target:
            target_dict = cls.parietal_deep_white_matter
            parcellation_dict = cls.parietal_deep_white_matter_parcellation
            prerequisite_dict = cls.parietal_deep_white_matter_prerequisite
        else:
            raise ValueError('target is not definition')
        #
        for k in target_dict:
            # 取出目標索引
            index = np.argwhere(label_array == target_dict[k])
            # 取出條件上邊界索引
            temp_index = np.argwhere(label_array == prerequisite_dict[k][0])
            temp_index_z = np.unique(temp_index[:, 2])
            parcellation_higher = temp_index_z.max()
            # 取出條件下邊界索引
            temp_index = np.argwhere(label_array == prerequisite_dict[k][1])
            temp_index_z = np.unique(temp_index[:, 2])
            parcellation_lower = temp_index_z.max()

            higher_index = index[index[:, 2] > parcellation_higher]
            middle_index = index[(index[:, 2] <= parcellation_higher) & (index[:, 2] >= parcellation_lower)]
            lower_index = index[index[:, 2] < parcellation_lower]

            new_label_array[higher_index[:, 0],
            higher_index[:, 1],
            higher_index[:, 2]] = parcellation_dict[k][0]
            new_label_array[middle_index[:, 0],
            middle_index[:, 1],
            middle_index[:, 2]] = parcellation_dict[k][1]
            new_label_array[lower_index[:, 0],
            lower_index[:, 1],
            lower_index[:, 2]] = parcellation_dict[k][2]
        return new_label_array

    @classmethod
    def brain_stem_hemi_parcellation(cls, label_array):
        # # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        BRAIN_STEM_mapping_list = list(cls.BRAIN_STEM_mapping.values())
        index = np.argwhere(np.isin(label_array, BRAIN_STEM_mapping_list))
        z_aixs_intersect = np.unique(index[:, 2])
        for z in z_aixs_intersect:
            temp_index = index[index[:, 2] == z]
            x_center = int(temp_index[:, 0].mean())
            y_center = int(temp_index[:, 1].mean())
            left_index = temp_index[(temp_index[:, 0] <= x_center)]
            right_index = temp_index[(temp_index[:, 0] >= x_center)]
            new_label_array[left_index[:, 0], left_index[:, 1], left_index[:, 2]] = \
                label_array[left_index[:, 0], left_index[:, 1], left_index[:, 2]] + cls.BRAIN_STEM_hemi_shift[
                    'left_hemi']
            new_label_array[right_index[:, 0], right_index[:, 1], right_index[:, 2]] = \
                label_array[right_index[:, 0], right_index[:, 1], right_index[:, 2]] + cls.BRAIN_STEM_hemi_shift[
                    'right_hemi']
        return new_label_array

    # 2023 10 24 改停
    # @classmethod
    # def brain_stem_parcellation(cls, label_array):
    #
    #     new_label_array = label_array.copy()
    #     output_label_array = np.zeros_like(new_label_array)
    #
    #     # to  pons、 medulla
    #     # brain-stem - midbrain
    #     index = np.argwhere(new_label_array == cls.BRAIN_STEM)
    #     values_x, count_x = np.unique(index[:, 0], return_counts=True)
    #     values_p = values_x[::-1]
    #     count_p = (count_x / count_x.max())[::-1]
    #     temp_count_p = count_p
    #     x_index = values_p[temp_count_p.argmax()]
    #     temp_index = index[index[:, 0] == x_index]
    #
    #     #2023 10 24 改
    #     y_axis = np.unique(temp_index[:, 1])
    #     y_axis_15 = np.percentile(y_axis, 15)
    #     y_axis_85 = np.percentile(y_axis, 85)
    #     y_axis_select = y_axis[(y_axis >= y_axis_15) & (y_axis <= y_axis_85)]
    #     norm_vector = []
    #     # for i in y_axis:
    #     for i in y_axis_select:
    #         vector = temp_index[temp_index[:, 1] == i][:, 2]
    #         if vector.shape[0] > 1:
    #             norm_vector.append([i, np.ptp(vector), vector.max(), vector.min()])
    #     norm_vector = np.array(norm_vector, dtype=np.float64)
    #     norm_vector = np.round(norm_vector, 8)
    #
    #     z_1_gradient = np.gradient(norm_vector[:, 1])
    #     z_2_gradient = np.gradient(z_1_gradient)
    #     y_index = norm_vector[z_2_gradient.argmax(), 0]
    #     z_index = index[(index[:, 0] == x_index) & (index[:, 1] == y_index), 2].min()
    #
    #     ###### 2023 10 24 改
    #     z_axis = np.unique(temp_index[:, 2])
    #     z_norm_vector = []
    #     for i in z_axis:
    #         vector = temp_index[temp_index[:, 2] == i][:, 1]
    #         if vector.shape[0] > 1:
    #             z_norm_vector.append([i, np.ptp(vector), vector.max(), vector.min()])
    #     z_norm_vector = np.array(z_norm_vector, dtype=np.float64)
    #     z_norm_vector = np.round(z_norm_vector, 8)
    #
    #     z_1_gradient = np.gradient(z_norm_vector[:, 1])
    #     z_2_gradient = np.gradient(z_1_gradient)
    #     z_index = np.where(z_2_gradient == z_2_gradient.max())[0][-1]
    #     higher_index = index[index[:, 2] > z_index]
    #     lower_index = index[index[:, 2] <= z_index]
    #     # new_label_array[higher_index[:, 0], higher_index[:, 1], higher_index[:, 2]] = cls.BRAIN_STEM_mapping['pons']
    #     # new_label_array[lower_index[:, 0], lower_index[:, 1], lower_index[:, 2]] = cls.BRAIN_STEM_mapping['medulla']
    #     output_label_array[higher_index[:, 0], higher_index[:, 1], higher_index[:, 2]] = cls.BRAIN_STEM_mapping['pons']
    #     output_label_array[lower_index[:, 0], lower_index[:, 1], lower_index[:, 2]] = cls.BRAIN_STEM_mapping['medulla']
    #
    #     # to midbrain
    #     index = np.argwhere(new_label_array == cls.BRAIN_STEM)
    #     ventral_DC_list = list(cls.ventral_DC.values())
    #     ventral_DC_z = np.argwhere(np.isin(new_label_array, ventral_DC_list))
    #     ventral_DC_z_min = ventral_DC_z[:, 2].min()
    #     ventral_DC_z_max = ventral_DC_z[:, 2].max()
    #     middle_index = index[(index[:, 2] <= ventral_DC_z_max) & (index[:, 2] >= ventral_DC_z_min)]
    #     # new_label_array[middle_index[:, 0],
    #     # middle_index[:, 1],
    #     # middle_index[:, 2]] = cls.BRAIN_STEM_mapping['midbrain']
    #     output_label_array[middle_index[:, 0],middle_index[:, 1],middle_index[:, 2]] = cls.BRAIN_STEM_mapping['midbrain']
    #
    #     # output_label_array = cls.brain_stem_hemi_parcellation(new_label_array)
    #     return output_label_array
    # 2023 10 24 改停
    @classmethod
    def brain_stem_parcellation(cls, label_array):

        new_label_array = label_array.copy()
        output_label_array = np.zeros_like(new_label_array)

        # to midbrain
        index = np.argwhere(new_label_array == cls.BRAIN_STEM)
        ventral_DC_list = list(cls.ventral_DC.values())
        ventral_DC_z = np.argwhere(np.isin(new_label_array, ventral_DC_list))
        ventral_DC_z_min = ventral_DC_z[:, 2].min()
        ventral_DC_z_max = ventral_DC_z[:, 2].max()
        middle_index = index[(index[:, 2] <= ventral_DC_z_max) & (index[:, 2] >= ventral_DC_z_min)]

        new_label_array[middle_index[:, 0], middle_index[:, 1], middle_index[:, 2]] = cls.BRAIN_STEM_mapping[
            'midbrain']

        #lower_index = index[index[:, 2] < ventral_DC_z_min]

        # to  pons、 medulla
        # brain-stem - midbrain
        index = np.argwhere(new_label_array == cls.BRAIN_STEM)
        values_x, count_x = np.unique(index[:, 0], return_counts=True)
        values_p = values_x[::-1]
        count_p = (count_x / count_x.max())[::-1]
        temp_count_p = count_p
        x_index = values_p[temp_count_p.argmax()]
        temp_index = index[index[:, 0] == x_index]

        ###### 2023 10 24 改
        z_axis = np.unique(temp_index[:, 2])
        z_norm_vector = []
        for i in z_axis:
            vector = temp_index[temp_index[:, 2] == i][:, 1]
            if vector.shape[0] > 1:
                z_norm_vector.append([i, np.ptp(vector), vector.max(), vector.min()])
        z_norm_vector = np.array(z_norm_vector, dtype=np.float64)
        z_norm_vector = np.round(z_norm_vector, 8)

        z_1_gradient = np.gradient(z_norm_vector[:, 1])
        z_2_gradient = np.gradient(z_1_gradient)
        z_index = np.where(z_2_gradient == z_2_gradient.max())[0][-1]
        # print(rf'z_index {np.where(z_2_gradient == z_2_gradient.max())[0]}')
        # print(rf'z_index {z_index}')
        higher_index = index[index[:, 2] > z_index]
        lower_index = index[index[:, 2] <= z_index]

        new_label_array[higher_index[:, 0], higher_index[:, 1], higher_index[:, 2]] = cls.BRAIN_STEM_mapping['pons']
        new_label_array[lower_index[:, 0], lower_index[:, 1], lower_index[:, 2]] = cls.BRAIN_STEM_mapping['medulla']

        midbrain_mask = new_label_array == cls.BRAIN_STEM_mapping['midbrain']
        pons_mask = new_label_array == cls.BRAIN_STEM_mapping['pons']
        medulla_mask = new_label_array == cls.BRAIN_STEM_mapping['medulla']
        output_label_array[midbrain_mask] = new_label_array[midbrain_mask]
        output_label_array[pons_mask] = new_label_array[pons_mask]
        output_label_array[medulla_mask] = new_label_array[medulla_mask]
        return output_label_array

    @classmethod
    def ventral_DC_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array)
        for k in cls.ventral_DC:
            index = np.argwhere(label_array == cls.ventral_DC[k])
            prerequisite_index = np.argwhere(label_array == cls.ventral_DC_prerequisite[k])

            index_z_unique = np.unique(index[:, 2])
            z_aixs_intersect = index_z_unique
            # Z 軸切片
            for j in z_aixs_intersect:
                index_sub = index[index[:, 2] == j]
                prerequisite_index_sub = prerequisite_index[prerequisite_index[:, 2] == j]
                if prerequisite_index_sub.shape[0] > 0:
                    if prerequisite_index_sub.shape[0] / index_sub.shape[0] >= cls.PREREQUISITE_THRESHOLD:
                        new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][
                            0]
                    else:
                        new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][
                            1]
                else:
                    new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][1]
        return new_label_array

    @classmethod
    def DPWM_redivide(cls, label_array):
        # 輸出 nii.gz 的 array
        new_label_array = label_array.copy()
        # 建立 誤差(距離)矩陣

        for k in cls.bullseye_DPWM_parcellation:
            # 取出 bullseye_DPWM 部分
            index = np.argwhere(label_array == cls.bullseye_DPWM[k])
            hemi_parcellation_mapping_keys = list(cls.bullseye_DPWM_parcellation[k].keys())
            np_loss = np.zeros((index.shape[0], len(hemi_parcellation_mapping_keys)))
            np_loss[:, :] = 999999
            # Z 軸
            z_aixs_intersect = np.unique(index[:, 2])
            for i in range(len(hemi_parcellation_mapping_keys)):
                label_index = np.argwhere(label_array == hemi_parcellation_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
                for j in z_aixs_intersect:
                    index_sub = index[index[:, 2] == j]
                    index_sub_arg = np.argwhere(index[:, 2] == j)
                    label_index_sub = label_index[label_index[:, 2] == j]
                    if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                        loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                        loss_list.append(loss_min.numpy())
                        index_sub_arg_list.append(index_sub_arg)
                    else:
                        continue
                if len(index_sub_arg_list) > 0:
                    index_sub_arg = np.concatenate(index_sub_arg_list)
                    np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
                else:
                    continue
            new_label = np_loss.argmin(axis=1)
            # 指定分類
            for ii in np.unique(new_label):
                select_index = index[np.argwhere(new_label == ii)].squeeze()
                new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                    cls.bullseye_DPWM_parcellation[k][hemi_parcellation_mapping_keys[ii]]
        return new_label_array

    @classmethod
    def left_right_translate(cls, slice):
        index_112 = np.argwhere(slice == 112)
        index_212 = np.argwhere(slice == 212)
        if index_112[:, 0].min() < index_212[:, 0].min():
            return False, slice
        else:
            return True, np.flip(slice, 0)

    @classmethod
    def inverse_left_right_translate(cls, flip, slice):
        if flip:
            return np.flip(slice, 0)
        else:
            return slice

    @classmethod
    def run(cls, label_array):
        FLIP, label_array_translate = cls.left_right_translate(label_array)
        redivide_label_array = cls.DPWM_redivide(label_array_translate)
        dwi_array_translate = cls.david_label_to_DWI_label(redivide_label_array)
        brain_stem_array = cls.brain_stem_parcellation(redivide_label_array)
        ventral_DC_array = cls.ventral_DC_parcellation(redivide_label_array)
        array_frontal_array = cls.DPWM_parcellation(redivide_label_array, target='frontal')
        array_parietal_array = cls.DPWM_parcellation(redivide_label_array, target='parietal')
        brain_stem_mask = brain_stem_array != 0
        ventral_DC_mask = ventral_DC_array != 0
        array_frontal_mask = array_frontal_array != 0
        array_parietal_mask = array_parietal_array != 0
        dwi_array_translate[array_frontal_mask] = array_frontal_array[array_frontal_mask]
        dwi_array_translate[array_parietal_mask] = array_parietal_array[array_parietal_mask]
        dwi_array_translate[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        dwi_array_translate[ventral_DC_mask] = ventral_DC_array[ventral_DC_mask]
        dwi_array = cls.inverse_left_right_translate(FLIP, dwi_array_translate)
        return dwi_array
        # redivide_label_array = cls.DPWM_redivide(label_array)
        # dwi_array = cls.david_label_to_DWI_label(redivide_label_array)
        # brain_stem_array = cls.brain_stem_parcellation(redivide_label_array)
        # ventral_DC_array = cls.ventral_DC_parcellation(redivide_label_array)
        # array_frontal_array = cls.DPWM_parcellation(redivide_label_array, target='frontal')
        # array_parietal_array = cls.DPWM_parcellation(redivide_label_array, target='parietal')
        # DPWM_array = cls.DPWM_parcellation(label_array)
        # brain_stem_mask = brain_stem_array != 0
        # ventral_DC_mask = ventral_DC_array != 0
        # array_frontal_mask = array_frontal_array != 0
        # array_parietal_mask = array_parietal_array != 0
        # dwi_array[array_frontal_mask] = array_frontal_array[array_frontal_mask]
        # dwi_array[array_parietal_mask] = array_parietal_array[array_parietal_mask]
        # dwi_array[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        # dwi_array[ventral_DC_mask] = ventral_DC_array[ventral_DC_mask]
        # return dwi_array


class WMHProcess:
    label_david_mapping_WMH = {
        'left_hemi': {
            112: 0,
            117: 0,
            119: 110,
            113: 0,
            120: 111,
            114: 0,
            121: 112,
            115: 0,
            122: 113,
            116: 0,
            123: 0,
            118: 0,
            124: 0,
            108: 0,
            109: 0,
            106: 0,
            110: 0,
            111: 0,
            125: 102,
            126: 103,
            127: 104,
            128: 105,
            101: 0,
            102: 0,
            104: 0,
            105: 0,
            103: 0,
            129: 106,
            130: 107,
            131: 108,
            132: 109,
            # 16: 101,
            14: 0,
            15: 0,
            24: 0,
        },
        'right_hemi': {
            212: 0,
            217: 0,
            219: 210,
            213: 0,
            220: 211,
            214: 0,
            221: 212,
            215: 0,
            222: 213,
            216: 0,
            223: 0,
            218: 0,
            224: 0,
            208: 0,
            209: 0,
            206: 0,
            210: 0,
            211: 0,
            225: 202,
            226: 203,
            227: 204,
            228: 205,
            201: 0,
            202: 0,
            204: 0,
            205: 0,
            203: 0,
            229: 206,
            230: 207,
            231: 208,
            232: 209,
        },
    }

    BRAIN_STEM = 16
    # 2023-10-25 BRAIN_STEM不分左右改為 301
    brain_stem_mapping = {
        'left_hemi': 301,
        'right_hemi': 301
    }
    # brain_stem_mapping = {
    #     'left_hemi': 101,
    #     'right_hemi': 201
    # }
    brain_stem_hemi_parcellation = {
        'left_hemi': {
            104: 101,
            105: 101,
        },
        'right_hemi': {
            204: 201,
            205: 201,
        }
    }
    decimal_places = 8

    @classmethod
    def brain_stem_parcellation(cls, label_array):

        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        # 取出 brain stem 部分
        index = np.argwhere(label_array == cls.BRAIN_STEM)
        # 建立 誤差(距離)矩陣
        hemi_list = list(cls.brain_stem_hemi_parcellation.keys())
        np_loss = np.zeros((index.shape[0], len(hemi_list)))
        np_loss[:, :] = 999999
        for k in cls.brain_stem_hemi_parcellation:
            if k == 'left_hemi':
                i = 0
            else:
                i = 1
            hemi_parcellation_mapping_keys = list(cls.brain_stem_hemi_parcellation[k].keys())
            # brain stem  的 Z 軸
            z_aixs_intersect = np.unique(index[:, 2])

            label_index = np.argwhere(np.isin(label_array, hemi_parcellation_mapping_keys))
            loss_list = []
            index_sub_arg_list = []
            # Z 軸切片
            for j in z_aixs_intersect:
                index_sub = index[index[:, 2] == j]
                index_sub_arg = np.argwhere(index[:, 2] == j)
                label_index_sub = label_index[label_index[:, 2] == j]
                if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                    loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                    loss_list.append(loss_min.numpy())
                    index_sub_arg_list.append(index_sub_arg)
                else:
                    continue
            if len(index_sub_arg_list) > 0:
                index_sub_arg = np.concatenate(index_sub_arg_list)
                np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
            else:
                continue
        new_label = np_loss.argmin(axis=1)
        # 指定分類
        for ii in np.unique(new_label):
            select_index = index[np.argwhere(new_label == ii)].squeeze()
            new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.brain_stem_mapping[
                hemi_list[ii]]
        return new_label_array

    @classmethod
    def david_label_to_WMH_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.label_david_mapping_WMH:
            # run every synseg label to freesurfer label
            for key in cls.label_david_mapping_WMH[hemi]:
                if cls.label_david_mapping_WMH[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.label_david_mapping_WMH[hemi][key]
        return new_label_array

    @classmethod
    def run(cls, label_array):
        brain_stem_array = cls.brain_stem_parcellation(label_array)
        wmh_array = cls.david_label_to_WMH_label(label_array)
        brain_stem_mask = brain_stem_array != 0
        wmh_array[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        return wmh_array


class CorpusCallosumParcellationForWMHProcess(CorpusCallosumParcellation):
    @classmethod
    def _get_df_lateral_ventricle(cls, left_lateral_ventricle_index, right_lateral_ventricle_index):
        # print('CorpusCallosumParcellationForWMHProcess _get_df_lateral_ventricle')
        lateral_ventricle_vector = []
        left_lateral_ventricle_z_axis = np.unique(left_lateral_ventricle_index[:, 2])
        right_lateral_ventricle_z_axis = np.unique(right_lateral_ventricle_index[:, 2])
        z_axis_intersect = np.intersect1d(left_lateral_ventricle_z_axis, right_lateral_ventricle_z_axis)
        for i in z_axis_intersect:
            left_vector_x = left_lateral_ventricle_index[left_lateral_ventricle_index[:, 2] == i][:, 1]
            right_vector_x = right_lateral_ventricle_index[right_lateral_ventricle_index[:, 2] == i][:, 1]

            lateral_ventricle_vector.append([i,
                                             np.percentile(left_vector_x, 50, axis=0).astype(int),
                                             np.percentile(right_vector_x, 50, axis=0).astype(int)])

        intersect_lateral_ventricle_array = np.array(lateral_ventricle_vector)
        df_lateral_ventricle = pd.DataFrame(intersect_lateral_ventricle_array, columns=['lateral_ventricle_z',
                                                                                        'left_lateral_ventricle_x',
                                                                                        'right_lateral_ventricle_x'])
        df_lateral_ventricle.index = df_lateral_ventricle['lateral_ventricle_z']
        return df_lateral_ventricle


def data_translate(slice):
    """Flip and swap axes for TMU scans."""
    slice = np.swapaxes(slice, 0, 1)
    # TMU scans need to be flipped
    slice = np.flip(slice, 0)
    slice = np.flip(slice, 1)
    return slice


def inverse_data_translate(slice):
    """Flip and swap axes for TMU scans."""
    slice = np.swapaxes(slice, 1, 0)
    # TMU scans need to be flipped
    slice = np.flip(slice, 0)
    slice = np.flip(slice, 1)
    return slice


def left_right_translate(slice):
    """Check and optionally flip for left-right consistency."""
    index_3001 = np.argwhere(slice == 3001)
    index_4001 = np.argwhere(slice == 4001)
    if index_3001[:, 1].min() < index_4001[:, 1].min():
        return False, slice
    else:
        return True, np.flip(slice, 1)

def inverse_left_right_translate(flip, slice):
    """Inverse the left-right flipping if necessary."""
    if flip:
        return np.flip(slice, 1)
    else:
        return slice



def process_file(file_path, depth_number, args, synthseg_array, synthseg33_array):
    """Process a single file for different algorithms."""
    seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
        synthseg_array=synthseg_array, synthseg33=synthseg33_array, depth_number=depth_number
    )

    if args.all or args.wm_file:
        save_nifti(seg_array, synthseg_array, args.wm_file_list, file_path)

    if args.all or args.cmb:
        cmb_array = CMBProcess.run(seg_array)
        save_nifti(cmb_array, synthseg_array, args.cmb_file_list, file_path)

    if args.all or args.dwi:
        dwi_array = DWIProcess.run(seg_array)
        save_nifti(dwi_array, synthseg_array, args.dwi_file_list, file_path)

    if args.all or args.wmh:
        wmh_array = run_wmh(synthseg_array=synthseg_array, synthseg_array_wm=synthseg_array_wm,
                            depth_number=depth_number)
        save_nifti(wmh_array, synthseg_array, args.wmh_file_list, file_path)


def save_nifti(data_array, reference_nii, file_list, file_path):
    """Save NIfTI file based on reference image."""
    out_nib = nib.Nifti1Image(data_array, reference_nii.affine, reference_nii.header)
    nib.save(out_nib, os.path.join(file_list, os.path.basename(file_path)))


def replace_suffix(filename, new_suffix):
    """Replace the .nii or .nii.gz suffix with a new one."""
    pattern = r'\.nii\.gz$|\.nii$'
    return re.sub(pattern, new_suffix, filename)

def prepare_file_lists(args, file_list):
    """Prepare output file lists based on arguments."""
    suffix_map = {
        'cmb_file': args.cmb_file,
        'dwi_file': args.dwi_file,
        'wmh_file': args.wmh_file
    }
    file_lists = {}
    for key, suffix in suffix_map.items():
        if getattr(args, key.split('_')[0]):
            file_lists[key] = [replace_suffix(f, f'_{suffix}.nii.gz') for f in file_list]
        else:
            file_lists[key] = []
    file_lists['wm_file'] = [replace_suffix(f, '_david.nii.gz') for f in file_list]
    return file_lists

def run(synthseg_array, depth_number):
    """Run the parcellation process."""
    synthseg_array = synthseg_array.round(0).astype(int)
    synthseg_array_wm = WhiteMatterParcellation.run(synthseg_array)
    synthseg_array_cc = CorpusCallosumParcellation.run(synthseg_array_wm)
    synthseg_array_ec = ECICParcellation.run(synthseg_array_wm)
    revert_array = WhiteMatterParcellation.re_run(synthseg_array_wm, synthseg_array_cc, synthseg_array_ec)
    re_white_matter_parcellation_array = WhiteMatterParcellation.re_white_matter_parcellation(revert_array)
    synthseg_array_bullseye = BullseyeProcess.run(re_white_matter_parcellation_array, synthseg_array, depth_number=depth_number)
    out_array = re_white_matter_parcellation_array.copy()
    cc_mask = synthseg_array_cc != 0
    ecic_mask = synthseg_array_ec != 0
    bullseye_mask = synthseg_array_bullseye != 0
    out_array[bullseye_mask] = synthseg_array_bullseye[bullseye_mask]
    out_array[ecic_mask] = synthseg_array_ec[ecic_mask]
    out_array[cc_mask] = synthseg_array_cc[cc_mask]
    david_label_array = WhiteMatterParcellation.label_to_david_label(out_array)
    return david_label_array, synthseg_array_wm


def run_with_WhiteMatterParcellation(synthseg_array, synthseg33, depth_number):
    """Run parcellation with WhiteMatterParcellation."""
    synthseg_array = synthseg_array.round(0).astype(int)
    synthseg33_array = synthseg33.round(0).astype(int)
    synthseg_array_wm = WhiteMatterParcellation2.run(synthseg_array=synthseg_array, synthseg33_array=synthseg33_array)
    synthseg_array_cc = CorpusCallosumParcellation.run(synthseg_array_wm)
    synthseg_array_ec = ECICParcellation.run(synthseg_array_wm)
    revert_array = WhiteMatterParcellation.re_run(synthseg_array_wm, synthseg_array_cc, synthseg_array_ec)
    re_white_matter_parcellation_array = WhiteMatterParcellation.re_white_matter_parcellation(revert_array)
    synthseg_array_bullseye = BullseyeProcess.run(re_white_matter_parcellation_array, synthseg_array, depth_number=depth_number)
    out_array = re_white_matter_parcellation_array.copy()
    cc_mask = synthseg_array_cc != 0
    ecic_mask = synthseg_array_ec != 0
    bullseye_mask = synthseg_array_bullseye != 0
    out_array[bullseye_mask] = synthseg_array_bullseye[bullseye_mask]
    out_array[ecic_mask] = synthseg_array_ec[ecic_mask]
    out_array[cc_mask] = synthseg_array_cc[cc_mask]
    david_label_array = WhiteMatterParcellation.label_to_david_label(out_array)
    return david_label_array, synthseg_array_wm

def run_cmb(synthseg_array, depth_number=5):
    """Run CMB process."""
    david_label_array, synthseg_array_wm = run(synthseg_array=synthseg_array, depth_number=depth_number)
    return CMBProcess.run(david_label_array)

def run_wmh(synthseg_array, synthseg_array_wm, depth_number=5):
    """Run WMH process."""
    synthseg_array = synthseg_array.round(0).astype(int)
    synthseg_array_cc = CorpusCallosumParcellationForWMHProcess.run(synthseg_array_wm)
    synthseg_array_ec = ECICParcellation.run(synthseg_array_wm)
    revert_array = WhiteMatterParcellation.re_run(synthseg_array_wm, synthseg_array_cc, synthseg_array_ec)
    re_white_matter_parcellation_array = WhiteMatterParcellation.re_white_matter_parcellation(revert_array)
    synthseg_array_bullseye = BullseyeProcess.run(re_white_matter_parcellation_array, synthseg_array, depth_number=depth_number)
    out_array = re_white_matter_parcellation_array.copy()
    cc_mask = synthseg_array_cc != 0
    ecic_mask = synthseg_array_ec != 0
    bullseye_mask = synthseg_array_bullseye != 0
    out_array[bullseye_mask] = synthseg_array_bullseye[bullseye_mask]
    out_array[ecic_mask] = synthseg_array_ec[ecic_mask]
    out_array[cc_mask] = synthseg_array_cc[cc_mask]
    david_label_array = WhiteMatterParcellation.label_to_david_label(out_array)
    return WMHProcess.run(david_label_array)

def run_dwi(synthseg_array, depth_number=5):
    """Run DWI process."""
    david_label_array, synthseg_array_wm = run(synthseg_array=synthseg_array, depth_number=depth_number)
    return DWIProcess.run(david_label_array)

def str_to_bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('false', 'no', 'n', 'f'):
        return False
    elif v.lower() in ('true', 'yes', 'y', 't'):
        return True
    else:
        raise argparse.ArgumentTypeError("Bool value expected")


def main(args):
    """Main function to process input arguments and run the parcellation."""
    # Prepare input and output paths
    file_list = [args.input] if args.input.endswith(('nii', 'nii.gz')) else glob.glob(f'{args.input}/*.nii*', recursive=True)
    assert file_list, 'No nii.gz files found'
    if args.input_name:
        file_list = [f for f in file_list if args.input_name in f]
    out_path = args.output if args.output else (args.input if os.path.isdir(args.input) else os.path.dirname(args.input))
    os.makedirs(out_path, exist_ok=True)

    # Prepare file lists
    args.cmb_file_list, args.dwi_file_list, args.wmh_file_list, args.wm_file_list = prepare_file_lists(args, file_list)
    depth_number = args.depth_number or 5

    # Process each file
    for file_path in file_list:
        synthseg_nii = nib.load(file_path)
        synthseg_array = np.array(synthseg_nii.dataobj)
        synthseg33_nii = nib.load(file_path.replace('synthseg.nii.gz', 'synthseg33.nii.gz'))
        synthseg33_array = np.array(synthseg33_nii.dataobj)
        try:
            process_file(file_path, depth_number, args, synthseg_array, synthseg33_array)
        except Exception as e:
            print(f'{file_path} processing error: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="Input synthseg file path or folder.")
    parser.add_argument('--input_name', help="Specific input file name to process.")
    parser.add_argument('-o', '--output', help="Output path for result files.")
    parser.add_argument('--all', type=str_to_bool, default=True, help="Run all algorithms.")
    parser.add_argument('--david', dest='wm_file', type=str_to_bool, default=True, help="Output white matter parcellation file.")
    parser.add_argument('--CMB', '--cmb', dest='cmb', type=str_to_bool, default=False, help="Output CMB Mask.")
    parser.add_argument('--CMBFile', '--cmbFile', dest='cmb_file', type=str, default='CMB', help="CMB Mask file name.")
    parser.add_argument('--DWI', '--dwi', dest='dwi', type=str_to_bool, default=False, help="Output DWI Mask.")
    parser.add_argument('--DWIFile', '--dwiFile', type=str, default='DWI', help="DWI Mask file name.")
    parser.add_argument('--WMH', '--wmh', dest='wmh', type=str_to_bool, default=False, help="Output WMH Mask.")
    parser.add_argument('--WMHFile', '--wmhFile', type=str, default='WMH', help="WMH Mask file name.")
    parser.add_argument('--depth_number', type=int, default=5, choices=range(4, 11), help="Deep white matter parameter.")
    args = parser.parse_args()
    main(args)
