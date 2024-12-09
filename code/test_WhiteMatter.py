import os
import glob
import pathlib
import re
import sys
import time
import traceback

import nibabel as nib
import argparse

import numpy as np

import tensorflow as tf
from pandas._libs import lib
from pandas._typing import npt
import log

def set_gpu(gpu_id='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        print(device)

    # select single GPU to use
    try:  # allow GPU memory growth
        tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    except:
        pass

    tf.config.set_visible_devices(devices=physical_devices[0], device_type='GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print('Seleted logical_devices:', logical_devices)

    tf.debugging.set_log_device_placement(True)
    tf.config.set_soft_device_placement(enabled=True)


set_gpu()


def pairwise_distance(a, b):
    return tf.reduce_sum(tf.pow(tf.abs(a - b), 2), axis=-1)


def loss_distance(index_sub, label_index_sub, decimal_places=8):
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



def white_matter_parcellation(label_array):
    # 輸出 nii.gz 的新 array
    new_label_array = label_array.copy()
    cerebral_white_matter = {
        'left_hemi': 2,
        'right_hemi': 41,
    }
    white_matter_mapping = {
        'left_hemi': {
            1001: 3001,
            1002: 3002,
            1003: 3003,
            1004: 3004,
            1005: 3005,
            1006: 3006,
            1007: 3007,
            1008: 3008,
            1009: 3009,
            1010: 3010,
            1011: 3011,
            1012: 3012,
            1013: 3013,
            1014: 3014,
            1015: 3015,
            1016: 3016,
            1017: 3017,
            1018: 3018,
            1019: 3019,
            1020: 3020,
            1021: 3021,
            1022: 3022,
            1023: 3023,
            1024: 3024,
            1025: 3025,
            1026: 3026,
            1027: 3027,
            1028: 3028,
            1029: 3029,
            1030: 3030,
            1031: 3031,
            1032: 3032,
            1033: 3033,
            1034: 3034,
            1035: 3035,

        },
        'right_hemi': {
            2001: 4001,
            2002: 4002,
            2003: 4003,
            2004: 4004,
            2005: 4005,
            2006: 4006,
            2007: 4007,
            2008: 4008,
            2009: 4009,
            2010: 4010,
            2011: 4011,
            2012: 4012,
            2013: 4013,
            2014: 4014,
            2015: 4015,
            2016: 4016,
            2017: 4017,
            2018: 4018,
            2019: 4019,
            2020: 4020,
            2021: 4021,
            2022: 4022,
            2023: 4023,
            2024: 4024,
            2025: 4025,
            2026: 4026,
            2027: 4027,
            2028: 4028,
            2029: 4029,
            2030: 4030,
            2031: 4031,
            2032: 4032,
            2033: 4033,
            2034: 4034,
            2035: 4035,
        },
    }
    for k in cerebral_white_matter:
        # 2 or 41
        index = np.argwhere(label_array == cerebral_white_matter[k])
        white_matter_mapping_keys = list(white_matter_mapping[k].keys())
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
                    loss_min = loss_distance(index_sub, label_index_sub, 8)
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
        # for i in np.unique(new_label):
        #     select_index = index[np.argwhere(new_label == i)].squeeze()
        #     new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
        #         white_matter_mapping[k][white_matter_mapping_keys[i]]
        # print('new_label', np.unique(new_label),new_label.shape)
        # print('temp_label', np.unique(temp_label), temp_label.shape)
        # print('index',index.shape)
        temp_label =  np.array(list(white_matter_mapping[k].values()))
        update_new_label(new_label=new_label,
                         new_label_array=new_label_array,
                         index=index,
                         temp_label=temp_label)
        # new_label_array[index[:, 0], index[:, 1], index[:, 2]] = temp_label[new_label]
    return new_label_array

@log.logging_time
def update_new_label(new_label,new_label_array,index,temp_label):
    # 指定分類
    # for i in np.unique(new_label):
    #     select_index = index[np.argwhere(new_label == i)].squeeze()
    #     new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
    #         white_matter_mapping[k][white_matter_mapping_keys[i]]
    # print('new_label', np.unique(new_label),new_label.shape)
    # print('temp_label', np.unique(temp_label), temp_label.shape)
    # print('index',index.shape)
    new_label_array[index[:, 0], index[:, 1], index[:, 2]] = temp_label[new_label]
    return new_label_array


@log.logging_time
def main(synthseg_file):
    try:
        start_time = time.time()
        synthseg_nii = nib.load(synthseg_file)
        synthseg_array = synthseg_nii.get_fdata()
        seg_array = white_matter_parcellation(synthseg_array)
        seg_array = seg_array.astype(np.int16)
        synthseg_nii.header.set_data_dtype('int16')
        out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
        nib.save(out_nib, R'D:\01_Lin\Task01_\seg\rMNI152_T1_1.5mm_whitematter_parcellation_0516.nii.gz')
        exec_time = time.time() - start_time
        print('exec time ', exec_time, ' seconds ')
    except Exception as e:
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0]  # 取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        print(errMsg)


# import pandas as pd
# class DataFrame(pd.DataFrame):
#     def to_numpy(self, dtype: npt.DTypeLike | None = None, copy: bool = False,
#                  na_value: object = lib.no_default) -> np.ndarray:
#         return super().to_numpy(dtype, copy, na_value)

def run_left(img,s):
    img_slice = img[:, :, s]
    distance_list_35 = []
    #     定義每一張slice的白質區域 左側先算
    white_matter_pos_left = np.argwhere(img_slice == 2)
    if white_matter_pos_left.shape[0] != 0:
        run_1001_1036(img_slice, white_matter_pos_left, distance_list_35)
        run_update_label(img, white_matter_pos_left, distance_list_35, 3001)


def run_right(img,s):
    img_slice = img[:, :, s]
    distance_list_35 = []
    #     定義每一張slice的白質區域 左側先算
    # 右側____________________________________________
    white_matter_pos_right = np.argwhere(img_slice==41)
    if white_matter_pos_right.shape[0] != 0:
        run_2001_2036(img_slice, white_matter_pos_right, distance_list_35)
        run_update_label(img, white_matter_pos_right, distance_list_35,4001)

if __name__ == '__main__':
    main(synthseg_file=r'D:\01_Lin\Task01_\seg\rMNI152_T1_1.5mm_synthseg.nii.gz',
         )
