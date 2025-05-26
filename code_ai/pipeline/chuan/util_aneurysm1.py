# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:59:21 2021

取名data util去記錄所有資料前處理的過程

@author: chuan
"""

#from logging.config import _RootLoggerConfiguration
import os
import time
import numpy as np
import pydicom
from pydicom.uid import JPEGLSLossless

import glob
import shutil
import nibabel as nib
import nibabel.processing
import matplotlib
import matplotlib.pyplot as plt
import sys
import logging
import cv2
import pandas as pd
import pydicom_seg
import SimpleITK as sitk
import matplotlib.colors as mcolors
#要安裝pillow 去開啟
from skimage import measure,color,morphology
from scipy.ndimage import rotate
from skimage.transform import resize
from skimage.measure import label, regionprops, regionprops_table
from scipy import ndimage
from collections import Counter
import pydicom_seg
import json
from collections import OrderedDict
from scipy import ndimage
from upload_orthanc import upload_data
import requests

import tensorflow as tf
import tensorflow.keras.backend as K
#print("Tensorflow version:", tf.__version__)
autotune = tf.data.experimental.AUTOTUNE
#s.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torchvision.transforms.functional import rotate as rotate_torch
from torchvision.transforms import InterpolationMode

from create_dicomseg_multi_file_json_claude import load_and_sort_dicom_files, make_study_json, MaskRequest


#會使用到的一些predict技巧
def data_translate(img, nii):
    img = np.swapaxes(img,0,1)
    img = np.flip(img,0)
    img = np.flip(img, -1)
    header = nii.header.copy() #抓出nii header 去算體積
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

def data_translate_back(img, nii):
    header = nii.header.copy() #抓出nii header 去算體積
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img,0)
    img = np.swapaxes(img,1,0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

#nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii

def reslice_nifti_pred_nobrain(path_nii, path_reslice):
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz')) #
    img = np.array(img_nii.dataobj) #讀出label的array矩陣      #
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz')) #
    original_affine = img_nii.affine.copy()  # 原始 affine

    img = data_translate(img, img_nii)
    y_i, x_i, z_i = img.shape #y, x, z

    header_img = img_nii.header.copy() #抓出nii header 去算體積
    pixdim_img = header_img['pixdim']  #可以借此從nii的header抓出voxel size

    #算出新版z軸要多少
    new_y_i = int(z_i * (pixdim_img[3] / pixdim_img[1]))

    #先把影像從 original*original*103轉成 original*original*103 * (pixdim_img[3] / pixdim_img[1])
    new_img_nii = nibabel.processing.conform(img_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 1) #影像用
    new_pred_nii = nibabel.processing.conform(pred_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用
    new_vessel_nii = nibabel.processing.conform(vessel_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用

    #讀出矩陣，然後第一值是負就反存，讀取 conform 後的 affine
    conformed_affine = new_img_nii.affine.copy()

    # 修正 affine，使其第一個 voxel dimension 方向與原始影像一致
    if np.sign(original_affine[0, 0]) != np.sign(conformed_affine[0, 0]):
        conformed_affine[0, :] *= -1  # 翻轉 X 軸方向

    # 建立新的影像
    fixed_img_nii = nib.Nifti1Image(new_img_nii.get_fdata(), conformed_affine, new_img_nii.header)
    fixed_pred_nii = nib.Nifti1Image(new_pred_nii.get_fdata().astype(int), conformed_affine, new_pred_nii.header)
    fixed_vessel_nii = nib.Nifti1Image(new_vessel_nii.get_fdata().astype(int), conformed_affine, new_vessel_nii.header)

    #輸出結果
    nib.save(fixed_img_nii, os.path.join(path_reslice, 'MRA_BRAIN.nii.gz'))
    nib.save(fixed_pred_nii, os.path.join(path_reslice, 'Pred.nii.gz'))
    nib.save(fixed_vessel_nii, os.path.join(path_reslice, 'Vessel.nii.gz'))

    return print('reslice OK!!!')


def create_MIP_pred(path_dcm, path_nii, path_png, gpu_num):

    #gpu_available = tf.config.list_physical_devices('GPU')
    #print(gpu_available)
    # set GPU
    #gpu_num = 0
    #tf.config.experimental.set_visible_devices(devices=gpu_available[gpu_num], device_type='GPU')

    #for gpu in gpu_available:
    #    tf.config.experimental.set_memory_growth(gpu, True)


    def img_to_MIPdicom(dcm, img, tag, series, SE, angle, count):
        y_i, x_i = img.shape
        dcm.PixelData = img.tobytes()  # 影像置換
        dcm[0x08,0x0008].value = ['DERIVED', 'SECONDARY', 'OTHER']
        dcm[0x08,0x103E].value = tag
        dcm[0x20,0x0011].value = SE

        dcm.add_new(0x00280006, 'US', 1)
        dcm[0x28,0x0010].value = y_i
        dcm[0x28,0x0011].value = x_i
        dcm[0x28,0x0100].value = 16
        dcm[0x28,0x0101].value = 16
        dcm[0x28,0x0102].value = 15

        try:
            del dcm[0x28,0x1050]
        except:
            pass
        try:
            del dcm[0x28,0x1051]
        except:
            pass

        seriesiu = dcm[0x20,0x000E].value
        new_seriesiu = seriesiu + '.' + str(series)
        dcm[0x20,0x000E].value = new_seriesiu
        sopiu = dcm[0x08,0x0018].value
        new_sopiu = sopiu + '.' + str(series) + str(angle)
        dcm[0x08,0x0018].value = new_sopiu

        PixelSpacing = dcm[0x28,0x0030].value
        ImagePosition = dcm[0x20,0x0032].value
        ImageOrientation = dcm[0x20,0x0037].value

        # 取得 Image Orientation 中的兩個向量
        row_vector = np.array(ImageOrientation[:3])
        col_vector = np.array(ImageOrientation[3:])

        # 計算法向量 (cross product)
        normal_vector = np.cross(row_vector, col_vector)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 標準化

        # 計算新的位置
        slice_thickness = PixelSpacing[0]  # 假設使用 PixelSpacing 的第一個值來定義每一層的厚度
        new_position = np.array(ImagePosition) - normal_vector * slice_thickness * count

        dcm[0x20,0x0032].value = [float(new_position[0]), float(new_position[1]), float(new_position[2])]
        dcm[0x20,0x1041].value = new_position[2]
        dcm[0x20,0x0013].value = count

        return dcm

    #跟gpu相關的function
    def dilation3d(x):
        kernel = np.ones((3, 3, 3), dtype=int) #3d dilation
        x_di = ndimage.binary_dilation(x, structure=kernel, iterations=15) #亂做n次，kernel越小算越快
        return x_di

    def rotate_tf(x, i, axes):
        c = np.zeros((int(x.shape[0]), int(x.shape[1]), 1))
        for j in range(0,i*5, i):
            r_x = rotate(x, j, axes, reshape=False)
            MIP_x = createMIP(r_x)
            c = np.concatenate([c, np.expand_dims(MIP_x, axis=-1)], axis = -1)
        #把第一個去除
        return c[:,:,1:]

    def rotation_3d(X, axis, theta, expand=True, fill=0.0, label=False, gpu=None):
        """
        The rotation is based on torchvision.transforms.functional.rotate, which is originally made for a 2d image rotation
        :param X: the data that should be rotated, a torch.tensor or an ndarray, with lenx * leny * lenz shape.
        :param axis: the rotation axis based on the keynote request. 0 for x axis, 1 for y axis, and 2 for z axis.
        :param expand:  (bool, optional) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
        :param fill:  (sequence or number, optional) –Pixel fill value for the area outside the transformed image. If given a number, the value is used for all bands respectively.
        :param theta: the rotation angle, Counter-clockwise rotation, [-180, 180] degrees.
        :return: rotated tensor.
        expand=True時要修正旋轉的大小
        原版是Z, H, W這樣的矩陣方向再去旋轉
        我這邊矩陣放 H, W, Z，所以在axis=0時變成左右轉
        """
        #先獲得x的長寬以用於更新尺寸
        #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        """
        初始化類並設置裝置。
        
        參數：
        - gpu：可選的整數，指定要使用的 GPU 設備編號（例如0、1、2、3等）。如果為 None，則根據 CUDA 是否可用選擇默認 GPU。
        """
        if gpu is not None:
            device = f'cuda:{gpu}'
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if isinstance(X, np.ndarray):
            # Make a copy of the array to avoid negative strides issue
            X = np.copy(X)
            X = np.copy(X)
            X = torch.from_numpy(X).float()

        X = X.to(device)

        if axis == 0:
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)

        elif axis == 1:
            X = X.permute((1, 0, 2))
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
            X = X.permute((1, 0, 2))
            X = torch.flip(X, [2])

        elif axis == 2:
            X = X.permute((2, 1, 0))
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=-theta, expand=expand, fill=fill)
            X = X.permute((2, 1, 0))
            X = torch.flip(X, [2]) #這個還要修正

        else:
            raise Exception('Not invalid axis')

        return X.squeeze(0)

    def createMIP_single(np_img):
        Mip_z = np.max(np_img, axis=-1)
        return Mip_z

    class createMIP:
        # 使用示例：
        # createMIP
        #createMIP = createMIP()

        # 假設已定義了translated_vessel_img、angle、angle_one、y_i和x_i
        # 呼叫process_images方法來處理圖像
        #processed_images = rotator.process_images(translated_vessel_img, angle, angle_one, y_i, x_i)

        def __init__(self):
            #__init__：初始化類並設置裝置（如果GPU可用則為'cuda:0'，否則為'cpu'）。
            #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            return

        def rotation_3d(self, X, axis, theta, expand=True, fill=0.0, label=False):
            """
            將一個3D tensor或numpy陣列沿指定軸進行旋轉。

            參數：
            - X：torch.tensor或np.ndarray，輸入的資料，形狀為(lenx, leny, lenz)。
            - axis：整數，旋轉軸（0表示x軸，1表示y軸，2表示z軸）。 0目前是水平轉, 1是垂直轉, 2是側手翻
            - theta：浮點數，旋轉角度（遞減角度為正）。
            - expand：布林值，可選，擴展輸出圖像大小（默認為True）。
            - fill：序列或數字，可選，填充值，用於轉換圖像外部區域。
            - label：布林值，可選，對標籤使用最近鄰插值（默認為False）。

            返回：
            - 旋轉後的tensor：torch.tensor，旋轉後的輸入tensor。
            """
            if isinstance(X, np.ndarray):
                X = np.copy(X)
                X = torch.from_numpy(X).float()

            X = X.to(self.device)

            if axis == 0:
                interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
                X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)

            elif axis == 1:
                X = X.permute((1, 0, 2))
                interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
                X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
                X = X.permute((1, 0, 2))
                X = torch.flip(X, [2]) #這個還要修正

            elif axis == 2:
                X = X.permute((2, 1, 0))
                interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
                X = rotate_torch(X, interpolation=interpolation_mode, angle=-theta, expand=expand, fill=fill)
                X = X.permute((2, 1, 0))
                X = torch.flip(X, [2]) #這個還要修正

            else:
                raise ValueError('無效的軸值。預期為0、1或2。')

            return X

        def process_images(self, translated_img, angle, angle_step, y_i, x_i, axis, label=False, gpu=None):
            """
            初始化類並設置裝置。

            參數：
            - gpu：可選的整數，指定要使用的 GPU 設備編號（例如0、1、2、3等）。如果為 None，則根據 CUDA 是否可用選擇默認 GPU。
            """
            if gpu is not None:
                self.device = f'cuda:{gpu}'
            else:
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            img_list = []

            #這邊補0的大小要跟axial view一樣


            angle_list = np.arange(0, angle, angle_step)

            for i in angle_list:
                if axis == 0:
                    output1 = self.rotation_3d(translated_img, 0, -float(i), expand=True, label=label) #水平
                elif axis == 1:
                    output1 = self.rotation_3d(translated_img, 1, -float(i), expand=True, label=label)  # 垂直旋轉

                MIP = torch.zeros(y_i, x_i).to(self.device)
                MIP_r = output1.amax(-1)
                y_r, x_r = MIP_r.shape[0], MIP_r.shape[1]

                if axis == 0:
                    if x_r > x_i:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_r - x_i) / 2)
                        MIP[slice_y:slice_y+y_r,:] = MIP_r[:,slice_x:slice_x+x_i]
                    else:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_i - x_r) / 2)
                        MIP[slice_y:slice_y+y_r,slice_x:slice_x+x_r] = MIP_r

                elif axis == 1:
                    if y_r > y_i:
                        slice_y = int((y_r - y_i) / 2)
                        slice_x = int((x_r - x_i) / 2)
                        MIP = MIP_r[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                    else:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_i - x_r) / 2)
                        MIP[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = MIP_r

                img_list.append(MIP)

            mip_img = torch.stack(img_list, dim=-1)
            #mip_img = torch.flip(mip_img, dims=[0])
            #mip_img = mip_img.permute(1, 2, 0)
            mip_img_np = mip_img.cpu().numpy()

            return mip_img_np


    #以下開始製作MIP跟label
    angle = 180
    angle_step = 3
    Series = ['MIP_Pitch', 'MIP_Yaw'] #用作2種方式
    #Series = ['MIP_Yaw', 'MIP_Pitch'] #用作2種方式

    #血管的mask也要跟著轉然後保存
    start = time.time()
    path_dcms = os.path.join(path_dcm, 'MRA_BRAIN')
    #先讀影像
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz')) #ADC要讀因為會取ADC值判斷
    img = np.array(img_nii.dataobj) #讀出label的array矩陣      #256*256*22
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #ADC要讀因為會取ADC值判斷
    pred = np.array(pred_nii.dataobj) #讀出label的array矩陣      #256*256*22
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz')) #ADC要讀因為會取ADC值判斷
    vessel = np.array(vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22

    img = data_translate(img, img_nii)
    pred = data_translate(pred, pred_nii)
    vessel = data_translate(vessel, vessel_nii)

    #以下改用tf的gpu測試
    vessel_tf = tf.convert_to_tensor(vessel, dtype=tf.bool)
    dilated_tf_npfunc = tf.numpy_function(dilation3d, [vessel_tf],tf.int32)
    vessel_di = dilated_tf_npfunc.numpy()

    #使用mask，brain mask要補齊vessel mask的部分，不然血管像被切掉
    vessel_img = (img * vessel_di).copy()

    #把影像的y軸裁切，使得旋轉時置中
    vessel_z = np.sum(vessel_img,  axis=(0, 1)) #對z軸sum
    vessel_z_list = [y for y, z in enumerate(vessel_z) if z > 0] #做前面那點label是在哪裡
    vessel_img = vessel_img[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    vessel = vessel[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    pred = pred[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    y_i, x_i, z_i = img.shape

    #讀取第一張dicom，剩下的資訊用這張來改
    reader = sitk.ImageSeriesReader()
    dcms_tofmra = reader.GetGDCMSeriesFileNames(path_dcms)

    #接下來做上下左右轉，需要先將矩陣轉置
    translated_vessel_img = np.swapaxes(vessel_img,0,-1).copy()
    #translated_vessel_img = np.flip(translated_vessel_img, 1)

    #label跟vessel還是要轉方向
    translated_vessel = np.swapaxes(vessel,0,-1).copy()
    #translated_vessel = np.flip(translated_vessel, 1)
    translated_pred = np.swapaxes(pred,0,-1).copy()

    print('img.shape:', img.shape,'translated_pred:', translated_pred.shape)
    for series in Series:
        path_vesselMIP = os.path.join(path_dcm, series)
        if not os.path.isdir(path_vesselMIP):
            os.mkdir(path_vesselMIP)

        #這邊讀取旋轉mark的圖片，準備做第一章的圖片疊圖
        png_landmark = cv2.imread(os.path.join(path_png, series + '.png'))
        png_landmark = png_landmark[:,:,0]

        if series == 'MIP_Pitch':
            axis = 1
            SE = 850
            series_uid = 3
        elif series == 'MIP_Yaw':
            axis = 0
            SE = 851
            series_uid = 6

        #先做垂直轉Pitch，再做水平轉Yaw
        start_img = time.time()
        createrMIP = createMIP()
        # 假設已定義了translated_vessel_img、angle、angle_one、y_i和x_i。 y_i和x_i是轉出目標影像的大小，以背景補0為主
        # 呼叫process_images方法來處理圖像
        MIP_images = createrMIP.process_images(translated_vessel_img, angle, angle_step, y_i, x_i, axis=axis, label=False, gpu=gpu_num).astype('int16')
        print('Time taken for {} sec\n'.format(time.time()-start_img), ' MIP_images is OK!!!, Next...')

        #血管確認是否有阻擋是改成前後一起看，所以可以當作MIP後確認重疊比例，所以可以label也做mip、vessel也做mip
        #MIP血管還是要做，因為要產生血管的mask
        MIP_vessels = createrMIP.process_images(translated_vessel, angle, angle_step, y_i, x_i, axis=axis, label=True, gpu=gpu_num).astype('int16')
        #print('Time taken for {} sec\n'.format(time.time()-start), 'MIP_images.shape:', MIP_images.shape)

        #label比較麻煩，需要重新根據旋轉的血管結果修正
        count_pred = 0 #紀錄現在是第幾張
        new_pred = np.zeros((y_i, x_i, 1)) #會跟mip的影像一樣大，雖已知大小，但跟for迴圈有關
        cover_ranges_pred = np.zeros((int(MIP_images.shape[-1]) , int(np.max(pred)))) #記錄哪張血管遮擋最少

        angle_list = np.arange(0, angle, angle_step)

        for i in angle_list:
            #結果還是要做出血管個別旋轉的圖
            if axis == 0:
                rotated_vessel = rotation_3d(translated_vessel,  axis, -float(i), expand=True, label=True, gpu=gpu_num).cpu().numpy().copy() #rotated_vessel大小會一直變動
            elif axis ==1:
                rotated_vessel = rotation_3d(translated_vessel,  axis, -float(i),  expand=True, label=True, gpu=gpu_num).cpu().numpy().copy() #rotated_vessel大小會一直變動

            #先找出投影后最前面那點pred的位置，以下都改成pred
            pred_num = np.max(pred)

            new_pred_slice = np.zeros((y_i, x_i)) #512*512
            for j in range(pred_num):
                pred_one = (translated_pred ==j+1).astype(int).copy()
                if axis == 0:
                    rotated_pred_one = rotation_3d(pred_one,  axis, -float(i), expand=True, label=True, gpu=gpu_num).cpu().numpy().copy() #rotated_vessel大小會一直變動
                elif axis == 1:
                    rotated_pred_one = rotation_3d(pred_one,  axis, -float(i),  expand=True, label=True, gpu=gpu_num).cpu().numpy().copy() #rotated_vessel大小會一直變動
                pred_z = np.sum(rotated_pred_one,  axis=(0, 1)) #對z軸sum
                pred_z_list = [y for y, z in enumerate(pred_z) if z > 0] #做前面那點label是在哪裡
                if len(pred_z_list) < 1:
                    pred_z_list = [1]
                #血管確認前面是否有阻擋，保底至少要維持3d矩陣
                if pred_z_list[0]-3 < 1:
                    index_front = 1
                else:
                    index_front = pred_z_list[0]-3 #保險一點

                if pred_z_list[-1]+3 > rotated_vessel.shape[-1] -1:
                    index_back = rotated_vessel.shape[-1] -1
                else:
                    index_back =  pred_z_list[-1]+3 #保險一點

                #對label做projection去獲得投影後的x,y
                pred_mip = createMIP_single(rotated_pred_one).astype(int)
                pred_mip_s = pred_mip.copy()
                vessel_mip_front = createMIP_single(rotated_vessel[:,:,:index_front])
                extra_pred = np.zeros((y_i, x_i))
                #print('extra_pred:', extra_pred.shape)
                y_r, x_r = pred_mip.shape
                pred_mip[vessel_mip_front > 0] = 0

                #這邊反而還要找一個血管是否有後面遮擋的，來計算遮擋比例，因此才能找比較好的mainSeg
                pred_cover = pred_mip.copy()
                vessel_mip_back = createMIP_single(rotated_vessel[:,:,index_back:])
                pred_cover[vessel_mip_back > 0] = 0

                if axis == 0:
                    if x_r > x_i:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_r - x_i) / 2)
                        extra_pred[slice_y:slice_y+y_r,:] = pred_mip[:,slice_x:slice_x+x_i]
                    else:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_i - x_r) / 2)
                        extra_pred[slice_y:slice_y+y_r,slice_x:slice_x+x_r] = pred_mip
                else:
                    if y_r > y_i:
                        slice_y = int((y_r - y_i)/2)
                        slice_x = int((x_r - x_i)/2)
                        extra_pred = pred_mip[slice_y:slice_y+y_i,slice_x:slice_x+x_i]
                    else:
                        slice_y = int((y_i - y_r)/2)
                        slice_x = int((x_i - x_r)/2)
                        extra_pred[slice_y:slice_y+y_r,slice_x:slice_x+x_r] = pred_mip

                new_pred_slice[extra_pred > 0] = int(j+1)

                #這邊先根據MIP計算出遮擋最大值，這邊統計一個數字是標註被血管擋掉的百分比(前後)，減少跑for迴圈
                #print('slice:', i, ' ane:', j, ' np.sum(label_cover):', np.sum(label_cover), ' np.sum(extra_label):', np.sum(extra_label))
                if np.sum(extra_pred) > 0:
                    cover_range = np.sum(pred_cover) / np.sum(pred_mip_s)
                    if cover_range > 0.98:
                        cover_range = 0.98
                else:
                    cover_range = 0
                cover_ranges_pred[count_pred, j] = cover_range
            count_pred += 1

            new_pred = np.concatenate([new_pred, np.expand_dims(new_pred_slice, axis=-1)], axis=-1)

        #這邊在第一張時多做一個含有landmark的圖
        MIP_mark = MIP_images[:,:,0].astype('int16').copy()
        fig_pitch = np.zeros((y_i, x_i)).astype('int16')
        #將圖帶入然後設定下大小，png_pitch縮小
        #因為輸入影像有大小章關係，所以用出動態的縮放值
        percentage_x = y_i / 512
        y_displacement = int(250*percentage_x)
        x_displacement = int(110*percentage_x)

        pitch_resize = resize(png_landmark, (int((png_landmark.shape[0]*percentage_x)/3),int((png_landmark.shape[1]*percentage_x)/3)), order=0, mode='symmetric', cval=0, clip=False, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)
        fig_pitch[y_displacement:pitch_resize.shape[0]+y_displacement, x_displacement:pitch_resize.shape[1]+x_displacement] = pitch_resize
        fig_pitch[fig_pitch > 0] = np.max(MIP_images[:,:,0]) #因為疊上去讓他報數值，這樣背景就會變暗
        output_pitch = cv2.add(MIP_mark, fig_pitch)  # 疊加第一張
        dcm_slice = pydicom.dcmread(dcms_tofmra[-1]) #dicom每次都會連著修改一下，所以範本要在迴圈中讀取
        new_dcm_seg = img_to_MIPdicom(dcm_slice, output_pitch, series, series_uid, SE, '00', 0) #從0開始
        new_dcm_seg.save_as(os.path.join(path_vesselMIP, series + '_' + str(0).rjust(4,'0') + '_.dcm')) #存出新dicom，同存同一層，因為有壓時間，所以不會互相覆蓋

        #因為做出新標註需要影像的affine matrix，所以這邊先做出dicom並轉出nifti
        for k in range(int(MIP_images.shape[-1])):
            dcm_slice = pydicom.dcmread(dcms_tofmra[-1]) #dicom每次都會連著修改一下，所以範本要在迴圈中讀取
            new_dcm_seg = img_to_MIPdicom(dcm_slice, MIP_images[:,:,k], series, series_uid, SE, k, k+1) #最後一項是image position所以多下降一次
            new_dcm_seg.save_as(os.path.join(path_vesselMIP, series + '_' + str(k).rjust(4,'0') + '.dcm')) #存出新dicom，同存同一層，因為有壓時間，所以不會互相覆蓋

        #先製作出影像的nifti，如後讀取後將相關資訊套用到新label中，以下跑dcm2niix
        bash_line = 'dcm2niix -z y -f ' + series + ' -o ' + path_nii + ' ' + path_vesselMIP #把tag檔案複製，-v : verbose (n/y or 0/1/2, default 0) [no, yes, logorrheic]
        #print('bash_line:', bash_line)
        os.system(bash_line)

        #讀取nifti取得新affine matrix
        img_pitch_nii = nib.load(os.path.join(path_nii, series + '.nii.gz')) #ADC要讀因為會取ADC值判斷

        #這邊先做血管mask存出mask，取完前三張，最後加上1張最頂端landmark，再重組回去
        vessel_landmark = np.zeros((y_i, x_i, 1))
        vessel_pitch = np.concatenate([vessel_landmark, MIP_vessels], axis=-1)
        #存出nifti
        vessel_pitch = data_translate_back(vessel_pitch, img_pitch_nii).astype('int16')
        vessel_pitch_nii = nii_img_replace(img_pitch_nii, vessel_pitch)
        nib.save(vessel_pitch_nii, os.path.join(path_nii, series + '_vessel.nii.gz'))

        #for迴圈完成的標註，這邊開始處理更新版標註，差異在只取重疊率最小加上+-1張
        new_pred = new_pred[:,:,1:]
        pred_top = np.zeros((new_pred.shape)).astype(int)
        #print('cover_ranges:', cover_ranges)
        #每顆標註只取top3張，規則是找沒有被1. 血管阻擋的 2. 之後最大面積
        for l in range(int(np.max(new_pred))):
            cover_list = cover_ranges_pred[:,l]
            cover_z = np.where(cover_list == np.max(cover_list))[0]
            predone = (new_pred ==l+1).astype(int).copy()
            areas = list(np.sum(predone[:,:,cover_z], axis = (0, 1))) #即可知道哪張切片面積最大，cover_z這幾張比大小

            index_area = cover_z[areas.index(max(areas))]
            pred_area = np.zeros((new_pred.shape))
            if index_area == 0:
                pred_area[:,:,0:3] = predone[:,:,0:3]
            elif index_area == new_pred.shape[-1]:
                pred_area[:,:,-3:] = predone[:,:,-3:]
            else:
                pred_area[:,:,index_area-1:index_area+2] = predone[:,:,index_area-1:index_area+2]
            pred_top[pred_area > 0] = int(l+1)
            # if series == 'MIP_Yaw' and l==1:
            #     print(series, ' cover_list:', cover_list)
            #     print('np.sum(label_area:):', np.sum(label_area))
            #     print('areas:', areas, ' index_area:', index_area)

        #取完前三張，最後加上1張最頂端landmark，再重組回去
        pred_landmark = np.zeros((y_i, x_i, 1))
        pred_top = np.concatenate([pred_landmark, pred_top], axis=-1)

        #存出nifti
        pred_top_save = data_translate_back(pred_top, img_pitch_nii).astype('int16')
        pred_top_save_nii = nii_img_replace(img_pitch_nii, pred_top_save)
        nib.save(pred_top_save_nii, os.path.join(path_nii, series + '_pred.nii.gz'))
        print('Time taken for {} sec\n'.format(time.time()-start))

#跟gpu相關的function
def dilation3d(x):
    kernel = np.ones((3, 3, 3), dtype=int) #3d dilation
    x_di = ndimage.binary_dilation(x, structure=kernel, iterations=1) #亂做n次，kernel越小算越快
    return x_di

def make_aneurysm_vessel_location_16labels_pred(path):
    #先做Pred
    label_nii = nib.load(os.path.join(path, 'Pred.nii.gz'))
    label = np.array(label_nii.dataobj) #讀出label的array矩陣      #256*256*22
    label = data_translate(label, label_nii)

    vessel_nii = nib.load(os.path.join(path, 'Vessel_16.nii.gz'))
    vessel = np.array(vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22
    vessel = data_translate(vessel, vessel_nii)

    new_label = np.zeros((label.shape)).astype(int)

    #一顆一顆做
    for i in range(int(np.max(label))):
        label_one = label == i+1
        if np.sum(label_one * vessel) > 0:
            ane_overlap = label_one.copy() * vessel.copy()
            #統計ane_overlap除0以外哪個數值出現最多次，提取所有非零的數值
            non_zero_elements = ane_overlap[ane_overlap != 0]

            # 使用 Counter 計算每個非零數值的出現次數
            counter = Counter(non_zero_elements)
            # 找到出現次數最多的非零數值及其出現次數
            most_common_value, most_common_count = counter.most_common(1)[0]
            #將值替換過去mask
            new_label[label_one > 0] = int(most_common_value)

        else:
            #需要針對aneurysm做dilation看什麼時候會跟血管重疊然後歸屬血管
            dilation_one = label_one.copy()

            while np.sum(dilation_one * vessel) == 0:
                dilation_one = dilation3d(dilation_one)

            #統計ane_overlap除0以外哪個數值出現最多次，提取所有非零的數值
            ane_overlap = dilation_one.copy() * vessel.copy()

            non_zero_elements = ane_overlap[ane_overlap != 0]

            # 使用 Counter 計算每個非零數值的出現次數
            counter = Counter(non_zero_elements)
            # 找到出現次數最多的非零數值及其出現次數
            most_common_value, most_common_count = counter.most_common(1)[0]
            #將值替換過去mask
            new_label[label_one > 0] = int(most_common_value)

            print('失敗慘了', i+1, ' 是沒有在血管mask上, 但最後分配給:', int(most_common_value))

    #最後存出新mask，存出nifti
    new_label_new = data_translate_back(new_label, label_nii).astype(int)
    new_label_new_nii = nii_img_replace(label_nii, new_label_new)
    nib.save(new_label_new_nii, os.path.join(path, 'Pred_Location16labels.nii.gz'))

def calculate_aneurysm_long_axis_make_pred(path_dcm, path_nii, path_excel, ID):
    #第3輪的資料是額外的，這邊改成建立EXCEL
    text = [['PatientID', 'StudyDate', 'AccessionNumber', 'Aneurysm_Number', 'Value', 'Size', 'Prob_max', 'Prob_mean',
            'Location4labels', 'Location6labels']] #一顆一排一直往下紀錄
    df = pd.DataFrame(text[1:], columns=text[0])
    df.to_excel(os.path.join(path_excel, 'Aneurysm_Pred_long_axis_list.xlsx'), index = False)

    PID = ID[:8]
    Sdate = ID[9:17]
    AN = str('_'.join(ID.split('/')[-1].split('_')[3:4]))

    #以下讀image
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz'))
    img_array = np.array(img_nii.dataobj)
    img_array = data_translate(img_array, img_nii)

    try:
        #先讀取img跟label
        if os.path.isdir(os.path.join(path_dcm, 'TOF_MRA')):
            img_list = sorted(os.listdir(os.path.join(path_dcm, 'TOF_MRA')))
            dcm = pydicom.dcmread(os.path.join(path_dcm, 'TOF_MRA', img_list[1])) #取第2個保險
        elif os.path.isdir(os.path.join(path_dcm, 'MRA_BRAIN')):
            img_list = sorted(os.listdir(os.path.join(path_dcm, 'MRA_BRAIN')))
            dcm = pydicom.dcmread(os.path.join(path_dcm, 'MRA_BRAIN', img_list[1])) #取第2個保險
        else:
            print('GG啦!!!!')
            #assert len(img_list) == 10000, print('沒找到任何影像')
        #取出'Pixel Spacing(mm)','Spacing Between Slices(mm)'，用以計算ml
        pixel_size = dcm[0x28,0x0030].value # Image Resolution mm2/pixel
        spacing = float(dcm[0x18,0x0088].value)
        ml_size = pixel_size[0] * pixel_size[1] * spacing / 1000
    except:
        #改讀取mifti的pixel_size資訊
        header_true = img_array.header.copy() #抓出nii header 去算體積
        pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size
        #print('pixdim:',pixdim)
        ml_size = (pixdim[1]*pixdim[2]*pixdim[3]) / 1000 #轉換成ml

    #以下讀label
    nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz'))
    mask_array = np.array(nii.dataobj) #讀出label的array矩陣      #256*256*22
    mask_array = data_translate(mask_array, nii)
    y_i, x_i, z_i = mask_array.shape

    prob_nii = nib.load(os.path.join(path_nii, 'Prob.nii.gz'))
    prob = np.array(prob_nii.dataobj) #讀出label的array矩陣      #256*256*22
    prob = data_translate(prob, prob_nii)

    #以下算出一個case中aneurysm的顆數
    ane_index = 0
    for i in range(int(np.max(mask_array))):
        y_cl, x_cl, z_cl = np.where(mask_array==(i+1))     #256*256*22
        #同時去獲得location的位置
        label_one = mask_array == i+1
        prob_one = prob[label_one > 0].copy()

        # labels4_list = labels4[label_one > 0]
        # # 得到唯一值的数量
        # num_unique_values_4labels = len(np.unique(labels4_list))
        # assert num_unique_values_4labels == 1, print('挫屎!!! 同一顆怎麼那麼多location')

        #labels6_list = labels6[label_one > 0]
        # 得到唯一值的数量
        #num_unique_values_6labels = len(np.unique(labels6_list))
        #assert num_unique_values_6labels == 1, print('挫屎!!! 同一顆怎麼那麼多location')

        if len(y_cl) > 0:
            cluster_size = len(y_cl)
            cluster_ml = cluster_size*ml_size #算出換算出的ml大小
            aneurysm_ml = float(str('%.2f'%cluster_ml))
            #做出立體矩陣，接下來用平面算出外接圓當作最長徑
            cluster_matrix = np.zeros(mask_array.shape)

            #下面計算aneurysm的x,y軸外接圓最大徑
            cluster_matrix[y_cl,x_cl,z_cl] = 254 #cv2的白，因為cv2是在 [0-255] base的，所以=255
            cluster_z_list = np.sum(cluster_matrix, axis=(0,1)) #對z軸sum
            z_cluster_l = np.where(cluster_z_list>0)
            z_c_l_num = len(z_cluster_l[0])
            z_cluster_m = np.zeros((z_c_l_num)) #做跟有aneurysm z軸等長的矩陣以方便np.max比較，真實長度
            z_cluster_int = np.zeros((z_c_l_num)) #做跟有aneurysm z軸等長的矩陣以方便np.max比較，pixel長度
            site = []
            for j in range(z_c_l_num):
                mask_fig = cluster_matrix[:,:,z_cluster_l[0][j]] #抓特定平面出來，去計算
                mask_fig = np.expand_dims(mask_fig, axis=-1)
                mask_fig = np.array(mask_fig,np.uint8)
                ret,thresh = cv2.threshold(mask_fig,127,255,cv2.THRESH_BINARY)
                contours,hierarchy=cv2.findContours(thresh,1,2)
                cnt = contours[0] #找出輪廓的座標
                #最小外接圆，函数cv2.minEnclosingCircle()可以帮我们找到一个对象的外接圆。它是所有能够包括对象的圆中面积最小的一个。
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                site.append([x,y,radius])
                diameter_mm = 2*radius*pixel_size[0] #直徑
                z_cluster_m[j] = diameter_mm #直接輸入矩陣當值
                diameter = 2*radius
                z_cluster_int[j] = diameter #直接輸入矩陣當值
                #center = (int(x),int(y))
                #radius = int(radius)
                #print('radius:',radius)
                #mask_fig = cv2.circle(mask_fig,center,radius,255,3)
                #cv2.imwrite('D:/' + str(j) + '.png', mask_fig)
            max_maxis = np.max(z_cluster_m)
            long_axis =  round(max_maxis, 1)
            max_maxis_int = np.max(z_cluster_int)
            long_axis_float = round(max_maxis_int, 1)

            #填入excel表格中，針對顆去填
            df.at[ane_index, 'PatientID'] = PID
            df.at[ane_index, 'StudyDate'] = Sdate
            df.at[ane_index, 'AccessionNumber'] = AN
            df.at[ane_index, 'Aneurysm_Number'] = int(np.max(mask_array))
            df.at[ane_index, 'Value'] = int(i+1)
            df.at[ane_index, 'Prob_max'] = round(np.max(prob_one), 2)
            df.at[ane_index, 'Prob_mean'] = round(np.mean(prob_one), 2)
            df.at[ane_index, 'Size'] = long_axis-0.5 #做修正
            df.at[ane_index, 'Location4labels'] = ''
            #df.at[ane_index, 'Location6labels'] = ''
            ane_index += 1

            #製作成excel
            df.to_excel(os.path.join(path_excel, 'Aneurysm_Pred_long_axis_list.xlsx'), index = False)

def make_table_row_patient_pred(path_excel, ID):
    excel_name2 = 'Aneurysm_Pred_long_axis_list.xlsx'

    PID1s = [ID[:8]]
    SDate1s = [ID[9:17]]
    AN1s = [str('_'.join(ID.split('/')[-1].split('_')[3:4]))]

    #先讀取excel，取得patientid跟accession_number當成資料夾檔名條件
    dtypes1 = {'ID': str, 'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'SubjectLabel': str, 'ExperimentLabel': str,
            'SeriesInstanceUID': str, 'url': str}


    dtypes2 = {'ID': str, 'PatientID': str, 'AccessionNumber': str}

    df2 = pd.read_excel(os.path.join(path_excel, excel_name2), dtype=dtypes1).fillna('')
    """
    ID2s = list(df2['ID'])

    for idx, ID2 in enumerate(ID2s):
        df2.at[idx, 'PatientID'] = str(ID2[:8]).rjust(8, '0') #將長度填入excel表中
        df2.at[idx, 'AccessionNumber'] = str(ID2[9:]) #將長度填入excel表中

    #存出新excel
    df2.to_excel(os.path.join(path_excel, excel_name2), index = False)
    """
    #ID2s = list(df2['ID'])
    PID2s = list(df2['PatientID'])
    AN2s = list(df2['AccessionNumber'])
    ID2s = [x + '_' + y for x, y in zip(PID2s, AN2s)]

    AneurysmNums = list(df2['Aneurysm_Number']) #label value => nifti的標籤值
    Values = list(df2['Value'])
    Sizes = list(df2['Size'])
    Prob_maxs = list(df2['Prob_max'])
    Prob_means = list(df2['Prob_mean'])
    Location4labels = list(df2['Location4labels'])
    #Location6labels = list(df2['Location6labels'])

    max_num  = max(AneurysmNums)

    #製作新列表的標頭名稱，加上長軸長度，列一個新列表
    mm_list = ['PatientID', 'StudyDate', 'AccessionNumber' , 'Aneurysm_Number']
    #XNAT Link	Doctor_Check	impression_str	aneurysm_number	1_Prob_mean	1_Prob_max	1_size	type_1	"1: ICA 2: P-com; 3: A-com; 4: MCA; 5: ACA; 6: PCA; 7: BA;8: VA;9: AICA/PICA/SCA
    #"	 1:cervical 2: supraclinoid 3: carotid siphone 4:cavernous 5:terminal 6:opthalmic 7: clinoid 8: communicating 9:distal 10:C4 11:C5 12:C6 13: C7 14:paraclinoid  15:bifurcation 16: frontal scalp	1_Comment	location_1

    #依照最大aneurysm數量增長list
    for i in range(max_num):
        #mm_list.append(str(i+1) + '_Value')
        mm_list.append(str(i+1) + '_size')
        mm_list.append(str(i+1) + '_Prob_max') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_Prob_mean') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_Confirm') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_type') #紀錄在nifti標註中label的數值
        #mm_list.append(str(i+1) + '_1: ICA 2: P-com; 3: A-com; 4: MCA; 5: ACA; 6: PCA; 7: BA;8: VA;9: AICA/PICA/SCA') #紀錄在nifti標註中label的數值
        #mm_list.append(str(i+1) + '_1:cervical 2: supraclinoid 3: carotid siphone 4:cavernous 5:terminal 6:opthalmic 7: clinoid 8: communicating 9:distal 10:C4 11:C5 12:C6 13: C7 14:paraclinoid  15:bifurcation 16: frontal scalp') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_Location') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_SubLocation') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_Location4labels') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_Location6labels') #紀錄在nifti標註中label的數值
        mm_list.append(str(i+1) + '_Comment') #紀錄在nifti標註中label的數值

    #將list字典化
    mm_dict = {key: [] for key in mm_list}


    for idx, (PID1, SDate1, AN1) in enumerate(zip(PID1s, SDate1s, AN1s)):
        #接下來針對有換算aneurysm個別大小的list，找出屬於他的aneurysm並排列大小
        print([idx], PID1 + '_' + AN1, ' Start...')
        ane_list = [jdx for jdx, y in enumerate(ID2s) if y == PID1 + '_' + AN1] #抓出相同病人的位置
        #print('ane_list:', ane_list)
        #以下紀錄資訊
        mm_dict['PatientID'].append(PID1)
        mm_dict['StudyDate'].append(SDate1)
        mm_dict['AccessionNumber'].append(AN1)

        if len(ane_list) > 0:
            mm_dict['Aneurysm_Number'].append(AneurysmNums[ane_list[0]])

            #更新long axis的dict
            for j, ane_inx in enumerate(ane_list):
                #mm_dict[str(j+1) + '_Value'].append(Values[ane_inx])
                mm_dict[str(j+1) + '_size'].append(round(Sizes[ane_inx], 3))
                mm_dict[str(j+1) + '_Prob_max'].append(Prob_maxs[ane_inx])
                mm_dict[str(j+1) + '_Prob_mean'].append(Prob_means[ane_inx])
                mm_dict[str(j+1) + '_Confirm'].append('')
                mm_dict[str(j+1) + '_type'].append('')
                #mm_dict[str(j+1) + '_1: ICA 2: P-com; 3: A-com; 4: MCA; 5: ACA; 6: PCA; 7: BA;8: VA;9: AICA/PICA/SCA'].append('')
                #mm_dict[str(j+1) + '_1:cervical 2: supraclinoid 3: carotid siphone 4:cavernous 5:terminal 6:opthalmic 7: clinoid 8: communicating 9:distal 10:C4 11:C5 12:C6 13: C7 14:paraclinoid  15:bifurcation 16: frontal scalp'].append('')
                mm_dict[str(j+1) + '_Location'].append('')
                mm_dict[str(j+1) + '_SubLocation'].append('')
                mm_dict[str(j+1) + '_Location4labels'].append('')
                mm_dict[str(j+1) + '_Location6labels'].append('')
                #mm_dict[str(j+1) + '_Location6labels'].append(int(Location6labels[ane_inx]))
                mm_dict[str(j+1) + '_Comment'].append('')

            for k in range(AneurysmNums[ane_list[0]], max_num):
                #mm_dict[str(k+1) + '_Value'].append('')
                mm_dict[str(k+1) + '_size'].append('')
                mm_dict[str(k+1) + '_Prob_max'].append('')
                mm_dict[str(k+1) + '_Prob_mean'].append('')
                mm_dict[str(k+1) + '_Confirm'].append('')
                mm_dict[str(k+1) + '_type'].append('')
                #mm_dict[str(k+1) + '_1: ICA 2: P-com; 3: A-com; 4: MCA; 5: ACA; 6: PCA; 7: BA;8: VA;9: AICA/PICA/SCA'].append('')
                #mm_dict[str(k+1) + '_1:cervical 2: supraclinoid 3: carotid siphone 4:cavernous 5:terminal 6:opthalmic 7: clinoid 8: communicating 9:distal 10:C4 11:C5 12:C6 13: C7 14:paraclinoid  15:bifurcation 16: frontal scalp'].append('')
                mm_dict[str(k+1) + '_Location'].append('')
                mm_dict[str(k+1) + '_SubLocation'].append('')
                mm_dict[str(k+1) + '_Location4labels'].append('')
                mm_dict[str(k+1) + '_Location6labels'].append('')
                mm_dict[str(k+1) + '_Comment'].append('')

        else:
            mm_dict['Aneurysm_Number'].append('')
            for j in range(max_num):
                #mm_dict[str(j+1) + '_Value'].append('')
                mm_dict[str(j+1) + '_size'].append('')
                mm_dict[str(j+1) + '_Prob_max'].append('')
                mm_dict[str(j+1) + '_Prob_mean'].append('')
                mm_dict[str(j+1) + '_Confirm'].append('')
                mm_dict[str(j+1) + '_type'].append('')
                #mm_dict[str(j+1) + '_1: ICA 2: P-com; 3: A-com; 4: MCA; 5: ACA; 6: PCA; 7: BA;8: VA;9: AICA/PICA/SCA'].append('')
                #mm_dict[str(j+1) + '_1:cervical 2: supraclinoid 3: carotid siphone 4:cavernous 5:terminal 6:opthalmic 7: clinoid 8: communicating 9:distal 10:C4 11:C5 12:C6 13: C7 14:paraclinoid  15:bifurcation 16: frontal scalp'].append('')
                mm_dict[str(j+1) + '_Location'].append('')
                mm_dict[str(j+1) + '_SubLocation'].append('')
                mm_dict[str(j+1) + '_Location4labels'].append('')
                mm_dict[str(j+1) + '_Location6labels'].append('')
                mm_dict[str(j+1) + '_Comment'].append('')


        #填入excel表格中，針對人去填
        #存成excel
        mm_dict_pd = pd.DataFrame(mm_dict)
        #製作成excel
        mm_dict_pd.to_excel(os.path.join(path_excel, 'Aneurysm_Pred_list.xlsx'), index=False)

def make_table_add_location(path_nii, path_excel):
    #讀取excel
    #讀取excel以獲得臨床資訊
    dtypes = {'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'PatientName': str, 'Doctor': str, 'birth_date': str,
            'study_time': str} #年齡在這裡就弄成數字吧
    #excel_name = 'Aneurysm_Pred_list_Val_merged_report.xlsx'

    #excel_file = 'Aneurysm_Label_list.xlsx'
    excel_file = 'Aneurysm_Pred_list.xlsx'

    df = pd.read_excel(os.path.join(path_excel, excel_file), dtype=dtypes).fillna('') #將指定欄位改成特定格式
    PIDs = list(df['PatientID'])
    Sdates = list(df['StudyDate'])
    ANs = list(df['AccessionNumber'])
    PAs = [x + '_' + y + '_MR_' + z for x,y, z in zip(PIDs, Sdates, ANs)]


    #等等用有無檔案排除pred失敗
    for idx, (PID, Sdate, AN, PA)  in enumerate(zip(PIDs, Sdates, ANs, PAs)):
        print([idx], PA, ' Start...')
        #先一致讀出標註檔跟影像檔，先排除 '00657109_20210413_MR_21004130157'
        if PA != '00657109_20210413_MR_21004130157':
            label_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #這邊預設不同影像也是同標註
            label = np.array(label_nii.dataobj) #讀出image的array矩陣
            label = data_translate(label, label_nii).astype('int16')
            y_l, x_l, z_l = label.shape

            labels16_nii = nib.load(os.path.join(path_nii, 'Pred_Location16labels.nii.gz')) #這邊預設不同影像也是同標註
            labels16 = np.array(labels16_nii.dataobj) #讀出image的array矩陣
            labels16 = data_translate(labels16, labels16_nii).astype('int16')

            Label_num = int(np.max(label))

            #將location資訊加入excel中
            for k in range(Label_num):
                #換數字
                label_one = label == k+1
                labels16_list = labels16[label_one > 0]
                # 得到唯一值的数量
                num_unique_values_16labels = len(np.unique(labels16_list))
                assert num_unique_values_16labels == 1, print('挫屎!!! 同一顆怎麼那麼多location')

                #接下來是用判別式判斷填入哪個location跟sublocation值
                if int(np.unique(labels16_list)[0]) == 1 or int(np.unique(labels16_list)[0]) == 3:
                    df.at[idx, str(k+1) + '_Location'] = 'ICA'
                    df.at[idx, str(k+1) + '_SubLocation'] = ''
                elif int(np.unique(labels16_list)[0]) == 2 or int(np.unique(labels16_list)[0]) == 4:
                    df.at[idx, str(k+1) + '_Location'] = 'MCA'
                    df.at[idx, str(k+1) + '_SubLocation'] = ''
                elif int(np.unique(labels16_list)[0]) == 5 or int(np.unique(labels16_list)[0]) == 6:
                    df.at[idx, str(k+1) + '_Location'] = 'ACA'
                    df.at[idx, str(k+1) + '_SubLocation'] = '1'
                elif int(np.unique(labels16_list)[0]) == 7:
                    df.at[idx, str(k+1) + '_Location'] = 'ACA'
                    df.at[idx, str(k+1) + '_SubLocation'] = ''
                elif int(np.unique(labels16_list)[0]) == 8 or int(np.unique(labels16_list)[0]) == 9:
                    df.at[idx, str(k+1) + '_Location'] = 'ICA'
                    df.at[idx, str(k+1) + '_SubLocation'] = 'p'
                elif int(np.unique(labels16_list)[0]) == 10 or int(np.unique(labels16_list)[0]) == 11:
                    df.at[idx, str(k+1) + '_Location'] = 'PCA'
                    df.at[idx, str(k+1) + '_SubLocation'] = ''
                elif int(np.unique(labels16_list)[0]) == 12 or int(np.unique(labels16_list)[0]) == 13:
                    df.at[idx, str(k+1) + '_Location'] = 'BA'
                    df.at[idx, str(k+1) + '_SubLocation'] = 's'
                elif int(np.unique(labels16_list)[0]) == 14:
                    df.at[idx, str(k+1) + '_Location'] = 'BA'
                    df.at[idx, str(k+1) + '_SubLocation'] = ''
                elif int(np.unique(labels16_list)[0]) == 15 or int(np.unique(labels16_list)[0]) == 16:
                    df.at[idx, str(k+1) + '_Location'] = 'VA'
                    df.at[idx, str(k+1) + '_SubLocation'] = ''

            df.to_excel(os.path.join(path_excel, excel_file), index = False)


def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)


def create_dicomseg_multi_file(path_code, path_img_dcm, path_nii, path_nii_resample, path_dcmseg, ID):
    example_file = os.path.join(path_code, 'example', 'SEG_20230210_160056_635_S3.dcm')
    #針對5類設計影像顏色
    model_name = 'Aneurysm_AI'

    labels_vessel = {
            1: {
                "SegmentLabel": "Vessel",
                "color":"red"
                }
        }

    #讀範例文件，其實就是抓出某些tag格式不想自己設定
    dcm_example = pydicom.dcmread(example_file)

    def get_dicom_seg_template(model, label_dict):
        # key is model output label
        # values is meta data
        unique_labels = list(label_dict.keys())

        segment_attributes = []
        for i, idx in enumerate(unique_labels):
            name = label_dict[idx]["SegmentLabel"]

            rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
            rgb = [int(y*255) for y in rgb_rate]


            segment_attribute = {
                    "labelID": int(idx),
                    "SegmentLabel": name,
                    "SegmentAlgorithmType": "MANUAL",
                    "SegmentAlgorithmName": "MONAILABEL",
                    "SegmentedPropertyCategoryCodeSequence": {
                        "CodeValue": "M-01000",
                        "CodingSchemeDesignator": "SRT",
                        "CodeMeaning": "Morphologically Altered Structure",
                    },
                    "SegmentedPropertyTypeCodeSequence": {
                        "CodeValue": "M-35300",
                        "CodingSchemeDesignator": "SRT",
                        "CodeMeaning": "Embolus",
                    },
                    "recommendedDisplayRGBValue": rgb,
                }

            segment_attributes.append(segment_attribute)

        template = {
            "ContentCreatorName": "Reader1",
            "ClinicalTrialSeriesID": "Session1",
            "ClinicalTrialTimePointID": "1",
            "SeriesDescription": model,
            "SeriesNumber": "300",
            "InstanceNumber": "1",
            "segmentAttributes": [segment_attributes],
            "ContentLabel": "SEGMENTATION",
            "ContentDescription": "MONAI Label - Image segmentation",
            "ClinicalTrialCoordinatingCenterName": "MONAI",
            "BodyPartExamined": "",
        }
        return template


    def make_dicomseg(mask, path_dcms, dcm_example, model_name, labels):
        #先做dicom-seg空格式
        template_json = get_dicom_seg_template(model_name, labels)
        template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

        #可以使用 編寫多類分割MultiClassWriter，它有一些可用的自定義選項。如果分割數據非常稀疏，這些選項可以大大減少生成的文件大小。
        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                    # x and y axes
            skip_empty_slices=True,  # Don't encode slices with only zeros
            skip_missing_segment=True,  # If a segment definition is missing in the
                                        # template, then raise an error instead of
                                        # skipping it.
        )

        #這邊把資料填入dicom-seg結構中
        y_l, x_l, z_l = mask.shape
        #先排序並生成一個原版的dicom-seg，沒任何AI標註True不適合用套件做dicom-seg
        if np.max(mask) > 0:
            #為了說明從分割數據創建 DICOM-SEG，典型的圖像分析工作流程如下所示。首先，圖像作為
            reader = sitk.ImageSeriesReader()
            dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))

            #Stack the 2D slices to form a 3D array representing the volume
            slices = [pydicom.dcmread(dcm) for dcm in dcms]

            slice_dcm = []
            #需要依照真實切片位置來更新順序
            for (slice, dcm_slice) in zip(slices, dcms):
                IOP = np.array(slice.ImageOrientationPatient)
                IPP = np.array(slice.ImagePositionPatient)
                normal = np.cross(IOP[0:3], IOP[3:])
                projection = np.dot(IPP, normal)
                slice_dcm += [{"d": projection, "dcm": dcm_slice}]

            #sort the slices，sort是由小到大，但頭頂是要由大到小，所以要反向
            #由於dicom-seg套件影像排序就是照reader.GetGDCMSeriesFileNames，所以不用再自己改頭到腳(原本腳到頭)
            slice_dcms = sorted(slice_dcm, key = lambda i: i['d'])

            new_dcms = [y['dcm'] for y in slice_dcms]
            new_dcms = new_dcms
            #print('new_dcms:', new_dcms)

            #接下來讀取第一張影像
            dcm_one = pydicom.dcmread(new_dcms[0], force=True) #dcm的切片

            #重建空影像給dicomSEG
            reader.SetFileNames(new_dcms)
            image = reader.Execute()
            #image_data = sitk.GetArrayFromImage(image)

            #將原本label的shape轉換成 z,y,x，把影像重新存入
            #mask = np.flip(mask, 0)
            #mask = np.flip(mask, -1) #製作dicom-seg時標註要腳到頭
            segmentation_data = mask.astype(np.uint8)
            #segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)
            #segmentation_data = np.swapaxes(segmentation_data,1,2)
            #segmentation_data = np.flip(segmentation_data, 1) #y軸
            #segmentation_data = np.flip(segmentation_data, 2)
            #segmentation_data = np.flip(segmentation_data, 0)
            #print('segmentation_data.shape:', segmentation_data.shape)

            #由於pydicom_seg期望 aSimpleITK.Image作為作者的輸入，因此需要將分割數據與有關圖像網格的所有相關信息一起封裝。
            #這對於編碼片段和引用相應的源 DICOM 文件非常重要。
            segmentation = sitk.GetImageFromArray(segmentation_data)
            segmentation.CopyInformation(image)


            #最後，可以執行 DICOM-SEG 的實際創建。因此，需要加載源 DICOM 文件pydicom，但出於優化目的，可以跳過像素數據
            source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in new_dcms]
            dcm_seg_ = writer.write(segmentation, source_images)

            #增加一些標註需要的資訊，作法，輸入範例資訊，再把實際case的值填入
            dcm_seg_[0x10,0x0010].value = dcm_one[0x10,0x0010].value #(0010,0010) Patient's Name <= (0010,0020) Patient ID
            dcm_seg_[0x20,0x0011].value = dcm_one[0x20,0x0011].value # (0020,0011) Series Number [3]
            #dcm_seg[0x62,0x0002].value = dcm_example[0x62,0x0002].value #Segment Sequence Attribute
            dcm_seg_[0x5200,0x9229].value = dcm_example[0x5200,0x9229].value
            dcm_seg_[0x5200,0x9229][0][0x20,0x9116][0][0x20,0x0037].value = dcm_one[0x20,0x0037].value #(0020,0037) Image Orientation (Patient)
            dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0050].value = dcm_one[0x18,0x0050].value #(0018,0050) Slice Thickness
            dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0088].value = dcm_one[0x18,0x0088].value #(0018,0088) Spacing Between Slices
            dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x28,0x0030].value = dcm_one[0x28,0x0030].value #(0028,0030) Pixel Spacing

        return dcm_seg_


    IDs = [ID]

    Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw'] #三軸都要做dicom-seg

    for idx, ID in enumerate(IDs):
        print([idx], ID, ' Start....')

        for idx_s, series in enumerate(Series):
            path_dcms = os.path.join(path_img_dcm, series)

            #讀取pred跟vessel
            if series == 'MRA_BRAIN':
                Pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz'))
                Pred = np.array(Pred_nii.dataobj) #讀出label的array矩陣      #256*256*22
                Vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz'))
                #Vessel_nii = nib.as_closest_canonical(Vessel_nii) #轉成RAS軸
                Vessel = np.array(Vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22
                #Pred = data_translate(Pred, Pred_nii)
                #Vessel = data_translate(Vessel, Vessel_nii)
                pred_nii_obj_axcodes = tuple(nib.aff2axcodes(Pred_nii.affine))
                Pred = do_reorientation(Pred, pred_nii_obj_axcodes, ('S', 'P', 'L'))

            else:
                Pred_nii = nib.load(os.path.join(path_nii_resample, series + '_pred.nii.gz'))
                Pred = np.array(Pred_nii.dataobj) #讀出label的array矩陣      #256*256*22
                Vessel_nii = nib.load(os.path.join(path_nii_resample, series + '_vessel.nii.gz'))
                Vessel = np.array(Vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22
                #Pred = data_translate(Pred, Pred_nii)
                # Vessel = data_translate(Vessel, Vessel_nii)
                pred_nii_obj_axcodes = tuple(nib.aff2axcodes(Pred_nii.affine))
                Pred = do_reorientation(Pred, pred_nii_obj_axcodes, ('S', 'P', 'L'))

            #vessel的dicom seg可以直接做
            #dcm_seg = make_dicomseg(Vessel.astype('uint8'), path_dcms, dcm_example, model_name, labels_vessel)
            #dcm_seg.save_as(os.path.join(path_dcmseg, ID + '_' + series + '_' + labels_vessel[1]['SegmentLabel'] + '.dcm')) #


            #以下做動脈瘤，要一顆一顆做，血管一定有標註所以有dicom-seg，動脈瘤沒標註就沒有dicom-seg
            for i in range(int(np.max(Pred))):
                new_Pred = np.zeros((Pred.shape))
                new_Pred[Pred==i+1] = 1

                labels_ane = {}
                labels_ane[1] = {'SegmentLabel': 'A' + str(i+1), 'color': 'red'}

                #底下做成dicom-seg
                dcm_seg = make_dicomseg(new_Pred.astype('uint8'), path_dcms, dcm_example, model_name, labels_ane)
                dcm_seg.save_as(os.path.join(path_dcmseg, ID + '_' + series + '_' + labels_ane[1]['SegmentLabel'] + '.dcm')) #

    return print('Dicom-SEG ok!!!')


def compress_dicom_into_jpeglossless(path_dcm_in, path_dcm_out):

    Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']

    for jdx, series in enumerate(Series):
        if not os.path.isdir(os.path.join(path_dcm_out, series)):
            os.mkdir(os.path.join(path_dcm_out, series))

        #讀取dicom影像，然後轉成jpeglossless
        dcms = sorted(os.listdir(os.path.join(path_dcm_in, series)))
        for dcm in dcms:
            dcm_slice = pydicom.dcmread(os.path.join(path_dcm_in, series, dcm)) #ADC的切片
            #(0002,0003) Media Stored SOP Instance UID  [1.2.840.113619.2.260.6945.3202356.28278.1273105263.493]
            SOPUID = dcm_slice[0x08, 0x0018].value #(0008,0018) SOP Instance UID [1.2.840.113619.2.260.6945.3202356.28278.1273105263.493]
            dcm_slice.compress(JPEGLSLossless)
            #dcm_slice[0x02, 0x0003].value = SOPUID
            dcm_slice[0x08, 0x0018].value = SOPUID

            # Need to set this flag to indicate the Pixel Data is compressed
            #dcm_slice['PixelData'].is_undefined_length = True

            # The Transfer Syntax UID needs to match the type of (JPEG) compression
            # For example, if you were to compress using JPEG Lossless, Hierarchical, First-Order Prediction (1.2.840.10008.1.2.4.70)
            #from pydicom.uid import JPEGLosslessSV1
            #dcm_slice.file_meta.TransferSyntaxUID = JPEGLosslessSV1

            dcm_slice.save_as(os.path.join(path_dcm_out, series, dcm))

    #最後複製dicom-seg
    shutil.copytree(os.path.join(path_dcm_in, 'Dicom-Seg'), os.path.join(path_dcm_out, 'Dicom-Seg'))

#建立json，如果有讀取excel，就會有xnat_link
def case_json(json_file_path, study_dict, sorted_dict, mask_dict):
    json_dict = OrderedDict()
    json_dict["study"] = study_dict
    json_dict["sorted"] = sorted_dict
    json_dict["mask"] = mask_dict

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False, default=int) #讓json能中文顯示

#建立json，如果有讀取excel，就會有xnat_link
def sort_json(json_file_path, data):
    json_dict = OrderedDict()
    json_dict["data"] = data

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False, default=int) #讓json能中文顯示


#黃色被拿去當成人工標註，所以先不要用 => 目前顏色順序就固定，先設定7個
#因為醫師還沒看過，預設select全部都打True
def make_pred_json(excel_file, path_dcm, path_nii, path_reslice, path_dicomseg, path_json, IDs, Series, group_id=49):

    colorbar = {
        "A1": "red",
        "A2": "green",
        "A3": "cyan",
        "A4": "fuchsia",
        "A5": "orange",
        "A6": "lawngreen",
        "A7": "blue",
        "A8": "purple",
        "A9": "salmon",
        "A10": "medium spring green",
        "A11": "dodger blue",
        "A12": "chocolate"
    }

    #顏色這邊改成只有紅跟黃
    colorbar1 = ["red", "green", "cyan", "fuchsia", "orange", "lawngreen", "blue", "purple", "salmon", "medium spring green", "dodger blue", "chocolate"] #只記錄顏色，因為要重新編號

    #讀取excel以獲得臨床資訊
    dtypes = {'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'PatientName': str, 'Doctor': str} #年齡在這裡就弄成數字吧
    #excel_name = 'Aneurysm_Pred_list_Val_merged_report.xlsx'
    df = pd.read_excel(excel_file, dtype=dtypes).fillna('') #將指定欄位改成特定格式
    PIDs = list(df['PatientID'])
    Sdates = list(df['StudyDate'])
    ANs = list(df['AccessionNumber'])
    AneurysmNums = list(df['Aneurysm_Number'])

    PAs = [x + '_' + y + '_MR_' + z for x,y, z in zip(PIDs, Sdates, ANs)]
    #Xlinks = list(df['XNAT_Link'])

    #tags = ['TOF_MRA', 'MIP_Pitch', 'MIP_Yaw']
    #file_endings = ['_TOF_MRA.nii.gz', '_MIP_Pitch.nii.gz', '_MIP_Yaw.nii.gz']

    for idx, ID in enumerate(IDs):
        print([idx], ID, ' mask json start...')

        study_index = PAs.index(ID)
        try:
            AneurysmNumber = int(df['Aneurysm_Number'][study_index])
        except:
            AneurysmNumber = 0

        if os.path.isfile(os.path.join(path_nii, 'Pred.nii.gz')):
            #先一致讀出標註檔跟影像檔
            sort_data = []
            for jdx, series in enumerate(Series):
                if series == 'MRA_BRAIN':
                    #沒有label，只要讀入Pred就好
                    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #這邊預設不同影像也是同標註
                    pred = np.array(pred_nii.dataobj) #讀出image的array矩陣
                    pred = data_translate(pred, pred_nii).astype('int16')
                    resolution_y, resolution_x, resolution_z = pred.shape #這裡獲得影像大小

                    #標註的顏色已經固定A1,A2...是啥了
                    #整理dicom的排序，tag目前只有1，不用迴圈顯示
                    reader = sitk.ImageSeriesReader()
                    files = sorted(reader.GetGDCMSeriesFileNames(os.path.join(path_dcm, series)))
                    #Stack the 2D slices to form a 3D array representing the volume
                    dcms = [pydicom.dcmread(os.path.join(path_dcm, series, y)) for y in files]

                    sort_slices = []
                    sort_instances = []
                    #需要依照真實切片位置來更新順序
                    for (dcm, file) in zip(dcms, files):
                        SOPuid = dcm[0x0008, 0x0018].value #SOPInstanceUID <- 以此為檔案命名
                        IOP = np.array(dcm.ImageOrientationPatient)
                        IPP = np.array(dcm.ImagePositionPatient)
                        normal = np.cross(IOP[0:3], IOP[3:])
                        projection = np.dot(IPP, normal)
                        sort_slices += [{"d": projection, "file": file}]
                        sort_instances += [{"sop_instance_uid": SOPuid, "projection": projection}]

                    #sort the slices，sort是由小到大，但頭頂是要由大到小，所以要反向
                    sort_slices = sorted(sort_slices, key = lambda i: i['d'])
                    sort_slices = sort_slices[::-1]
                    New_dcms = [y['file'] for y in sort_slices]

                    sorted_dcms, image, first_dcm, source_images = load_and_sort_dicom_files(os.path.join(path_dcm, series))
                    ai_team_request = make_study_json(source_images)
                    ai_team_request.study.aneurysm_lession = str(AneurysmNumber)
                    ai_team_request.study.group_id = str(group_id)

                    #接下來只要自行額外製作mask的dict即可將3個dict合併製成json
                    #跟有沒有Predict結果的內容我就放這裡拉~~
                    if os.path.isfile(os.path.join(path_nii, 'Pred.nii.gz')) and AneurysmNumber > 0:
                        pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #這邊預設不同影像也是同標註
                        pred = np.array(pred_nii.dataobj) #讀出image的array矩陣
                        pred = data_translate(pred, pred_nii).astype('int16')
                        y_l, x_l, z_l = pred.shape

                        study_index = PAs.index(ID)
                        AneurysmNumber = int(df['Aneurysm_Number'][study_index])

                        #這邊要跑2輪for迴圈去處理label跟pred
                        max_pred = int(np.max(pred))

                        #讀出全部的sop instance uid來與影像對應
                        SOPs = [pydicom.dcmread(y, force=True)[0x08,0x0018].value for y in New_dcms] #(0008,0018) SOP Instance UID [1.2.840.113619.2.353.6945.3202356.14286.1484267644.90]

                        #這邊要先取得標註的哪張是isMainSeg，採用單點最大機率，建立list紀錄單顆資訊
                        seg_dict = {}
                        seg_dict["series_instance_uid"] = ai_team_request.sorted.series[0].series_instance_uid
                        seg_dict["series_type"] = series
                        """
                        if series == 'MIP_Pitch':
                            seg_dict["series_type"] = '2'
                        elif series == 'MIP_Yaw':
                            seg_dict["series_type"] = '3'
                        """

                        #for每一顆去做，做label那一run，沒有機率可以控制，所以MainSEGSlice用中間張
                        #以下做是pred的那幾筆資料
                        print('AneurysmNumber:', AneurysmNumber)

                        Instances = [] #做出json要保留的資訊，因為dicom-seg關係，不要一張一張作，變成一顆一顆做
                        #接下來做pred的部分
                        for o in range(AneurysmNumber):
                            #先找isMainSeg張，這邊處理pred
                            pred_one = (pred==(o+1)).copy()
                            have_labels = np.where(np.sum(pred_one, axis=(0,1)) > 0)[0]
                            #以下加載Diameter, Type, Loc, SubLoc, MaxProb，找出excel對應顆
                            study_index = PAs.index(ID)
                            try:
                                SubLocation = str(int(df[str(o+1) + '_SubLocation'][study_index]))
                            except:
                                SubLocation = str(df[str(o+1) + '_SubLocation'][study_index])

                            #根據ID讀取dicom-seg，去保存DICOM-SEG_SeriesInstanceUid跟DICOM-SEG_SOPInstanceUid
                            #(0008,0018)	SOP Instance UID	1.2.826.0.1.3680043.8.498.68320971665105720436602317481415467891
                            #(0020,000E)	Series Instance UID	1.2.826.0.1.3680043.8.498.41029033783570649947753616927742160422
                            dcm_seg = pydicom.dcmread(os.path.join(path_dicomseg, ID + '_' + series + '_' + 'A' + str(int(o+1)) + '.dcm'), force=True)

                            #SOP就是關鍵張 #type為假設帶定 囊狀,
                            segment_attribute = {
                            "mask_index": str(int(o+1)),
                            "mask_name": 'A' + str(int(o+1)),
                            "diameter": str(round(df[str(o+1) + '_size'][study_index], 1)),
                            "type": 'saccular',
                            "location": str(df[str(o+1) + '_Location'][study_index]),
                            "sub_location": SubLocation,
                            "checked": "1",
                            "prob_max": str(round(df[str(o+1) + '_Prob_max'][study_index], 2)),
                            "is_ai": "1",
                            "seg_series_instance_uid": dcm_seg[0x20,0x000e].value,
                            "seg_sop_instance_uid": dcm_seg[0x08,0x0018].value,
                            "dicom_sop_instance_uid": SOPs[int(np.median(have_labels))],
                            "main_seg_slice": int(np.median(have_labels)+1),
                            "is_main_seg": "1"
                            }
                            Instances.append(segment_attribute)

                        #血管先不做
                        """
                        #最後這邊讀取血管跟加入血管，血管main-seg放第二張
                        dcm_seg = pydicom.dcmread(os.path.join(path_dicomseg, ID + '_' + series + '_' + 'Vessel.dcm'), force=True)
                        rgb = mcolors.to_rgb("yellow")

                        segment_attribute = {
                            "SOPInstanceUid": SOPs[int(1)],
                            "maskIndex": int(AneurysmNumber+1),
                            "maskName": 'vessel',
                            "maskColorName": 'yellow',
                            "maskColorRGB": [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)],
                            "maskColorHex":'#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)),
                            "isMainSeg": True,
                            "MainSegSlice": int(2),
                            "Diameter": '',
                            "Type": 'vessel',  
                            "Location": '',
                            "SubLocation": '',
                            "Prob_max": 'AI',
                            "is_ai": True, 
                            "Select":  True,
                            "maskArray": '',
                            "DICOM-SEG_SeriesInstanceUid": dcm_seg[0x20,0x000e].value,
                            "DICOM-SEG_SOPInstanceUid":  dcm_seg[0x08,0x0018].value
                            }
                            
                        Instances.append(segment_attribute)
                        """

                        seg_dict["instances"] = Instances
                        #最後把dict合併，初始化MaskRequest
                        ai_team_request.mask = MaskRequest(
                            study_instance_uid=ai_team_request.study.study_instance_uid,
                            group_id=str(group_id),
                            series=[seg_dict]
                        )
                else:
                    #是 MIP_Pitch or MIP_Yaw
                    pred_nii = nib.load(os.path.join(path_reslice, series + '_pred.nii.gz')) #這邊預設不同影像也是同標註
                    pred = np.array(pred_nii.dataobj) #讀出image的array矩陣
                    pred = data_translate(pred, pred_nii).astype('int16')
                    resolution_y, resolution_x, resolution_z = pred.shape #這裡獲得影像大小

                    #如果有標註才畫出他的json檔
                    #重新計算顏色的分配
                    #整理dicom的排序，tag目前只有1，不用迴圈顯示
                    reader = sitk.ImageSeriesReader()
                    files = sorted(reader.GetGDCMSeriesFileNames(os.path.join(path_dcm, series)))
                    #Stack the 2D slices to form a 3D array representing the volume
                    dcms = [pydicom.dcmread(os.path.join(path_dcm, series, y)) for y in files]

                    sort_slices = []
                    sort_instances = []
                    #需要依照真實切片位置來更新順序
                    for (dcm, file) in zip(dcms, files):
                        SOPuid = dcm[0x0008, 0x0018].value #SOPInstanceUID <- 以此為檔案命名
                        IOP = np.array(dcm.ImageOrientationPatient)
                        IPP = np.array(dcm.ImagePositionPatient)
                        normal = np.cross(IOP[0:3], IOP[3:])
                        projection = np.dot(IPP, normal)
                        sort_slices += [{"d": projection, "file": file}]
                        sort_instances += [{"sop_instance_uid": SOPuid, "projection": projection}]

                    #sort the slices，sort是由小到大，但頭頂是要由大到小，所以要反向
                    sort_slices = sorted(sort_slices, key = lambda i: i['d'])
                    sort_slices = sort_slices[::-1]
                    New_dcms = [y['file'] for y in sort_slices]

                    sorted_dcms, image, first_dcm, source_images = load_and_sort_dicom_files(os.path.join(path_dcm, series))
                    ai_team_request_mip = make_study_json(source_images)

                    #取出mip中的sorted，加入at_team_request中
                    ai_team_request.sorted.series.append(ai_team_request_mip.sorted.series[0])

                    #跟有沒有Predict結果的內容我就放這裡拉~~
                    if os.path.isfile(os.path.join(path_reslice, series + '_pred.nii.gz')) and AneurysmNumber > 0:
                        pred_nii = nib.load(os.path.join(path_reslice, series + '_pred.nii.gz')) #這邊預設不同影像也是同標註
                        pred = np.array(pred_nii.dataobj) #讀出image的array矩陣
                        pred = data_translate(pred, pred_nii).astype('int16')
                        y_l, x_l, z_l = pred.shape

                        #讀出全部的sop instance uid來與影像對應
                        SOPs = [pydicom.dcmread(y, force=True)[0x08,0x0018].value for y in New_dcms] #(0008,0018) SOP Instance UID [1.2.840.113619.2.353.6945.3202356.14286.1484267644.90]

                        #這邊要先取得標註的哪張是isMainSeg，採用單點最大機率，建立list紀錄單顆資訊
                        seg_dict = {}
                        seg_dict["series_instance_uid"] = ai_team_request_mip.sorted.series[0].series_instance_uid
                        if series == 'MIP_Pitch':
                            seg_dict["series_type"] = '2'
                        elif series == 'MIP_Yaw':
                            seg_dict["series_type"] = '3'

                        #這邊要先取得標註的哪張是isMainSeg，採用單點最大機率，建立list紀錄單顆資訊
                        Instances = [] #做出json要保留的資訊，因為dicom-seg關係，不要一張一張作，變成一顆一顆做
                        #for每一顆去做，做label那一run，沒有機率可以控制，所以MainSEGSlice用中間張

                        for o in range(AneurysmNumber):
                            #先找isMainSeg張，這邊處理pred
                            pred_one = (pred==(o+1)).copy()
                            have_labels = np.where(np.sum(pred_one, axis=(0,1)) > 0)[0]
                            #以下加載Diameter, Type, Loc, SubLoc, MaxProb，找出excel對應顆
                            study_index = PAs.index(ID)

                            try:
                                SubLocation = str(int(df[str(o+1) + '_SubLocation'][study_index]))
                            except:
                                SubLocation = str(df[str(o+1) + '_SubLocation'][study_index])

                            #根據ID讀取dicom-seg，去保存DICOM-SEG_SeriesInstanceUid跟DICOM-SEG_SOPInstanceUid
                            #(0008,0018)	SOP Instance UID	1.2.826.0.1.3680043.8.498.68320971665105720436602317481415467891
                            #(0020,000E)	Series Instance UID	1.2.826.0.1.3680043.8.498.41029033783570649947753616927742160422
                            dcm_seg = pydicom.dcmread(os.path.join(path_dicomseg, ID + '_' + series + '_' + 'A' + str(int(o+1)) + '.dcm'), force=True)

                            #SOP就是關鍵張 #type為假設帶定 囊狀,
                            segment_attribute = {
                            "mask_index": str(int(o+1)),
                            "mask_name": 'A' + str(int(o+1)),
                            "diameter": str(round(df[str(o+1) + '_size'][study_index], 1)),
                            "type": 'saccular',
                            "location": str(df[str(o+1) + '_Location'][study_index]),
                            "sub_location": SubLocation,
                            "checked": "1",
                            "prob_max": str(round(df[str(o+1) + '_Prob_max'][study_index], 2)),
                            "is_ai": "1",
                            "seg_series_instance_uid": dcm_seg[0x20,0x000e].value,
                            "seg_sop_instance_uid": dcm_seg[0x08,0x0018].value,
                            "dicom_sop_instance_uid": SOPs[int(np.median(have_labels))],
                            "main_seg_slice": int(np.median(have_labels)+1),
                            "is_main_seg": "1"
                            }
                            Instances.append(segment_attribute)

                        """               
                        #最後這邊讀取血管跟加入血管，血管main-seg放第二張
                        dcm_seg = pydicom.dcmread(os.path.join(path_dicomseg, ID + '_' + series + '_' + 'Vessel.dcm'), force=True)
                        rgb = mcolors.to_rgb("yellow")

                        segment_attribute = {
                            "SOPInstanceUid": SOPs[int(1)],
                            "maskIndex": int(AneurysmNumber+1),
                            "maskName": 'vessel',
                            "maskColorName": 'yellow',
                            "maskColorRGB": [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)],
                            "maskColorHex":'#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)),
                            "isMainSeg": True,
                            "MainSegSlice": int(2),
                            "Diameter": '',
                            "Type": 'vessel',  
                            "Location": '',
                            "SubLocation": '',
                            "Prob_max": 'AI',
                            "is_ai": True, 
                            "Select":  True,
                            "maskArray": '',
                            "DICOM-SEG_SeriesInstanceUid": dcm_seg[0x20,0x000e].value,
                            "DICOM-SEG_SOPInstanceUid": dcm_seg[0x08,0x0018].value,
                            }
                        
                        Instances.append(segment_attribute)
                        """
                        #最後把dict合併
                        seg_dict["instances"] = Instances
                        #mask_dict["series"] = seg_dict
                        ai_team_request.mask.series.append(seg_dict)

        #最後存出sort_json，一定會有且一定成功
        platform_json_file = os.path.join(path_json, ID + '_platform_json.json')
        #ai_team_request.model_validate()
        mask_request = MaskRequest.model_validate(ai_team_request.mask)
        #print('mask_request:', mask_request)

        with open(platform_json_file, 'w') as f:
            f.write(ai_team_request.model_dump_json())

        # 驗證為 MaskRequest 格式


    return print("Processing complete!")

def orthanc_zip_upload(path_dicom, path_zip, Series):
    upload_data(path_dicom, path_zip, Series)

def build_upload_study(json_data):
    data = {}
    data['study[0][study_name]'] = json_data['StudyDescription']
    data['study[0][age]'] = str(json_data['Age'])
    data['study[0][patient_id]'] = json_data['PatientID']
    data['study[0][study_date]'] = json_data['StudyDate']
    data['study[0][gender]'] = json_data['Gender']
    data['study[0][patient_name]'] = json_data['PatientName']
    #data['study[0][xnat_link]'] = json_data['XNAT_Link']
    data['study[0][aneurysm_lession]'] = str(json_data['Aneurysm_Number'])
    data['study[0][resolution_x]'] = json_data['resolution_x']
    data['study[0][resolution_y]'] = json_data['resolution_y']
    data['study[0][study_instance_uid]'] = json_data['StudyInstanceUID']
    data['study[0][group_id]'] = json_data['group_id']
    return data

def upload_json(path_json):
    url = 'http://localhost:84/api/cornerstone/demoStudy'
    Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']

    files = sorted(os.listdir(path_json))

    IDs = [y.split('_')[0:1][0] + '_' + y.split('_')[1:2][0] + '_' + y.split('_')[2:3][0] + '_' + y.split('_')[3:4][0] for y in files]
    IDs = sorted(list(set(IDs)))

    for idx, ID in enumerate(IDs):
        print([idx], ID, 'Upload JSON Start...')
        for jdx, series in enumerate(Series):
            #先讀取json，因為是以study為主，所以只要讀取MRA_BRAIN就好
            if series == 'MRA_BRAIN':
                json_file = os.path.join(path_json, ID + '_' + series + '.json')
                #讀取json以獲得資料集參數
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                study_data = build_upload_study(json_data)
                # 发出 POST 请求
                response = requests.post(url, data=study_data, headers={'Accept': 'application/json'})

                # 打印响应内容
                print('Status Code:', response.status_code)
                print('Response Text:', response.text)

                #上傳某 series 的 mask json 到 DB，有mask才上傳
                if len(str(json_data['Aneurysm_Number'])) > 0:
                    if json_data['Aneurysm_Number'] > 0:
                        bash_line = 'cd /var/www/shh-pacs && php artisan import:mask ' + os.path.join(path_json, ID + '_' + series + '.json')
                        uploadJSON_line_line  = os.popen(bash_line)
                        print('uploadJSON_line_line:', uploadJSON_line_line.read())
                        time.sleep(2)

            else:
                #這邊就不用上傳study list了
                json_file = os.path.join(path_json, ID + '_' + series + '.json')
                #讀取json以獲得資料集參數
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                #上傳某 series 的 mask json 到 DB，有mask才上傳
                if len(str(json_data['Aneurysm_Number'])) > 0:
                    if json_data['Aneurysm_Number'] > 0:
                        bash_line = 'cd /var/www/shh-pacs && php artisan import:mask ' + os.path.join(path_json, ID + '_' + series + '.json')
                        uploadJSON_line_line  = os.popen(bash_line)
                        print('uploadJSON_line_line:', uploadJSON_line_line.read())
                        time.sleep(2)

        #sort可以放到最後上傳mask後做，上傳完study list，接下來 快取某 series 的 imageIds 順序到 DB
        bash_line = 'cd /var/www/shh-pacs && php artisan import:sorted-image-ids ' + os.path.join(path_json, ID + '_sort.json')
        sortDcm_line  = os.popen(bash_line)
        print('sortDcm_line:', sortDcm_line.read())
        time.sleep(2)

import warnings
import httpx
import orjson

warnings.filterwarnings("ignore")  # 忽略警告输出
import os
import argparse

def upload_json_aiteam(json_file):
    UPLOAD_DATA_JSON_URL = 'http://localhost:84/api/ai_team/create_or_update'
    client = httpx.Client()
    if isinstance(json_file, str):
        with open(json_file, 'rb') as f:
            data = orjson.loads(f.read())
        with client:
            response = client.post(UPLOAD_DATA_JSON_URL, json=data)
            print(f"Uploaded single file: Status {response.status_code}")

    if isinstance(json_file, list):
        input_list = json_file
        with client:
            for inputs in input_list:
                with open(inputs, 'rb') as f:
                    data = orjson.loads(f.read())