#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

predict model，非常重要:使用這個程式，安裝的版本都需要一致，才能運行
conda create -n orthanc-stroke tensorflow-gpu=2.3.0 anaconda python=3.7

@author: chuan
"""
import warnings
warnings.filterwarnings("ignore") # 忽略警告输出

import os
import time
import numpy as np
import nibabel as nib
from random import uniform
import math
import argparse
import logging
import json
import pynvml #导包
from collections import OrderedDict
from .nii_transforms import nii_img_replace
import onnxruntime as ort

def case_json(json_file_path, ret, code, msg, chr_no, request_date, pat_name, AccessionNumber, StudyID, PatientBirth, PatientAge, PatientSex, PatientWeight, MagneticFieldStrength,
                  StudyDescription, ImagesAcquisition, PixelSpacing, SpacingBetweenSlices, InfarctSliceNum, InfarctVoxel, volume, MeanADC, Report):
        json_dict = OrderedDict()
        json_dict["ret"] = ret #使用的程式是哪一支python api
        json_dict["code"] = code #目前程式執行的結果狀態 0: 成功 1: 失敗
        json_dict["msg"] = msg #描述狀態訊息
        json_dict["chr_no"] = chr_no #patient id
        json_dict["request_date"] = request_date #studydate
        json_dict["pat_name"] = pat_name #病人姓名
        json_dict["AccessionNumber"] = AccessionNumber 
        json_dict["StudyID"] = StudyID 
        json_dict["PatientBirth"] = PatientBirth 
        json_dict["PatientAge"] = PatientAge         
        json_dict["PatientSex"] = PatientSex 
        json_dict["PatientWeight"] = PatientWeight 
        json_dict["MagneticFieldStrength"] = MagneticFieldStrength 
        json_dict["StudyDescription"] = StudyDescription 
        json_dict["ImagesAcquisition"] = ImagesAcquisition 
        json_dict["PixelSpacing"] = PixelSpacing 
        json_dict["SpacingBetweenSlices"] = SpacingBetweenSlices 
        json_dict["InfarctSliceNum"] = InfarctSliceNum 
        json_dict["InfarctVoxel"] = InfarctVoxel #infarct core 體積(voxel數)
        json_dict["volume"] = volume #infarct core 體積(ml) 
        json_dict["MeanADC"] = MeanADC 
        json_dict["Report"] = Report
        with open(json_file_path, 'w', encoding='utf8') as json_file:
             json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

def model_predict_stroke(path_code, path_process, model_unet, case_name, path_log, gpu_n):
        #以log紀錄資訊，先建置log
    localt = time.localtime(time.time()) # 取得 struct_time 格式的時間
    #以下加上時間註記，建立唯一性
    time_str_short = str(localt.tm_year) + str(localt.tm_mon).rjust(2,'0') + str(localt.tm_mday).rjust(2,'0')
    log_file = os.path.join(path_log, time_str_short + '.log')
    if not os.path.isfile(log_file):  #如果log檔不存在
        f = open(log_file, "a+") #a+	可讀可寫	建立，不覆蓋
        f.write("")        #寫入檔案，設定為空
        f.close()      #執行完結束

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    logging.info('!!! ' + case_name + ' gpu_stroke call.')

    #把一些json要記錄的資訊弄出空集合或欲填入的值，這樣才能留空資訊
    ret = "gpu_stroke.py " #使用的程式是哪一支python api
    code_pass = 0 #確定是否成功
    msg = "ok" #描述狀態訊息
    
    #%% Deep learning相關
    # pynvml.nvmlInit() #初始化
    # handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)#获取GPU i的handle，后续通过handle来处理
    # memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
    # gpumRate = memoryInfo.used/memoryInfo.total
    #print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code
    # plt.ion()    # 開啟互動模式，畫圖都是一閃就過
    # 一些記憶體的配置

    # %%
    # 建立Infarct model之前先運行SynthSEG，應該可以使其用於同一張顯卡
    # 底下正式開始predict任務
    ROWS = 256  # y
    COLS = 256  # x

    path_nii = os.path.join(path_process, 'nii')
    path_test_img = os.path.join(path_process, 'test_case')
    path_predict = os.path.join(path_process, 'predict_map')
    path_npy = os.path.join(path_test_img, 'npy')

    if not os.path.isdir(path_test_img):
        os.mkdir(path_test_img)
    if not os.path.isdir(path_predict):
        os.mkdir(path_predict)
    if not os.path.isdir(path_npy):
        os.mkdir(path_npy)

        # load model

    # model_unet = ort.InferenceSession("./infarct_unet_256_model.onnx",providers=["CPUExecutionProvider"])
    # model_unet = ort.InferenceSession("./infarct_unet_256_model.onnx",providers=["CUDAExecutionProvider"])

    # %%
    # 會使用到的一些predict技巧
    def data_translate(img):
        img = np.swapaxes(img, 0, 1)
        img = np.flip(img, 0)
        img = np.flip(img, -1)
        # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
        return img

    # 會使用到的一些predict技巧
    def data_translate_back(img):
        img = np.flip(img, -1)
        img = np.flip(img, 0)
        img = np.swapaxes(img, 1, 0)
        # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
        return img

    # 在gpu程式中就要把predction用nifti保存才能接resample，先讀取nifti影像
    img_nii = nib.load(os.path.join(path_nii, case_name + '_DWI1000.nii.gz'))  # 讀取影像

    # 以上前處理完成，開始predict => .png
    ADC_array = np.load(
        os.path.join(path_npy, case_name + '_ADC.npy'))  # 讀出image的array矩陣，用nii讀影像的話會有背景問題   #256*256*22
    DWI1000_array = np.load(
        os.path.join(path_npy, case_name + '_DWI1000.npy'))  # 讀出image的array矩陣，用nii讀影像的話會有背景問題
    ADC_array = data_translate(ADC_array)  # change image to use cnn
    DWI1000_array = data_translate(DWI1000_array)  # change image to use cnn
    y_i, x_i, z_i = ADC_array.shape  # 得出尺寸

    slice_y = int((ROWS - y_i) / 2)
    slice_x = int((COLS - x_i) / 2)

    img_test = np.zeros((2, ROWS, COLS, z_i))  # 2 because ADC and DWI1000.
    img_test[0, slice_y:slice_y + y_i, slice_x:slice_x + x_i, :] = ADC_array
    img_test[1, slice_y:slice_y + y_i, slice_x:slice_x + x_i, :] = DWI1000_array
    img_test = img_test.swapaxes(0, 3)  # 交换轴0和3，因為model_predict_batch 是 張數,y,x,c

    # y_pred = model_unet.predict(img_test) #
    img_test = img_test.astype(np.float32)
    y_pred = model_unet.run(["activation_1"], {"input_2": img_test})[0]
    y_pred_unet = y_pred[:, :, :, 0]
    np.save(os.path.join(path_predict, case_name + '.npy'), y_pred_unet)

    # 以下轉向
    Y_pred = np.expand_dims(y_pred_unet, axis=-1)
    Y_pred = Y_pred.swapaxes(0, 3)  # 交换轴0和3，因為model_predict_batch 是 張數,y,x,c
    Y_pred = Y_pred[0, :, :, :]
    Y_pred = data_translate_back(Y_pred)
    Y_pred_nii = nii_img_replace(img_nii, Y_pred)
    nib.save(Y_pred_nii, os.path.join(path_predict, case_name + '_prediction.nii.gz'))

    # 以json做輸出
    # time.sleep(1)
    # logging.info('!!! ' + str(case_name) + ' gpu_stroke finish.')
    #
    # if gpumRate < 0.6 :
    #     #plt.ion()    # 開啟互動模式，畫圖都是一閃就過
    #     #一些記憶體的配置
    #
    #     #%%
    #     #建立Infarct model之前先運行SynthSEG，應該可以使其用於同一張顯卡
    #     #底下正式開始predict任務
    #     ROWS=256 #y
    #     COLS=256 #x
    #
    #     path_nii = os.path.join(path_process, 'nii')
    #     path_test_img = os.path.join(path_process, 'test_case')
    #     path_predict = os.path.join(path_process, 'predict_map')
    #     path_npy = os.path.join(path_test_img, 'npy')
    #
    #     if not os.path.isdir(path_test_img):
    #         os.mkdir(path_test_img)
    #     if not os.path.isdir(path_predict):
    #         os.mkdir(path_predict)
    #     if not os.path.isdir(path_npy):
    #         os.mkdir(path_npy)
    #
    #     # load model
    #     # model_unet = ort.InferenceSession("./infarct_unet_256_model.onnx",providers=["CPUExecutionProvider"])
    #     # model_unet = ort.InferenceSession("./infarct_unet_256_model.onnx",providers=["CUDAExecutionProvider"])
    #
    #     #%%
    #     #會使用到的一些predict技巧
    #     def data_translate(img):
    #         img = np.swapaxes(img,0,1)
    #         img = np.flip(img,0)
    #         img = np.flip(img, -1)
    #         # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    #         return img
    #
    #     #會使用到的一些predict技巧
    #     def data_translate_back(img):
    #         img = np.flip(img, -1)
    #         img = np.flip(img,0)
    #         img = np.swapaxes(img,1,0)
    #         # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    #         return img
    #
    #     #在gpu程式中就要把predction用nifti保存才能接resample，先讀取nifti影像
    #     img_nii = nib.load(os.path.join(path_nii, case_name + '_DWI1000.nii.gz')) #讀取影像
    #
    #     #以上前處理完成，開始predict => .png
    #     ADC_array = np.load(os.path.join(path_npy, case_name + '_ADC.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題   #256*256*22
    #     DWI1000_array = np.load(os.path.join(path_npy, case_name + '_DWI1000.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題
    #     ADC_array = data_translate(ADC_array)    #change image to use cnn
    #     DWI1000_array = data_translate(DWI1000_array)    #change image to use cnn
    #     y_i, x_i , z_i = ADC_array.shape #得出尺寸
    #
    #     slice_y = int((ROWS - y_i)/2)
    #     slice_x = int((COLS - x_i)/2)
    #
    #     img_test = np.zeros((2,ROWS,COLS,z_i)) #2 because ADC and DWI1000.
    #     img_test[0,slice_y:slice_y+y_i,slice_x:slice_x+x_i,:] = ADC_array
    #     img_test[1,slice_y:slice_y+y_i,slice_x:slice_x+x_i,:] = DWI1000_array
    #     img_test = img_test.swapaxes(0,3) #交换轴0和3，因為model_predict_batch 是 張數,y,x,c
    #
    #     # y_pred = model_unet.predict(img_test) #
    #     img_test = img_test.astype(np.float32)
    #     y_pred = model_unet.run(["activation_1"], {"input_2": img_test})[0]
    #     y_pred_unet = y_pred[:,:,:,0]
    #     np.save(os.path.join(path_predict, case_name + '.npy'), y_pred_unet)
    #
    #     #以下轉向
    #     Y_pred = np.expand_dims(y_pred_unet, axis=-1)
    #     Y_pred = Y_pred.swapaxes(0,3) #交换轴0和3，因為model_predict_batch 是 張數,y,x,c
    #     Y_pred = Y_pred[0,:,:,:]
    #     Y_pred = data_translate_back(Y_pred)
    #     Y_pred_nii = nii_img_replace(img_nii, Y_pred)
    #     nib.save(Y_pred_nii, os.path.join(path_predict, case_name + '_prediction.nii.gz'))
    #
    #     #以json做輸出
    #     time.sleep(1)
    #     logging.info('!!! ' + str(case_name) +  ' gpu_stroke finish.')
    #
    # else:
    #     logging.error('!!! ' + str(case_name) + ' Insufficient GPU Memory.')
    #     #以json做輸出
    #     code_pass = 1
    #     msg = "Insufficient GPU Memory"
    #
    #     # #刪除資料夾
    #     # if os.path.isdir(path_process):  #如果資料夾存在
    #     #     shutil.rmtree(path_process) #清掉整個資料夾

    return

