#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-28 10:40

Python 3.10.13
tensorflow==2.14.0
numpy==1.26.0
SimpleITK==2.3.1
nibabel==5.1.0
scikit-image==0.22.0
pynvml==12.0.0

@author: sean Ho
"""
import glob
import re
import shutil
import warnings
warnings.filterwarnings("ignore")  # 忽略警告输出
import os
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import logging
import pynvml  # 导包
import tensorflow as tf
from code_ai.pipeline import study_id_pattern

autotune = tf.data.experimental.AUTOTUNE


def get_study_id(file_name:str) -> Optional[str]:
    result = study_id_pattern.match(file_name)
    if result is not None:
        return result.groups()[0]
    return ""


def pipeline_synthseg(ID :str,
                      file_path_str :str,
                      path_output :str,
                      path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/sean/code/',
                      path_processModel = '/mnt/d/wsl_ubuntu/pipeline/sean/sean/process/Deep_CMB/',
                      path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/sean/json/',
                      path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/sean/log/',
                      gpu_n = 0):
    # 當使用gpu有錯時才確認
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # 以log紀錄資訊，先建置log
    localt = time.localtime(time.time())  # 取得 struct_time 格式的時間
    # 以下加上時間註記，建立唯一性
    time_str_short = str(localt.tm_year) + str(localt.tm_mon).rjust(2, '0') + str(localt.tm_mday).rjust(2, '0')
    log_file = os.path.join(path_log, time_str_short + '.log')
    if not os.path.isfile(log_file):  # 如果log檔不存在
        f = open(log_file, "a+")  # a+	可讀可寫	建立，不覆蓋
        f.write("")  # 寫入檔案，設定為空
        f.close()  # 執行完結束

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  # 日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    logging.info('!!! Pred CMB call.')
    path_processID = os.path.join(path_processModel, ID)  # 前處理dicom路徑(test case)
    os.makedirs(path_processID,exist_ok=True) # 如果資料夾不存在就建立
    print(ID, ' Start...')

    try:
        # %% Deep learning相關
        pynvml.nvmlInit()  # 初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)  # 获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
        gpumRate = memoryInfo.used / memoryInfo.total
        # print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code

        if gpumRate < 0.6:
            # plt.ion()    # 開啟互動模式，畫圖都是一閃就過
            # 一些記憶體的配置
            autotune = tf.data.experimental.AUTOTUNE
            # print(keras.__version__)
            # print(tf.__version__)
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            # print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            gpu_line = 'python {} -i {} --output {} --all False --WMH TRUE'.format(
                os.path.join(os.path.dirname(__file__), 'main.py'),
                file_path_str,
                path_processID)

            print('gpu_line',gpu_line)
            os.system(gpu_line)

            path_output_dir = os.path.join(path_output, ID)
            os.makedirs(path_output_dir, exist_ok=True)
            original_file_path_list = glob.glob('{}/*_original_*.nii.gz'.format(path_processID))
            for original_file_path_str in original_file_path_list:
                if not os.path.exists(original_file_path_str):
                    continue
                temp_path_basename = os.path.basename(original_file_path_str)
                temp_path_basename = temp_path_basename.replace(get_study_id(temp_path_basename), '')
                shutil.copy(original_file_path_str, os.path.join(path_output_dir, temp_path_basename))

            logging.info('!!! ' + str(ID) + ' gpu_synthseg finish.')

        else:
            logging.error('!!! ' + str(ID) + ' Insufficient GPU Memory.')
            # 以json做輸出
            code_pass = 1
            msg = "Insufficient GPU Memory"

            # #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾

    except:
        logging.error('!!! ' + str(ID) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)
        # 以json做輸出
        code_pass = 1
        msg = "have error code"
        # 刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾

    return


# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default='00971591_20160503_MR_250425032',
                        help='目前執行的case的patient_id or study id')

    parser.add_argument('--Inputs', type=str, nargs='+',
                        default=[
                                 '/mnt/d/wsl_ubuntu/pipeline/sean/example_input/00971591_20160503_MR_250425032/T2FLAIR_AXI.nii.gz', ],
                        help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
                        help='用於輸出結果的資料夾')
    # parser.add_argument('--Inputs', type=str, nargs='+', default = ['/mnt/d/wsl_ubuntu/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/ADC.nii.gz',
    #                                                                  '/mnt/d/wsl_ubuntu/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/DWI0.nii.gz',
    #                                                                    '/mnt/d/wsl_ubuntu/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/DWI1000.nii.gz',
    #                                                                      '/mnt/d/wsl_ubuntu/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/synthseg_DWI0_original_DWI.nii.gz'], help='用於輸入的檔案')
    # parser.add_argument('--Output_folder', type=str, default = '/mnt/d/wsl_ubuntu/pipeline/chuan/example_output/',help='用於輸出結果的資料夾')

    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    # 下面設定各個路徑
    path_output = str(args.Output_folder)
    path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
    path_process = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
    path_processModel = os.path.join(path_process, 'Deep_synthseg')  # 前處理dicom路徑(test case)
    path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
    path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
    gpu_n = 0  # 使用哪一顆gpu

    file_path_str = Inputs[0]

    # 建置資料夾
    os.makedirs(path_processModel,exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    # 直接當作function的輸入
    pipeline_synthseg(ID, file_path_str, path_output, path_code, path_processModel,
                      path_json, path_log, gpu_n)

