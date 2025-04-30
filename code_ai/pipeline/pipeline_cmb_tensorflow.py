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

autotune = tf.data.experimental.AUTOTUNE
from code_ai import PYTHON3

from code_ai.pipeline.cmb import CMBServiceTF
from code_ai.pipeline import study_id_pattern


def get_study_id(file_name:str) -> Optional[str]:
    result = study_id_pattern.match(file_name)
    if result is not None:
        return result.groups()[0]
    return ""



def pipeline_cmb(ID :str,
                 swan_file :str,
                 t1_file :str,
                 path_output :str,
                 path_code = '/mnt/d/wsl_ubuntu/pipeline/sean/code/',
                 path_processModel = '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_CMB/',
                 path_json = '/mnt/d/wsl_ubuntu/pipeline/sean/json/',
                 path_log = '/mnt/d/wsl_ubuntu/pipeline/sean/log/',
                 path_synthseg = '/mnt/d/wsl_ubuntu/pipeline_synthseg/',
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

            # 底下正式開始predict任務

            # gpu_line = 'bash ' + os.path.join(path_code, 'SynthSEG_stroke.sh ') + case_name
            # 以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
            # python main.py -i './forler/Ax SWAN.nii.gz' --template './forler/Sag_FSPGR_BRAVO.nii.gz' --all False --CMB TRUE
            # gpu_line = 'python ' + os.path.join(path_synthseg, 'main.py -i ') + path_nii + ' --all False --CMB TRUE'
            gpu_line = '{} {} -i {} --template {} --output {} --all False --CMB TRUE'.format(
                PYTHON3,
                os.path.join(path_synthseg, 'main.py'),
                swan_file,
                t1_file,
                path_processID)

            print(gpu_line)
            os.system(gpu_line)

            temp_path_str = glob.glob('{}/*SWAN_original_CMB*.nii.gz'.format(path_processID))[0]
            if not os.path.exists(temp_path_str):
                raise FileNotFoundError(temp_path_str)
            path_output_dir = os.path.join(path_output, ID)
            os.makedirs(path_output_dir, exist_ok=True)
            temp_path_basename = os.path.basename(temp_path_str)
            temp_path_basename = temp_path_basename.replace(get_study_id(temp_path_basename), '')
            shutil.copy(temp_path_str,os.path.join(path_output_dir, temp_path_basename))
            output_nii_path_str = os.path.join(path_output_dir ,'Pred_CMB.nii.gz')
            output_json_path_str = os.path.join(path_output_dir, 'Pred_CMB.json')
            #
            cmb_pipeline = CMBServiceTF()
            cmb_pipeline.cmb_classify(swan_path_str=swan_file,
                                      temp_path_str=temp_path_str,
                                      output_nii_path_str=output_nii_path_str,
                                      output_json_path_str=output_json_path_str
                                      )
            logging.info('!!! ' + str(ID) + ' gpu_cmb finish.')

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
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    # print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # python /mnt/d/wsl_ubuntu/pipeline/sean/code/pipeline_cmb_tensorflow.py --ID 00971591_20160503_MR_250425032 --Inputs /mnt/d/wsl_ubuntu/pipeline/sean/example_input/00971591_20160503_MR_250425032/SWAN.nii.gz /mnt/d/wsl_ubuntu/pipeline/sean/example_input/00971591_20160503_MR_250425032/T1FLAIR_AXI.nii.gz
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default='12292196_20200223_MR_20902230007',
                        help='目前執行的case的patient_id or study id')

    parser.add_argument('--Inputs', type=str, nargs='+',
                        default=['/mnt/d/wsl_ubuntu/pipeline/sean/rename_nii/12292196_20200223_MR_20902230007/SWAN.nii.gz',
                                 '/mnt/d/wsl_ubuntu/pipeline/sean/rename_nii/12292196_20200223_MR_20902230007/T1FLAIR_AXI.nii.gz', ],
                        help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--InputsDicomDir', type=str, nargs='+',
                        default=[
                            '/mnt/d/wsl_ubuntu/pipeline/sean/rename_dicom/12292196_20200223_MR_20902230007/SWAN',
                            '/mnt/d/wsl_ubuntu/pipeline/sean/rename_dicom/12292196_20200223_MR_20902230007/T1FLAIR_AXI', ],
                        help='用於輸入的檔案')

    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    # 下面設定各個路徑
    path_output = str(args.Output_folder)
    path_code     = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'
    path_process  = '/mnt/d/wsl_ubuntu/pipeline/sean/process/'  # 前處理dicom路徑(test case)
    path_processModel = os.path.join(path_process, 'Deep_CMB')  # 前處理dicom路徑(test case)
    path_json     = '/mnt/d/wsl_ubuntu/pipeline/sean/json/'  # 存放json的路徑，回傳執行結果
    path_log      = '/mnt/d/wsl_ubuntu/pipeline/sean/log/'  # log資料夾
    path_synthseg = '/mnt/d/wsl_ubuntu/pipeline/sean/code/'

    gpu_n = 0  # 使用哪一顆gpu

    swan_path_str = Inputs[0]
    t1_path_str = Inputs[1]

    # 這裡先沒有dicom
    # path_json = '/mnt/d/wsl_ubuntu/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    # json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    # path_log = '/mnt/d/wsl_ubuntu/pipeline/chuan/log/'  #log資料夾
    # cuatom_model = '/mnt/d/wsl_ubuntu/pipeline/chuan/code/model_weights/Unet-0194-0.17124-0.14731-0.86344.h5'  #model路徑+檔案

    # 建置資料夾
    os.makedirs(path_processModel,exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    # 直接當作function的輸入
    pipeline_cmb(ID, swan_path_str, t1_path_str, path_output, path_code, path_processModel,
                     path_json, path_log, path_synthseg, gpu_n)
    # dicom_seg()
    # upload_dicom_seg()
    # upload_json()

