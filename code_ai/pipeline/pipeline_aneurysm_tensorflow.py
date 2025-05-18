# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

這邊分別去執行3個

@author: chuan
"""
import pathlib
import subprocess
import warnings

from code_ai import PYTHON3

warnings.filterwarnings("ignore")  # 忽略警告输出
from typing import Tuple,Optional
import os
import numpy as np
import logging
import shutil
import time
import json
import tensorflow as tf
from collections import OrderedDict
import pynvml  # 导包
from code_ai.pipeline import pipeline_parser, upload_json,InferenceEnum
from code_ai.pipeline.chuan.gpu_aneurysm import model_predict_aneurysm
from code_ai.pipeline.chuan.util_aneurysm import reslice_nifti_pred_nobrain, create_MIP_pred, \
    make_aneurysm_vessel_location_16labels_pred, \
    calculate_aneurysm_long_axis_make_pred, make_table_row_patient_pred, make_table_add_location, \
    create_dicomseg_multi_file, make_pred_json

# 會使用到的一些predict技巧
def data_translate(img, nii):
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    header = nii.header.copy()  # 抓出nii header 去算體積
    pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
        # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img


def data_translate_back(img, nii):
    header = nii.header.copy()  # 抓出nii header 去算體積
    pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img


# case_json(json_path_name)
def case_json(json_file_path, ID):
    json_dict = OrderedDict()
    json_dict["PatientID"] = ID  # 使用的程式是哪一支python api

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '),
                  ensure_ascii=False)  # 讓json能中文顯示


def pipeline_aneurysm(ID,
                      MRA_BRAIN_file,
                      path_output,
                      path_code='/mnt/e/pipeline/chuan/code/',
                      path_processModel='/mnt/e/pipeline/chuan/process/Deep_Aneurysm/',
                      path_outdcm='',
                      path_json='/mnt/e/pipeline/chuan/json/',
                      path_log='/mnt/e/pipeline/chuan/log/',
                      path_cuatom_model='/mnt/e/pipeline/code/model_weights',
                      gpu_n=0
                      )->Tuple[(Optional[str|pathlib.Path],
                                Optional[str|pathlib.Path],
                                Optional[str|pathlib.Path],
                                Optional[str|pathlib.Path],
                                Optional[str|pathlib.Path],)]:

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

    logging.info('!!! Pre_Aneurysm call.')

    path_processID = os.path.join(path_processModel, ID)  # 前處理dicom路徑(test case)
    if not os.path.isdir(path_processID):  # 如果資料夾不存在就建立
        os.mkdir(path_processID)  # 製作nii資料夾

    print(ID, ' Start...')

    # 依照不同情境拆分try需要小心的事項 <= 重要
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
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            # 先判斷有無影像，複製過去
            # print('ADC_file:', ADC_file, ' copy:', os.path.join(path_nii, ID + '_ADC.nii.gz'))
            shutil.copy(MRA_BRAIN_file, os.path.join(path_processID, 'MRA_BRAIN.nii.gz'))

            # 因為松諭會用排程，所以這邊改成call function不管gpu了
            model_predict_aneurysm(path_cuatom_model, path_processID, ID, path_log, gpu_n)

            # 接下來做mip影像
            path_dcm = os.path.join(path_processID, 'Dicom')
            path_nii = os.path.join(path_processID, 'Image_nii')
            path_reslice = os.path.join(path_processID, 'Image_reslice')
            path_excel = os.path.join(path_processID, 'excel')
            os.makedirs(path_dcm, exist_ok=True)
            os.makedirs(path_nii, exist_ok=True)
            os.makedirs(path_reslice, exist_ok=True)
            os.makedirs(path_excel, exist_ok=True)

            # 複製nii到nii資料夾
            shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nii, 'MRA_BRAIN.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Pred.nii.gz'), os.path.join(path_nii, 'Pred.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nii, 'Vessel.nii.gz'))

            # reslice
            reslice_nifti_pred_nobrain(path_nii, path_reslice)

            # 做mip影像
            # 複製dicom影像
            if not os.path.isdir(os.path.join(path_dcm, 'MRA_BRAIN')):  # 如果資料夾不存在就建立
                shutil.copytree(path_outdcm, os.path.join(path_dcm, 'MRA_BRAIN'))
            path_png = os.path.join(path_code, 'png')
            #path_png = os.path.join(path_processID, 'png')
            create_MIP_pred(path_dcm, path_reslice, path_png, gpu_n)
            # 接下來是計算動脈瘤的各項數據
            make_aneurysm_vessel_location_16labels_pred(path_processID)
            calculate_aneurysm_long_axis_make_pred(path_dcm, path_processID, path_excel, ID)
            make_table_row_patient_pred(path_excel, ID)
            make_table_add_location(path_processID, path_excel)
            # 接下來製作dicom-seg
            path_dicomseg = os.path.join(path_dcm, 'Dicom-Seg')
            os.makedirs(path_dicomseg, exist_ok=True)
            # 上線版不用做vessel
            create_dicomseg_multi_file(path_code, path_dcm, path_nii, path_reslice, path_dicomseg, ID)
            # 將dicom壓縮不包含dicom-seg  Dicom_JPEGlossless => 由於要用numpy > 2.0，之後補強
            path_dcmjpeglossless = os.path.join(path_processID, 'Dicom_JPEGlossless')
            os.makedirs(path_dcmjpeglossless, exist_ok=True)
            # compress_dicom_into_jpeglossless(path_dcm, path_dcmjpeglossless)

            # 建立json檔
            path_json_out = os.path.join(path_processID, 'JSON')
            os.makedirs(path_json_out, exist_ok=True)
            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']
            group_id = 49
            excel_file = os.path.join(path_excel, 'Aneurysm_Pred_list.xlsx')
            make_pred_json(excel_file, path_dcm, path_nii, path_reslice, path_dicomseg, path_json_out, [ID], Series,
                           group_id)

            # 接下來上傳dicom到orthanc
            # path_zip = os.path.join(path_processID, 'Dicom_zip')
            # os.makedirs(path_zip, exist_ok=True)

            # Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw', 'Dicom-Seg']
            # orthanc_zip_upload(path_dcm, path_zip, Series)

            # 接下來，上傳json
            # upload_json(path_json_out)

            # 把json跟nii輸出到out資料夾
            path_output_dir = str(os.path.join(path_output, ID))
            os.makedirs(path_output_dir, exist_ok=True)
            output_tuple = (os.path.join(path_output_dir, 'Pred_Aneurysm.nii.gz'),
                            os.path.join(path_output_dir, 'Prob_Aneurysm.nii.gz'),
                            os.path.join(path_output_dir, 'Pred_Aneurysm_vessel.nii.gz'),
                            os.path.join(path_output_dir, 'Pred_Aneurysm_vessel16.nii.gz'),
                            os.path.join(path_output_dir, 'Pred_Aneurysm.json'))
            shutil.copy(os.path.join(path_processID, 'Pred.nii.gz'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Prob.nii.gz'),
                        os.path.join(path_output_dir, 'Prob_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm_vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm_vessel16.nii.gz'))
            shutil.copy(os.path.join(path_json_out, ID + '_MRA_BRAIN.json'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm.json'))

            # 刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾

            logging.info('!!! ' + ID + ' post_aneurysm finish.')
            return output_tuple
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
        # 刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾

    print('end!!!')
    return None, None, None, None, None


if __name__ == '__main__':
    from code_ai.pipeline.chuan import CUATOM_MODEL_ANEURYSM

    parser = pipeline_parser()
    args = parser.parse_args()

    ID = str(args.ID)
    path_DcmDir = args.InputsDicomDir  # 對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)
    MRA_BRAIN_file = args.Inputs[0]

    # 需要安裝 pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg => 先不壓縮，因為壓縮需要numpy > 2

    # 下面設定各個路徑
    path_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chuan')
    path_process = os.getenv("PATH_PROCESS")
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")
    path_synthseg = os.getenv("PATH_SYNTHSEG")
    path_processModel = os.path.join(path_process, 'Deep_Aneurysm')  # 前處理dicom路徑(test case)

    cuatom_model = CUATOM_MODEL_ANEURYSM
    # 自訂模型
    gpu_n = 0  # 使用哪一顆gpu

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True)  # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output, exist_ok=True)

    # 直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    Pred_Aneurysm_path, Prob_Aneurysm_path, \
        Pred_Aneurysm_vessel_path, Pred_Aneurysm_vessel16_path, \
        Pred_Aneurysm_json_path = pipeline_aneurysm(ID=ID,
                                                    MRA_BRAIN_file=MRA_BRAIN_file,
                                                    path_output=path_output,
                                                    path_code=path_code,
                                                    path_processModel=path_processModel,
                                                    path_outdcm=path_DcmDir,
                                                    path_json=path_json,
                                                    path_log=path_log,
                                                    path_cuatom_model=CUATOM_MODEL_ANEURYSM,
                                                    gpu_n=gpu_n)
    #

    path_dcm = os.path.join(path_processModel,ID,'Dicom')
    upload_dicom_dir_tuple = (os.path.join(path_dcm, 'Dicom-Seg'),
                              os.path.join(path_dcm, 'MIP_Pitch'),
                              os.path.join(path_dcm, 'MIP_Yaw'))
    for dicom_dir in upload_dicom_dir_tuple:
        cmd_str = ('export PYTHONPATH={} && '
                   '{} code_ai/pipeline/upload_dicom_seg.py '
                   '--Input {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                        PYTHON3,
                                        dicom_dir
                                        )
               )
        process = subprocess.Popen(args=cmd_str, shell=True,
                                   # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(cmd_str,stdout)
    upload_json(ID,InferenceEnum.Aneurysm)
