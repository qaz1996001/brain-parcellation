# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

這邊分別去執行3個

@author: chuan
"""
import pathlib
import subprocess
import warnings
warnings.filterwarnings("ignore")  # 忽略警告输出
from typing import Optional, List
import os
import logging
import shutil
import time
import tensorflow as tf
import pynvml  # 导包
from code_ai import PYTHON3
from code_ai.pipeline import pipeline_parser
from code_ai.pipeline.chuan.util_aneurysm import reslice_nifti_pred_nobrain, create_MIP_pred, \
    make_aneurysm_vessel_location_16labels_pred, \
    calculate_aneurysm_long_axis_make_pred, make_table_row_patient_pred, make_table_add_location, \
    create_dicomseg_multi_file, make_pred_json, orthanc_zip_upload, upload_json_aiteam


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
                      ) -> Optional[tuple[str, str, str, str, str]]:
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
            # 一些記憶體的配置
            autotune = tf.data.experimental.AUTOTUNE
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            # 先判斷有無影像，複製過去
            shutil.copy(MRA_BRAIN_file, os.path.join(path_processID, 'MRA_BRAIN.nii.gz'))

            # 因為松諭會用排程，所以這邊改成call function不管gpu了
            # 定義要傳入的參數，建立指令

            cmd_str = ('export PYTHONPATH={} && '
                       '{} code_ai/pipeline/chuan/gpu_aneurysm.py '
                       '--path_code {} '
                       '--path_process {} '
                       '--case {} '
                       '--path_log {} '
                       '--gpu_n {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                            PYTHON3,
                                            CUATOM_MODEL_ANEURYSM,
                                            path_processID,
                                            ID,
                                            path_log,
                                            str(gpu_n))
                       )
            print('cmd',cmd_str)
            process = subprocess.Popen(args=cmd_str, shell=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print('gpu_aneurysm.py',stdout, stderr)

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
            GROUP_ID_ANEURYSM = os.getenv("GROUP_ID_ANEURYSM", 50)
            excel_file = os.path.join(path_excel, 'Aneurysm_Pred_list.xlsx')
            make_pred_json(excel_file,
                           path_dcm,
                           path_nii,
                           path_reslice,
                           path_dicomseg,
                           path_json_out, [ID], Series,
                           GROUP_ID_ANEURYSM)

            # 接下來上傳dicom到orthanc
            path_zip = os.path.join(path_processID, 'Dicom_zip')
            os.makedirs(path_zip, exist_ok=True)

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw', 'Dicom-Seg']
            orthanc_zip_upload(path_dcm, path_zip, Series)

            # 把json跟nii輸出到out資料夾
            path_output_dir = str(os.path.join(path_output, ID))
            os.makedirs(path_output_dir, exist_ok=True)

            output_tuple = (os.path.join(path_output_dir, 'Pred_Aneurysm.nii.gz'),
                            os.path.join(path_json_out, ID + '_' + Series[0] + '.json'),
                            os.path.join(path_json_out, ID + '_' + Series[1] + '.json'),
                            os.path.join(path_json_out, ID + '_' + Series[2] + '.json'),
                            os.path.join(path_json_out, ID + '_sort.json')
                            )

            shutil.copy(os.path.join(path_processID, 'Pred.nii.gz'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Prob.nii.gz'),
                        os.path.join(path_output_dir, 'Prob_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm_vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm_vessel16.nii.gz'))
            shutil.copy(os.path.join(path_json_out, ID + '_platform_json.json'),
                        os.path.join(path_output_dir, 'Pred_Aneurysm_platform_json.json'))
            upload_json_aiteam(os.path.join(path_output_dir, 'Pred_Aneurysm_platform_json.json'))

            logging.info('!!! ' + ID + ' post_aneurysm finish.')
            return output_tuple
        else:
            logging.error('!!! ' + str(ID) + ' Insufficient GPU Memory.')
            code_pass = 1
            msg = "Insufficient GPU Memory"

    except:
        logging.error('!!! ' + str(ID) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)

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
    pred_aneurysm_path , mra_brain_json_path,  mip_pitch_json_path, \
        mip_yaw_json_path,  sort_json_path =  pipeline_aneurysm(ID=ID,
                                                                MRA_BRAIN_file=MRA_BRAIN_file,
                                                                path_output=path_output,
                                                                path_code=path_code,
                                                                path_processModel=path_processModel,
                                                                path_outdcm=path_DcmDir,
                                                                path_json=path_json,
                                                                path_log=path_log,
                                                                path_cuatom_model=CUATOM_MODEL_ANEURYSM,
                                                                gpu_n=gpu_n)



