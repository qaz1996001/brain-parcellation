# -*- coding: utf-8 -*-
"""
Created on 2025-05-19
@author: sean
"""
import argparse
import json
import os.path
import pathlib
import traceback
from typing import Optional, List, Union

from code_ai.pipeline.dicomseg.create_dicomseg_multi_file_json_claude import InstanceRequest, GROUP_ID
from code_ai.pipeline.dicomseg.create_dicomseg_multi_file_json_claude import MaskSeriseRequest,MaskRequest
from code_ai.pipeline.dicomseg.create_dicomseg_multi_file_json_claude import SortedRequest,StudyRequest,AITeamRequest
from code_ai.pipeline.dicomseg.create_dicomseg_multi_file_json_claude import SeriesRequest,SeriesTypeEnum
from code_ai.pipeline import get_study_id


def load_mask_json(old_json_path: str, group_id:int = GROUP_ID) -> Optional[MaskRequest]:
    with open(old_json_path) as f:
        data_dict = json.load(f)
    try:
        basename = os.path.basename(old_json_path).split('.')[0]
        print('basename',basename)
        ser_name = basename.replace(get_study_id(basename),'').removeprefix('_').removesuffix('_')
        new_data_dict = { 'study_instance_uid' : data_dict['StudyInstanceUID'],
                          'group_id':group_id,
                          }
        print('ser_name',ser_name)
        mask_instance_dict_list = []
        for instance in data_dict['Instances']:
            mask_instance_dict = {'mask_index'          : instance['maskIndex'],
                                  'mask_name'           : instance['maskName'],
                                  'diameter'            : instance['Diameter'],
                                  'type'                : instance['Type'],
                                  'location'            : instance['Location'],
                                  'sub_location'        : instance['SubLocation'],
                                  'prob_max'            : instance['Prob_max'],
                                  'checked'             : '1',
                                  'is_ai'               : '1',
                                  # 'series_instance_uid' : instance['DICOM-SEG_SeriesInstanceUid'],
                                  # 'sop_instance_uid'    : instance['DICOM-SEG_SOPInstanceUid'],
                                  'is_main_seg'         : instance['isMainSeg'],
                                  'main_seg_slice'      : instance['MainSegSlice'],
                                  'series_type'         : ser_name
                                  }
            mask_instance_dict_list.append(MaskSeriseRequest.model_validate(mask_instance_dict))
        print(mask_instance_dict_list[0])
        new_data_dict.update({'instances':mask_instance_dict_list})
        mask_request = MaskRequest.model_validate(new_data_dict)
        return mask_request
    except:
        traceback.print_exc()
        return None



def load_study_json(old_json_path: str, group_id:int = GROUP_ID) -> Optional[StudyRequest]:
    with open(old_json_path) as f:
        data_dict = json.load(f)
    try:
        new_data_dict = { 'group_id'           : GROUP_ID,
                          'study_date'         : data_dict['StudyDate'],
                          'gender'             : data_dict['Gender'],
                          'age'                : data_dict['Age'],
                          'study_name'         : data_dict['StudyDescription'],
                          'patient_name'       : data_dict['PatientName'],
                          'aneurysm_lession'   : data_dict['Aneurysm_Number'],
                          'aneurysm_status'    : 1,
                          'resolution_x'       : data_dict['resolution_x'],
                          'resolution_y'       : data_dict['resolution_y'],
                          'study_instance_uid' : data_dict['StudyInstanceUID'],
                          'patient_id'         : data_dict['PatientID'],
                          }
        study_request = StudyRequest.model_validate(new_data_dict)
        return study_request, data_dict['SeriesInstanceUID']
    except:
        traceback.print_exc()
        return None



def load_sort_dicom_json(sort_json_path: str) -> Optional[Union[List[SortedRequest],SortedRequest]]:
    """載入並排序DICOM檔案，只需執行一次"""
    with open(sort_json_path) as f:
        data_dict = json.load(f)
    try:
        sort_request_list = []
        for i in range(len(data_dict['data'])):
            new_data_dict = { 'study_instance_uid':data_dict['data'][i]['study_instance_uid'],
                              'series':[{'series_instance_uid':data_dict['data'][i]['series_instance_uid'],
                                        'instance':data_dict['data'][i]['instances']
                                        }]
                              }
            sort_request_list.append(SortedRequest.model_validate(new_data_dict))
        return sort_request_list
    except:
        traceback.print_exc()
        return None


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="處理NIFTI檔案並創建DICOM-SEG檔案")
    parser.add_argument('--ID', type=str, default='10516407_20231215_MR_21210200091',
                        help='目前執行的case的patient_id or study id')
    parser.add_argument('--input_nii', type=str,
                        default='/mnt/e/rename_dicom_202505051/10516407_20231215_MR_21210200091/T2FLAIR_AXI',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--old_json_path', type=str, nargs='+',
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--sort_json_path', type=str,
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
                        help='用於輸出結果的資料夾')

    args = parser.parse_args()
    ID = args.ID
    path_nii           = pathlib.Path(args.input_nii)
    path_json_list     = pathlib.Path(args.old_json_path)
    path_sort_json     = pathlib.Path(args.sort_json_path)

    platform_json_path = path_nii.parent.joinpath(path_nii.name.replace('.nii.gz', '_platform_json.json'))
    sort_request_list  = load_sort_dicom_json(path_sort_json)
    study_request, series_instance_uid = load_study_json(path_json)
    filter_sort_request = next(filter(lambda x:x.series[0].series_instance_uid == series_instance_uid,sort_request_list))
    mask_json           = load_mask_json(old_json_path=path_json)
    at_team_request      = AITeamRequest(study=study_request,
                                        sorted=filter_sort_request,
                                        mask=mask_json)
    with open(platform_json_path, 'w') as f:
        f.write(at_team_request.model_dump_json())
    print('platform_json_path',platform_json_path)
    print("Processing complete!")


if __name__ == '__main__':
    main()
