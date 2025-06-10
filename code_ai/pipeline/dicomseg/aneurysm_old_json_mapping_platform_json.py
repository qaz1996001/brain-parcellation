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

from code_ai import load_dotenv
from code_ai.pipeline.dicomseg.schema import MaskRequest
from code_ai.pipeline.dicomseg.schema import SortedRequest,StudyRequest,AITeamRequest
from code_ai.pipeline.dicomseg.schema import SortedSeriesRequest
from code_ai.pipeline import get_study_id


def load_mask_json(old_json_path: str, group_id:int) -> Optional[MaskRequest]:
    with open(old_json_path) as f:
        data_dict = json.load(f)
    try:
        basename = os.path.basename(old_json_path).split('.')[0]
        print('basename',basename)
        ser_name = basename.replace(get_study_id(basename),'').removeprefix('_').removesuffix('_')
        series_instance_uid =  data_dict['SeriesInstanceUID']
        mask_dict = { 'study_instance_uid' : data_dict['StudyInstanceUID'],
                      'group_id':group_id,}
        mask_series = {'series_instance_uid': data_dict['SeriesInstanceUID'],
                       'series_type': ser_name, }

        print('ser_name',ser_name)
        mask_instance_dict_list = []
        for instance in data_dict['Instances']:
            mask_instance_dict = {'mask_index'              : instance['maskIndex'],
                                  'mask_name'               : instance['maskName'],
                                  'diameter'                : instance['Diameter'],
                                  'type'                    : instance['Type'],
                                  'location'                : instance['Location'],
                                  'sub_location'            : instance['SubLocation'],
                                  'prob_max'                : instance['Prob_max'],
                                  'checked'                 : '1',
                                  'is_ai'                   : '1',
                                  'seg_series_instance_uid' : instance['DICOM-SEG_SeriesInstanceUid'],
                                  'seg_sop_instance_uid'    : instance['DICOM-SEG_SOPInstanceUid'],
                                  'dicom_sop_instance_uid'  :series_instance_uid,
                                  'main_seg_slice'          : instance['MainSegSlice'],
                                  'is_main_seg': instance['isMainSeg'],

                                  }
            mask_instance_dict_list.append(mask_instance_dict)
        mask_series.update({'instances':mask_instance_dict_list})
        mask_dict.update({'series':[mask_series]})
        mask_request = MaskRequest.model_validate(mask_dict)
        return mask_request
    except:
        traceback.print_exc()
        return None



def load_study_json(old_json_path: str, group_id:int) -> Optional[StudyRequest]:
    with open(old_json_path) as f:
        data_dict = json.load(f)
    try:
        new_data_dict = { 'group_id'           : group_id,
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
        study_instance_uid = data_dict['data'][0]['study_instance_uid']
        sort_series_list = []
        for i in range(len(data_dict['data'])):
            new_data_dict = {'series_instance_uid':data_dict['data'][i]['series_instance_uid'],
                             'instance':data_dict['data'][i]['instances']}

            sort_series_list.append(SortedSeriesRequest.model_validate(new_data_dict))


        sort_mask_dict = { 'study_instance_uid' :study_instance_uid,
                           'series'             :sort_series_list
                           }

        return SortedRequest.model_validate(sort_mask_dict)
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
    path_nii           = pathlib.Path(args.input_nii)
    path_json_list     = list(map(lambda x:pathlib.Path(x), args.old_json_path))
    path_sort_json     = pathlib.Path(args.sort_json_path)
    GROUP_ID           = os.getenv("GROUP_ID_ANEURYSM",50)
    platform_json_path = path_nii.parent.joinpath(path_nii.name.replace('.nii.gz',
                                                                        '_platform_json.json'))
    sort_request  = load_sort_dicom_json(sort_json_path = path_sort_json)
    study_request, series_instance_uid = load_study_json(old_json_path=path_json_list[0],
                                                         group_id=GROUP_ID)
    new_mask_list :List[Optional[MaskRequest]] = []
    for path_json in path_json_list:
        mask_json   = load_mask_json(old_json_path=path_json,
                                     group_id=GROUP_ID)
        print(mask_json)
        new_mask_list.append(mask_json)

    first_new_mask = new_mask_list[0]
    for new_mask in new_mask_list[1:]:
        first_new_mask.series.extend(new_mask.series)
    at_team_request = AITeamRequest(study=study_request,
                                    sorted=sort_request,
                                    mask=first_new_mask)
    with open(platform_json_path, 'w') as f:
        f.write(at_team_request.model_dump_json())
    print('platform_json_path',platform_json_path)
    print("Processing complete!")


if __name__ == '__main__':
    load_dotenv()
    main()
