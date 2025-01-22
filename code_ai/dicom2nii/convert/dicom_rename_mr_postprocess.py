import argparse
import pathlib
import re
import traceback
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from tqdm.auto import tqdm
from pydicom import dcmread, FileDataset, Dataset,DataElement
import orjson
from .config import MRSeriesRenameEnum,DSCSeriesRenameEnum,ASLSEQSeriesRenameEnum

class ProcessingStrategy(metaclass=ABCMeta):

    @abstractmethod
    def process(self, study_path: pathlib.Path, *args, **kwargs):
        pass


class MRProcessingStrategy(ProcessingStrategy):
    exclude_dicom_tag = {
        '00020012': 'Implementation Class UID',
        '00020013': 'Implementation Version Name',
        '00080005': 'Specific Character Set',
        '00080008': 'Image Type',
        '00080016': 'SOP Class UID',
        '00080020': 'Study Date',
        '00080021': 'Series Date',
        '00080022': 'Acquisition Date',
        '00080023': 'Study Date',
        '00080030': 'Study Time',
        '00080031': 'Series Time',
        '00080032': 'Acquisition Time',
        '00080033': 'Study Time',
        '00080050': 'Accession Number',
        '00080060': 'Modality',
        '00080070': 'Manufacturer',
        '00080080': 'Institution Name',
        '00080090': 'Referring Physician Name',
        '00081010': 'Station Name',
        '00081030': 'Study Description',
        '0008103e': 'Series Description',
        '00081090': 'Manufacturer Model Name',
        '00081111': 'Referenced Performed Procedure Step Sequence',
        '00081140': 'Referenced Image Sequence',
        '00082218': 'Anatomic Region Sequence',

        '00100010': 'Patient Name',
        '00100020': 'Patient ID',
        '00100030': 'Patient Birth Date',
        '00100040': 'Patient Sex',
        '00101010': 'Patient Age',
        '00101030': 'Patient Weight',
        '001021b0': 'Additional Patient History',

        '00180015': 'Body Part Examined',
        '00180020': 'Scanning Sequence',
        '00180021': 'Sequence Variant',
        '00180022': 'Scan Options',
        '00180023': 'MR Acquisition Type',
        '00180025': 'Angio Flag',
        '00181020': 'Software Versions',
        '00181030': 'Protocol Name',

        '0020000D': 'Study Instance UID',
        '0020000E': 'Series Instance UID',
        '00200010': 'Study ID',
        '00200011': 'Series Number',
        '00200012': 'Acquisition Number',

        '00210010': 'Private Creator',
        '00230010': 'Private Creator',
        '00231080': 'Private Creator',

        '00250010': 'Private Creator',
        '00270010': 'Private Creator',
        '00290010': 'Private Creator',

        '00380010': 'Admission ID',
        '00400242': 'Performed Station Name',
        '00400243': 'Performed Location',
        '00400244': 'Performed Procedure Step Start Date',
        '00400245': 'Performed Procedure Step Start Time',
        '00400252': 'Performed Procedure Step ID',
        '00400254': 'Performed Procedure Step Descriptio',
        '00400275': 'Request Attributes Sequence',
    }

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        series_folder_list = list(filter(lambda x: x.is_dir() and x.name != '.meta', study_path.iterdir()))
        meta_folder = study_path.joinpath('.meta')
        meta_folder.mkdir(exist_ok=True)
        for series_folder in tqdm(series_folder_list,
                                  desc=f'json :{study_path.name}', ):
            try:
                jsonlines_path = meta_folder.joinpath(f'{series_folder.name}.jsonlines')
                if jsonlines_path.exists():
                    continue
                dicom_header = {}
                dicom_list = list(
                    map(lambda x: dcmread(str(x), force=True, stop_before_pixels=True), series_folder.iterdir()))
                for i in dicom_list:
                    result = dicom_header.get(str(i.SeriesInstanceUID))
                    if result:
                        temp_json_dict = i.to_json_dict()
                        temp_dict = {}
                        for temp_key, temp_value in temp_json_dict.items():
                            if self.exclude_dicom_tag.get(temp_key):
                                continue
                            else:
                                if temp_value.get('Value'):
                                    temp_dict[temp_key] = temp_value['Value']
                        result.append(temp_dict)
                    else:
                        dicom_header.update({str(i.SeriesInstanceUID): [i.to_json_dict()]})
                with open(f'{jsonlines_path}', 'wb') as file:
                    file.write(orjson.dumps(dicom_header, option=orjson.OPT_APPEND_NEWLINE))
            except :
                print(str(series_folder))
                print(traceback.print_exc())


class MRDicomProcessingStrategy(ProcessingStrategy):
    exclude_dicom_series = {
        MRSeriesRenameEnum.RESTING.value,
        MRSeriesRenameEnum.RESTING2000.value,
        MRSeriesRenameEnum.CVR.value,
        MRSeriesRenameEnum.CVR1000.value,
        MRSeriesRenameEnum.CVR2000.value,
        MRSeriesRenameEnum.CVR2000_EAR.value,
        MRSeriesRenameEnum.CVR2000_EYE.value,

        MRSeriesRenameEnum.eADC.value,
        MRSeriesRenameEnum.eSWAN.value,
        MRSeriesRenameEnum.DTI32D.value,
        MRSeriesRenameEnum.DTI64D.value,
        MRSeriesRenameEnum.MRAVR_NECK.value,
        MRSeriesRenameEnum.MRAVR_BRAIN.value,

        DSCSeriesRenameEnum.DSC.value,
        DSCSeriesRenameEnum.rCBV.value,
        DSCSeriesRenameEnum.rCBF.value,
        DSCSeriesRenameEnum.MTT.value,

        ASLSEQSeriesRenameEnum.ASLSEQ.value,


        ASLSEQSeriesRenameEnum.ASLSEQATT.value,
        ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR.value,

        ASLSEQSeriesRenameEnum.ASLSEQCBF.value,
        ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR.value,

        ASLSEQSeriesRenameEnum.ASLSEQPW.value,

        ASLSEQSeriesRenameEnum.ASLPROD.value,
        ASLSEQSeriesRenameEnum.ASLPRODCBF.value,
        ASLSEQSeriesRenameEnum.ASLPRODCBF_COLOR.value,
    }

    @classmethod
    def validate_age(cls, input_string: str):
        pattern = re.compile(r'^(\d{3})Y$')
        if pattern.match(input_string):
            return False,input_string
        else:
            if input_string.isnumeric():
                return True,f"{int(input_string):03d}Y"
            else:
                return True,f"{int(input_string.split('Y')[0]):03d}Y"

    @classmethod
    def validate_date(cls, input_string: str):
        pattern = re.compile(r'^(\d{8})$')
        if pattern.match(input_string):
            return input_string
        else:
            pass

    @classmethod
    def validate_time(cls, input_string: str):
        pattern = re.compile(r'^(\d{6})$')
        if pattern.match(input_string):
            return False,input_string
        else:
            return True,f'{int(input_string):06d}'

    def revise_age(self, dicom_ds: FileDataset):
        # (0010,1010) Patient Age 057Y
        age = dicom_ds.get((0x10, 0x1010))
        if age:
            flag,age_value = self.validate_age(age.value)
            if flag:
                dicom_ds[0x10, 0x1010].value = age_value
        return flag,dicom_ds

    def revise_time(self, dicom_ds: FileDataset):
        # (0008,0030) Study Time 141107
        # (0008,0031) Series Time 142808
        # (0008,0032) Acquisition Time 142808
        # (0008,0033) Content Time 142808
        study_time = dicom_ds.get((0x08, 0x30))
        series_time = dicom_ds.get((0x08, 0x31))
        acquisition_time = dicom_ds.get((0x08, 0x32))
        content_time = dicom_ds.get((0x08, 0x33))
        flag_list = []
        if study_time:
            study_time_str = study_time.value.split('.')[0]
            flag, study_time_str = self.validate_time(study_time_str)
            if flag:
                dicom_ds[(0x08, 0x30)].value = study_time_str
                flag_list.append(flag)
        if series_time:
            series_time_str = series_time.value.split('.')[0]
            flag, series_time_str = self.validate_time(series_time_str)
            if flag:
                dicom_ds[(0x08, 0x31)].value = series_time_str
                flag_list.append(flag)
        if acquisition_time:
            acquisition_time_str = acquisition_time.value.split('.')[0]
            flag, acquisition_time_str = self.validate_time(acquisition_time_str)
            if flag:
                dicom_ds[(0x08, 0x32)].value = acquisition_time_str
                flag_list.append(flag)
        if content_time:
            content_time_str = content_time.value.split('.')[0]
            flag, content_time_str = self.validate_time(content_time_str)
            if flag:
                dicom_ds[(0x08, 0x33)].value = content_time_str
                flag_list.append(flag)
        return any(flag_list),dicom_ds

    # def revise_date(self, dicom_ds: FileDataset):
    #     # (0008,0020)	Study Date 20220920
    #     # (0008,0021)	Series Date 20220920
    #     # (0008,0022)	Acquisition Date 20220920
    #     # (0008,0023)	Content Date 20220920
    #     # (0010,0030)	Patient Birth Date 19641228
    #
    #     study_date       = dicom_ds.get((0x08, 0x20))
    #     series_date      = dicom_ds.get((0x08, 0x21))
    #     acquisition_date = dicom_ds.get((0x08, 0x22))
    #     content_date     = dicom_ds.get((0x08, 0x23))
    #     birth_date       = dicom_ds.get((0x10, 0x30))
    #     pass

    def get_series_folder_list(self, study_path: pathlib.Path) -> List[pathlib.Path]:
        # series_folder_list = list(study_path.iterdir())
        series_folder_list = []
        for series_folder in study_path.iterdir():
            if series_folder.is_dir() and series_folder.name != '.meta':
                if series_folder.name not in self.exclude_dicom_series:
                    series_folder_list.append(series_folder)
        return series_folder_list

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        series_folder_list = self.get_series_folder_list(study_path=study_path)
        for series_folder in series_folder_list:
            # dicom_list = list(map(lambda x: (dcmread(str(x), force=True), x), series_folder.iterdir()))
            dicom_list = []
            for x in series_folder.iterdir():
                with open(x, mode='rb') as dcm:
                    dicom_list.append((dcmread(dcm, force=True), x))
            for row in dicom_list:
                dicom = row[0]
                file_name = row[1]
                flag_age,dicom = self.revise_age(dicom_ds=dicom)
                flag_time,dicom = self.revise_time(dicom_ds=dicom)
                if any([flag_age,flag_time]):
                    dicom.save_as(filename=f'{file_name}')




class PostProcessManager:
    processing_strategy_list: List[ProcessingStrategy] = [MRDicomProcessingStrategy(),
                                                          # DWI ADC acquisition_time
                                                          # MRProcessingStrategy()
                                                          ]

    def __init__(self, *args, **kwargs):
        pass

    def post_process(self, study_path):
        for processing_strategy in self.processing_strategy_list:
            processing_strategy.process(study_path=study_path)