import argparse
import pathlib
import re
import traceback
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union,Tuple
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
        flag = False
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
                    dicom.save_as(str(file_name),)



class ADCProcessingStrategy(MRDicomProcessingStrategy):

    @staticmethod
    def has_image_position_patient_with_dicom(dicom_ds:FileDataset) -> bool:
        image_position_patient = dicom_ds.get((0x20, 0x32))
        image_type             = dicom_ds.get((0x08, 0x08))
        # (0008,0008)	Image Type	DERIVED\SECONDARY\COMBINED
        if image_position_patient and image_type[2] == 'ADC':
            return True
        else:
            return False

    @staticmethod
    def read_dicom(dicom_path: Union[pathlib.Path, str]) -> FileDataset:
        with open(dicom_path, mode='rb') as dcm:
            dicom_ds: FileDataset = dcmread(dcm, force=True)
        return dicom_ds


    def process(self, study_path: pathlib.Path, *args, **kwargs):
        series_folder_list = self.get_series_folder_list(study_path=study_path)

        #
        filter_series_folder_list = sorted(filter(lambda x: (x.name == MRSeriesRenameEnum.ADC.value) \
                                                  or        (x.name == MRSeriesRenameEnum.DWI1000.value),
                                                  series_folder_list))

        if filter_series_folder_list and len(filter_series_folder_list) >= 2:
            adc_series_folder = study_path.joinpath(MRSeriesRenameEnum.ADC.value)
            dwi_1000_series_folder = study_path.joinpath(MRSeriesRenameEnum.DWI1000.value)

            adc_series_dicom_list      = sorted(adc_series_folder.glob('*.dcm'))
            dwi_1000_series_dicom_list = sorted(dwi_1000_series_folder.glob('*.dcm'))

            adc_dicom_list: List[Tuple[FileDataset, pathlib.Path]] = []
            dwi_dicom_list: List[Tuple[FileDataset, pathlib.Path]] = []
            if len(adc_series_dicom_list) == len(dwi_1000_series_dicom_list):

                for index,dicom_path in enumerate(adc_series_dicom_list) :
                    # read ADC
                    dicom_ds : FileDataset = self.read_dicom(dicom_path=dicom_path)
                    # get  (0020,0032)	Image Position Patient	-111.946\-124.966\-98.7531
                    if self.has_image_position_patient_with_dicom(dicom_ds):
                        break
                    else:
                        adc_dicom_list.append((dicom_ds, dicom_path))
                # read DWI1000
                temp_dwi_list: List[Tuple[FileDataset, pathlib.Path]] = [(self.read_dicom(dicom_path),dicom_path)
                                                                         for dicom_path in dwi_1000_series_dicom_list]
                dwi_dicom_list.extend(temp_dwi_list)

            elif len(adc_series_dicom_list)//2 == len(dwi_1000_series_dicom_list):
                dicom_list = [(dcmread(x, force=True), x) for x in adc_series_dicom_list]
                adc_dicom_list = list(filter(lambda x: not self.has_image_position_patient_with_dicom(x[0]),dicom_list))
                # read DWI1000
                temp_dwi_list = [(self.read_dicom(dicom_path), dicom_path) for dicom_path in dwi_1000_series_dicom_list]
                dwi_dicom_list.extend(temp_dwi_list)
            # Sort slices by Instance Number
            try:
                # (0020,0013)	Instance Number	4
                sorted_adc_dicom_list: List[Tuple[FileDataset, pathlib.Path]]      = sorted(adc_dicom_list,
                                                                                            key=lambda x: x[0][0x20, 0x13].value)
                sorted_dwi_1000_dicom_list: List[Tuple[FileDataset, pathlib.Path]] = sorted(dwi_dicom_list,
                                                                                            key=lambda x: x[0][0x20, 0x13].value)
            except:
                # (0027,1041)	Unknown  Tag &  Data  -73.76799 (Image location (0027,1041) FL
                sorted_adc_dicom_list: List[Tuple[FileDataset, pathlib.Path]]      = sorted(adc_dicom_list,
                                                                                            key=lambda x: x[0][0x27, 0x1041].value)
                sorted_dwi_1000_dicom_list: List[Tuple[FileDataset, pathlib.Path]] = sorted(dwi_dicom_list,
                                                                                            key=lambda x: x[0][0x27, 0x1041].value)

            for index, (dwi_1000_dicom_ds, dwi_1000_dicom_path) in enumerate(sorted_dwi_1000_dicom_list):
                try:
                    adc_dicom_ds = sorted_adc_dicom_list[index][0]
                    adc_dicom_path = sorted_adc_dicom_list[index][1]

                    # (0028,0030)	Pixel Spacing	0.8984\0.8984
                    # (0018,0050)	Slice Thickness	4
                    # (0018,0088)	Spacing Between Slices	4
                    # (0020,1041)	Slice Location	-85.72914886
                    # (0020,0032)	Image Position Patient	-111.946\-124.966\-98.7531
                    # (0020,0037)	Image Orientation Patient	0.997335\-0.0447198\0.057639\0.0414903\0.997565\0.0560564

                    adc_dicom_ds[0x28, 0x30] = dwi_1000_dicom_ds[0x28, 0x30]
                    adc_dicom_ds[0x18, 0x50] = dwi_1000_dicom_ds[0x18, 0x50]
                    adc_dicom_ds[0x18, 0x88] = dwi_1000_dicom_ds[0x18, 0x88]
                    adc_dicom_ds[0x20, 0x1041] = dwi_1000_dicom_ds[0x20, 0x1041]
                    adc_dicom_ds[0x20, 0x32] = dwi_1000_dicom_ds[0x20, 0x32]
                    adc_dicom_ds[0x20, 0x37] = dwi_1000_dicom_ds[0x20, 0x37]
                    adc_dicom_ds.save_as(str(adc_dicom_path), )
                except:
                    traceback.print_exc()
                    continue
        else:
            pass



class PostProcessManager:
    processing_strategy_list: List[ProcessingStrategy] = [MRDicomProcessingStrategy(),
                                                          # DWI ADC acquisition_time
                                                          # MRProcessingStrategy()
                                                          ADCProcessingStrategy()
                                                          ]

    def __init__(self, *args, **kwargs):
        pass

    def post_process(self, study_path):
        for processing_strategy in self.processing_strategy_list:
            processing_strategy.process(study_path=study_path)
