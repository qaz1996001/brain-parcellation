import argparse
import pathlib
import re
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np

from .config import *


class ProcessingStrategy(metaclass=ABCMeta):
    pattern: re.Pattern
    suffix_pattern: re.Pattern
    FILE_SIZE: int = 1 * 1024 * 1024  # 1MB
    CHAR_OFFSET: int = 95  # a = 97，b = 98 ，a:2 b:3

    @abstractmethod
    def process(self, input_path: pathlib.Path, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        super().__call__()
        self.process(*args, **kwargs)

    def del_file(self, study_path):
        for series_path in study_path.iterdir():
            swan_pattern = self.pattern.match(series_path.name)
            if swan_pattern:
                if series_path.stat().st_size < self.FILE_SIZE:
                    print('del_file', series_path)
                    json_file_path = series_path.parent.joinpath(series_path.name.replace(r'.nii.gz', '.json'))
                    if json_file_path.exists():
                        json_file_path.unlink()
                    if series_path.exists():
                        series_path.unlink()

    def rename_file_suffix(self, series_path: pathlib.Path, pattern: re.Pattern):
        pattern_result = pattern.match(series_path.name)
        if pattern_result:
            groups = pattern_result.groups()
            # print(f'groups {groups}')
            if len(groups[1]) > 0:
                suffix_char = pattern_result.groups()[1]
                suffix_int = ord(suffix_char) - self.CHAR_OFFSET
                new_file_name = rf'{pattern_result.groups()[0]}_{suffix_int}.nii.gz'
                new_file_path = series_path.parent.joinpath(new_file_name)
                return new_file_path

    def rename_file_only(self, series_path: pathlib.Path, pattern: re.Pattern):
        pattern_result = pattern.match(series_path.name)
        if pattern_result:
            new_file_name = rf'{pattern_result.groups()[0]}.nii.gz'
            new_file_path = series_path.parent.joinpath(new_file_name)
            return new_file_path

    def rename_file(self, study_path: pathlib.Path):
        file_list = []
        for series_path in study_path.iterdir():
            swan_pattern = self.pattern.match(series_path.name)
            if swan_pattern:
                file_list.append(series_path)
        print(file_list)
        for series_path in file_list:
            if series_path.exists():
                if len(file_list) == 1:
                    new_file_path = self.rename_file_only(series_path=series_path, pattern=self.suffix_pattern)
                else:
                    new_file_path = self.rename_file_suffix(series_path=series_path, pattern=self.suffix_pattern)

                file_base_name = series_path.name.replace('.nii.gz', '')
                all_rename_file_list = list(filter(lambda x: x.stem == file_base_name, study_path.iterdir()))
                if new_file_path:
                    print(series_path, new_file_path)
                    new_file_base_name = new_file_path.name.replace('.nii.gz', '')
                    for rename_file in all_rename_file_list:
                        new_rename_file = new_file_path.parent.joinpath(f"{new_file_base_name}{rename_file.suffix}")
                        rename_file.rename(new_rename_file)
                    series_path.rename(new_file_path)


class ADCProcessingStrategy(ProcessingStrategy):
    pattern = re.compile(r'(?<!e)(ADC[a-z]{0,2}?)(\.nii\.gz)$', re.IGNORECASE)
    suffix_pattern = re.compile(r'(?<!e)(ADC)([a-z]{0,2}?)(\.nii\.gz)$', re.IGNORECASE)
    dwi_pattern = re.compile(r'(DWI0)(.*\.nii\.gz)$', re.IGNORECASE)
    FILE_SIZE = 100 * 1024  # 100kB

    def update_header(self, study_path: pathlib.Path, *args, **kwargs):
        adc_file_list = []
        dwi_file_list = []
        for series_path in study_path.iterdir():
            adc_pattern = self.pattern.match(series_path.name)
            dwi_pattern = self.dwi_pattern.match(series_path.name)
            if adc_pattern:
                adc_file_list.append(series_path.name)
            if dwi_pattern:
                dwi_file_list.append(series_path.name)
        if len(dwi_file_list) > 0 and len(adc_file_list) > 0:
            for dwi_file in dwi_file_list:
                dwi_nii = nib.load(str(study_path.joinpath(dwi_file)))
                for series_path in adc_file_list:
                    adc_path_str = str(study_path.joinpath(series_path))
                    image_nii = nib.load(adc_path_str)
                    if dwi_nii.get_fdata().shape == image_nii.get_fdata().shape:
                        new_header = image_nii.header.copy()
                        new_header['pixdim'] = dwi_nii.header['pixdim']
                        new_affine = dwi_nii.affine
                        data = image_nii.get_fdata()
                        output_nii = nib.Nifti1Image(data, new_affine, new_header)
                        nib.save(output_nii, adc_path_str)

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        self.update_header(study_path=study_path)
        self.rename_file(study_path=study_path)

    def rename_file(self, study_path: pathlib.Path):
        adc_file_list = []
        dwi_file_list = []
        for series_path in study_path.iterdir():
            adc_pattern = self.pattern.match(series_path.name)
            dwi_pattern = self.dwi_pattern.match(series_path.name)
            if adc_pattern:
                adc_file_list.append(series_path)
            if dwi_pattern:
                dwi_file_list.append(series_path)
        for series_path in adc_file_list:
            if series_path.exists():
                if len(adc_file_list) == 1:
                    new_file_path = self.rename_file_only(series_path=series_path, pattern=self.suffix_pattern)
                    if new_file_path:
                        file_base_name = series_path.name.replace('.nii.gz', '')
                        new_file_base_name = new_file_path.name.replace('.nii.gz', '')
                        all_rename_file_list = list(filter(lambda x: x.stem == file_base_name, study_path.iterdir()))
                        for rename_file in all_rename_file_list:
                            new_rename_file = new_file_path.parent.joinpath(f"{new_file_base_name}{rename_file.suffix}")
                            rename_file.rename(new_rename_file)
                        series_path.rename(new_file_path)
                else:
                    adc_nii_file_list = []
                    for adc_file in adc_file_list:
                        adc_nii = nib.load(str(adc_file))
                        data = adc_nii.get_fdata().round(0).astype(np.int32)
                        adc_nii_file_list.append((adc_nii, data, adc_file))
                    for i in adc_nii_file_list:
                        adc_nii = i[0]
                        data = i[1]
                        adc_file = i[2]
                        for dwi_file in dwi_file_list:
                            dwi_nii = nib.load(str(dwi_file))
                            if (dwi_nii.affine == adc_nii.affine).all():
                                print(dwi_file)
                                adc_file.unlink()
                                adc_file_str = dwi_file.name.replace('DWI0', 'ADC')
                                adc_file_path = adc_file.parent.joinpath(adc_file_str)
                                output_nii = nib.Nifti1Image(data, adc_nii.affine, adc_nii.header)
                                nib.save(output_nii, str(adc_file_path))

                                raw_bval_file_str = adc_file.name.replace('.nii.gz', '.bval')
                                bval_file_path = adc_file.parent.joinpath(raw_bval_file_str)
                                if bval_file_path.exists():
                                    new_bval_file_str = adc_file_path.name.replace('.nii.gz', '.bval')
                                    new_bval_file_path = adc_file_path.parent.joinpath(new_bval_file_str)
                                    bval_file_path.rename(new_bval_file_path)

                                raw_bvec_file_str = adc_file.name.replace('.nii.gz', '.bvec')
                                bvec_file_path = adc_file.parent.joinpath(raw_bvec_file_str)
                                if bvec_file_path.exists():
                                    new_bvec_file_str = adc_file_path.name.replace('.nii.gz', '.bvec')
                                    new_bvec_file_path = adc_file_path.parent.joinpath(new_bvec_file_str)
                                    bvec_file_path.rename(new_bvec_file_path)
                                break


class SWANProcessingStrategy(ProcessingStrategy):
    pattern = re.compile(r'(?<!e)(SWAN[a-z]{0,2}?)(\.nii\.gz)$', re.IGNORECASE)
    suffix_pattern = re.compile(r'(?<!e)(SWAN)([a-z]{0,2}?)(\.nii\.gz)$', re.IGNORECASE)
    FILE_SIZE = 800 * 1024  # 800kB

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        self.del_file(study_path=study_path)
        self.rename_file(study_path=study_path)


class T1ProcessingStrategy(ProcessingStrategy):
    pattern = re.compile(r'(T1.*)(\.nii\.gz)$')
    suffix_pattern = re.compile(r'(T1.*)(AXIr?|CORr?|SAGr?)([a-z]{0,1})(\.nii\.gz)$')
    FILE_SIZE = 800 * 1024  # 800kB

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        self.del_file(study_path=study_path)
        self.rename_file(study_path=study_path)

    def rename_file_suffix(self, series_path: pathlib.Path, pattern: re.Pattern):
        pattern_result = pattern.match(series_path.name)
        if pattern_result:
            groups = pattern_result.groups()
            if len(groups[2]) > 0:
                suffix_char = pattern_result.groups()[2]
                suffix_int = ord(suffix_char) - self.CHAR_OFFSET
                new_file_name = rf'{pattern_result.groups()[0]}{pattern_result.groups()[1]}_{suffix_int}.nii.gz'
                new_file_path = series_path.parent.joinpath(new_file_name)
                return new_file_path

    def rename_file_only(self, series_path: pathlib.Path, pattern: re.Pattern):
        pattern_result = pattern.match(series_path.name)
        if pattern_result:
            new_file_name = rf'{pattern_result.groups()[0]}{pattern_result.groups()[1]}.nii.gz'
            new_file_path = series_path.parent.joinpath(new_file_name)
            return new_file_path


class T2ProcessingStrategy(ProcessingStrategy):
    pattern = re.compile(r'(T2.*)(\.nii\.gz)$')
    suffix_pattern = re.compile(r'(T2.*)(AXIr?|CORr?|SAGr?)([a-z]{0,1})(\.nii\.gz)$')
    FILE_SIZE = 800 * 1024  # 800kB

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        self.del_file(study_path=study_path)
        self.rename_file(study_path=study_path)

    def rename_file_suffix(self, series_path: pathlib.Path, pattern: re.Pattern):
        pattern_result = pattern.match(series_path.name)
        if pattern_result:
            groups = pattern_result.groups()
            if len(groups[2]) > 0:
                suffix_char = pattern_result.groups()[2]
                suffix_int = ord(suffix_char) - self.CHAR_OFFSET
                new_file_name = rf'{pattern_result.groups()[0]}{pattern_result.groups()[1]}_{suffix_int}.nii.gz'
                new_file_path = series_path.parent.joinpath(new_file_name)
                return new_file_path

    def rename_file_only(self, series_path: pathlib.Path, pattern: re.Pattern):
        pattern_result = pattern.match(series_path.name)
        if pattern_result:
            new_file_name = rf'{pattern_result.groups()[0]}{pattern_result.groups()[1]}.nii.gz'
            new_file_path = series_path.parent.joinpath(new_file_name)
            return new_file_path


class DwiProcessingStrategy(ProcessingStrategy):
    pattern = re.compile(r'(DWI.*)(\.nii\.gz)$')
    suffix_pattern = re.compile(r'(DWI.*)(?<![a-z])([a-z]{0,2}?)(\.nii\.gz)$')
    FILE_SIZE = 550 * 1024  # 800kB

    def process(self, study_path: pathlib.Path, *args, **kwargs):
        self.del_file(study_path=study_path)
        self.rename_file(study_path=study_path)


class PostProcessManager:
    processing_strategy_list: List[ProcessingStrategy] = [DwiProcessingStrategy(),
                                                          ADCProcessingStrategy(),
                                                          SWANProcessingStrategy(),
                                                          T1ProcessingStrategy(),
                                                          T2ProcessingStrategy()]

    def __init__(self, input_path: Union[str, pathlib.Path],*args, **kwargs):
        self._input_path = pathlib.Path(input_path)

    def post_process(self, study_path):
        self.del_json_file(study_path=study_path)
        for processing_strategy in self.processing_strategy_list:
            processing_strategy.process(study_path=study_path)

    def del_json_file(self, study_path: pathlib.Path):
        json_path_list = filter(lambda x: x.name.endswith('json'), study_path.iterdir())
        for json_path in json_path_list:
            json_path.unlink()

    def run(self, executor: Union[ThreadPoolExecutor, None] = None):
        is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_path.iterdir())))
        if is_dir_flag:
            study_path_list = list(self.input_path.iterdir())
            for study_path in study_path_list:
                self.post_process(study_path=study_path)
        else:
            self.post_process(study_path=self.input_path)
            # break

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value: str):
        self._input_path = pathlib.Path(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help="input rename nifti folder.\r\n")
    args = parser.parse_args()
    nifti_path = pathlib.Path(args.input)

    post_process_manager = PostProcessManager(input_path=nifti_path)
    post_process_manager.run(executor=None)
