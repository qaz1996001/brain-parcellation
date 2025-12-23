import argparse
import os
import pathlib
import re
from typing import Union

import numpy as np
from pydicom import dcmread, FileDataset, Dataset, DataElement
import nibabel as nib
import orjson

from concurrent.futures import ProcessPoolExecutor, Executor, ThreadPoolExecutor

from tqdm import tqdm

from config import MRSeriesRenameEnum


class Nifti2DicmConverter:
    study_folder_name_pattern = re.compile('^(\d{8})_(\d{8})_(MR|CT)_(.*)$', re.IGNORECASE)

    def __init__(self, input_path, output_path):
        """
        Initialize the Nifti2DicmConverter.

        Parameters
        ----------
        input_path : str or pathlib.Path
            Path to the input Nifti files.
        output_path : str or pathlib.Path
            Path to the output directory for Dicm folder.
        """
        self.input_path = pathlib.Path(input_path)
        self.output_path = pathlib.Path(output_path)
        self.exclude_set = {
            MRSeriesRenameEnum.MRAVR_BRAIN,
            MRSeriesRenameEnum.MRAVR_NECK,
            MRSeriesRenameEnum.DTI32D,
            MRSeriesRenameEnum.DTI64D,
            MRSeriesRenameEnum.RESTING,
            MRSeriesRenameEnum.RESTING2000,

            # Add more excluded values as needed
        }

    @staticmethod
    def get_study_folder_name(input_path: pathlib.Path):

        if input_path.name.endswith('nii.gz'):
            return input_path.parent.name
        else:
            if input_path.is_dir():
                match_result = input_path.match(input_path.name)
                if match_result:
                    return input_path.name
        return ValueError('input path is not study folder')

    def run(self, executor: Union[ThreadPoolExecutor, None] = None):
        """Run the Nifti file to DICOM process.

        Parameters:
        executor (Union[ThreadPoolExecutor, None]): Thread pool executor for parallel processing.
        """
        input_path = self.input_path
        is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_path.iterdir())))
        if is_dir_flag:
            study_path_list = list(self.input_path.iterdir())
            print(study_path_list)
            print(self.output_path)
            for study_path in study_path_list:
                nifti_file_path_list = list(filter(lambda x: x.name.endswith('nii.gz'), study_path.iterdir()))
                for i in range(len(nifti_file_path_list)):
                    meta_file_name = nifti_file_path_list[i].name.replace('nii.gz', 'jsonlines')
                    meta_folder_path = nifti_file_path_list[i].parent.joinpath('.meta')
                    meta_file_path = meta_folder_path.joinpath(meta_file_name)
                    if meta_file_path.stem in self.exclude_set:
                        continue
                    output_folder_path = self.output_path.joinpath(nifti_file_path_list[i].parent.name)
                    print(output_folder_path)
                    self.nii_to_dicom(nifti_file_path=nifti_file_path_list[i],
                                      meta_file_path=meta_file_path,
                                      output_folder_path=output_folder_path)
        else:
            study_path = self.input_path
            nifti_file_path_list = list(filter(lambda x: x.name.endswith('nii.gz'), study_path.iterdir()))
            # meta_file_path_list = list(
            # filter(lambda x: x.name.endswith('jsonlines'), study_path.joinpath('.meta').iterdir()))
            for i in range(len(nifti_file_path_list)):
                meta_file_name = nifti_file_path_list[i].name.replace('nii.gz', 'jsonlines')
                meta_folder_path = nifti_file_path_list[i].parent.joinpath('.meta')
                meta_file_path = meta_folder_path.joinpath(meta_file_name)
                if meta_file_path.stem in self.exclude_set:
                    continue
                output_folder_path = self.output_path.joinpath(nifti_file_path_list[i].parent.name)
                self.nii_to_dicom(nifti_file_path=nifti_file_path_list[i],
                                  meta_file_path=meta_file_path,
                                  output_folder_path=output_folder_path)

    def nii_to_dicom(self,
                     nifti_file_path: pathlib.Path,
                     meta_file_path: pathlib.Path,
                     output_folder_path: pathlib.Path):
        if 'COR' in nifti_file_path.name or 'SAG' in nifti_file_path.name:
            return
        nifti_obj = nib.load(str(nifti_file_path))
        nifti_array = nifti_obj.get_fdata().round(0).astype(np.int16)
        nifti_obj_axcodes = tuple(nib.aff2axcodes(nifti_obj.affine))  # ('R', 'A', 'S') ('L', 'A', 'S')
        pixdim = nifti_obj.header.get('pixdim')
        if pixdim[0] == -1:
            nifti_array = self.do_reorientation(nifti_array, nifti_obj_axcodes, ('S', 'P', 'L'))
        elif pixdim[0] == 1 and nifti_obj_axcodes == ('R', 'A', 'S'):
            nifti_array = self.do_reorientation(nifti_array, nifti_obj_axcodes, ('S', 'P', 'L'))
            # nifti_array       = do_reorientation(nifti_array, nifti_obj_axcodes, ('I', 'P', 'L'))

        with open(f'{meta_file_path}', encoding='utf8') as file:
            orjson_dict = orjson.loads(file.read())
        dicom_header_list = []
        exclude_dicom_tag_set = {'0018A001', '00081070', '00081110'}
        for key, value in orjson_dict.items():
            if len(value) == nifti_array.shape[0]:
                dicom_header_list.append(Dataset.from_json(value[0]))
                for i in range(1, len(value)):
                    temp_dicom_header = Dataset.from_json(value[0])
                    temp_orjson_dict = value[i]
                    for j_key, j_value in temp_orjson_dict.items():
                        idx_1 = int(j_key[:4], base=16)
                        idx_2 = int(j_key[4:], base=16)
                        if j_key in exclude_dicom_tag_set:
                            continue
                        else:
                            # print(j_key,idx_1,idx_2,j_value)
                            temp_dicom_header[idx_1, idx_2].value = j_value
                    dicom_header_list.append(temp_dicom_header)
                # break
                break
        if dicom_header_list[0].get((0x0020, 0x0032)):
            new_list = sorted(dicom_header_list, key=lambda x: x[0x0020, 0x0032].value[-1])
            # new_list = sorted(value, key=lambda x: x['00200032']['Value'][-1])
        else:
            new_list = sorted(dicom_header_list, key=lambda x: x[0x0020, 0x0013].value)
        os.makedirs(f'{output_folder_path}/{meta_file_path.stem}', exist_ok=True)
        for i in range(len(new_list)):
            ds = new_list[i]
            arr = nifti_array[i]
            ds.PixelData = arr.tobytes()
            ds.is_little_endian = True
            ds.is_implicit_VR = True
            ds.save_as(f"{output_folder_path}/{meta_file_path.stem}/{ds.SOPInstanceUID}.dcm")

    @classmethod
    def compute_orientation(cls,init_axcodes, final_axcodes):
        """
        A thin wrapper around ``nib.orientations.ornt_transform``

        :param init_axcodes: Initial orientation codes
        :param final_axcodes: Target orientation codes
        :return: orientations array, start_ornt, end_ornt
        """
        ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
        ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

        ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

        return ornt_transf, ornt_init, ornt_fin

    @classmethod
    def do_reorientation(cls,data_array, init_axcodes, final_axcodes):
        """
        source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
        Performs the reorientation (changing order of axes)

        :param data_array: 3D Array to reorient
        :param init_axcodes: Initial orientation
        :param final_axcodes: Target orientation
        :return data_reoriented: New data array in its reoriented form
        """
        ornt_transf, ornt_init, ornt_fin = cls.compute_orientation(init_axcodes, final_axcodes)
        if np.array_equal(ornt_init, ornt_fin):
            return data_array

        return nib.orientations.apply_orientation(data_array, ornt_transf)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help="input rename nifti folder.\r\n or input rename nifti file")
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help="output rename dicom folder.\r\n"
                             "Example: python convert_nifti.py -i input_path -o output_path")
    parser.add_argument('--work', dest='work', type=int, default=4,
                        help="Process cont max values is 8 .\r\n"
                             "Example: python convert_nifti.py -i input_path -o output_path --work 8")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    work = min(2, max(1, args.work))
    executor = ProcessPoolExecutor(max_workers=work)
    with executor:
        converter = Nifti2DicmConverter(input_path=args.input, output_path=args.output)
        converter.run(executor=executor)
