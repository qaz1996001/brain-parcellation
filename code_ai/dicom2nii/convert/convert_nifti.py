import argparse
import os
import pathlib
import re
import shutil

import dcm2niix
import subprocess
from concurrent.futures import ProcessPoolExecutor, Executor
from .config import MRSeriesRenameEnum,DSCSeriesRenameEnum,ASLSEQSeriesRenameEnum


class Dicm2NiixConverter:
    def __init__(self, input_path, output_path):
        """
        Initialize the Dcm2NiixConverter.

        Parameters
        ----------
        input_path : str or pathlib.Path
            Path to the input DICOM files.
        output_path : str or pathlib.Path
            Path to the output directory for NIfTI files.
        """
        self.input_path = pathlib.Path(input_path)
        self.output_path = pathlib.Path(output_path)
        self.exclude_set = {
            MRSeriesRenameEnum.MRAVR_BRAIN.value,
            MRSeriesRenameEnum.MRAVR_NECK.value,

            # DSCSeriesRenameEnum.DSC.value,
            # DSCSeriesRenameEnum.rCBV.value,
            # DSCSeriesRenameEnum.rCBF.value,
            # DSCSeriesRenameEnum.MTT.value,
            #
            # ASLSEQSeriesRenameEnum.ASLSEQ.value,
            # ASLSEQSeriesRenameEnum.ASLPROD.value,
            #
            # ASLSEQSeriesRenameEnum.ASLSEQATT.value,
            # ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR.value,
            #
            # ASLSEQSeriesRenameEnum.ASLSEQCBF.value,
            # ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR.value,
            #
            # ASLSEQSeriesRenameEnum.ASLSEQPW.value,
        }

    def run_cmd(self, output_series_path, series_path):
        """
        Run the dcm2niix command to convert DICOM to NIfTI.

        Parameters
        ----------
        output_series_path : pathlib.Path
            Path to the output series.
        series_path : pathlib.Path
            Path to the input DICOM series.

        Returns
        -------
        str
            The result of the conversion.
        """
        output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
        cmd_str = f'{dcm2niix.bin} -z y -f {output_series_path.name} -o {output_series_path.parent} {series_path}'

        completed_process = subprocess.run(cmd_str, capture_output=True)
        pattern = re.compile(r"DICOM as (.*)\s[(]", flags=re.MULTILINE)
        match_result = pattern.search(completed_process.stdout.decode())
        str_result = match_result.groups()[0]
        dcm2niix_output_path = pathlib.Path(f'{str_result}.nii.gz')

        if dcm2niix_output_path.name != output_series_path:
            try:
                # Rename the output file and corresponding JSON file
                dcm2niix_output_path.rename(output_series_file_path)
                dcm2niix_json_path = pathlib.Path(str(dcm2niix_output_path).replace('.nii.gz', '.json'))
                output_series_json_path = pathlib.Path(str(output_series_file_path).replace('.nii.gz', '.json'))
                dcm2niix_json_path.rename(output_series_json_path)
            except FileExistsError:
                print(rf'FileExistsError {series_path}')
        return str_result

    def copy_meta_dir(self, study_path: pathlib.Path):
        meta_path = study_path.joinpath('.meta')
        output_study_path = pathlib.Path(f'{str(study_path).replace(str(study_path.parent), str(self.output_path))}')
        if meta_path.exists():
            shutil.copytree(meta_path,output_study_path.joinpath('.meta'),dirs_exist_ok=True)

    def convert_dicom_to_nifti(self, executor: Executor = None):
        """
        Convert DICOM files to NIfTI format.

        Parameters
        ----------
        executor : concurrent.futures.Executor, optional
            Executor for parallel execution.
        """
        instances = list(self.input_path.rglob('*.dcm'))
        study_list = list(set(list(map(lambda x: x.parent.parent, instances))))
        future_list = []
        for study_path in study_list:
            series_list = list(filter(lambda series_path : series_path.name != '.meta' ,study_path.iterdir()))
            for series_path in series_list:
                if series_path.name in self.exclude_set:
                    continue
                output_series_path = pathlib.Path(
                    f'{str(series_path).replace(str(study_path.parent), str(self.output_path))}')
                output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
                if output_series_file_path.exists():
                    print(output_series_file_path)
                    continue
                else:
                    os.makedirs(output_series_path.parent, exist_ok=True)
                    if executor:
                        future = executor.submit(self.run_cmd, output_series_path, series_path)
                        future_list.append(future)
                    else:
                        self.run_cmd(output_series_path=output_series_path, series_path=series_path)
            self.copy_meta_dir(study_path=study_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help="input rename dicom folder.\r\n")
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help="output rename nifti folder.\r\n"
                             "Example: python convert_nifti.py -i input_path -o output_path")
    parser.add_argument('--work', dest='work', type=int, default=4,
                        help="Process cont max values is 8 .\r\n"
                             "Example: python convert_nifti.py -i input_path -o output_path --work 8")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    work = min(8, max(1, args.work))
    executor = ProcessPoolExecutor(max_workers=work)
    with executor:
        converter = Dicm2NiixConverter(input_path=args.input, output_path=args.output)
        converter.convert_dicom_to_nifti(executor=executor)
