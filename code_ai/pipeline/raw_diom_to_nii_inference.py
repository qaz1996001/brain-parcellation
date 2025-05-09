#!/usr/bin/env python3
"""
DICOM處理工具 - 提供DICOM到NIfTI的轉換、重命名和推理功能
"""

import os
import argparse
import pathlib
from typing import List, Optional, Callable, Any, Union, Dict


class DicomProcessor:
    """DICOM處理工具類別，封裝DICOM相關處理功能"""

    def __init__(self):
        from code_ai.task.task_dicom2nii import dicom_to_nii, dicom_rename
        from code_ai.task.task_pipeline import task_pipeline_inference
        from code_ai.task.schema.intput_params import Dicom2NiiParams

        self.dicom_to_nii:Callable = dicom_to_nii
        self.dicom_rename:Callable = dicom_rename
        self.task_pipeline_inference = task_pipeline_inference
        self.Dicom2NiiParams = Dicom2NiiParams
        self.result_list = []

    @staticmethod
    def get_subdirectories(directory_path: Optional[pathlib.Path]) -> List[pathlib.Path]:
        """獲取指定路徑下的所有子目錄

        Args:
            directory_path: 目標目錄路徑

        Returns:
            目錄列表
        """
        if not directory_path or not directory_path.exists():
            return []
        directories = sorted(directory_path.iterdir())
        return list(filter(lambda x: x.is_dir(), directories))

    def process_directories(
            self,
            input_dirs: Union[List[pathlib.Path], pathlib.Path, None],
            task_func: Any,
            params_generator: Callable[[pathlib.Path], Any]
    ) -> List[Any]:
        """處理目錄並執行任務

        Args:
            input_dirs: 輸入目錄列表或單個目錄
            task_func: 要執行的任務函數
            params_generator: 產生任務參數的函數

        Returns:
            任務結果列表
        """
        results = []

        if not input_dirs:
            return results

        # 處理單個目錄的情況
        if isinstance(input_dirs, pathlib.Path):
            params = params_generator(input_dirs)
            results.append(task_func.push(params.get_str_dict()))
            return results

        # 處理目錄列表
        for dir_path in input_dirs:
            params = params_generator(dir_path)
            results.append(task_func.push(params.get_str_dict()))

        return results

    def dicom_to_nifti_conversion(
            self,
            input_dicom_path: Optional[pathlib.Path],
            output_dicom_path: pathlib.Path,
            output_nifti_path: pathlib.Path
    ) -> List[Any]:
        """執行DICOM到NIfTI的轉換

        Args:
            input_dicom_path: 輸入DICOM目錄
            output_dicom_path: 輸出DICOM目錄
            output_nifti_path: 輸出NIfTI目錄

        Returns:
            任務結果列表
        """
        input_dirs = self.get_subdirectories(input_dicom_path)
        if not input_dirs and input_dicom_path:
            input_dirs = input_dicom_path

        def generate_params(dir_path):
            return self.Dicom2NiiParams(
                sub_dir=dir_path,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path
            )

        return self.process_directories(input_dirs, self.dicom_to_nii, generate_params)

    def dicom_rename_only(
            self,
            input_dicom_path: pathlib.Path,
            output_dicom_path: pathlib.Path
    ) -> List[Any]:
        """執行DICOM重命名

        Args:
            input_dicom_path: 輸入DICOM目錄
            output_dicom_path: 輸出DICOM目錄

        Returns:
            任務結果列表
        """
        input_dirs = self.get_subdirectories(input_dicom_path)

        def generate_params(dir_path):
            return self.Dicom2NiiParams(
                sub_dir=dir_path,
                output_dicom_path=output_dicom_path,
                output_nifti_path=None
            )

        return self.process_directories(input_dirs, self.dicom_rename, generate_params)

    def renamed_dicom_to_nifti(
            self,
            output_dicom_path: pathlib.Path,
            output_nifti_path: pathlib.Path
    ) -> List[Any]:
        """將已重命名的DICOM轉換為NIfTI

        Args:
            output_dicom_path: DICOM目錄
            output_nifti_path: 輸出NIfTI目錄

        Returns:
            任務結果列表
        """
        input_dirs = self.get_subdirectories(output_dicom_path)

        def generate_params(dir_path):
            return self.Dicom2NiiParams(
                sub_dir=None,
                output_dicom_path=dir_path,
                output_nifti_path=output_nifti_path
            )

        return self.process_directories(input_dirs, self.dicom_to_nii, generate_params)

    def run_pipeline_inference(
            self,
            result_paths: List[str],
            output_dicom_path: pathlib.Path,
            output_nifti_path: pathlib.Path
    ) -> None:
        """執行管道推理

        Args:
            result_paths: 結果路徑列表
            output_dicom_path: DICOM輸出目錄
            output_nifti_path: NIfTI輸出目錄
        """
        output_nifti_path_list = [output_nifti_path.joinpath(os.path.basename(x)) for x in result_paths]

        for nifti_study_path in output_nifti_path_list:
            dicom_study_path = output_dicom_path.joinpath(nifti_study_path.name)
            self.task_pipeline_inference.push({
                'nifti_study_path': str(nifti_study_path),
                'dicom_study_path': str(dicom_study_path),
            })

    def process_tasks(self, args: argparse.Namespace) -> None:
        """根據命令行參數處理任務

        Args:
            args: 解析後的命令行參數
        """
        input_dicom_path = args.input_dicom and pathlib.Path(args.input_dicom)
        output_dicom_path = args.output_dicom and pathlib.Path(args.output_dicom)
        output_nifti_path = args.output_nifti and pathlib.Path(args.output_nifti)

        # 根據不同的參數組合執行不同的任務
        if all((args.input_dicom, args.output_dicom, args.output_nifti)):
            self.result_list = self.dicom_to_nifti_conversion(
                input_dicom_path, output_dicom_path, output_nifti_path
            )

        elif all((args.input_dicom, args.output_dicom)):
            self.result_list = self.dicom_rename_only(
                input_dicom_path, output_dicom_path
            )

        elif all((args.output_dicom, args.output_nifti)):
            self.result_list = self.renamed_dicom_to_nifti(
                output_dicom_path, output_nifti_path
            )

        else:
            raise ValueError(
                f'Missing required arguments. input_dicom: {args.input_dicom}, '
                f'output_dicom: {args.output_dicom}, output_nifti: {args.output_nifti}'
            )

        # 處理異步任務結果
        for async_result in self.result_list:
            async_result.set_timeout(3600)

        result_paths = [async_result.result for async_result in self.result_list]
        print('result_paths', result_paths)

        # 執行後續的管道推理
        if result_paths and output_nifti_path and output_dicom_path:
            self.run_pipeline_inference(result_paths, output_dicom_path, output_nifti_path)


def main():
    """主函數，解析命令行參數並執行處理"""
    parser = argparse.ArgumentParser(description='DICOM處理工具')
    parser.add_argument('--input_dicom', type=str, help='輸入DICOM目錄，例如：/mnt/e/raw_dicom')
    parser.add_argument('--output_dicom', type=str, help='輸出DICOM目錄，例如：/mnt/e/rename_dicom_0407')
    parser.add_argument('--output_nifti', type=str, help='輸出NIfTI目錄，例如：/mnt/e/rename_nifti_0407')

    args = parser.parse_args()
    processor = DicomProcessor()
    try:
        processor.process_tasks(args)
    except Exception as e:
        print(f"處理失敗: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

# import argparse
# import os.path
# import pathlib
#
# if __name__ == '__main__':
#     from code_ai.task.task_dicom2nii import dicom_to_nii, dicom_rename
#     from code_ai.task.task_pipeline import task_pipeline_inference
#     from code_ai.task.schema.intput_params import Dicom2NiiParams
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dicom', type=str,
#                         help='input_dicom  /mnt/e/raw_dicom ')
#     parser.add_argument('--output_dicom', type=str,
#                         help='output_dicom /mnt/e/rename_dicom_0407 ')
#     parser.add_argument('--output_nifti', type=str,
#                         help='output_nifti /mnt/e/rename_nifti_0407')
#
#
#     args = parser.parse_args()
#     result_list = []
#     if all((args.input_dicom, args.output_dicom, args.output_nifti)):
#         input_dicom  = pathlib.Path(args.input_dicom)
#         output_dicom_path = pathlib.Path(args.output_dicom)
#         output_nifti_path = pathlib.Path(args.output_nifti)
#
#         input_dicom_list = sorted(input_dicom.iterdir())
#         input_dicom_list = list(filter(lambda x: x.is_dir(),input_dicom_list))
#         if len(input_dicom_list) == 0:
#             task_params = Dicom2NiiParams(
#                 sub_dir=input_dicom,
#                 output_dicom_path=output_dicom_path,
#                 output_nifti_path=output_nifti_path, )
#             task = dicom_to_nii.push(task_params.get_str_dict())
#             result_list.append(task)
#         else:
#             for input_dicom_path in input_dicom_list:
#                 task_params = Dicom2NiiParams(
#                     sub_dir=input_dicom_path,
#                     output_dicom_path=output_dicom_path,
#                     output_nifti_path=output_nifti_path,)
#                 task = dicom_to_nii.push(task_params.get_str_dict())
#                 result_list.append(task)
#     elif all((args.input_dicom, args.output_dicom)):
#         input_dicom = pathlib.Path(args.input_dicom)
#         output_dicom_path = pathlib.Path(args.output_dicom)
#
#         input_dicom_list = sorted(input_dicom.iterdir())
#         input_dicom_list = list(filter(lambda x: x.is_dir(), input_dicom_list))
#
#         for input_dicom_path in input_dicom_list:
#             task_params = Dicom2NiiParams(
#                 sub_dir=input_dicom_path,
#                 output_dicom_path=output_dicom_path,
#                 output_nifti_path=None, )
#             result_list.append(dicom_rename.push(task_params.get_str_dict()))
#     elif all((args.output_dicom, args.output_nifti)):
#         output_dicom_path = pathlib.Path(args.output_dicom)
#         output_nifti_path = pathlib.Path(args.output_nifti)
#
#         input_dicom_list = sorted(output_dicom_path.iterdir())
#         input_dicom_list = list(filter(lambda x: x.is_dir(), input_dicom_list))
#
#         for input_dicom_path in input_dicom_list:
#             task_params = Dicom2NiiParams(
#                 sub_dir=None,
#                 output_dicom_path=input_dicom_path,
#                 output_nifti_path=output_nifti_path, )
#             result_list.append(dicom_to_nii.push(task_params.get_str_dict()))
#     else:
#         raise ValueError(f'input_dicom {args.input_dicom} or {args.output_dicom}')
#     for async_result in result_list:
#         async_result.set_timeout(3600)
#
#     result_list = [async_result.result for async_result in result_list]
#     print('result_list',result_list)
#     if len(result_list) > 0:
#         output_nifti_path = pathlib.Path(args.output_nifti)
#         dicom_path = pathlib.Path(args.output_dicom)
#         output_nifti_path_list = list(map(lambda x:output_nifti_path.joinpath(os.path.basename(x)),
#                                           result_list))
#         for nifti_study_path in output_nifti_path_list:
#             dicom_study_path = dicom_path.joinpath(nifti_study_path.name)
#             task_pipeline_result = task_pipeline_inference.push({'nifti_study_path':str(nifti_study_path),
#                                                                  'dicom_study_path':str(dicom_study_path),
#                                                                  })
