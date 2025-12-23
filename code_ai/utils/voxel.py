import os
import time
from typing import Optional,Union
import numpy as np
import pandas as pd
import nibabel as nib
import pathlib
from .inference import InferenceEnum


class VolumeProcessor:
    label_index_mapping_name_dict = {
        InferenceEnum.Area:{
            0: "Background",
            14: "3rd ventricle",
            15: "4th ventricle",
            16: "brain-stem",
            24: "csf",
            101: "left cerebral cortex",
            102: "left lateral ventricle",
            103: "left ventral DC",
            104: "left cerebellum white matter",
            105: "left cerebellum cortex",
            106: "left accumbens area",
            108: "left thalamus",
            109: "left caudate",
            110: "left putamen",
            111: "left pallidum",
            112: "left frontal",
            113: "left parietal",
            114: "left occipital",
            115: "left temporal",
            116: "left cingulate",
            118: "left insula",
            119: "left frontal white matter",
            120: "left parietal white matter",
            121: "left occipital white matter",
            122: "left temporal white matter",
            123: "left cingulate white matter",
            124: "left insula white matter",
            125: "left External Capsule",
            126: "left Internal Capsule",
            127: "left CorpusCallosum",
            128: "left DPWM(Bullseye)",
            129: "left frontal deep white matter",
            130: "left parietal deep white matter",
            131: "left occipital deep white matter",
            132: "left temporal deep white matter",
            201: "right cerebral cortex",
            202: "right lateral ventricle",
            203: "right ventral DC",
            204: "right cerebellum white matter",
            205: "right cerebellum cortex",
            206: "right accumbens area",
            208: "right thalamus",
            209: "right caudate",
            210: "right putamen",
            211: "right pallidum",
            212: "right frontal",
            213: "right parietal",
            214: "right occipital",
            215: "right temporal",
            216: "right cingulate",
            218: "right insula",
            219: "right frontal white matter",
            220: "right parietal white matter",
            221: "right occipital white matter",
            222: "right temporal white matter",
            223: "right cingulate white matter",
            224: "right insula white matter",
            225: "right External Capsule",
            226: "right Internal Capsule",
            227: "right CorpusCallosum",
            228: "right DPWM(Bullseye)",
            229: "right frontal deep white matter",
            230: "right parietal deep white matter",
            231: "right occipital deep white matter",
            232: "right temporal deep white matter",
        },
        InferenceEnum.WMH_PVS: {
            0: "Background",
            301: "brainstem",
            101: "left brainstem",
            102: "left external capsule",
            103: "left internal capsule",
            104: "left corpus callosum",
            105: "left periventricular WM",
            106: "left frontal deep WM",
            107: "left parietal deep WM",
            108: "left occipital deep WM",
            109: "left temporal deep WM",
            110: "left frontal subcortical",
            111: "left parietal subcortical",
            112: "left occipital subcortical",
            113: "left temporal subcortical",
            201: "right brainstem",
            202: "right external capsule",
            203: "right internal capsule",
            204: "right corpus callosum",
            205: "right periventricular WM",
            206: "right frontal deep WM",
            207: "right parietal deep WM",
            208: "right occipital deep WM",
            209: "right temporal deep WM",
            210: "right frontal subcortical",
            211: "right parietal subcortical",
            212: "right occipital subcortical",
            213: "right temporal subcortical",
        },
        InferenceEnum.DWI: {
            0: "Background",
            1: "CSF",
            301: "midbrain",
            302: "pons",
            303: "medulla",
            101: "left midbrain",
            102: "left pons",
            103: "left medulla",
            104: "left cerebellum",
            105: "left caudate",
            106: "left putamen",
            107: "left globus pallidus",
            108: "left thalamus",
            109: "left external capsule",
            110: "left internal capsule",
            111: "left corpus callosum",
            112: "left centrum semioval",
            113: "left corona radiata",
            114: "left frontal WM",
            115: "left parietal WM",
            116: "left occipital WM",
            117: "left temporal WM",
            118: "left frontal",
            119: "left parietal",
            120: "left occipital",
            121: "left temporal",
            122: "left insula",
            123: "left cingulate",
            201: "right midbrain",
            202: "right pons",
            203: "right medulla",
            204: "right cerebellum",
            205: "right caudate",
            206: "right putamen",
            207: "right globus pallidus",
            208: "right thalamus",
            209: "right external capsule",
            210: "right internal capsule",
            211: "right corpus callosum",
            212: "right centrum semioval",
            213: "right corona radiata",
            214: "right frontal WM",
            215: "right parietal WM",
            216: "right occipital WM",
            217: "right temporal WM",
            218: "right frontal",
            219: "right parietal",
            220: "right occipital",
            221: "right temporal",
            222: "right insula",
            223: "right cingulate",
        },
        InferenceEnum.CMB: {
            0: "Background",
            1: "CSF",
            301: "brainstem",
            101: "left brainstem",
            102: "left cerebellum",
            103: "left basal ganglion",
            104: "left thalamus",
            105: "left internal capsule",
            106: "left external capsule",
            107: "left corpus callosum",
            108: "left DPWM",
            109: "left frontal",
            110: "left parietal",
            111: "left occipital",
            112: "left temporal",
            113: "left insular",
            114: "left cingulate",
            201: "right brainstem",
            202: "right cerebellum",
            203: "right basal ganglion",
            204: "right thalamus",
            205: "right internal capsule",
            206: "right external capsule",
            207: "right corpus callosum",
            208: "right DPWM",
            209: "right frontal",
            210: "right parietal",
            211: "right occipital",
            212: "right temporal",
            213: "right insular",
            214: "right cingulate",
        },
        InferenceEnum.Infarct: {},
        InferenceEnum.WMH: {},
        InferenceEnum.Aneurysm: {},
    }


    @classmethod
    def process(cls,maks_file_path:pathlib.Path,mode:InferenceEnum) -> Optional[Union[str,pathlib.Path]]:
        label_index_mapping = cls.label_index_mapping_name_dict.get(mode,None)
        if label_index_mapping is None:
            return
        return cls.calculate_volume(maks_file_path,label_index_mapping)

    @classmethod
    def calculate_volume(cls, mask_file_path:pathlib.Path, label_index_mapping:dict, ) -> Optional[Union[pathlib.Path,str]]:
        mask_nii = nib.load(mask_file_path)
        mask_array: np.ndarray = np.array(mask_nii.dataobj)
        pixdim = mask_nii.header['pixdim']
        spacing = pixdim[3]
        pixel_size = pixdim[1:]
        ml_size = (pixdim[1] * pixdim[2] * pixdim[3]) / 1000
        unique_values, values_count = np.unique(mask_array, return_counts=True)
        mask_size = values_count * ml_size
        df_mask_size = pd.DataFrame(mask_size, index=unique_values).T
        return


if __name__ == '__main__':

    # input_path = pathlib.Path(r'D:\00_Chen\Task06_\Study_SVD_20220916_20230718_out_8nii_T2')
    #dir_list = ['Study_SVD_20220916_20230718_out_8nii_T1','Study_SVD_20220916_20230718_out_8nii_T2','Study_SVD_20190726_20220105_out_7nii_T1','Study_SVD_20190726_20220105_out_7nii_T2']
    dir_list = ['Study_SVD_20220113_20220902_rawdata_T1']
    # dir_list = ['Study_SVD_20220113_20220902_rawdata_T2']
    for dir_path in dir_list:
        input_path = pathlib.Path(rf'D:\00_Chen\Task06_SVD\{dir_path}')

        # input_path = pathlib.Path(r'D:\00_Chen\Task06_\Study_SVD_20190726_20220105_out_7nii_T1')
        synthseg_list = list(input_path.glob(rf'*synthseg.nii.gz'))
        mask_size_list = []
        for file_path in synthseg_list:
            start_time = time.time()
            base_name = os.path.basename(file_path)
            synthseg_path = file_path
            mask_nii = nib.load(synthseg_path)
            mask_array: np.ndarray = mask_nii.get_fdata()
            mask_array = mask_array.astype(int)
            pixdim = mask_nii.header['pixdim']
            spacing = pixdim[3]
            pixel_size = pixdim[1:]
            ml_size = (pixdim[1] * pixdim[2] * pixdim[3]) / 1000
            # 算出換算出的ml大小
            # cluster_ml = cluster_size * ml_size
            unique_values, values_count = np.unique(mask_array, return_counts=True)
            mask_size = values_count * ml_size
            df_mask_size = pd.DataFrame(mask_size,index=unique_values).T
            mask_size_list.append(df_mask_size)

            # 結束時間
            end_time = time.time()
            # 計算運行時間
            execution_time = end_time - start_time
            # 輸出運行時間
            print(f"{base_name} 程式運行時間：", execution_time, "秒")
        index_list = list(map(lambda x: str(x.name.replace('.nii.gz','')),synthseg_list))
        df = pd.concat(mask_size_list)
        df.index = index_list
        df.to_csv(input_path.parent.joinpath(f'{input_path.name}_mask_size.csv'))
