import numpy as np
from ..core import loss_distance,Process


class WMHProcess(Process):
    label_david_mapping_WMH = {
        'left_hemi': {
            112: 0,
            117: 0,
            119: 110,
            113: 0,
            120: 111,
            114: 0,
            121: 112,
            115: 0,
            122: 113,
            116: 0,
            123: 0,
            118: 0,
            124: 0,
            108: 0,
            109: 0,
            106: 0,
            110: 0,
            111: 0,
            125: 102,
            126: 103,
            127: 104,
            128: 105,
            101: 0,
            102: 0,
            104: 0,
            105: 0,
            103: 0,
            129: 106,
            130: 107,
            131: 108,
            132: 109,
            # 16: 101,
            14: 0,
            15: 0,
            24: 0,
        },
        'right_hemi': {
            212: 0,
            217: 0,
            219: 210,
            213: 0,
            220: 211,
            214: 0,
            221: 212,
            215: 0,
            222: 213,
            216: 0,
            223: 0,
            218: 0,
            224: 0,
            208: 0,
            209: 0,
            206: 0,
            210: 0,
            211: 0,
            225: 202,
            226: 203,
            227: 204,
            228: 205,
            201: 0,
            202: 0,
            204: 0,
            205: 0,
            203: 0,
            229: 206,
            230: 207,
            231: 208,
            232: 209,
        },
    }

    BRAIN_STEM = 16
    # 2023-10-25 BRAIN_STEM不分左右改為 301
    brain_stem_mapping = {
        'left_hemi': 301,
        'right_hemi': 301
    }
    # brain_stem_mapping = {
    #     'left_hemi': 101,
    #     'right_hemi': 201
    # }
    brain_stem_hemi_parcellation = {
        'left_hemi': {
            104: 101,
            105: 101,
        },
        'right_hemi': {
            204: 201,
            205: 201,
        }
    }
    decimal_places = 8

    @classmethod
    def brain_stem_parcellation(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        # 取出 brain stem 部分
        index = np.argwhere(label_array == cls.BRAIN_STEM)
        # 建立 誤差(距離)矩陣
        hemi_list = list(cls.brain_stem_hemi_parcellation.keys())
        np_loss = np.zeros((index.shape[0], len(hemi_list)))
        np_loss[:, :] = 999999
        for k in cls.brain_stem_hemi_parcellation:
            if k == 'left_hemi':
                i = 0
            else:
                i = 1
            hemi_parcellation_mapping_keys = list(cls.brain_stem_hemi_parcellation[k].keys())
            # brain stem  的 Z 軸
            z_aixs_intersect = np.unique(index[:, 2])

            label_index = np.argwhere(np.isin(label_array, hemi_parcellation_mapping_keys))
            loss_list = []
            index_sub_arg_list = []
            # Z 軸切片
            for j in z_aixs_intersect:
                index_sub = index[index[:, 2] == j]
                index_sub_arg = np.argwhere(index[:, 2] == j)
                label_index_sub = label_index[label_index[:, 2] == j]
                if (index_sub.shape[0] > 0) and (label_index_sub.shape[0] > 0):
                    loss_min = loss_distance(index_sub, label_index_sub, cls.decimal_places)
                    loss_list.append(loss_min.numpy())
                    index_sub_arg_list.append(index_sub_arg)
                else:
                    continue
            if len(index_sub_arg_list) > 0:
                index_sub_arg = np.concatenate(index_sub_arg_list)
                np_loss[index_sub_arg, i] = np.concatenate(loss_list).reshape(-1, 1)
            else:
                continue
        new_label = np_loss.argmin(axis=1)
        # 指定分類
        for ii in np.unique(new_label):
            select_index = index[np.argwhere(new_label == ii)].squeeze()
            new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.brain_stem_mapping[
                hemi_list[ii]]
        return new_label_array

    @classmethod
    def david_label_to_WMH_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.label_david_mapping_WMH:
            # run every synseg label to freesurfer label
            for key in cls.label_david_mapping_WMH[hemi]:
                if cls.label_david_mapping_WMH[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.label_david_mapping_WMH[hemi][key]
        return new_label_array

    @classmethod
    def run(cls, label_array:np.ndarray) -> np.ndarray:
        brain_stem_array = cls.brain_stem_parcellation(label_array)
        wmh_array = cls.david_label_to_WMH_label(label_array)
        brain_stem_mask = brain_stem_array != 0
        wmh_array[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        return wmh_array