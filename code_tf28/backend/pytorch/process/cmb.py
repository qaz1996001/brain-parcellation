import numpy as np
from ..core import loss_distance, Process


class CMBProcess(Process):
    # brain stem(16) 分左右 (104，204)
    # left ventral DC (103) 分 (left thalamus 104)、(to left brainstem 101)
    # ventral DC 分 thalamus and brainstem
    label_david_mapping_CMB = {
        'left_hemi': {
            112: 109,
            117: 109,
            119: 109,
            113: 110,
            120: 110,
            114: 111,
            121: 111,
            115: 112,
            122: 112,
            116: 114,
            123: 114,
            118: 113,
            124: 113,
            108: 104,
            109: 103,
            106: 103,
            110: 103,
            111: 103,
            125: 106,
            126: 105,
            127: 107,
            128: 108,
            102: 1,
            104: 102,
            105: 102,
            129: 108,
            130: 108,
            131: 108,
            132: 108,
            14: 1,
            15: 1,
            24: 1,
        },
        'right_hemi': {
            212: 209,
            217: 209,
            219: 209,
            213: 210,
            220: 210,
            214: 211,
            221: 211,
            215: 212,
            222: 212,
            216: 214,
            223: 214,
            218: 213,
            224: 213,
            208: 204,
            209: 203,
            206: 203,
            210: 203,
            211: 203,
            225: 206,
            226: 205,
            227: 207,
            228: 208,
            202: 1,
            204: 202,
            205: 202,
            229: 208,
            230: 208,
            231: 208,
            232: 208,
        }
    }
    ventral_DC = {
        'left_hemi': 103,
        'right_hemi': 203
    }
    # 2023-10-24 BRAIN_STEM不分左右改為 301
    ventral_DC_mapping = {
        'left_hemi': [104, 301],
        'right_hemi': [204, 301]
    }
    # ventral_DC_mapping = {
    #     'left_hemi': [104, 101],
    #     'right_hemi': [204, 201]
    # }
    ventral_DC_prerequisite = {
        'left_hemi': 108,
        'right_hemi': 208
    }
    PREREQUISITE_THRESHOLD = 0.5

    BRAIN_STEM = 16

    # 2023-10-24 BRAIN_STEM不分左右改為 301
    brain_stem_mapping_CMB = {
        'left_hemi': 301,
        'right_hemi': 301
    }
    # brain_stem_mapping_CMB = {
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
    def david_label_to_CMB_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.label_david_mapping_CMB:
            # run every synseg label to freesurfer label
            for key in cls.label_david_mapping_CMB[hemi]:
                if cls.label_david_mapping_CMB[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.label_david_mapping_CMB[hemi][key]
        return new_label_array

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
            new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.brain_stem_mapping_CMB[
                hemi_list[ii]]
        return new_label_array

    @classmethod
    def ventral_DC_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array)
        for k in cls.ventral_DC:
            index = np.argwhere(label_array == cls.ventral_DC[k])
            prerequisite_index = np.argwhere(label_array == cls.ventral_DC_prerequisite[k])

            index_z_unique = np.unique(index[:, 2])
            # prerequisite_index_z_unique = np.unique(prerequisite_index[:, 2])
            # z_aixs_intersect = np.intersect1d(index_z_unique, prerequisite_index_z_unique)
            z_aixs_intersect = index_z_unique
            # Z 軸切片
            for j in z_aixs_intersect:
                index_sub = index[index[:, 2] == j]
                prerequisite_index_sub = prerequisite_index[prerequisite_index[:, 2] == j]
                if prerequisite_index_sub.shape[0] > 0:
                    if prerequisite_index_sub.shape[0] / index_sub.shape[0] >= cls.PREREQUISITE_THRESHOLD:
                        new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][
                            0]
                    else:
                        new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][
                            1]
                else:
                    new_label_array[index_sub[:, 0], index_sub[:, 1], index_sub[:, 2]] = cls.ventral_DC_mapping[k][1]
        return new_label_array

    @classmethod
    def run(cls, label_array:np.ndarray) -> np.ndarray:
        brain_stem_array = cls.brain_stem_parcellation(label_array)
        ventral_DC_array = cls.ventral_DC_parcellation(label_array)
        cmd_array = cls.david_label_to_CMB_label(label_array)
        brain_stem_mask = brain_stem_array != 0
        ventral_DC_mask = ventral_DC_array != 0

        cmd_array[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        cmd_array[ventral_DC_mask] = ventral_DC_array[ventral_DC_mask]
        return cmd_array

