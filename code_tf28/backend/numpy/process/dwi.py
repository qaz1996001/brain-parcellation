import numpy as np
from ..core import loss_distance,Process


class DWIProcess(Process):
    label_david_mapping_DWI = {
        'left_hemi': {
            112: 118,
            117: 118,
            119: 118,
            113: 119,
            120: 119,
            114: 120,
            121: 120,
            115: 121,
            122: 121,
            116: 123,
            123: 123,
            118: 122,
            124: 122,
            108: 108,
            109: 105,
            106: 105,
            110: 106,
            111: 107,
            125: 109,
            126: 110,
            127: 111,
            # 128: 111,
            101: 0,
            102: 1,
            104: 104,
            105: 104,
            # 103 use method
            # 129 use method
            # 130 use method
            131: 116,
            132: 117,
            # 16 use method
            14: 1,
            15: 1,
            24: 1,
        },
        'right_hemi': {
            212: 218,
            217: 218,
            219: 218,
            213: 219,
            220: 219,
            214: 220,
            221: 220,
            215: 221,
            222: 221,
            216: 223,
            223: 223,
            218: 222,
            224: 222,
            208: 208,
            209: 205,
            206: 205,
            210: 206,
            211: 207,
            225: 209,
            226: 210,
            227: 211,
            228: 211,
            201: 0,
            202: 1,
            204: 204,
            205: 204,
            # 203 use method
            # 229 use method
            # 230 use method
            231: 216,
            232: 217,
            # 16 use method
        },
    }

    ventral_DC = {
        'left_hemi': 103,
        'right_hemi': 203
    }

    # 2023-10-24 BRAIN_STEM不分左右改為 301
    ventral_DC_mapping = {
        'left_hemi': [108, 301],
        'right_hemi': [208, 301]
    }
    # ventral_DC_mapping = {
    #     'left_hemi': [108, 101],
    #     'right_hemi': [208, 201]
    # }
    ventral_DC_prerequisite = {
        'left_hemi': 108,
        'right_hemi': 208
    }
    PREREQUISITE_THRESHOLD = 0.5

    BRAIN_STEM = 16

    BRAIN_STEM_mapping = {
        'midbrain': 301,
        'pons': 302,
        'medulla': 303
    }

    BRAIN_STEM_hemi_shift = {
        'left_hemi': 50,
        'right_hemi': 150,
    }

    brain_stem_mapping_DWI = {
        # midbrain 、pons、 medulla
        'left_hemi': [101, 102, 103],
        'right_hemi': [201, 202, 203],
    }

    brain_stem_hemi = {
        'left_hemi': {
            103: 101,
            104: 101,
            105: 101,
            112: 101,
            115: 101,
        },
        'right_hemi': {
            203: 201,
            204: 201,
            205: 201,
            212: 201,
            215: 201,
        }
    }

    frontal_deep_white_matter = {
        'left_hemi': 129,
        'right_hemi': 229
    }
    frontal_deep_white_matter_parcellation = {
        'left_hemi': [112, 113, 114],
        'right_hemi': [212, 213, 214],
    }
    frontal_deep_white_matter_prerequisite = {
        'left_hemi': [102, 110],
        'right_hemi': [202, 210],
    }
    parietal_deep_white_matter = {
        'left_hemi': 130,
        'right_hemi': 230
    }
    parietal_deep_white_matter_parcellation = {
        'left_hemi': [112, 113, 115],
        'right_hemi': [212, 213, 215],
    }
    parietal_deep_white_matter_prerequisite = {
        'left_hemi': [102, 110],
        'right_hemi': [202, 210],
    }

    decimal_places = 8

    bullseye_DPWM = {
        'left_hemi': 128,
        'right_hemi': 228,
    }
    bullseye_DPWM_parcellation = {
        'left_hemi': {
            129: 129,
            130: 130,
            131: 131,
            132: 132,
        },
        'right_hemi': {
            229: 229,
            230: 230,
            231: 231,
            232: 232,
        },
    }

    @classmethod
    def david_label_to_DWI_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.label_david_mapping_DWI:
            # run every synseg label to freesurfer label
            for key in cls.label_david_mapping_DWI[hemi]:
                if cls.label_david_mapping_DWI[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.label_david_mapping_DWI[hemi][key]
        return new_label_array

    @classmethod
    def DPWM_parcellation(cls, label_array, target='frontal'):
        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        # 設定 target 、parcellation
        if 'frontal' == target:
            target_dict = cls.frontal_deep_white_matter
            parcellation_dict = cls.frontal_deep_white_matter_parcellation
            prerequisite_dict = cls.frontal_deep_white_matter_prerequisite
        elif 'parietal' == target:
            target_dict = cls.parietal_deep_white_matter
            parcellation_dict = cls.parietal_deep_white_matter_parcellation
            prerequisite_dict = cls.parietal_deep_white_matter_prerequisite
        else:
            raise ValueError('target is not definition')
        #
        for k in target_dict:
            # 取出目標索引
            index = np.argwhere(label_array == target_dict[k])
            # 取出條件上邊界索引
            temp_index = np.argwhere(label_array == prerequisite_dict[k][0])
            temp_index_z = np.unique(temp_index[:, 2])
            parcellation_higher = temp_index_z.max()
            # 取出條件下邊界索引
            temp_index = np.argwhere(label_array == prerequisite_dict[k][1])
            temp_index_z = np.unique(temp_index[:, 2])
            parcellation_lower = temp_index_z.max()

            higher_index = index[index[:, 2] > parcellation_higher]
            middle_index = index[(index[:, 2] <= parcellation_higher) & (index[:, 2] >= parcellation_lower)]
            lower_index = index[index[:, 2] < parcellation_lower]

            new_label_array[higher_index[:, 0],
            higher_index[:, 1],
            higher_index[:, 2]] = parcellation_dict[k][0]
            new_label_array[middle_index[:, 0],
            middle_index[:, 1],
            middle_index[:, 2]] = parcellation_dict[k][1]
            new_label_array[lower_index[:, 0],
            lower_index[:, 1],
            lower_index[:, 2]] = parcellation_dict[k][2]
        return new_label_array

    @classmethod
    def brain_stem_hemi_parcellation(cls, label_array):
        # # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(label_array)
        BRAIN_STEM_mapping_list = list(cls.BRAIN_STEM_mapping.values())
        index = np.argwhere(np.isin(label_array, BRAIN_STEM_mapping_list))
        z_aixs_intersect = np.unique(index[:, 2])
        for z in z_aixs_intersect:
            temp_index = index[index[:, 2] == z]
            x_center = int(temp_index[:, 0].mean())
            y_center = int(temp_index[:, 1].mean())
            left_index = temp_index[(temp_index[:, 0] <= x_center)]
            right_index = temp_index[(temp_index[:, 0] >= x_center)]
            new_label_array[left_index[:, 0], left_index[:, 1], left_index[:, 2]] = \
                label_array[left_index[:, 0], left_index[:, 1], left_index[:, 2]] + cls.BRAIN_STEM_hemi_shift[
                    'left_hemi']
            new_label_array[right_index[:, 0], right_index[:, 1], right_index[:, 2]] = \
                label_array[right_index[:, 0], right_index[:, 1], right_index[:, 2]] + cls.BRAIN_STEM_hemi_shift[
                    'right_hemi']
        return new_label_array

    @classmethod
    def brain_stem_parcellation(cls, label_array):

        new_label_array = label_array.copy()
        output_label_array = np.zeros_like(new_label_array)

        # to midbrain
        index = np.argwhere(new_label_array == cls.BRAIN_STEM)
        ventral_DC_list = list(cls.ventral_DC.values())
        ventral_DC_z = np.argwhere(np.isin(new_label_array, ventral_DC_list))
        ventral_DC_z_min = ventral_DC_z[:, 2].min()
        ventral_DC_z_max = ventral_DC_z[:, 2].max()
        middle_index = index[(index[:, 2] <= ventral_DC_z_max) & (index[:, 2] >= ventral_DC_z_min)]

        new_label_array[middle_index[:, 0], middle_index[:, 1], middle_index[:, 2]] = cls.BRAIN_STEM_mapping[
            'midbrain']

        #lower_index = index[index[:, 2] < ventral_DC_z_min]

        # to  pons、 medulla
        # brain-stem - midbrain
        index = np.argwhere(new_label_array == cls.BRAIN_STEM)
        values_x, count_x = np.unique(index[:, 0], return_counts=True)
        values_p = values_x[::-1]
        count_p = (count_x / count_x.max())[::-1]
        temp_count_p = count_p
        x_index = values_p[temp_count_p.argmax()]
        temp_index = index[index[:, 0] == x_index]

        ###### 2023 10 24 改
        z_axis = np.unique(temp_index[:, 2])
        z_norm_vector = []
        for i in z_axis:
            vector = temp_index[temp_index[:, 2] == i][:, 1]
            if vector.shape[0] > 1:
                z_norm_vector.append([i, np.ptp(vector), vector.max(), vector.min()])
        z_norm_vector = np.array(z_norm_vector, dtype=np.float64)
        z_norm_vector = np.round(z_norm_vector, 8)

        z_1_gradient = np.gradient(z_norm_vector[:, 1])
        z_2_gradient = np.gradient(z_1_gradient)
        z_index = np.where(z_2_gradient == z_2_gradient.max())[0][-1]
        # print(rf'z_index {np.where(z_2_gradient == z_2_gradient.max())[0]}')
        # print(rf'z_index {z_index}')
        higher_index = index[index[:, 2] > z_index]
        lower_index = index[index[:, 2] <= z_index]

        new_label_array[higher_index[:, 0], higher_index[:, 1], higher_index[:, 2]] = cls.BRAIN_STEM_mapping['pons']
        new_label_array[lower_index[:, 0], lower_index[:, 1], lower_index[:, 2]] = cls.BRAIN_STEM_mapping['medulla']

        midbrain_mask = new_label_array == cls.BRAIN_STEM_mapping['midbrain']
        pons_mask = new_label_array == cls.BRAIN_STEM_mapping['pons']
        medulla_mask = new_label_array == cls.BRAIN_STEM_mapping['medulla']
        output_label_array[midbrain_mask] = new_label_array[midbrain_mask]
        output_label_array[pons_mask] = new_label_array[pons_mask]
        output_label_array[medulla_mask] = new_label_array[medulla_mask]
        return output_label_array

    @classmethod
    def ventral_DC_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array)
        for k in cls.ventral_DC:
            index = np.argwhere(label_array == cls.ventral_DC[k])
            prerequisite_index = np.argwhere(label_array == cls.ventral_DC_prerequisite[k])

            index_z_unique = np.unique(index[:, 2])
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
    def DPWM_redivide(cls, label_array):
        # 輸出 nii.gz 的 array
        new_label_array = label_array.copy()
        # 建立 誤差(距離)矩陣

        for k in cls.bullseye_DPWM_parcellation:
            # 取出 bullseye_DPWM 部分
            index = np.argwhere(label_array == cls.bullseye_DPWM[k])
            hemi_parcellation_mapping_keys = list(cls.bullseye_DPWM_parcellation[k].keys())
            np_loss = np.zeros((index.shape[0], len(hemi_parcellation_mapping_keys)))
            np_loss[:, :] = 999999
            # Z 軸
            z_aixs_intersect = np.unique(index[:, 2])
            for i in range(len(hemi_parcellation_mapping_keys)):
                label_index = np.argwhere(label_array == hemi_parcellation_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
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
                new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                    cls.bullseye_DPWM_parcellation[k][hemi_parcellation_mapping_keys[ii]]
        return new_label_array

    @classmethod
    def left_right_translate(cls, slice):
        index_112 = np.argwhere(slice == 112)
        index_212 = np.argwhere(slice == 212)
        if index_112[:, 0].min() < index_212[:, 0].min():
            return False, slice
        else:
            return True, np.flip(slice, 0)

    @classmethod
    def inverse_left_right_translate(cls, flip, slice):
        if flip:
            return np.flip(slice, 0)
        else:
            return slice

    @classmethod
    def run(cls, label_array: np.ndarray)  -> np.ndarray:
        FLIP, label_array_translate = cls.left_right_translate(label_array)
        redivide_label_array = cls.DPWM_redivide(label_array_translate)
        dwi_array_translate = cls.david_label_to_DWI_label(redivide_label_array)
        brain_stem_array = cls.brain_stem_parcellation(redivide_label_array)
        ventral_DC_array = cls.ventral_DC_parcellation(redivide_label_array)
        array_frontal_array = cls.DPWM_parcellation(redivide_label_array, target='frontal')
        array_parietal_array = cls.DPWM_parcellation(redivide_label_array, target='parietal')
        brain_stem_mask = brain_stem_array != 0
        ventral_DC_mask = ventral_DC_array != 0
        array_frontal_mask = array_frontal_array != 0
        array_parietal_mask = array_parietal_array != 0
        dwi_array_translate[array_frontal_mask] = array_frontal_array[array_frontal_mask]
        dwi_array_translate[array_parietal_mask] = array_parietal_array[array_parietal_mask]
        dwi_array_translate[brain_stem_mask] = brain_stem_array[brain_stem_mask]
        dwi_array_translate[ventral_DC_mask] = ventral_DC_array[ventral_DC_mask]
        dwi_array = cls.inverse_left_right_translate(FLIP, dwi_array_translate)
        return dwi_array
