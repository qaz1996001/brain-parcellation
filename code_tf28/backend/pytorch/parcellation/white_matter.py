import numpy as np



class WhiteMatterParcellation:
    synseg_label_freesurfer_GM_mapping = {
        'left_hemi': {
            1003: 1001,
            1032: 1001,
            1012: 1001,
            1014: 1001,
            1018: 1001,
            1019: 1001,
            1020: 1001,
            1024: 1001,
            1027: 1001,
            1028: 1001,
            1017: 1001,
            1008: 1006,
            1022: 1006,
            1025: 1006,
            1029: 1006,
            1031: 1006,
            1005: 1004,
            1011: 1004,
            1013: 1004,
            1021: 1004,
            1001: 1005,
            1006: 1005,
            1007: 1005,
            1009: 1005,
            1015: 1005,
            1016: 1005,
            1030: 1005,
            1033: 1005,
            1034: 1005,
            1002: 1003,
            1010: 1003,
            1023: 1003,
            1026: 1003,
            1035: 1007,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
            17: 1005,
            18: None, },
        'right_hemi': {
            2003: 2001,
            2012: 2001,
            2014: 2001,
            2018: 2001,
            2019: 2001,
            2020: 2001,
            2024: 2001,
            2027: 2001,
            2028: 2001,
            2032: 2001,
            2017: 2001,
            2008: 2006,
            2022: 2006,
            2025: 2006,
            2029: 2006,
            2031: 2006,
            2005: 2004,
            2011: 2004,
            2013: 2004,
            2021: 2004,
            2001: 2005,
            2006: 2005,
            2007: 2005,
            2009: 2005,
            2015: 2005,
            2016: 2005,
            2030: 2005,
            2033: 2005,
            2034: 2005,
            2002: 2003,
            2010: 2003,
            2023: 2003,
            2026: 2003,
            2035: 2007,
            49: 49,
            50: 50,
            51: 51,
            52: 52,
            53: 2005,
            54: None
        },
    }
    white_matter_mapping = {
        'left_hemi': {1001: 3001,
                      1003: 3003,
                      1004: 3004,
                      1005: 3005,
                      1006: 3006,
                      1007: 3007,
                      12: 3007,
                      13: 3007
                      },
        'right_hemi': {2001: 4001,
                       2003: 4003,
                       2004: 4004,
                       2005: 4005,
                       2006: 4006,
                       2007: 4007,
                       51: 4007,
                       52: 4007
                       }
    }
    cerebral_white_matter = {
        'left_hemi': 2,
        'right_hemi': 41,
    }

    synseg_label_david_mapping = {
        'left_hemi': {
            1003: 112,
            1032: 112,
            1012: 112,
            1014: 112,
            1018: 112,
            1019: 112,
            1020: 112,
            1024: 112,
            1027: 112,
            1028: 112,
            1017: 117,
            1008: 113,
            1022: 113,
            1025: 113,
            1029: 113,
            1031: 113,
            1005: 114,
            1011: 114,
            1013: 114,
            1021: 114,
            1001: 115,
            1006: 115,
            1007: 115,
            1009: 115,
            1015: 115,
            1016: 115,
            1030: 115,
            1033: 115,
            1034: 115,
            1002: 116,
            1010: 116,
            1023: 116,
            1026: 116,
            1035: 118,
            10: 108,
            11: 109,
            12: 110,
            13: 111,
            17: 115,
            18: 115,

        },
        'right_hemi': {
            2003: 212,
            2012: 212,
            2014: 212,
            2018: 212,
            2019: 212,
            2020: 212,
            2024: 212,
            2027: 212,
            2028: 212,
            2032: 212,
            2017: 217,
            2008: 213,
            2022: 213,
            2025: 213,
            2029: 213,
            2031: 213,
            2005: 214,
            2011: 214,
            2013: 214,
            2021: 214,
            2001: 215,
            2006: 215,
            2007: 215,
            2009: 215,
            2015: 215,
            2016: 215,
            2030: 215,
            2033: 215,
            2034: 215,
            2002: 216,
            2010: 216,
            2023: 216,
            2026: 216,
            2035: 218,
            49: 208,
            50: 209,
            51: 210,
            52: 211,
            53: 215,
            54: 215,

        }
    }

    white_matter_david_mapping = {
        'left_hemi': {3001: 119,
                      3006: 120,
                      3004: 121,
                      3005: 122,
                      3003: 123,
                      3007: 124,
                      3008: 125,
                      3009: 126,
                      3010: 127,
                      3011: 128,
                      3031: 129,
                      3036: 130,
                      3034: 131,
                      3035: 132,
                      },
        'right_hemi': {4001: 219,
                       4006: 220,
                       4004: 221,
                       4005: 222,
                       4003: 223,
                       4007: 224,
                       4008: 225,
                       4009: 226,
                       4010: 227,
                       4011: 228,
                       4031: 229,
                       4036: 230,
                       4034: 231,
                       4035: 232,
                       }
    }

    gray_matter_david_mapping = {
        'left_hemi': {1001: 112,
                      1006: 113,
                      1004: 114,
                      1005: 115,
                      1003: 116,
                      1007: 118,
                      10: 108,
                      11: 109,
                      12: 110,
                      13: 111,
                      17: 115,
                      18: 115,
                      3: 101,
                      4: 102,
                      5: 102,
                      7: 104,
                      8: 105,
                      26: 106,
                      28: 103,
                      },
        'right_hemi': {2001: 212,
                       2006: 213,
                       2004: 214,
                       2005: 215,
                       2003: 216,
                       2007: 218,
                       49: 208,
                       50: 209,
                       51: 210,
                       52: 211,
                       53: 215,
                       54: 215,
                       42: 201,
                       43: 202,
                       44: 202,
                       46: 204,
                       47: 205,
                       58: 206,
                       60: 203,
                       }
    }

    decimal_places = 8
    scaling_factor = 10 ** decimal_places

    re_white_matter_mapping = {
        'left_hemi': {1001: 3001,
                      1004: 3004,
                      1005: 3005,
                      1006: 3006,
                      3001: 3001,
                      3004: 3004,
                      3005: 3005,
                      3006: 3006,
                      },
        'right_hemi': {2001: 4001,
                       2004: 4004,
                       2005: 4005,
                       2006: 4006,
                       4001: 4001,
                       4004: 4004,
                       4005: 4005,
                       4006: 4006,
                       }
    }

    @classmethod
    def label_to_david_label(cls, label_array):
        # left hemi and right hemi
        new_label_array = label_array.copy()
        for hemi in cls.gray_matter_david_mapping:
            # run every synseg label to freesurfer label
            for key in cls.gray_matter_david_mapping[hemi]:
                if cls.gray_matter_david_mapping[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.gray_matter_david_mapping[hemi][key]
                # left hemi and right hemi
        for hemi in cls.white_matter_david_mapping:
            # run every synseg label to freesurfer label
            for key in cls.white_matter_david_mapping[hemi]:
                if cls.white_matter_david_mapping[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.white_matter_david_mapping[hemi][key]
        return new_label_array

    @classmethod
    def synseg_label_to_freesurfer_GM(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = label_array.copy()
        # left hemi and right hemi
        for hemi in cls.synseg_label_freesurfer_GM_mapping:
            # run every synseg label to freesurfer label
            for key in cls.synseg_label_freesurfer_GM_mapping[hemi]:
                if cls.synseg_label_freesurfer_GM_mapping[hemi][key] is not None:
                    index_mask = np.argwhere(label_array == key)
                    new_label_array[index_mask[:, 0], index_mask[:, 1], index_mask[:, 2]] = \
                        cls.synseg_label_freesurfer_GM_mapping[hemi][key]
        return new_label_array

    @classmethod
    def white_matter_parcellation(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = label_array.copy()

        for k in cls.cerebral_white_matter:
            # 2 or 41
            index = np.argwhere(label_array == cls.cerebral_white_matter[k])
            white_matter_mapping_keys = list(cls.white_matter_mapping[k].keys())
            np_loss = np.zeros((index.shape[0], len(white_matter_mapping_keys)))
            np_loss[:, :] = 999999
            for i in range(len(white_matter_mapping_keys)):
                # 3001 ... 4007
                label_index = np.argwhere(label_array == white_matter_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
                # Z 軸切片
                for j in np.unique(label_index[:, 2]):
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
            for i in np.unique(new_label):
                select_index = index[np.argwhere(new_label == i)].squeeze()
                new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                    cls.white_matter_mapping[k][white_matter_mapping_keys[i]]
        return new_label_array

    @classmethod
    def re_run(cls, synthseg_array_wm, synthseg_array_cc, synthseg_array_ec):
        out_array = synthseg_array_wm.copy()
        for k in CorpusCallosumParcellation.cc_white_matter:
            cc_target = CorpusCallosumParcellation.cc_target[k]
            base_mask = (out_array == CorpusCallosumParcellation.cc_white_matter[k])
            target_mask = np.isin(synthseg_array_cc, cc_target)
            diff_mask = np.logical_and(base_mask,
                                       np.logical_not(target_mask))
            out_array[diff_mask] = WhiteMatterParcellation.cerebral_white_matter[k]
        for k in ECICParcellation.ec_ic_white_matter:
            ec_ic_target_list = list(set(ECICParcellation.ec_ic_parcellation_mapping[k].values()))
            base_mask = (out_array == ECICParcellation.ec_ic_white_matter[k])
            target_mask = np.isin(synthseg_array_ec, ec_ic_target_list)
            diff_mask = np.logical_and(base_mask,
                                       np.logical_not(target_mask))
            out_array[diff_mask] = WhiteMatterParcellation.cerebral_white_matter[k]
        return out_array

    @classmethod
    def re_white_matter_parcellation(cls, label_array):
        # 輸出 nii.gz 的新 array
        new_label_array = label_array.copy()
        for k in cls.cerebral_white_matter:
            # 2 or 41
            index = np.argwhere(label_array == cls.cerebral_white_matter[k])
            white_matter_mapping_keys = list(cls.re_white_matter_mapping[k].keys())
            np_loss = np.zeros((index.shape[0], len(white_matter_mapping_keys)))
            np_loss[:, :] = 999999
            for i in range(len(white_matter_mapping_keys)):
                # 3001 ... 4007
                label_index = np.argwhere(label_array == white_matter_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
                # Z 軸切片
                for j in np.unique(label_index[:, 2]):
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
            for i in np.unique(new_label):
                select_index = index[np.argwhere(new_label == i)].squeeze()
                if select_index.ndim == 2:
                    new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = \
                        cls.re_white_matter_mapping[k][white_matter_mapping_keys[i]]
                else:
                    new_label_array[select_index[0], select_index[1], select_index[2]] = \
                        cls.re_white_matter_mapping[k][white_matter_mapping_keys[i]]
        return new_label_array

    @classmethod
    def run(cls, synthseg_array):
        synthseg_freesurfer_array = cls.synseg_label_to_freesurfer_GM(synthseg_array)
        synthseg_freesurfer_array_wm = cls.white_matter_parcellation(synthseg_freesurfer_array)
        return synthseg_freesurfer_array_wm


class WhiteMatterParcellation2(WhiteMatterParcellation):
    synthseg_left_label = np.array([2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28,
                                    1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
                                    1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
                                    1030, 1031, 1032, 1033, 1034, 1035, ])
    synthseg_right_label = np.array([41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
                                     2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                     2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
                                     2030, 2031, 2032, 2033, 2034, 2035, ])

    synthseg33_left_label = np.array([2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28])
    synthseg33_right_label = np.array([41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])

    @classmethod
    def hemi_revise(cls, synthseg_array, synthseg33_array):
        synthseg33_array_right_mask = np.isin(synthseg33_array, cls.synthseg33_right_label)
        synthseg33_array_left_mask = np.isin(synthseg33_array, cls.synthseg33_left_label)
        synthseg_array_right_mask = np.isin(synthseg_array, cls.synthseg_right_label)
        synthseg_array_left_mask = np.isin(synthseg_array, cls.synthseg_left_label)
        temp_array = synthseg_array.copy()

        temp_array[(synthseg33_array_left_mask & synthseg_array_right_mask)] = 100
        temp_array[(synthseg33_array_right_mask & synthseg_array_left_mask)] = 200
        ## 2023-10-25 debug by 16041934_20230913 case
        # temp_array[(synthseg33_array_right_mask & synthseg_array_left_mask)] = 100
        # temp_array[(synthseg33_array_left_mask & synthseg_array_right_mask)] = 200
        kk = 3
        while np.logical_or(np.any(temp_array == 100), np.any(temp_array == 200)):
            kk += 1
            window_size = kk
            half_window = window_size // 2
            temp_array_index = np.argwhere(temp_array == 100)
            for row in temp_array_index:
                center_x = row[0]
                center_y = row[1]
                center_z = row[2]
                neighborhood = temp_array[center_x - half_window:center_x + half_window + 1,
                               center_y - half_window:center_y + half_window + 1,
                               center_z - half_window:center_z + half_window + 1, ]
                neighborhood = np.where(neighborhood < 1000, np.nan, neighborhood)
                neighborhood = np.where(neighborhood > 2000, np.nan, neighborhood)
                unique, counts = np.unique(neighborhood, return_counts=True, equal_nan=False)
                if len(unique) > 0:
                    if np.isnan(unique[counts.argmax()]):
                        pass
                    else:
                        temp_array[center_x, center_y, center_z] = unique[counts.argmax()]
            temp_array_index = np.argwhere(temp_array == 200)
            for row in temp_array_index:
                center_x = row[0]
                center_y = row[1]
                center_z = row[2]
                neighborhood = temp_array[center_x - half_window:center_x + half_window + 1,
                               center_y - half_window:center_y + half_window + 1,
                               center_z - half_window:center_z + half_window + 1, ]
                neighborhood = np.where(neighborhood < 2000, np.nan, neighborhood)
                neighborhood = np.where(neighborhood > 3000, np.nan, neighborhood)
                unique, counts = np.unique(neighborhood, return_counts=True, equal_nan=False)
                if len(unique) > 0:
                    if np.isnan(unique[counts.argmax()]):
                        pass
                    else:
                        temp_array[center_x, center_y, center_z] = unique[counts.argmax()]
        return temp_array

    @classmethod
    def run(cls, synthseg_array, *args, **kwargs):
        synthseg33_array = kwargs.get('synthseg33_array')
        synthseg_array_revise = cls.hemi_revise(synthseg_array=synthseg_array,
                                                synthseg33_array=synthseg33_array)
        synthseg_freesurfer_array = cls.synseg_label_to_freesurfer_GM(synthseg_array_revise)
        synthseg_freesurfer_array_wm = cls.white_matter_parcellation(synthseg_freesurfer_array)

        return synthseg_freesurfer_array_wm

