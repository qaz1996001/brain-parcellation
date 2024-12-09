import numpy as np
from ..core import loss_distance,Process


class ECICParcellation(Process):
    ec_ic_parcellation_mapping = {
        'left_hemi': {1007: 3008,
                      11: 3009,
                      10: 3009,
                      13: 3009,
                      26: 3009,
                      },
        'right_hemi': {2007: 4008,
                       49: 4009,
                       50: 4009,
                       52: 4009,
                       58: 4009,
                       }
    }
    ec_ic_prerequisite = {'left_hemi': {'start': 12,
                                        'end': 13
                                        },
                          'right_hemi': {'start': 51,
                                         'end': 52
                                         }
                          }

    ec_ic_white_matter = {
        'left_hemi': 3007,
        'right_hemi': 4007,
    }

    decimal_places = 8
    scaling_factor = 10 ** decimal_places

    @classmethod
    def ec_ic_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array, dtype=np.int64)
        for k in cls.ec_ic_white_matter:
            # 取出putamen開始與停止層
            prerequisite_start_index = np.argwhere(label_array == cls.ec_ic_prerequisite[k]['start'])
            prerequisite_end_index = np.argwhere(label_array == cls.ec_ic_prerequisite[k]['end'])
            prerequisite_end_index = prerequisite_end_index[:, 2].min()
            unique_prerequisite_index = np.unique(prerequisite_start_index[:, 2])

            # 計算距離的各群
            ec_ic_parcellation_mapping_keys = list(cls.ec_ic_parcellation_mapping[k].keys())

            # 被分類的
            index = np.argwhere(label_array == cls.ec_ic_white_matter[k])
            unique_label_index = np.unique(index[:, 2])

            # 被分類的Z軸索引與 putamen
            z_aixs_intersect = np.intersect1d(unique_prerequisite_index, unique_label_index)
            z_aixs_intersect = z_aixs_intersect[z_aixs_intersect >= prerequisite_end_index]
            index = index[np.isin(index[:, 2], z_aixs_intersect)]

            np_loss = np.zeros((index.shape[0], len(ec_ic_parcellation_mapping_keys)))
            np_loss[:, :] = 999999
            for i in range(len(ec_ic_parcellation_mapping_keys)):
                label_index = np.argwhere(label_array == ec_ic_parcellation_mapping_keys[i])
                loss_list = []
                index_sub_arg_list = []
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
                    cls.ec_ic_parcellation_mapping[k][ec_ic_parcellation_mapping_keys[i]]
        return new_label_array

    @classmethod
    def run(cls, label_array):
        new_label_array = cls.ec_ic_parcellation(label_array)
        return new_label_array