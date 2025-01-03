import numpy as np
import pandas as pd
from skimage import measure
from ..core import Process,data_translate,left_right_translate,inverse_data_translate,inverse_left_right_translate


class CorpusCallosumParcellation(Process):
    cc_prerequisite = {'left_hemi': {'start': 1003,
                                     'end': None
                                     },
                       'right_hemi': {'start': 2003,
                                      'end': None
                                      }
                       }
    lateral_ventricle = {
        'left_hemi': 4,
        'right_hemi': 43,
    }
    cc_white_matter = {
        'left_hemi': 3003,
        'right_hemi': 4003,
    }
    cc_target = {
        'left_hemi': 3010,
        'right_hemi': 4010,
    }

    @classmethod
    def get_cluster_min_max(cls, prerequisite_cluster):
        prerequisite_cluster_unique = np.unique(prerequisite_cluster)
        cluster_x_list = []
        cluster_y_list = []
        for i in prerequisite_cluster_unique:
            if i == 0:
                continue
            else:
                y_max = np.argwhere(prerequisite_cluster == i)[:, 0].max()
                y_min = np.argwhere(prerequisite_cluster == i)[:, 0].min()
                x_max = np.argwhere(prerequisite_cluster == i)[:, 1].max()
                x_min = np.argwhere(prerequisite_cluster == i)[:, 1].min()
                cluster_x_list.append(x_min)
                cluster_x_list.append(x_max)
                cluster_y_list.append(y_min)
                cluster_y_list.append(y_max)
        return sorted(cluster_x_list), sorted(cluster_y_list)

    @classmethod
    def get_prerequisite_cluster(cls, measure_np):
        prerequisite_cluster = measure.label(measure_np, connectivity=2)
        prerequisite_cluster_unique = np.unique(prerequisite_cluster)
        prerequisite_cluster_index_count = np.sum(prerequisite_cluster != 0)
        prerequisite_cluster_list = []
        if prerequisite_cluster_unique.shape[0] > 2:
            for i in prerequisite_cluster_unique:
                if i == 0:
                    prerequisite_cluster_list.append([0, 0])
                else:
                    prerequisite_cluster_i_count = np.sum(prerequisite_cluster == i)
                    prerequisite_cluster_list.append(
                        [i, (prerequisite_cluster_i_count / prerequisite_cluster_index_count)])
            prerequisite_df = pd.DataFrame(prerequisite_cluster_list)
            prerequisite_df = prerequisite_df.sort_values(by=1)
            del_cluster = prerequisite_df.iloc()[:-2, 0].to_numpy()
            select_cluster = prerequisite_df.iloc()[-2:, 0].to_numpy()
            prerequisite_cluster = np.where(np.isin(prerequisite_cluster, del_cluster), 0, prerequisite_cluster)
            return prerequisite_cluster, select_cluster
        else:
            return prerequisite_cluster, prerequisite_cluster_unique

    @classmethod
    def cc_parcellation(cls, label_array):
        new_label_array = np.zeros_like(label_array)
        for k in cls.cc_white_matter:
            index = np.argwhere(label_array == cls.cc_white_matter[k])
            prerequisite_start_index = np.argwhere(label_array == cls.cc_prerequisite[k]['start'])
            index_z_unique = np.unique(index[:, 2])
            prerequisite_index_z_unique = np.unique(prerequisite_start_index[:, 2])
            z_aixs_intersect = np.intersect1d(index_z_unique, prerequisite_index_z_unique)
            z_aixs_intersect = z_aixs_intersect[::-1]
            # Z 軸切片
            start_status = False
            for j in z_aixs_intersect:
                measure_np = np.zeros(label_array.shape[:2])
                index_sub = index[index[:, 2] == j]
                prerequisite_index_sub = prerequisite_start_index[prerequisite_start_index[:, 2] == j]
                measure_np[prerequisite_index_sub[:, 0], prerequisite_index_sub[:, 1]] = j
                prerequisite_cluster, prerequisite_cluster_unique = cls.get_prerequisite_cluster(measure_np)

                if prerequisite_cluster_unique.shape[0] >= 2:
                    cluster_x_list, cluster_y_list = cls.get_cluster_min_max(prerequisite_cluster)
                    if start_status:
                        if 'left_hemi' == k:
                            select_index = index_sub[
                                (index_sub[:, 1] >= cluster_x_list[1]) &
                                (index_sub[:, 0] >= cluster_y_list[1]) &
                                (index_sub[:, 0] <= cluster_y_list[-2])]
                        else:
                            select_index = index_sub[
                                (index_sub[:, 1] <= cluster_x_list[-2]) &
                                (index_sub[:, 0] >= cluster_y_list[1]) &
                                (index_sub[:, 0] <= cluster_y_list[-2])]
                        new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.cc_target[k]
                    else:
                        index_x_max = np.max(index_sub[:, 1])
                        prerequisite_index_x_max = np.max(prerequisite_index_sub[:, 1])
                        prerequisite_index_x_max = int(prerequisite_index_x_max * 0.97)
                        if index_x_max >= prerequisite_index_x_max:
                            start_status = True
                            if 'left_hemi' == k:
                                select_index = index_sub[
                                    (index_sub[:, 1] >= cluster_x_list[1]) &
                                    (index_sub[:, 0] >= cluster_y_list[1]) &
                                    (index_sub[:, 0] <= cluster_y_list[-2])]
                            else:
                                select_index = index_sub[
                                    (index_sub[:, 1] <= cluster_x_list[-2]) &
                                    (index_sub[:, 0] >= cluster_y_list[1]) &
                                    (index_sub[:, 0] <= cluster_y_list[-2])]
                            new_label_array[select_index[:, 0], select_index[:, 1], select_index[:, 2]] = cls.cc_target[
                                k]
        return new_label_array

    @classmethod
    def _get_cc_x_y_axis_index(cls, cc_index):
        z_axis = np.unique(cc_index[:, 2])
        norm_vector = []
        for i in z_axis:
            vector_x = cc_index[cc_index[:, 2] == i][:, 1]
            vector_y = cc_index[cc_index[:, 2] == i][:, 0]
            if vector_x.shape[0] > 1:
                norm_vector.append([i, vector_x.min(),
                                    vector_y.max(),
                                    vector_y.min(), ])
        norm_vector = np.array(norm_vector)
        norm_vector = norm_vector[::-1]
        return norm_vector

    @classmethod
    def _get_df_lateral_ventricle(cls, left_lateral_ventricle_index, right_lateral_ventricle_index):
        lateral_ventricle_vector = []
        left_lateral_ventricle_z_axis = np.unique(left_lateral_ventricle_index[:, 2])
        right_lateral_ventricle_z_axis = np.unique(right_lateral_ventricle_index[:, 2])
        z_axis_intersect = np.intersect1d(left_lateral_ventricle_z_axis, right_lateral_ventricle_z_axis)
        for i in z_axis_intersect:
            left_vector_x = left_lateral_ventricle_index[left_lateral_ventricle_index[:, 2] == i][:, 1]
            right_vector_x = right_lateral_ventricle_index[right_lateral_ventricle_index[:, 2] == i][:, 1]
            # lateral_ventricle_vector.append([i,
            #                                  np.percentile(left_vector_x, 10, axis=0).astype(int),
            #                                  np.percentile(right_vector_x, 90, axis=0).astype(int)])
            lateral_ventricle_vector.append([i,
                                             np.percentile(left_vector_x, 20, axis=0).astype(int),
                                             np.percentile(right_vector_x, 80, axis=0).astype(int)])

        intersect_lateral_ventricle_array = np.array(lateral_ventricle_vector)
        df_lateral_ventricle = pd.DataFrame(intersect_lateral_ventricle_array, columns=['lateral_ventricle_z',
                                                                                        'left_lateral_ventricle_x',
                                                                                        'right_lateral_ventricle_x'])
        df_lateral_ventricle.index = df_lateral_ventricle['lateral_ventricle_z']
        return df_lateral_ventricle

    @classmethod
    def _get_df_cc(cls, left_cc_index, right_cc_index) -> pd.DataFrame:
        left_cc_axis_vector = cls._get_cc_x_y_axis_index(left_cc_index)
        right_cc_axis_vector = cls._get_cc_x_y_axis_index(right_cc_index)
        z_axis_intersect = np.intersect1d(left_cc_axis_vector[:, 0], right_cc_axis_vector[:, 0])[::-1]
        z_axis_intersect = z_axis_intersect.reshape(-1, 1)
        intersect_cc_array = np.concatenate([z_axis_intersect,
                                             left_cc_axis_vector[
                                                 np.isin(left_cc_axis_vector[:, 0], z_axis_intersect)][:, 1:],
                                             right_cc_axis_vector[
                                                 np.isin(right_cc_axis_vector[:, 0], z_axis_intersect)][:, 1:], ],
                                            axis=1)
        df_cc = pd.DataFrame(intersect_cc_array, columns=['cc_z',
                                                          'left_cc_x', 'left_cc_y_max', 'left_cc_y_min',
                                                          'right_cc_x', 'right_cc_y_max', 'right_cc_y_min'])
        df_cc.index = df_cc['cc_z']
        return df_cc

    @classmethod
    def cc_adapt(cls, cc_array, label_array):
        new_label_array = np.zeros_like(label_array)
        left_cc_index = np.argwhere(cc_array == cls.cc_target['left_hemi'])
        right_cc_index = np.argwhere(cc_array == cls.cc_target['right_hemi'])
        left_lateral_ventricle_index = np.argwhere(label_array == cls.lateral_ventricle['left_hemi'])
        right_lateral_ventricle_index = np.argwhere(label_array == cls.lateral_ventricle['right_hemi'])

        df_cc = cls._get_df_cc(left_cc_index=left_cc_index, right_cc_index=right_cc_index)
        df_lateral_ventricle = cls._get_df_lateral_ventricle(left_lateral_ventricle_index=left_lateral_ventricle_index,
                                                             right_lateral_ventricle_index=right_lateral_ventricle_index
                                                             )
        df = df_cc.join(df_lateral_ventricle, how='inner')
        df = df.drop(columns=['lateral_ventricle_z'])

        left_index = np.argwhere(label_array == cls.cc_white_matter['left_hemi'])
        right_index = np.argwhere(label_array == cls.cc_white_matter['right_hemi'])
        for row in df.index:
            cc_z = df.loc()[row, 'cc_z']
            left_lateral_ventricle_x = df.loc()[row, 'left_lateral_ventricle_x']
            right_lateral_ventricle_x = df.loc()[row, 'right_lateral_ventricle_x']

            left_cc_y_max = df.loc()[row, 'left_cc_y_max']
            left_cc_y_min = df.loc()[row, 'left_cc_y_min']
            right_cc_y_max = df.loc()[row, 'right_cc_y_max']
            right_cc_y_min = df.loc()[row, 'right_cc_y_min']

            select_left_index = left_index[(left_index[:, 2] == cc_z) & (left_index[:, 1] > left_lateral_ventricle_x) &
                                           (left_index[:, 0] >= left_cc_y_min) & (left_index[:, 0] <= left_cc_y_max)
                                           ]
            select_right_index = right_index[
                (right_index[:, 2] == cc_z) & (right_index[:, 1] < right_lateral_ventricle_x) &
                (right_index[:, 0] >= right_cc_y_min) & (right_index[:, 0] <= right_cc_y_max)
                ]
            new_label_array[select_left_index[:, 0],
            select_left_index[:, 1],
            select_left_index[:, 2]] = cls.cc_target['left_hemi']
            new_label_array[select_right_index[:, 0],
            select_right_index[:, 1],
            select_right_index[:, 2]] = cls.cc_target['right_hemi']
        return new_label_array

    @classmethod
    def run(cls, label_array):
        label_array_translate = data_translate(label_array)
        FLIP, label_array_translate = left_right_translate(label_array_translate)
        cc_array_translate = cls.cc_parcellation(label_array_translate)
        cc_array_translate = cls.cc_adapt(cc_array_translate, label_array_translate)
        cc_array = inverse_left_right_translate(FLIP, cc_array_translate)
        out_label_array = inverse_data_translate(cc_array)
        return out_label_array



class CorpusCallosumParcellationForWMHProcess(CorpusCallosumParcellation):
    @classmethod
    def _get_df_lateral_ventricle(cls, left_lateral_ventricle_index, right_lateral_ventricle_index):
        # print('CorpusCallosumParcellationForWMHProcess _get_df_lateral_ventricle')
        lateral_ventricle_vector = []
        left_lateral_ventricle_z_axis = np.unique(left_lateral_ventricle_index[:, 2])
        right_lateral_ventricle_z_axis = np.unique(right_lateral_ventricle_index[:, 2])
        z_axis_intersect = np.intersect1d(left_lateral_ventricle_z_axis, right_lateral_ventricle_z_axis)
        for i in z_axis_intersect:
            left_vector_x = left_lateral_ventricle_index[left_lateral_ventricle_index[:, 2] == i][:, 1]
            right_vector_x = right_lateral_ventricle_index[right_lateral_ventricle_index[:, 2] == i][:, 1]

            lateral_ventricle_vector.append([i,
                                             np.percentile(left_vector_x, 50, axis=0).astype(int),
                                             np.percentile(right_vector_x, 50, axis=0).astype(int)])

        intersect_lateral_ventricle_array = np.array(lateral_ventricle_vector)
        df_lateral_ventricle = pd.DataFrame(intersect_lateral_ventricle_array, columns=['lateral_ventricle_z',
                                                                                        'left_lateral_ventricle_x',
                                                                                        'right_lateral_ventricle_x'])
        df_lateral_ventricle.index = df_lateral_ventricle['lateral_ventricle_z']
        return df_lateral_ventricle