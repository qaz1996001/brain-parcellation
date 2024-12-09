import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure, \
    distance_transform_edt
# from ..core import Process

# To DO

class BullseyeProcess:
    synthseg_wm_output = {
        'left_hemi': {4: 3010,
                      5: 3010,
                      7: 3010,
                      8: 3010,
                      10: 3010,
                      11: 3010,
                      12: 3010,
                      13: 3010,
                      18: 3010,
                      26: 3010,
                      28: 3010,

                      1001: 3010,
                      1003: 3010,
                      1004: 3010,
                      1005: 3010,
                      1006: 3010,
                      1007: 3010,

                      3001: 3010,
                      3003: 3010,
                      3004: 3010,
                      3005: 3010,
                      3006: 3010,
                      3007: 3010,
                      },
        'right_hemi': {43: 4010,
                       44: 4010,
                       46: 4010,
                       47: 4010,
                       49: 4010,
                       50: 4010,
                       51: 4010,
                       52: 4010,
                       54: 4010,
                       58: 4010,
                       60: 4010,

                       2001: 4010,
                       2003: 4010,
                       2004: 4010,
                       2005: 4010,
                       2006: 4010,
                       2007: 4010,

                       4001: 4010,
                       4003: 4010,
                       4004: 4010,
                       4005: 4010,
                       4006: 4010,
                       4007: 4010},
    }
    bullseye_parcellation = {
        'left_hemi': 3010,
        'right_hemi': 4010,
    }
    lateral_ventricle = {
        'left_hemi': 4,
        'right_hemi': 43,
    }
    depth_number = 6
    inner_size = 2
    outer_size = 5
    decimal_places = 8

    @classmethod
    def filter_labels(cls, in0, include_superlist, fixed_id=None, map_pairs_list=None):
        """filters-out labels not in the include-superset. Merges labels within superset. Transforms label-ids according
        to mappings (or fixed id)"""

        # read label file and create output
        out0 = np.zeros(in0.shape, dtype=in0.dtype)

        # for each group of labels in subset assign them the same id (either 1st in subset or fixed-id, in case given)
        for labels_list in include_superlist:
            for label in labels_list:
                value = labels_list[0]
                if fixed_id is not None: value = fixed_id[0]
                out0[in0 == label] = value

        # transform label-ids in case mapping is specified
        if map_pairs_list is not None:
            out1 = np.copy(out0)
            for map_pair in map_pairs_list:
                out1[out0 == map_pair[0]] = map_pair[1]

        # save output
        out_final = out0 if not map_pairs_list else out1

        return out_final

    @classmethod
    def norm_dist_map(cls, orig, dest):
        """compute normalized distance map given an origin and destination masks, resp."""

        dist_orig = distance_transform_edt(np.logical_not(orig.astype(np.bool_)))
        dist_dest = distance_transform_edt(np.logical_not(dest.astype(np.bool_)))

        # normalized distance (0 in origin to 1 in dest)
        ndist = dist_orig / (dist_orig + dist_dest)
        return ndist

    @classmethod
    def create_shells(cls, ndist, mask, n_shells=4):
        """creates specified number of shells given normalized distance map. When mask is given, output in mask == 0 is
        set to zero"""

        out = np.zeros(ndist.shape, dtype=np.int32)

        limits = np.linspace(0., 1., n_shells + 1)
        for i in np.arange(n_shells) + 1:
            # compute shell and assing increasing label-id
            mask2 = np.logical_and(ndist >= limits[i - 1], ndist < limits[i])
            if mask is not None:  # maskout regions outside mask
                mask2 = np.logical_and(mask2, mask)
            out[mask2] = i
        out[np.isclose(ndist, 0.)] = 0  # need to assign zero to ventricles because of >= above
        return out

    @classmethod
    def merge_labels(cls, in1, in2, intersect=False):
        """merges labels from two input labelmaps, optionally computing intersection"""
        in1 = in1.round(0).astype(int)
        in2 = in2.round(0).astype(int)

        # if not intersection, simply include labels from 'in2' into 'in1'
        if not intersect:

            out = np.zeros(in1.shape, dtype=np.int32)

            out[:] = in1[:]
            mask = in2 > 0
            out[mask] = in2[mask]  # overwrite in1 where in2 > 0

        # if intersection, create new label-set as cartesian product of the two sets
        else:
            out = np.zeros(in1.shape, dtype=np.int32)

            u1_set = np.unique(in1.ravel())
            u2_set = np.unique(in2.ravel())

            u1_set = u1_set.astype(int)
            u2_set = u2_set.astype(int)

            for u1 in u1_set:
                if u1 == 0: continue
                mask1 = in1 == u1
                for u2 in u2_set:
                    if u2 == 0: continue
                    mask2 = in2 == u2
                    mask3 = np.logical_and(mask1, mask2)
                    if not np.any(mask3): continue
                    out[mask3] = int(str(u1) + str(u2))  # new label id by concatenating [u1, u2]
        return out

    @classmethod
    def generate_wmparc(cls, incl_aux, ndist, label, incl_labels=None, verbose=False):
        """generates wmparc by propagating labels in 'label_file' down the gradient defined by distance map in
        'ndist_file'. Labels are only propagated in regions where 'incl_file' > 0 (or 'incl_file' == incl_labels[i],
        if 'incl_labels is provided).
        """
        connectivity = generate_binary_structure(3, 2)

        # create inclusion mask
        if incl_labels is None:
            incl_mask = incl_aux > 0
        else:
            incl_mask = np.zeros(incl_aux.shape, dtype=np.bool_)
            for lab in incl_labels:
                incl_mask[incl_aux == lab] = True

        # get DONE and processing masks
        DONE_mask = label > 0  # this is for using freesurfer wmparc
        proc_mask = np.logical_and(np.logical_and(ndist > 0., ndist < 1.), incl_mask)

        # setup the ouptut vol
        out = np.zeros(label.shape, dtype=label.dtype)

        # initialize labels in cortex
        out[DONE_mask] = label[DONE_mask]  # this is for using freesurfer wmparc

        # start with connectivity 1
        its_conn = 1

        # main loop
        while not np.all(DONE_mask[proc_mask]):

            if verbose:
                print('%0.1f done' % (100. * float(DONE_mask[proc_mask].sum()) / float(proc_mask.sum())))

            # loop to increase connectivity for non-reachable TO-DO points
            while True:

                # dilate the SOLVED area
                aux = binary_dilation(DONE_mask, iterate_structure(connectivity, its_conn))
                # next TO-DO: close to DONE, in the processing mask and not yet done
                TODO_mask = np.logical_and(np.logical_and(aux, proc_mask), np.logical_not(DONE_mask))

                if TODO_mask.sum() > 0:
                    break

                if verbose:
                    print('Non-reachable points. Increasing connectivity')

                its_conn += 1

            # sort TO-DO points by ndist
            Idx_TODO = np.argwhere(TODO_mask)
            Idx_ravel = np.ravel_multi_index(Idx_TODO.T, label.shape)
            I_sort = np.argsort(ndist.ravel()[Idx_ravel])

            # iterate along TO-DO points
            for idx in Idx_TODO[I_sort[::-1]]:

                max_dist = -1.

                # process each neighbor
                for off in np.argwhere(iterate_structure(connectivity, its_conn)) - its_conn:

                    try:

                        # if it is not DONE then skip
                        if not DONE_mask[idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]]:
                            continue

                        # if it is the largest distance (ie, largest gradient)
                        cur_dist = ndist[idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]]
                        if cur_dist > max_dist:
                            out[idx[0], idx[1], idx[2]] = out[idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]]
                            max_dist = cur_dist

                    except:
                        print('something wrong with neighbor at: (%d, %d, %d)' % (
                            idx[0] + off[0], idx[1] + off[1], idx[2] + off[2]))
                        pass

                if max_dist < 0.: print("something went wrong with point: (%d, %d, %d)" % (idx[0], idx[1], idx[2]))

                # mark as solved and remove from visited
                DONE_mask[idx[0], idx[1], idx[2]] = True
        return out

    @classmethod
    def bullsey_hemi_parcellation(cls, depth_array, synthseg_array):
        # 輸出 nii.gz 的新 array
        new_label_array = np.zeros_like(depth_array, dtype=depth_array.dtype)
        temp_array = np.zeros_like(depth_array, dtype=depth_array.dtype)
        # 取出 bullseye 部分
        index = np.argwhere(depth_array != 0)
        temp_array[index[:, 0], index[:, 1], index[:, 2]] = synthseg_array[index[:, 0], index[:, 1], index[:, 2]]

        left_mask = (temp_array // 1000) == 3
        right_mask = (temp_array // 1000) == 4
        new_label_array[left_mask] = cls.bullseye_parcellation['left_hemi']
        new_label_array[right_mask] = cls.bullseye_parcellation['right_hemi']
        new_label_array = new_label_array + depth_array
        return new_label_array

    @classmethod
    def bullseye_depth_merge(cls, bullsey_parcellation_array, synthseg_array):
        out_array = np.zeros_like(bullsey_parcellation_array)
        for k in cls.lateral_ventricle:
            lateral_ventricle_index = np.argwhere(synthseg_array == cls.lateral_ventricle[k])
            lateral_ventricle_z = np.unique(lateral_ventricle_index[:, 2])
            z_higher = lateral_ventricle_z.max()
            z_lower = lateral_ventricle_z.min()
            bullseye_cluster = np.unique(cls.bullseye_parcellation[k])
            bullseye_cluster = bullseye_cluster[bullseye_cluster != 0]
            for i in bullseye_cluster:
                bullseye_depth_mask = (bullsey_parcellation_array > i) & (
                        bullsey_parcellation_array < (i + cls.depth_number))
                bullseye_index = np.argwhere(bullseye_depth_mask)
                bullseye_index_higher = bullseye_index[bullseye_index[:, 2] > z_higher]
                bullseye_index_lower = bullseye_index[bullseye_index[:, 2] < z_lower]

                bullseye_between_inner_mask = (bullsey_parcellation_array > (i)) & (
                        bullsey_parcellation_array <= (i + cls.inner_size))
                bullseye_between_inner_index = np.argwhere(bullseye_between_inner_mask == True)
                bullseye_between_inner_index_between = bullseye_between_inner_index[
                    (bullseye_between_inner_index[:, 2] <= z_higher) &
                    (bullseye_between_inner_index[:, 2] >= z_lower)]

                bullseye_between_outer_mask = (bullsey_parcellation_array > (i + cls.inner_size)) & (
                        bullsey_parcellation_array <= (i + cls.outer_size))
                bullseye_between_outer_index = np.argwhere(bullseye_between_outer_mask == True)
                bullseye_between_outer_index_between = bullseye_between_outer_index[
                    (bullseye_between_outer_index[:, 2] <= z_higher) &
                    (bullseye_between_outer_index[:, 2] >= z_lower)]
                out_array[bullseye_index_higher[:, 0],
                bullseye_index_higher[:, 1],
                bullseye_index_higher[:, 2]] = i + 2
                out_array[bullseye_index_lower[:, 0],
                bullseye_index_lower[:, 1],
                bullseye_index_lower[:, 2]] = i + 2
                out_array[bullseye_between_inner_index_between[:, 0],
                bullseye_between_inner_index_between[:, 1],
                bullseye_between_inner_index_between[:, 2]] = i + 1
                out_array[bullseye_between_outer_index_between[:, 0],
                bullseye_between_outer_index_between[:, 1],
                bullseye_between_outer_index_between[:, 2]] = i + 2
        return out_array

    @classmethod
    def bullseye_depth_synthseg_label(cls, bullsey_depth_merge_array, label_array, synthseg_array):
        out = np.zeros(bullsey_depth_merge_array.shape, dtype=np.int32)
        u1 = np.unique(bullsey_depth_merge_array)
        mask2 = (synthseg_array == 2)
        mask41 = (synthseg_array == 41)
        for k in u1:
            mask1 = (bullsey_depth_merge_array == k)
            out_mask = np.logical_or(np.logical_and(mask1, mask2),
                                     np.logical_and(mask1, mask41))
            if (k == 3011) or (k == 4011):
                out[out_mask] = bullsey_depth_merge_array[out_mask]
            else:
                inner_mask = np.logical_and((bullsey_depth_merge_array >= 3000), out_mask)
                out[out_mask] = label_array[out_mask]
                out[inner_mask] = label_array[inner_mask] + 30
        return out

    @classmethod
    def get_depth_wmparc_np(cls, label_array, n_shells):
        filter_labels_include_superlist = [[3001, 3007], [4001, 4007], [3004], [4004], [3005], [4005], [3006],
                                           [4006]]  # lobar labels in WM
        filter_labels_fixed_id = None
        filter_labels_map_pairs_list = [[3001, 11], [4001, 21], [3004, 12], [4004, 22], [3005, 13], [4005, 23],
                                        [3006, 14], [4006, 24]]
        filter_lobes_np = cls.filter_labels(label_array,
                                            include_superlist=filter_labels_include_superlist,
                                            fixed_id=filter_labels_fixed_id,
                                            map_pairs_list=filter_labels_map_pairs_list)

        ventricles_include_superlist = [[43, 4]]
        ventricles_fixed_id = [1]
        ventricles_map_pairs_list = None
        ventricles_np = cls.filter_labels(label_array,
                                          include_superlist=ventricles_include_superlist,
                                          fixed_id=ventricles_fixed_id, map_pairs_list=ventricles_map_pairs_list)

        cortex_include_superlist = [[1001, 2001, 1004, 2004, 1005, 2005, 1006, 2006]]  # lobar labels in cortex
        cortex_fixed_id = [1]
        cortex_map_pairs_list = None
        cortex_np = cls.filter_labels(label_array,
                                      include_superlist=cortex_include_superlist,
                                      fixed_id=cortex_fixed_id, map_pairs_list=cortex_map_pairs_list)
        bgt_include_superlist = [[10, 49, 11, 12, 50, 51, 26, 58, 13, 52]]  # basal ganglia + thalamus
        bgt_fixed_id = [5]
        bgt_map_pairs_list = None
        bgt_np = cls.filter_labels(label_array,
                                   include_superlist=bgt_include_superlist,
                                   fixed_id=bgt_fixed_id, map_pairs_list=bgt_map_pairs_list)

        ndist_np = cls.norm_dist_map(ventricles_np, cortex_np)
        generate_wmparc_incl_labels = [3003, 4003, 5001, 5002]  # the labels that need to be 'filled'
        gen_wmparc_np = cls.generate_wmparc(incl_aux=label_array, ndist=ndist_np,
                                            label=filter_lobes_np,
                                            incl_labels=generate_wmparc_incl_labels, verbose=False)
        lobe_wmparc_np = cls.merge_labels(in1=gen_wmparc_np, in2=bgt_np, intersect=False)
        depth_wmparc_np = cls.create_shells(ndist=ndist_np, mask=lobe_wmparc_np, n_shells=n_shells)
        return depth_wmparc_np

    @classmethod
    def run(cls, label_array, synthseg_array, depth_number):
        cls.depth_number = depth_number
        cls.outer_size = depth_number - 1
        depth_np = cls.get_depth_wmparc_np(label_array=label_array, n_shells=depth_number)
        bullsey_parcellation_array = cls.bullsey_hemi_parcellation(depth_array=depth_np, synthseg_array=label_array)
        bullsey_depth_merge_array = cls.bullseye_depth_merge(bullsey_parcellation_array, label_array)
        bullseye_depth_synthseg_array = cls.bullseye_depth_synthseg_label(
            bullsey_depth_merge_array=bullsey_depth_merge_array,
            label_array=label_array,
            synthseg_array=synthseg_array)
        return bullseye_depth_synthseg_array
