import os
import pathlib
import sys
import numpy as np
import argparse

import tensorflow as tf

import traceback
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# project imports
from SynthSeg.predict import get_flip_indices

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models

import nibabel as nib
import SimpleITK as sitk

# from tensorflow.keras.models import Model, Sequential, load_model
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=8192)])
        # memory_limit=20480)])
    except RuntimeError as e:
        print(e)


class SynthSeg:
    LABELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'labels_classes_priors')
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'models')

    def __init__(self, ):
        self.args = None
        self.net_unet2 = None
        self.net_convert = None
        self.net_parcellation = None
        pass

    def prepare_output_files(self, path_images, out_seg, recompute):
        # check inputs
        assert path_images is not None, 'please specify an input file/folder (--i)'
        assert out_seg is not None, 'please specify an output file/folder (--o)'

        # convert path to absolute paths
        path_images = os.path.abspath(path_images)
        basename = os.path.basename(path_images)
        out_seg = os.path.abspath(out_seg)

        # path_images is a text file
        if basename[-4:] == '.txt':

            # input images
            if not os.path.isfile(path_images):
                raise Exception('provided text file containing paths of input images does not exist' % path_images)
            with open(path_images, 'r') as f:
                path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

            # define helper to deal with outputs
            def text_helper(path, name):
                if path is not None:
                    assert path[-4:] == '.txt', 'if path_images given as text file, so must be %s' % name
                    with open(path, 'r') as ff:
                        path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                    recompute_files = [not os.path.isfile(p) for p in path]
                else:
                    path = [None] * len(path_images)
                    recompute_files = [False] * len(path_images)
                unique_file = False
                return path, recompute_files, unique_file

            # use helper on all outputs
            out_synthseg, recompute_synthseg, _ = text_helper(out_seg, 'path_segmentations')
            out_synthseg33, recompute_synthseg33, _ = text_helper(out_seg, 'path_segmentations')

        # path_images is a folder
        elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & (
                '.npz' not in basename):

            # input images
            if os.path.isfile(path_images):
                raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
            path_images = utils.list_images_in_folder(path_images)

            # define helper to deal with outputs
            def helper_dir(path, name, file_type, suffix):
                unique_file = False
                if path is not None:
                    assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                    if file_type == 'csv':
                        if path[-4:] != '.csv':
                            print('%s provided without csv extension. Adding csv extension.' % name)
                            path += '.csv'
                        path = [path] * len(path_images)
                        recompute_files = [True] * len(path_images)
                        unique_file = True
                    else:
                        if (path[-7:] == '.nii.gz') | (path[-4:] == '.nii') | (path[-4:] == '.mgz') | (
                                path[-4:] == '.npz'):
                            raise Exception('Output FOLDER had a FILE extension' % path)
                        path = [os.path.join(path, os.path.basename(p)) for p in path_images]
                        path = [p.replace('.nii', '_%s.nii' % suffix) for p in path]
                        path = [p.replace('.mgz', '_%s.mgz' % suffix) for p in path]
                        path = [p.replace('.npz', '_%s.npz' % suffix) for p in path]
                        recompute_files = [not os.path.isfile(p) for p in path]
                    utils.mkdir(os.path.dirname(path[0]))
                else:
                    path = [None] * len(path_images)
                    recompute_files = [False] * len(path_images)
                return path, recompute_files, unique_file

            # use helper on all outputs
            out_synthseg, recompute_synthseg, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg')
            out_synthseg33, recompute_synthseg33, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg33')

        # path_images is an image
        else:

            # input images
            assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                                'please make sure the path and the extension are correct' % path_images
            path_images = [path_images]

            # define helper to deal with outputs
            def helper_im(path, name, file_type, suffix):
                unique_file = False
                if path is not None:
                    assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                    if file_type == 'csv':
                        if path[-4:] != '.csv':
                            print('%s provided without csv extension. Adding csv extension.' % name)
                            path += '.csv'
                        recompute_files = [True]
                        unique_file = True
                    else:
                        if ('.nii.gz' not in path) & ('.nii' not in path) & ('.mgz' not in path) & ('.npz' not in path):
                            file_name = os.path.basename(path_images[0]).replace('.nii', '_%s.nii' % suffix)
                            file_name = file_name.replace('.mgz', '_%s.mgz' % suffix)
                            file_name = file_name.replace('.npz', '_%s.npz' % suffix)
                            path = os.path.join(path, file_name)
                        recompute_files = [not os.path.isfile(path)]
                    utils.mkdir(os.path.dirname(path))
                else:
                    recompute_files = [False]
                path = [path]
                return path, recompute_files, unique_file

            # use helper on all outputs
            out_synthseg, recompute_synthseg, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
            out_synthseg33, recompute_synthseg33, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg33')

        recompute_list = [recompute | re_seg for (re_seg, re_seg33) in zip(recompute_synthseg, recompute_synthseg33)]

        return path_images, out_synthseg, out_synthseg33, recompute_list

    def build_unet_model(self, path_model_segmentation, n_labels_seg, n_groups):
        # build first UNet
        net = nrn_models.unet(input_shape=[None, None, None, 1],
                              nb_labels=n_groups,
                              nb_levels=5,
                              nb_conv_per_level=2,
                              conv_size=3,
                              nb_features=24,
                              feat_mult=2,
                              activation='elu',
                              batch_norm=-1,
                              name='unet')

        # transition between the two networks: one_hot -> argmax -> one_hot (it simulates how the network was trained)
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor, name='')

        # build denoiser
        net = nrn_models.unet(input_model=net,
                              input_shape=[None, None, None, 1],
                              nb_labels=n_groups,
                              nb_levels=5,
                              nb_conv_per_level=2,
                              conv_size=5,
                              nb_features=16,
                              feat_mult=2,
                              activation='elu',
                              batch_norm=-1,
                              skip_n_concatenations=2,
                              name='l2l')

        # transition between the two networks: one_hot -> argmax -> one_hot, and concatenate input image and labels
        input_image = net.inputs[0]
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        if n_groups <= 2:
            last_tensor = KL.Lambda(lambda x: x[..., 1:])(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = Model(inputs=net.inputs, outputs=last_tensor)

        # build 2nd network
        net = nrn_models.unet(input_model=net,
                              input_shape=[None, None, None, 2],
                              nb_labels=n_labels_seg,
                              nb_levels=5,
                              nb_conv_per_level=2,
                              conv_size=3,
                              nb_features=24,
                              feat_mult=2,
                              activation='elu',
                              batch_norm=-1,
                              name='unet2')
        net.load_weights(path_model_segmentation, by_name=True)
        return net

    def build_conver_model(self, labels_segmentation, n_labels_seg, net_unet2, ):
        input_image = KL.Input(shape=net_unet2.input.shape[1:])
        input_tensor = KL.Input(shape=net_unet2.output.shape[1:])
        last_tensor1 = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(input_tensor)
        last_tensor1 = layers.ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor1)
        parcellation_masking_values = np.array([1 if ((ll == 3) | (ll == 42)) else 0 for ll in labels_segmentation])
        last_tensor1 = layers.ConvertLabels(labels_segmentation, parcellation_masking_values)(last_tensor1)
        last_tensor1 = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor1)
        last_tensor1 = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor1])
        net_convert = Model(inputs=[input_tensor, input_image], outputs=last_tensor1, name='net_convert')
        return net_convert

    def build_model(self, path_model_segmentation,
                    path_model_parcellation,
                    labels_segmentation,
                    labels_denoiser,
                    labels_parcellation, ):
        assert os.path.isfile(path_model_segmentation), "The provided model path does not exist."

        # get labels
        n_labels_seg = len(labels_segmentation)
        n_groups = len(labels_denoiser)

        # build UNet 2
        net_unet2 = self.build_unet_model(path_model_segmentation=path_model_segmentation,
                                          n_labels_seg=n_labels_seg,
                                          n_groups=n_groups)

        net_unet2.load_weights(path_model_segmentation, by_name=True)

        name_segm_prediction_layer = 'unet2_prediction'
        n_labels_parcellation = len(labels_parcellation)

        net_convert = self.build_conver_model(labels_segmentation=labels_segmentation,
                                              n_labels_seg=n_labels_seg,
                                              net_unet2=net_unet2)

        net_parcellation = nrn_models.unet(  # input_model=net_unet2,
            input_shape=[None, None, None, 3],
            nb_labels=n_labels_parcellation,
            nb_levels=5,
            nb_conv_per_level=2,
            conv_size=3,
            nb_features=24,
            feat_mult=2,
            activation='elu',
            batch_norm=-1,
            name='unet_parc')
        net_parcellation.load_weights(path_model_parcellation, by_name=True)
        # # smooth predictions
        last_tensor = net_parcellation.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = layers.GaussianBlur(sigma=0.5)(last_tensor)
        net_parcellation = Model(inputs=net_parcellation.inputs,
                                 outputs=last_tensor,
                                 name='unet_parc')

        return net_unet2, net_convert, net_parcellation

    def postprocess(self, post_patch_seg, post_patch_parc, shape, pad_idx, crop_idx,
                    labels_segmentation, labels_parcellation, aff, im_res, fast, topology_classes, v1):
        # get posteriors
        post_patch_seg = np.squeeze(post_patch_seg)
        if fast | (topology_classes is None):
            post_patch_seg = edit_volumes.crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)

        # keep biggest connected component
        tmp_post_patch_seg = post_patch_seg[..., 1:]
        post_patch_seg_mask = np.sum(tmp_post_patch_seg, axis=-1) > 0.25
        post_patch_seg_mask = edit_volumes.get_largest_connected_component(post_patch_seg_mask)
        post_patch_seg_mask = np.stack([post_patch_seg_mask] * tmp_post_patch_seg.shape[-1], axis=-1)
        tmp_post_patch_seg = edit_volumes.mask_volume(tmp_post_patch_seg, mask=post_patch_seg_mask, return_copy=False)
        post_patch_seg[..., 1:] = tmp_post_patch_seg

        # reset posteriors to zero outside the largest connected component of each topological class
        if (not fast) & (topology_classes is not None):
            post_patch_seg_mask = post_patch_seg > 0.25
            for topology_class in np.unique(topology_classes)[1:]:
                tmp_topology_indices = np.where(topology_classes == topology_class)[0]
                tmp_mask = np.any(post_patch_seg_mask[..., tmp_topology_indices], axis=-1)
                tmp_mask = edit_volumes.get_largest_connected_component(tmp_mask)
                for idx in tmp_topology_indices:
                    post_patch_seg[..., idx] *= tmp_mask
            post_patch_seg = edit_volumes.crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)
        else:
            post_patch_seg_mask = post_patch_seg > 0.2
            post_patch_seg[..., 1:] *= post_patch_seg_mask[..., 1:]

        # get hard segmentation
        post_patch_seg /= np.sum(post_patch_seg, axis=-1)[..., np.newaxis]
        seg_patch = labels_segmentation[post_patch_seg.argmax(-1).astype('int32')].astype('int32')

        # postprocess parcellation
        if post_patch_parc is not None:
            post_patch_parc = np.squeeze(post_patch_parc)
            post_patch_parc = edit_volumes.crop_volume_with_idx(post_patch_parc, pad_idx, n_dims=3, return_copy=False)
            mask = (seg_patch == 3) | (seg_patch == 42)
            post_patch_parc[..., 0] = np.ones_like(post_patch_parc[..., 0])
            post_patch_parc[..., 0] = edit_volumes.mask_volume(post_patch_parc[..., 0], mask=mask < 0.1,
                                                               return_copy=False)
            post_patch_parc /= np.sum(post_patch_parc, axis=-1)[..., np.newaxis]
            parc_patch = labels_parcellation[post_patch_parc.argmax(-1).astype('int32')].astype('int32')
            seg_patch[mask] = parc_patch[mask]

        # paste patches back to matrix of original image size
        if crop_idx is not None:
            # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
            seg = np.zeros(shape=shape, dtype='int32')
            posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
            posteriors[..., 0] = np.ones(shape)  # place background around patch
            seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
            posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = post_patch_seg
        else:
            seg = seg_patch
            posteriors = post_patch_seg

        # align prediction back to first orientation
        seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
        posteriors = edit_volumes.align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)

        # compute volumes
        volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
        if not v1:
            volumes = np.concatenate([np.array([np.sum(volumes)]), volumes])
        if post_patch_parc is not None:
            volumes_parc = np.sum(post_patch_parc[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
            total_volume_cortex = np.sum(
                volumes[np.where((labels_segmentation == 3) | (labels_segmentation == 42))[0] - 1])
            volumes_parc = volumes_parc / np.sum(volumes_parc) * total_volume_cortex
            volumes = np.concatenate([volumes, volumes_parc])
        volumes = np.around(volumes * np.prod(im_res), 3)

        return seg, posteriors, volumes

    def preprocess(self, path_image, ct, target_res=1., n_levels=5, crop=None, min_pad=None):
        # read image and corresponding info
        im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)
        if n_dims == 2 and 1 < n_channels < 4:
            raise Exception('either the input is 2D with several channels, or is 3D with at most 3 slices. '
                            'Either way, results are going to be poor...')
        elif n_dims == 2 and 3 < n_channels < 11:
            print('warning: input with very few slices')
            n_dims = 3
        elif n_dims < 3:
            raise Exception('input should have 3 dimensions, had %s' % n_dims)
        elif n_dims == 4 and n_channels == 1:
            n_dims = 3
            im = im[..., 0]
        elif n_dims > 3:
            raise Exception('input should have 3 dimensions, had %s' % n_dims)
        elif n_channels > 1:
            print('WARNING: detected more than 1 channel, only keeping the first channel.')
            im = im[..., 0]

        # align image
        im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
        shape = list(im.shape[:n_dims])

        # crop image if necessary
        if crop is not None:
            crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
            crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
            im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
        else:
            crop_idx = None

        # normalise image
        if ct:
            im = np.clip(im, 0, 80)
        im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

        # pad image
        input_shape = im.shape[:n_dims]
        pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
        im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

        # add batch and channel axes
        im = utils.add_axis(im, axis=[0, -1])

        return im, aff, h, im_res, shape, pad_idx, crop_idx

    def load_parameter(self):
        if self.args is None:
            args = {'robust': True, 'parc': True, 'fast': False, 'v1': False, 'ct': False}

            if args['robust']:
                args['path_model_segmentation'] = os.path.join(self.MODEL_DIR, 'synthseg_robust_2.0.h5')
            else:
                args['path_model_segmentation'] = os.path.join(self.MODEL_DIR, 'synthseg_2.0.h5')

            args['path_model_parcellation'] = os.path.join(self.MODEL_DIR, 'synthseg_parc_2.0.h5')
            args['labels_segmentation'] = os.path.join(self.LABELS_DIR, 'synthseg_segmentation_labels_2.0.npy')
            args['labels_denoiser'] = os.path.join(self.LABELS_DIR, 'synthseg_denoiser_labels_2.0.npy')
            args['labels_parcellation'] = os.path.join(self.LABELS_DIR, 'synthseg_parcellation_labels.npy')
            args['names_segmentation_labels'] = os.path.join(self.LABELS_DIR,
                                                             'synthseg_segmentation_labels_2.0_david2.npy')
            args['names_parcellation_labels'] = os.path.join(self.LABELS_DIR, 'synthseg_parcellation_names.npy')
            args['topology_classes'] = os.path.join(self.LABELS_DIR, 'synthseg_topological_classes_2.0.npy')
            args['n_neutral_labels'] = 19
            args['crop'] = 192

            self.args = args
        else:
            pass
        return self.args

    def run(self, path_images, path_segmentations, path_segmentations33):
        args = self.load_parameter()
        path_model_segmentation = args['path_model_segmentation']
        labels_segmentation = args['labels_segmentation']
        robust = args['robust']
        fast = args['fast']
        v1 = args['v1']
        n_neutral_labels = args['n_neutral_labels']
        labels_denoiser = args['labels_denoiser']
        path_model_parcellation = args['path_model_parcellation']
        labels_parcellation = args['labels_parcellation']
        cropping = args['crop']
        ct = args['ct']
        topology_classes = args['topology_classes']
        # get label lists
        labels_segmentation, _ = utils.get_list_labels(label_list=labels_segmentation)
        if (n_neutral_labels is not None) & (not fast) & (not robust):
            labels_segmentation, flip_indices, unique_idx = get_flip_indices(labels_segmentation, n_neutral_labels)
        else:
            labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
            flip_indices = None

        if topology_classes is not None:
            topology_classes = utils.load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]
        labels_denoiser = np.unique(utils.get_list_labels(labels_denoiser)[0])

        labels_parcellation, unique_i_parc = np.unique(utils.get_list_labels(labels_parcellation)[0],
                                                       return_index=True)
        # set cropping/padding
        if cropping is not None:
            cropping = utils.reformat_to_list(cropping, length=3, dtype='int')
            min_pad = cropping
        else:
            min_pad = 128

        if self.net_unet2 is None:
            net_unet2, net_convert, net_parcellation = self.build_model(path_model_segmentation=path_model_segmentation,
                                                                        path_model_parcellation=path_model_parcellation,
                                                                        labels_segmentation=labels_segmentation,
                                                                        labels_denoiser=labels_denoiser,
                                                                        labels_parcellation=labels_parcellation, )
            self.net_unet2 = net_unet2
            self.net_convert = net_convert
            self.net_parcellation = net_parcellation
        else:
            net_unet2 = self.net_unet2
            net_convert = self.net_convert
            net_parcellation = self.net_parcellation

        image, aff, h, im_res, shape, pad_idx, crop_idx = self.preprocess(path_image=path_images,
                                                                          ct=ct,
                                                                          crop=cropping,
                                                                          min_pad=min_pad)

        unet2_output = net_unet2.predict(image)

        parc_input = net_convert.predict([unet2_output,image])
        #parc_input = net_convert.predict([[unet2_output], [image]])
        post_patch_parcellation = net_parcellation.predict(parc_input)

        #post_patch_parcellation = net_parcellation.predict(parc_input[0])

        seg, posteriors, volumes = self.postprocess(post_patch_seg=unet2_output,
                                                    post_patch_parc=post_patch_parcellation,
                                                    shape=shape,
                                                    pad_idx=pad_idx,
                                                    crop_idx=crop_idx,
                                                    labels_segmentation=labels_segmentation,
                                                    labels_parcellation=labels_parcellation,
                                                    aff=aff,
                                                    im_res=im_res,
                                                    fast=fast,
                                                    topology_classes=topology_classes,
                                                    v1=v1)
        utils.save_volume(seg, aff, h, path_segmentations, dtype='int32')

        seg, posteriors, volumes = self.postprocess(post_patch_seg=unet2_output,
                                                    post_patch_parc=None,
                                                    shape=shape,
                                                    pad_idx=pad_idx,
                                                    crop_idx=crop_idx,
                                                    labels_segmentation=labels_segmentation,
                                                    labels_parcellation=labels_parcellation,
                                                    aff=aff,
                                                    im_res=im_res,
                                                    fast=fast,
                                                    topology_classes=topology_classes,
                                                    v1=v1)
        utils.save_volume(seg, aff, h, path_segmentations33, dtype='int32')

    def run_segmentations33(self, path_images, path_segmentations33):
        args = self.load_parameter()
        path_model_segmentation = args['path_model_segmentation']
        labels_segmentation = args['labels_segmentation']
        robust = args['robust']
        fast = args['fast']
        v1 = args['v1']
        n_neutral_labels = args['n_neutral_labels']
        labels_denoiser = args['labels_denoiser']
        path_model_parcellation = args['path_model_parcellation']
        labels_parcellation = args['labels_parcellation']
        cropping = args['crop']
        ct = args['ct']
        topology_classes = args['topology_classes']
        # get label lists
        labels_segmentation, _ = utils.get_list_labels(label_list=labels_segmentation)
        if (n_neutral_labels is not None) & (not fast) & (not robust):
            labels_segmentation, flip_indices, unique_idx = get_flip_indices(labels_segmentation, n_neutral_labels)
        else:
            labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
            flip_indices = None

        if topology_classes is not None:
            topology_classes = utils.load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]
        labels_denoiser = np.unique(utils.get_list_labels(labels_denoiser)[0])

        labels_parcellation, unique_i_parc = np.unique(utils.get_list_labels(labels_parcellation)[0],
                                                       return_index=True)
        # set cropping/padding
        if cropping is not None:
            cropping = utils.reformat_to_list(cropping, length=3, dtype='int')
            min_pad = cropping
        else:
            min_pad = 128

        if self.net_unet2 is None:
            net_unet2, net_convert, net_parcellation = self.build_model(path_model_segmentation=path_model_segmentation,
                                                                        path_model_parcellation=path_model_parcellation,
                                                                        labels_segmentation=labels_segmentation,
                                                                        labels_denoiser=labels_denoiser,
                                                                        labels_parcellation=labels_parcellation, )
            self.net_unet2 = net_unet2
            self.net_convert = net_convert
            self.net_parcellation = net_parcellation
        else:
            net_unet2 = self.net_unet2

        image, aff, h, im_res, shape, pad_idx, crop_idx = self.preprocess(path_image=path_images,
                                                                          ct=ct,
                                                                          crop=cropping,
                                                                          min_pad=min_pad)

        unet2_output = net_unet2.predict(image)
        seg, posteriors, volumes = self.postprocess(post_patch_seg=unet2_output,
                                                    post_patch_parc=None,
                                                    shape=shape,
                                                    pad_idx=pad_idx,
                                                    crop_idx=crop_idx,
                                                    labels_segmentation=labels_segmentation,
                                                    labels_parcellation=labels_parcellation,
                                                    aff=aff,
                                                    im_res=im_res,
                                                    fast=fast,
                                                    topology_classes=topology_classes,
                                                    v1=v1)
        utils.save_volume(seg, aff, h, path_segmentations33, dtype='int32')


def resample_one(input_file_path, output_file_path):
    # 1. 讀取nii.gz檔為影像
    image = sitk.ReadImage(input_file_path)
    # 2. 將影像的體素塊尺寸Resample為1x1x1並創建新的影像
    new_spacing = [1.0, 1.0, 1.0]  # 目標體素塊尺寸
    original_spacing = image.GetSpacing()  # 原始體素塊尺寸
    original_size = image.GetSize()  # 原始尺寸
    transform = sitk.Transform()
    transform.SetIdentity()
    # 計算Resample的大小
    new_size = [int(sz * spc / new_spc + 0.5) for sz, spc, new_spc in zip(original_size, original_spacing, new_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)  # 使用線性插值器，可以根據需要更改插值方法
    resampler.SetTransform(transform)
    # 進行Resample
    new_image = resampler.Execute(image)
    new_image.SetSpacing(new_spacing)
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection(image.GetDirection())
    # xyzt_units
    new_image.SetMetaData("xyzt_units", image.GetMetaData('xyzt_units'))
    # 保存新的影像
    sitk.WriteImage(new_image, output_file_path)
    return output_file_path


def resample_to_original(resample_file_path, original_file_path, output_file_path):
    # 1. 讀取nii.gz檔為影像
    resample_image = sitk.ReadImage(resample_file_path)
    original_image = sitk.ReadImage(original_file_path)
    # 2. 將影像的體素塊尺寸Resample為1x1x1並創建新的影像
    original_origin = original_image.GetOrigin()
    original_direction = original_image.GetDirection()
    original_spacing = original_image.GetSpacing()
    original_size = original_image.GetSize()
    # 建立一个 Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(original_size)
    resampler.SetOutputSpacing(original_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # 进行 Resample
    resampled_to_original_image = resampler.Execute(resample_image)
    resampled_to_original_image.CopyInformation(original_image)
    sitk.WriteImage(resampled_to_original_image, output_file_path)
    return output_file_path


# def main(args):
#     synth_seg = SynthSeg()
#     synth_seg.run()
#
#
if __name__ == '__main__':

    # test resample_to_original
    root_dir_path = pathlib.Path(r'D:\00_Chen\Task03_\50_\seg')
    original_file_path = root_dir_path.joinpath('rMNI152_T1_1.5mm.nii')
    file_list = list(root_dir_path.glob('*.nii.gz'))

    output_file_list = list(map(lambda x: root_dir_path.joinpath(x.name.replace('.nii.gz','original.nii.gz'))
                                ,file_list))
    for i in range(len(file_list)):
        resample_file_path = file_list[i]
        output_file_path = output_file_list[i]
        resample_to_original(resample_file_path, original_file_path, output_file_path)
