import datetime
import itertools
import pathlib
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy import ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table
import bentoml
import cupy as cp




INFARCT_PATH_CODE         =  '/mnt/e/data/pipeline/chuan/code/'
INFARCT_PATH_PROCESSMODEL = '/mnt/e/data/pipeline/chuan/process/Deep_Infarct/'
INFARCT_PATH_JSON         ='/mnt/e/data/pipeline/chuan/json/'
INFARCT_PATH_LOG          ='/mnt/e/data/pipeline/chuan/log/'



@bentoml.service(
    name="shh-model",
    traffic={
        "timeout": 360,
        "concurrency": 2,
    },
    resources={
        "gpu": 1,
        "cpu": 4,

    },
    workers = 2
)
class Model:

    # cmb_unet_26_model_300
    # cmb_unet_64_model

    BENTOML_MODEL1_TAG = "cmb_unet_64_model"
    BENTOML_MODEL2_TAG = "cmb_unet_26_model_300"
    BENTOML_MODEL3_TAG = "synthsegrobust2_trace"
    BENTOML_MODEL4_TAG = "synthsegparcmodel"

    BENTOML_INFARCT_MODEL_TAG = 'infarct_unet_256_model:5x3btgxm66gzwaav'

    label_index_name_mapping_dict = {0: "Background",
                                     1: "CSF",
                                     301: "brainstem",
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
                                     214: "right cingulate", }
    patch_size = (64, 64, 64)
    blend_mode = "gaussian"
    sigma = 1 / 8  # 0.125
    n_class = 1
    overlap = 0.5
    name_rule = 'file'

    # @bentoml.on_deployment
    # def prepare():
    #     pass

    # @bentoml.on_startup
    # def init_resources(self):
    #     # This runs once per worker

    def __init__(self) -> None:
        from code_ai.utils_synthsegOnnx import SynthSegOnnx
        # self.model1 = bentoml.onnx.get(self.BENTOML_MODEL1_TAG).load_model(providers=['CUDAExecutionProvider'])
        # self.model2 = bentoml.onnx.get(self.BENTOML_MODEL2_TAG).load_model(providers=['CUDAExecutionProvider'])
        # self.model3 = bentoml.onnx.get(self.BENTOML_MODEL3_TAG).load_model(providers=['CUDAExecutionProvider'])
        # self.model4 = bentoml.onnx.get(self.BENTOML_MODEL4_TAG).load_model(providers=['CUDAExecutionProvider'])
        self.synth_seg_model = SynthSegOnnx()
        self.importance_kernel = self.get_importance_kernel(self.patch_size, self.blend_mode, self.sigma)
        self.importance_map = cp.tile(cp.reshape(self.importance_kernel,
                                                 newshape=[1, *self.patch_size, 1]),
                                      [1, 1, 1, 1, self.n_class],)

    @bentoml.api
    async def cmb_classify(self, swan_path_str: str,
                                 temp_path_str: str,
                                 output_nii_path_str: str,
                                 output_json_path_str:str) -> str:
        '''
        Classify input image to label
        '''
        swan_path = pathlib.Path(swan_path_str)
        temp_path = pathlib.Path(temp_path_str)

        image_arr, spacing, aff = self.load_volume(swan_path, dtype='float32')
        target_spacing = np.array([spacing[0], spacing[1], min(spacing)])  # isotropical z-axis only
        target_spacing = target_spacing / 2 if target_spacing[0] > 0.6 else target_spacing  # force upsampling?
        image_arr = self.resize_volume(image_arr, spacing, target_spacing, dtype='float32')
        if temp_path is not None:
            model1 = bentoml.onnx.get(self.BENTOML_MODEL1_TAG).load_model(providers=['CUDAExecutionProvider'])
            model2 = bentoml.onnx.get(self.BENTOML_MODEL2_TAG).load_model(providers=['CUDAExecutionProvider'])
            seg_arr, _ = self.load_synthseg_seg(seg_path=str(temp_path),
                                                target_size=image_arr.shape)
            brain_mask = seg_arr > 0
            image_arr = self.custom_normalize_1(image_arr, brain_mask)

            output = self.sliding_window_inference(inputs=image_arr[np.newaxis, :, :, :, np.newaxis],
                                                   roi_size=self.patch_size, model=model1,
                                                   overlap=self.overlap,
                                                   n_class=self.n_class, importance_map=self.importance_map,
                                                   mask=brain_mask[np.newaxis, :, :, :, np.newaxis], )
            pred_map = cp.asnumpy(cp.squeeze(output)) * brain_mask
            df_pred, pred_label = self.object_analysis(image_arr, pred_map, target_spacing, model2)
            self.save_nii_trio(nifti_file = swan_path, pred_label = pred_label,
                               output_path = output_nii_path_str)
            df_label = self.save_label_table(swan_path, df_pred, pred_label, seg_arr,output_json_path_str)

            del model1, model2
        gc.collect()
        return output_nii_path_str

    @bentoml.api
    async def synthseg_classify(self,
                                path_images:str,
                                path_segmentations:str,
                                path_segmentations33:str):
        print('synthseg_classify path_images',path_images)
        segmentations_path   = pathlib.Path(path_segmentations)
        segmentations33_path = pathlib.Path(path_segmentations33)
        if segmentations_path.exists() and segmentations33_path.exists():
            pass
        else:
            model3 = bentoml.onnx.get(self.BENTOML_MODEL3_TAG).load_model(providers=['CUDAExecutionProvider'])
            model4 = bentoml.onnx.get(self.BENTOML_MODEL4_TAG).load_model(providers=['CUDAExecutionProvider'])
            self.synth_seg_model.run(path_images=path_images, path_segmentations=path_segmentations,
                                     path_segmentations33=path_segmentations33,
                                     net_unet2 = model3,
                                     net_parcellation = model4)
            del model3, model4
        gc.collect()

    @bentoml.api
    async def infarct_classify(self,adc_file:str,
                               dwi0_file:str,
                               dwi1000_file:str,
                               synthseg_file:str,
                               output_path:str):
        from .pipeline_infarct_onnx import pipeline_infarct
        print('infarct_classify adc_file',adc_file)
        print('infarct_classify dwi0_file',dwi0_file)
        print('infarct_classify dwi1000_file',dwi1000_file)
        print('infarct_classify synthseg_file',synthseg_file)
        model = bentoml.onnx.get(self.BENTOML_INFARCT_MODEL_TAG).load_model(providers=['CPUExecutionProvider'])
        adc_path = pathlib.Path(adc_file)
        id = adc_path.parent.name
        pipeline_infarct(ID                = id,
                         ADC_file          = adc_file,
                         DWI0_file         = dwi0_file ,
                         DWI1000_file      = dwi1000_file,
                         SynthSEG_file     = synthseg_file,
                         path_output       = output_path,
                         cuatom_model      = model,
                         path_code         = INFARCT_PATH_CODE,
                         path_processModel = INFARCT_PATH_PROCESSMODEL,
                         path_json         = INFARCT_PATH_JSON,
                         path_log          = INFARCT_PATH_LOG,
                         )

    @bentoml.api
    async def wmh_classify(self, ):
        pass

    @classmethod
    def run_model(cls, sess, input1, importance_map, mode='cmb_unet_64_model'):
        if 'cmb_unet_64_model' == mode:
            if isinstance(input1, cp.ndarray):
                np_input1 = cp.asnumpy(input1)
            else:
                np_input1 = input1
            temp = sess.run(["predictions"], {"input_1": np_input1})[0]
            return cp.round(cp.asarray(temp) * importance_map, 6)
            # return cp.asarray(temp) * importance_map
        if 'cmb_unet_26_model_300' == mode:
            if isinstance(input1, cp.ndarray):
                np_input1 = cp.asnumpy(input1)
            else:
                np_input1 = input1
            temp = sess.run(["prob_out"], {"input_1": np_input1})[0]
            return np.round(temp, 6)
        return input1


    @classmethod
    def sliding_window_inference(cls, inputs, roi_size, model, overlap, n_class, importance_map, strategy="overlap_inside",
                                 mask=None, **kwargs):
        image_size = tuple(inputs.shape[1:-1])
        roi_size = tuple(roi_size)

        # Padding to ensure image size is at least roi_size
        padded_image_size = tuple(max(image_size[i], roi_size[i]) for i in range(3))
        padding_size = [image_x - input_x for image_x, input_x in zip(image_size, padded_image_size)]
        paddings = [(0, 0)] + [(x // 2, x - x // 2) for x in padding_size] + [(0, 0)]

        if isinstance(inputs,np.ndarray):
            inputs = cp.asarray(inputs)
        input_padded = cp.pad(inputs, paddings)
        if mask is not None:
            mask_padded = cp.pad(cp.asarray(mask, dtype=cp.bool_), paddings)

        output_shape = (1, *padded_image_size, n_class)
        output_sum = cp.zeros(output_shape, dtype=cp.float32)
        output_weight_sum = cp.ones(output_shape, dtype=cp.float32)
        window_slices = cls.get_window_slices(padded_image_size, roi_size, overlap, strategy)
        for index, window_slice in enumerate(window_slices):
            start_indices = tuple(ws for ws in window_slice[0])
            end_indices = list(window_slice[0][ws] + window_slice[1][ws] for ws in range(len(start_indices)))
            end_indices = tuple(end_indices)
            if mask is None or cp.any(mask_padded[start_indices[0]:end_indices[0],
                                      start_indices[1]:end_indices[1],
                                      start_indices[2]:end_indices[2],
                                      start_indices[3]:end_indices[3]
                                      ]):
                window = input_padded[start_indices[0]:end_indices[0],
                         start_indices[1]:end_indices[1],
                         start_indices[2]:end_indices[2],
                         start_indices[3]:end_indices[3]]
                pred = cls.run_model(model, window, importance_map)
                padding = [(start, output_size - (start + size)) for start, size, output_size in
                           zip(*window_slice, output_shape)]
                padding = padding[:-1] + [(0, 0)]
                output_sum += cp.pad(pred, padding)
                output_weight_sum += cp.pad(importance_map, padding)
        output = output_sum / output_weight_sum
        crop_slice = tuple(slice(pad[0], pad[0] + input_x) for pad, input_x in zip(paddings, inputs.shape[:-1]))
        return output[crop_slice]
    @classmethod
    def load_volume(cls, path_volume, im_only=False, squeeze=True, dtype=None, LPS_coor=True):
        """
        Load volume file.
        :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
        If npz format, 1) the variable name is assumed to be 'vol_data',
        2) the volume is associated with an identity affine matrix and blank header.
        :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
        :param squeeze: (optional) whether to squeeze the volume when loading.
        :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
        The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
        :return: the volume, with corresponding affine matrix and header if im_only is False.
        """
        path_volume = str(path_volume)
        assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

        if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
            x = nib.load(path_volume)
            x = nib.as_closest_canonical(x)  # to RAS space
            if squeeze:
                volume = np.squeeze(x.get_fdata())
            else:
                volume = x.get_fdata()
            aff = x.affine
            header = x.header
            spacing = list(x.header.get_zooms())
        else:  # npz
            volume = np.load(path_volume)['vol_data']
            if squeeze:
                volume = np.squeeze(volume)
            aff = np.eye(4)
            header = nib.Nifti1Header()
            spacing = [1., 1., 1.]
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
        if LPS_coor:
            volume = volume[::-1, ::-1, :]

        if im_only:
            return volume
        else:
            return volume, spacing, aff

    @classmethod
    def resize_volume(cls, arr, spacing=None, target_spacing=None, target_size=None, dtype='float32'):
        order = 1 if 'float' in dtype else 0
        if (spacing is not None) and (target_spacing is not None):
            scale = np.array(spacing) / np.array(target_spacing)
            out_vol = ndi.zoom(arr, zoom=scale, order=order, prefilter=True, grid_mode=False)
        elif target_size is not None:
            scale = np.array(target_size) / np.array(arr.shape)
            out_vol = ndi.zoom(arr, zoom=scale, output=np.zeros(target_size, dtype=dtype), order=order, prefilter=False,
                               grid_mode=False)
        return out_vol.astype(dtype)

    @classmethod
    def load_synthseg_seg(cls, seg_path='synthseg33.nii.gz', target_size=None, verbose=False):
        # Load synthseg result
        x = nib.load(seg_path)
        x = nib.as_closest_canonical(x)  # to RAS space
        spacing = list(x.header.get_zooms())
        aff = x.affine
        seg_arr = x.get_fdata().astype('uint16')[::-1, ::-1, :]  # RAS to LPS orientation

        if target_size != None:
            seg_arr = cls.resize_volume(seg_arr, target_size=target_size, dtype='uint16')
            spacing = None

        return seg_arr, spacing

    @classmethod
    def custom_normalize_1(cls, volume, mask=None, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5,
                           use_positive_only=True):
        """This function linearly rescales a volume between new_min and new_max.
        :param volume: a numpy array
        :param new_min: (optional) minimum value for the rescaled image.
        :param new_max: (optional) maximum value for the rescaled image.
        :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
        where 0 = np.min
        :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
        where 100 = np.max
        :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
        :return: rescaled volume
        """
        # select intensities
        new_volume = volume.copy()
        new_volume = new_volume.astype("float32")
        if (mask is not None) and (use_positive_only):
            intensities = new_volume[mask].ravel()
            intensities = intensities[intensities > 0]
        elif mask is not None:
            intensities = new_volume[mask].ravel()
        elif use_positive_only:
            intensities = new_volume[new_volume > 0].ravel()
        else:
            intensities = new_volume.ravel()

        # define min and max intensities in original image for normalisation
        robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
        robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

        # trim values outside range
        new_volume = np.clip(new_volume, robust_min, robust_max)

        # rescale image
        if robust_min != robust_max:
            return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
        else:  # avoid dividing by zero
            return np.zeros_like(new_volume)

    @classmethod
    def get_histogram_xy(cls, vol_arr, brain_mask):
        hist, bin_edges = np.histogram(vol_arr[brain_mask].ravel(), bins=100)
        bins_mean = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(100)]
        return [bins_mean, hist]

    # @title Model1 sliding_window_inference
    # sliding_window functions (https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/nnUNet/models/sliding_window.py)
    @classmethod
    def get_window_slices(cls, image_size, roi_size, overlap, strategy):
        dim_starts = []
        for image_x, roi_x in zip(image_size, roi_size):
            interval = roi_x if roi_x == image_x else int(roi_x * (1 - overlap))
            starts = list(range(0, image_x - roi_x + 1, interval))
            if strategy == "overlap_inside" and starts[-1] + roi_x < image_x:
                starts.append(image_x - roi_x)
            dim_starts.append(starts)
        slices = [(starts + (0,), roi_size + (-1,)) for starts in itertools.product(*dim_starts)]
        batched_window_slices = [((0,) + start, (1,) + roi_size) for start, roi_size in slices]
        return batched_window_slices

    @classmethod
    def gaussian_kernel(cls, roi_size, sigma):
        gauss = signal.windows.gaussian(roi_size[0], std=sigma * roi_size[0])
        for s in roi_size[1:]:
            gauss = np.outer(gauss, signal.windows.gaussian(s, std=sigma * s))
        gauss = np.reshape(gauss, roi_size)
        gauss = np.power(gauss, 1 / len(roi_size))
        gauss /= gauss.max()
        return cp.asarray(gauss)

    @classmethod
    def get_importance_kernel(cls, roi_size, blend_mode, sigma):
        if blend_mode == "constant":
            return cp.ones(roi_size, dtype=cp.float32)
        elif blend_mode == "gaussian":
            return cls.gaussian_kernel(roi_size, sigma)
        else:
            raise ValueError(f'Invalid blend mode: {blend_mode}. Use either "constant" or "gaussian".')


    @classmethod
    def center_crop(cls, volume, centroid, size=26):
        """ center crop a cube with desired size,
        if the target area is smaller than the desired size, it will be padded with the minimum
        input:: volume is a 3d-array
        input:: centroid is center coordinates (x, y, z)
        output:: cropped cube with desired size ex.(28, 28, 28)
        """
        w, h, d = volume.shape
        r = int(size // 2)
        x0 = int(centroid[0])
        y0 = int(centroid[1])
        z0 = int(centroid[2])
        if (r < x0 < w - r - 1) and (r < y0 < h - r - 1) and (r < z0 < d - r - 1):
            return volume[x0 - r:x0 - r + size, y0 - r:y0 - r + size, z0 - r:z0 - r + size]
        else:
            volume = np.pad(volume, ((r, r), (r, r), (r, r)), 'minimum')
            return volume[x0:x0 + size, y0:y0 + size, z0:z0 + size]

    @classmethod
    def object_analysis(cls, image_arr, pred_map, spacing, model, min_th=0.084, FP_reduction_th=0.357, uncertain_th=0.5175):
        pred_label = label(pred_map > 0.05)
        # pred_label = get_watershed_label(pred_map, 0.05)  # separate connected labels
        # gaussian = gaussian_kernel(roi_size=(26, 26, 26), sigma=1/8).numpy().astype('float16')
        gaussian = cls.gaussian_kernel(roi_size=(26, 26, 26), sigma=1 / 8)
        if (pred_label.max() > 0) and (FP_reduction_th > 0):
            # FP_reduction model predict object by object
            props = regionprops(pred_label)
            TP_conf = []
            for p in props:
                x0, y0, z0, x1, y1, z1 = p.bbox
                center = [(x0 + x1) // 2, (y0 + y1) // 2, (z0 + z1) // 2]
                # crop_image = center_crop(image_arr, center).astype('float16')
                crop_image = cp.asarray(cls.center_crop(image_arr, center))
                x = cp.stack([crop_image, gaussian], axis=-1, dtype=cp.float32)[cp.newaxis, ...]
                TP_conf.append(cls.run_model(model, x, importance_map=None, mode='cmb_unet_26_model_300'))
                # TP_conf.append(model(x, training=False).numpy()[0][0])  # <-- model2 prediction
                print(".", end="")
            TP_conf = np.squeeze(np.stack(TP_conf).astype('float32') / 2)  # covert to [0.0:1.0]
            pred_type = ['CMB' if x * 2 > uncertain_th else 'Uncertain' for x in TP_conf]
            print("。")
        else:
            TP_conf = np.ones((pred_label.max(),), dtype='float32')
            pred_type = 'other'
        # pred measurement
        props = regionprops_table(pred_label, pred_map, properties=('label', 'bbox', 'intensity_mean'))

        df_pred = pd.DataFrame({'ori_Pred_label': props['label'],
                                'Pred_diameter': ((props['bbox-3'] - props['bbox-0']) * spacing[0] + (
                                            props['bbox-4'] - props['bbox-1']) * spacing[1]) / 2,
                                'Pred_mean': props['intensity_mean'] / 0.6,
                                'TP_conf': TP_conf, 'CMB_prob': (props['intensity_mean'] / 0.6 + TP_conf) / 2,
                                'Pred_type': pred_type})
        # sort and false-positive filtering
        df_pred = df_pred[(df_pred['CMB_prob'] > min_th)]  # filter too small CMB_prob
        df_pred = df_pred.sort_values(by='CMB_prob', ascending=False)
        df_pred['Pred_label'] = np.arange(1, df_pred.shape[0] + 1)
        df_pred.loc[df_pred['CMB_prob'] < FP_reduction_th, 'Pred_type'] = 'other'
        # remap pred_label array
        new_pred_label = np.zeros_like(pred_label)
        for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
            new_pred_label[pred_label == k] = v
        return df_pred, new_pred_label

    @classmethod
    def save_nii_trio(cls, nifti_file, pred_label,output_path = 'Pred.nii.gz',
                       verbose=False):
        # name output folder
        image_file_name = str(nifti_file).split('/')[-1].split('.')[0]

        ## load original image
        img = nib.load(nifti_file)
        img = nib.as_closest_canonical(img)  # to RAS space
        aff = img.affine
        hdr = img.header
        spacing = tuple(img.header.get_zooms())
        shape = tuple(img.header.get_data_shape())

        ## CMB label-map
        pred_label = cls.resize_volume(pred_label, target_size=shape, dtype='uint16')
        # build NifTi1 image
        pred_label = pred_label[::-1, ::-1, :]  # LPS to RAS orientation
        x = nib.nifti1.Nifti1Image(pred_label, affine=aff)
        label_arr = x.get_fdata().astype('uint16')
        nib.save(x, output_path)
        if verbose:
            print(
                f"size={x.header.get_data_shape()} spacing={x.header.get_zooms()}  {label_arr.dtype}:{label_arr.min()}-{label_arr.max()}")
            print("affine matrix =\n", x.affine)

    @classmethod
    def save_label_table(cls, nifti_file, df_pred, label_arr, seg_arr, output_json_path_str,):
        def get_location(lb):
            counts = np.bincount(seg_arr[label_arr == lb].ravel())
            mode = np.argmax(counts)
            return mode

        # write position
        cols = ['label#', 'class_name',
                'review_class_name', 'type', 'type_name', 'LPS_coordinates', 'pred_diameter',
                # 'cube_shape','mask_in_cube','image_size',
                'CAD_method', 'complete', 'Doctor', 'c-time', 'Reviewer', 'r-time']
        df_tsv = pd.DataFrame(columns=cols)
        for props in regionprops(label_arr):
            lb                                = props.label
            df_tsv.loc[lb, 'label#']          = lb
            df_tsv.loc[lb, 'class_name']      = df_pred.loc[df_pred['Pred_label'] == lb, 'Pred_type'].values[0]
            label_index                       = get_location(lb)
            df_tsv.loc[lb, 'type']            = f"C{label_index}"  # location
            df_tsv.loc[lb, 'type_name']       = cls.label_index_name_mapping_dict.get(label_index, '')
            df_tsv.loc[lb, 'LPS_coordinates'] = ",".join([str(x) for x in props.bbox])
            df_tsv.loc[lb, 'pred_diameter']   = df_pred.loc[df_pred['Pred_label'] == lb, 'Pred_diameter'].values[0]
            # df_tsv.loc[lb, 'cube_shape']    = ",".join([str(x) for x in props.image.shape])
            # df_tsv.loc[lb, 'mask_in_cube']  = ",".join(['1' if x > 0 else '0' for x in props.image.ravel()])
        # df_tsv['image_size']                = ",".join([str(x) for x in label_arr.shape])
        df_tsv['CAD_method'] = "AI_v3"
        df_tsv['Doctor']     = "AI"
        df_tsv['c-time']     = datetime.datetime.now().strftime('%Y/%m/%d %H:%M')

        with open(output_json_path_str, 'w') as f:
            df_tsv.to_json(path_or_buf = f,
                           orient='records',
                           index=False)
        # df_tsv.to_csv(f"{out_root}/{case_name}_label.tsv", index=False, sep='\t')
        return df_tsv