import os

import nibabel as nib
import nibabel.processing
import SimpleITK as sitk
import pathlib

import numpy as np


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

#
def resampleSynthSEG2original_(path, case_name, series, SynthSEGmodel):
    #先讀取原始影像並抓出SpacingBetweenSlices跟PixelSpacing
    img_nii = nib.load(os.path.join(path, case_name, case_name + '_' + series + '.nii.gz')) #256*256*22
    img_array = np.array(img_nii.dataobj)
    img_array = data_translate(img_array, img_nii) #拿來對
    img_1mm_nii = nib.load(os.path.join(path, case_name, case_name + '_' + series + '_resample.nii.gz')) #230*230*140
    img_1mm_array = np.array(img_1mm_nii.dataobj)
    SynthSEG_1mm_nii = nib.load(os.path.join(path, case_name, case_name + '_' + series + '_' + SynthSEGmodel + '.nii.gz')) #230*230*140
    SynthSEG33_1mm_nii = nib.load(os.path.join(path, case_name, case_name + '_' + series + '_synthseg33.nii.gz')) #230*230*140
    david_1mm_nii = nib.load(os.path.join(path, case_name, case_name + '_' + series + '_david.nii.gz')) #230*230*140

    y_i, x_i, z_i = img_array.shape
    y_i1, x_i1, z_i1 = img_1mm_array.shape

    header_img = img_nii.header.copy() #抓出nii header 去算體積
    pixdim_img = header_img['pixdim']  #可以借此從nii的header抓出voxel size
    header_img_1mm = img_1mm_nii.header.copy() #抓出nii header 去算體積
    pixdim_img_1mm = header_img_1mm['pixdim']  #可以借此從nii的header抓出voxel size

    #先把影像從230*230*140轉成 original*original*140
    img_1mm_ori_nii = nibabel.processing.conform(img_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 1) #影像用，256*256*140
    img_1mm_ori = np.array(img_1mm_ori_nii.dataobj)
    img_1mm_ori = data_translate(img_1mm_ori, img_1mm_ori_nii)

    #再將SynthSEG從230*230*140轉成 original*original*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 0) #256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)
    SynthSEG33_1mm_ori_nii = nibabel.processing.conform(SynthSEG33_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 0) #256*256*140
    SynthSEG33_1mm_ori = np.array(SynthSEG33_1mm_ori_nii.dataobj)
    SynthSEG33_1mm_ori = data_translate(SynthSEG33_1mm_ori, SynthSEG33_1mm_ori_nii)
    david_1mm_ori_nii = nibabel.processing.conform(david_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 0) #256*256*140
    david_1mm_ori = np.array(david_1mm_ori_nii.dataobj)
    david_1mm_ori = data_translate(david_1mm_ori, david_1mm_ori_nii)

    #以下將1mm重新組回otiginal
    new_array = np.zeros(img_array.shape)
    new_array33 = np.zeros(img_array.shape)
    new_arraydavid = np.zeros(img_array.shape)
    img_repeat = np.expand_dims(img_array, -1).repeat(z_i1, axis=-1)
    img_1mm_repeat = np.expand_dims(img_1mm_ori, 2).repeat(z_i, axis=2)
    diff = np.sum(np.abs(img_1mm_repeat - img_repeat), axis = (0,1))
    argmin = np.argmin(diff, axis=1)
    new_array = SynthSEG_1mm_ori[:,:,argmin]
    new_array33 = SynthSEG33_1mm_ori[:,:,argmin]
    new_arraydavid = david_1mm_ori[:,:,argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    nib.save(new_SynthSeg_nii, os.path.join(path, case_name, case_name + '_' + series + '_' + SynthSEGmodel + '_to' + series + '.nii.gz'))
    new_array33_save = data_translate_back(new_array33, img_nii)
    new_SynthSeg33_nii = nii_img_replace(img_nii, new_array33_save)
    nib.save(new_SynthSeg33_nii, os.path.join(path, case_name, case_name + '_' + series + '_synthseg33_to' + series + '.nii.gz'))
    new_arraydavid_save = data_translate_back(new_arraydavid, img_nii)
    new_arraydavid_save_nii = nii_img_replace(img_nii, new_arraydavid_save)
    nib.save(new_arraydavid_save_nii, os.path.join(path, case_name, case_name + '_' + series + '_david_to' + series + '.nii.gz'))

    return new_array


def resampleSynthSEG2original_0204(raw_file:pathlib.Path,
                              resample_image_file:pathlib.Path,
                              resample_seg_file:pathlib.Path):
    # 會使用到的一些predict技巧
    img_nii = nib.load(str(raw_file))  # 256*256*22
    img_array = np.array(img_nii.dataobj)

    img_1mm_nii = nib.load(resample_image_file)  # 230*230*140
    img_1mm_array = np.array(img_1mm_nii.dataobj)

    SynthSEG_1mm_nii = nib.load(resample_seg_file)  # 230*230*140

    y_i, x_i, z_i = img_array.shape
    y_i1, x_i1, z_i1 = img_1mm_array.shape


    header_img = img_nii.header.copy()  # 抓出nii header 去算體積
    pixdim_img = header_img['pixdim']  # 可以借此從nii的header抓出voxel size
    header_img_1mm = img_1mm_nii.header.copy()  # 抓出nii header 去算體積
    pixdim_img_1mm = header_img_1mm['pixdim']  # 可以借此從nii的header抓出voxel size

    # 先把影像從230*230*140轉成 original*original*140
    img_1mm_ori_nii = nibabel.processing.conform(img_1mm_nii,
                                                 ((y_i, x_i, z_i1)),
                                                 (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),
                                                 order=1)  # 影像用，256*256*140
    img_1mm_ori = np.array(img_1mm_ori_nii.dataobj)

    # 再將SynthSEG從230*230*140轉成 original*original*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii,
                                                      ((y_i, x_i, z_i1)),
                                                      (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3])
                                                      , order=0)  # 256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)
    img_repeat = np.expand_dims(img_array, -1).repeat(z_i1, axis=-1)
    img_1mm_repeat = np.expand_dims(img_1mm_ori, 2).repeat(z_i, axis=2)

    diff = np.sum(np.abs(img_1mm_repeat - img_repeat), axis=(0, 1))
    argmin = np.argmin(diff, axis=1)
    new_array = SynthSEG_1mm_ori[:, :, argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    original_seg_file = resample_seg_file.parent.joinpath(f"synthseg_{resample_seg_file.name.replace('resample', 'original')}")
    nib.save(new_SynthSeg_nii,original_seg_file)
    return original_seg_file


def resampleSynthSEG2original(raw_file: pathlib.Path,
                              resample_image_file: pathlib.Path,
                              resample_seg_file: pathlib.Path):
    img_nii, img_array, y_i, x_i, z_i, header_img, pixdim_img = get_volume_info(str(raw_file))
    img_1mm_nii, img_1mm_array, y_i1, x_i1, z_i1, header_img_1mm, pixdim_img_1mm = get_volume_info(str(resample_image_file))

    # 先把影像從230*230*140轉成 original*original*140
    img_1mm_ori_nii = nibabel.processing.conform(img_1mm_nii,
                                                 ((y_i, x_i, z_i1)),
                                                 (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),
                                                 order=1)  # 影像用，256*256*140
    img_1mm_ori = np.array(img_1mm_ori_nii.dataobj)

    if z_i < 64:
        img_repeat = np.expand_dims(img_array, -1).repeat(z_i1, axis=-1)
        img_1mm_repeat = np.expand_dims(img_1mm_ori, 2).repeat(z_i, axis=2)
        diff = np.sum(np.abs(img_1mm_repeat - img_repeat), axis=(0, 1))
        argmin = np.argmin(diff, axis=1)
    else:
        argmin_list = []
        broadcast_shape = img_1mm_ori.shape
        for z_index in range(z_i):
            img_array_broadcast = np.broadcast_to(np.expand_dims(img_array[:, :, z_index], -1), broadcast_shape)
            diff = np.sum(np.abs(img_array_broadcast - img_1mm_ori), axis=(0, 1))
            argmin_list.append(np.argmin(diff))
        argmin = np.array(argmin_list)

    SynthSEG_1mm_nii = nib.load(str(resample_seg_file))  # 230*230*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii,
                                                      ((y_i, x_i, z_i1)),
                                                      (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3])
                                                      , order=0)  # 256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)

    new_array = SynthSEG_1mm_ori[:, :, argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    original_seg_file = resample_seg_file.parent.joinpath(
        f"synthseg_{resample_seg_file.name.replace('resample', 'original')}")
    nib.save(new_SynthSeg_nii, original_seg_file)
    return original_seg_file


def data_translate(img, nii):
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    header = nii.header.copy()  # 抓出nii header 去算體積
    pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
        # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

def data_translate_back(img, nii):
    header = nii.header.copy()  # 抓出nii header 去算體積
    pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img


# nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii


def get_volume_info(file_path_str:str):
    img_nii = nib.load(file_path_str)  # 256*256*22
    img_array = np.array(img_nii.dataobj)
    y_i, x_i, z_i = img_array.shape
    header_img = img_nii.header.copy()  # 抓出nii header 去算體積
    pixdim_img = header_img['pixdim']  # 可以借此從nii的header抓出voxel size
    return img_nii,img_array,y_i, x_i, z_i,header_img, pixdim_img