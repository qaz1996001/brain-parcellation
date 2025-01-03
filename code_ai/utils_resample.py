import SimpleITK as sitk


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

