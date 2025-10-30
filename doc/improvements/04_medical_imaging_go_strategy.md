# 醫學影像處理 Go 遷移專用策略

## 📋 文件資訊
- **版本**: 1.0.0
- **建立日期**: 2025年9月24日
- **重點**: 保留 NiBabel、SimpleITK 等醫學影像處理庫在 Python
- **策略**: 精確分離 Go 和 Python 職責

## 🎯 醫學影像處理現狀分析

### 發現的關鍵依賴
基於程式碼分析，發現以下關鍵醫學影像處理依賴：

#### Python 專用醫學影像庫使用情況
| 庫名 | 使用檔案數 | 主要功能 | 遷移難度 |
|------|-----------|---------|----------|
| **NiBabel** | 15+ 檔案 | NIfTI 檔案讀寫、座標轉換 | 🔴 極高 |
| **SimpleITK** | 12+ 檔案 | 影像重採樣、配準、濾波 | 🔴 極高 |
| **PyDICOM** | 20+ 檔案 | DICOM 解析、標籤提取 | 🟡 中等 |
| **dcm2niix** | 5+ 檔案 | DICOM→NIfTI 轉換 | 🟢 低 (系統調用) |

#### 具體使用案例分析
```python
# 發現的 NiBabel 核心使用模式
# 1. NIfTI 檔案讀寫 (15個檔案)
nifti_obj = nib.load(str(nifti_file_path))
nifti_array = nifti_obj.get_fdata()
out_nib = nib.Nifti1Image(result_array, nifti_obj.affine, nifti_obj.header)
nib.save(out_nib, output_path)

# 2. 座標系統轉換 (8個檔案) 
nifti_obj_axcodes = tuple(nib.aff2axcodes(nifti_obj.affine))
ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

# 3. 影像重新定向 (6個檔案)
data_reoriented = nib.orientations.apply_orientation(data_array, ornt_transf)

# SimpleITK 核心使用模式
# 1. 影像重採樣 (12個檔案)
image = sitk.ReadImage(input_file_path)
resampler = sitk.ResampleImageFilter()
new_image = resampler.Execute(image)
sitk.WriteImage(new_image, output_file_path)

# 2. 影像配準和變換 (5個檔案)
transform = sitk.Transform()
resampler.SetTransform(transform)
```

## 🏗️ 修訂的混合架構設計

### 精確的職責分離
```
┌─────────────────────────────────────────────────────────────┐
│                     Go 微服務層                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   API Gateway   │  │  User Service   │  │  File Service   │  │
│  │    (純 Go)      │  │    (純 Go)      │  │    (純 Go)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ DICOM Metadata  │  │ Task Scheduler  │  │  Cache Service  │  │
│  │ Service (Go)    │  │    (純 Go)      │  │    (純 Go)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Python 醫學影像專用服務                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ NIfTI Processor │  │ Image Resampler │  │ Coordinate Conv │  │
│  │   (NiBabel)     │  │   (SimpleITK)   │  │   (NiBabel)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   AI Inference  │  │ Image Analysis  │  │ Report Generator│  │
│  │ (TensorFlow/PT) │  │ (SciPy/Skimage) │  │   (Matplotlib)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 精確的服務職責劃分

### Go 服務職責 (高效能、並行處理)
```go
// 1. API Gateway Service - 純 Go
type APIGatewayService struct {
    userClient   UserServiceClient
    fileClient   FileServiceClient
    taskClient   TaskServiceClient
    imageClient  ImageProcessingServiceClient // gRPC 到 Python
}

// 2. User Management Service - 純 Go
type UserService struct {
    repo   UserRepository      // GORM
    cache  CacheService       // go-redis
    auth   AuthService        // JWT
    logger *zap.Logger
}

// 3. File Management Service - 純 Go (僅元資料和檔案 I/O)
type FileService struct {
    repo     FileRepository    // GORM
    storage  StorageManager    // 檔案系統操作
    metadata MetadataExtractor // 基礎 DICOM 元資料
    logger   *zap.Logger
}

// 4. Task Scheduling Service - 純 Go
type TaskService struct {
    repo      TaskRepository
    queue     TaskQueue        // Asynq/River
    scheduler JobScheduler
    clients   PythonServiceClients // gRPC 客戶端
}

// 5. DICOM Metadata Service - 部分 Go
type DicomMetadataService struct {
    parser DicomParser        // 基礎解析用 go-dicom
    client ImageServiceClient // 複雜處理調用 Python
}
```

### Python 服務職責 (醫學影像專業處理)
```python
# 1. NIfTI Processing Service - 純 Python
class NiftiProcessingService:
    """NIfTI 檔案專業處理服務"""
    
    def load_nifti(self, file_path: str) -> NiftiData:
        """載入 NIfTI 檔案"""
        nifti_obj = nib.load(file_path)
        return NiftiData(
            data=nifti_obj.get_fdata(),
            affine=nifti_obj.affine,
            header=nifti_obj.header,
            axcodes=tuple(nib.aff2axcodes(nifti_obj.affine))
        )
    
    def save_nifti(self, data: np.ndarray, affine: np.ndarray, 
                   header: nib.Nifti1Header, output_path: str):
        """儲存 NIfTI 檔案"""
        out_nib = nib.Nifti1Image(data, affine, header)
        nib.save(out_nib, output_path)
    
    def reorient_image(self, data: np.ndarray, 
                      from_axcodes: tuple, to_axcodes: tuple) -> np.ndarray:
        """重新定向影像"""
        ornt_init = nib.orientations.axcodes2ornt(from_axcodes)
        ornt_fin = nib.orientations.axcodes2ornt(to_axcodes)
        ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
        return nib.orientations.apply_orientation(data, ornt_transf)

# 2. SimpleITK Processing Service - 純 Python
class SimpleITKProcessingService:
    """SimpleITK 專業影像處理服務"""
    
    def resample_image(self, input_path: str, output_path: str, 
                      new_spacing: List[float] = [1.0, 1.0, 1.0]):
        """影像重採樣"""
        image = sitk.ReadImage(input_path)
        
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # 計算新尺寸
        new_size = [int(sz * spc / new_spc + 0.5) 
                   for sz, spc, new_spc in zip(original_size, original_spacing, new_spacing)]
        
        # 設置重採樣器
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # 執行重採樣
        new_image = resampler.Execute(image)
        sitk.WriteImage(new_image, output_path)
        
        return output_path
    
    def register_images(self, fixed_path: str, moving_path: str, 
                       output_path: str) -> RegistrationResult:
        """影像配準"""
        fixed_image = sitk.ReadImage(fixed_path)
        moving_image = sitk.ReadImage(moving_path)
        
        # 使用 SimpleITK 的配準算法
        registration_method = sitk.ImageRegistrationMethod()
        
        # 設置相似性度量
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-6, numberOfIterations=500
        )
        
        # 執行配準
        transform = registration_method.Execute(fixed_image, moving_image)
        
        # 應用變換
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        
        registered_image = resampler.Execute(moving_image)
        sitk.WriteImage(registered_image, output_path)
        
        return RegistrationResult(
            output_path=output_path,
            transform_parameters=transform.GetParameters(),
            metric_value=registration_method.GetMetricValue()
        )

# 3. AI Inference Service - 純 Python (保留現有)
class AIInferenceService:
    """AI 推理服務 - 保留所有現有功能"""
    
    def __init__(self):
        self.synthseg_model = self.load_synthseg_model()
        self.wmh_model = self.load_wmh_model()
        self.cmb_model = self.load_cmb_model()
    
    def process_synthseg(self, image_path: str) -> SynthSegResult:
        """SynthSeg 腦部分割 - 保留現有實作"""
        # 載入影像 (使用 NiBabel)
        nifti_obj = nib.load(image_path)
        image_data = nifti_obj.get_fdata()
        
        # 執行 SynthSeg
        segmentation = self.synthseg_model.predict(image_data)
        
        # 儲存結果 (使用 NiBabel)
        output_path = image_path.replace('.nii.gz', '_synthseg.nii.gz')
        out_nib = nib.Nifti1Image(segmentation, nifti_obj.affine, nifti_obj.header)
        nib.save(out_nib, output_path)
        
        return SynthSegResult(
            output_path=output_path,
            segmentation_volume=np.sum(segmentation > 0),
            processing_time=time.time() - start_time
        )
```

## 🔗 gRPC 服務介面設計

### 醫學影像專用 gRPC 定義
```protobuf
// medical_imaging.proto
syntax = "proto3";

package medical_imaging;

// NIfTI 處理服務
service NiftiProcessingService {
    rpc LoadNifti(LoadNiftiRequest) returns (LoadNiftiResponse);
    rpc SaveNifti(SaveNiftiRequest) returns (SaveNiftiResponse);
    rpc ReorientImage(ReorientRequest) returns (ReorientResponse);
    rpc GetImageInfo(ImageInfoRequest) returns (ImageInfoResponse);
}

// SimpleITK 處理服務
service SimpleITKProcessingService {
    rpc ResampleImage(ResampleRequest) returns (ResampleResponse);
    rpc RegisterImages(RegistrationRequest) returns (RegistrationResponse);
    rpc ApplyTransform(TransformRequest) returns (TransformResponse);
    rpc FilterImage(FilterRequest) returns (FilterResponse);
}

// AI 推理服務
service AIInferenceService {
    rpc ProcessSynthSeg(SynthSegRequest) returns (SynthSegResponse);
    rpc DetectWMH(WMHRequest) returns (WMHResponse);
    rpc DetectCMB(CMBRequest) returns (CMBResponse);
    rpc DetectAneurysm(AneurysmRequest) returns (AneurysmResponse);
}

// 請求/回應訊息定義
message LoadNiftiRequest {
    string file_path = 1;
    bool load_data = 2;
    bool load_header = 3;
}

message LoadNiftiResponse {
    bool success = 1;
    string error_message = 2;
    NiftiInfo info = 3;
    bytes image_data = 4;  // 序列化的 numpy 陣列
}

message NiftiInfo {
    repeated float affine = 1;      // 4x4 仿射矩陣
    repeated int32 shape = 2;       // 影像形狀
    repeated float spacing = 3;     // 體素間距
    repeated string axcodes = 4;    // 座標軸代碼
    map<string, string> header_info = 5; // 標頭資訊
}

message ResampleRequest {
    string input_path = 1;
    string output_path = 2;
    repeated float new_spacing = 3;
    string interpolation = 4;  // "linear", "nearest", "cubic"
}

message ResampleResponse {
    bool success = 1;
    string error_message = 2;
    string output_path = 3;
    ResampleInfo info = 4;
}

message ResampleInfo {
    repeated int32 original_size = 1;
    repeated int32 new_size = 2;
    repeated float original_spacing = 3;
    repeated float new_spacing = 4;
}
```

## 🔧 Go 服務實作 (簡化版)

### DICOM 元資料服務 (Go)
```go
// Go 只處理基礎 DICOM 解析和元資料提取
package dicom

import (
    "fmt"
    "github.com/suyashkumar/dicom"
    "github.com/suyashkumar/dicom/pkg/tag"
)

type DicomMetadataService struct {
    imageClient ImageProcessingServiceClient // gRPC 到 Python
}

// 基礎 DICOM 元資料提取 (Go)
func (s *DicomMetadataService) ExtractBasicMetadata(filePath string) (*BasicDicomInfo, error) {
    dataset, err := dicom.ParseFile(filePath, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to parse DICOM: %w", err)
    }
    
    info := &BasicDicomInfo{
        FilePath: filePath,
    }
    
    // 安全提取基礎標籤
    info.PatientID = s.safeGetStringTag(dataset, tag.PatientID, "UNKNOWN")
    info.StudyInstanceUID = s.safeGetStringTag(dataset, tag.StudyInstanceUID, "")
    info.SeriesInstanceUID = s.safeGetStringTag(dataset, tag.SeriesInstanceUID, "")
    info.SeriesDescription = s.safeGetStringTag(dataset, tag.SeriesDescription, "")
    info.Modality = s.safeGetStringTag(dataset, tag.Modality, "")
    info.ProtocolName = s.safeGetStringTag(dataset, tag.ProtocolName, "")
    
    // REFORMATTED 檢測 (Go)
    info.IsReformatted, info.ReformattedConfidence = s.detectReformatted(info)
    
    return info, nil
}

// 複雜影像處理委託給 Python
func (s *DicomMetadataService) ProcessDicomImage(ctx context.Context, 
                                                filePath string,
                                                options ProcessingOptions) (*ImageProcessingResult, error) {
    // 調用 Python 服務進行複雜處理
    req := &pb.DicomProcessingRequest{
        FilePath: filePath,
        Options: &pb.ProcessingOptions{
            ConvertToNifti:    options.ConvertToNifti,
            ResampleSpacing:   options.ResampleSpacing,
            ReorientAxes:      options.ReorientAxes,
            ExtractBrainMask:  options.ExtractBrainMask,
        },
    }
    
    return s.imageClient.ProcessDicomImage(ctx, req)
}

func (s *DicomMetadataService) safeGetStringTag(dataset dicom.Dataset, tagID tag.Tag, defaultValue string) string {
    elem, err := dataset.FindElementByTag(tagID)
    if err != nil || elem.Value == nil || len(elem.Value) == 0 {
        return defaultValue
    }
    
    if str, ok := elem.Value[0].GetValue().(string); ok {
        return str
    }
    
    return defaultValue
}
```

### 檔案服務 (Go) - 僅處理檔案 I/O 和元資料
```go
// 檔案服務只負責檔案管理，不處理影像內容
type FileService struct {
    repo          FileRepository
    storage       StorageManager
    dicomMetadata DicomMetadataService
    imageClient   ImageProcessingServiceClient // gRPC 到 Python
}

func (s *FileService) ProcessUploadedFile(ctx context.Context, 
                                        file multipart.File,
                                        header *multipart.FileHeader,
                                        userID uint) (*File, error) {
    // 1. 基礎檔案處理 (Go)
    fileInfo, err := s.saveFileToStorage(file, header)
    if err != nil {
        return nil, err
    }
    
    // 2. 檔案類型檢測 (Go)
    fileType := s.detectFileType(header.Filename, fileInfo.Checksum)
    
    // 3. 建立檔案記錄 (Go)
    fileRecord := &File{
        Filename:         header.Filename,
        OriginalFilename: header.Filename,
        FileType:         fileType,
        FileSize:         header.Size,
        FilePath:         fileInfo.StoragePath,
        Checksum:         fileInfo.Checksum,
        Status:           UploadedStatus,
        CreatedBy:        userID,
        UploadedAt:       timePtr(time.Now()),
    }
    
    // 4. 如果是 DICOM，提取基礎元資料 (Go)
    if fileType == DicomFileType {
        metadata, err := s.dicomMetadata.ExtractBasicMetadata(fileInfo.StoragePath)
        if err != nil {
            s.logger.Warn("Failed to extract DICOM metadata", zap.Error(err))
        } else {
            fileRecord.PatientID = &metadata.PatientID
            fileRecord.StudyInstanceUID = &metadata.StudyInstanceUID
            fileRecord.SeriesInstanceUID = &metadata.SeriesInstanceUID
            fileRecord.Modality = &metadata.Modality
            fileRecord.IsReformatted = &metadata.IsReformatted
        }
    }
    
    // 5. 儲存檔案記錄 (Go)
    if err := s.repo.Create(ctx, fileRecord); err != nil {
        s.storage.DeleteFile(fileInfo.StoragePath) // 清理
        return nil, err
    }
    
    // 6. 如果需要影像處理，非同步調用 Python 服務
    if fileType == DicomFileType || fileType == NiftiFileType {
        s.enqueueImageProcessing(fileRecord.ID, fileInfo.StoragePath)
    }
    
    return fileRecord, nil
}

// 委託複雜影像處理給 Python
func (s *FileService) ProcessImageFile(ctx context.Context, fileID uint) error {
    file, err := s.repo.GetByID(ctx, fileID)
    if err != nil {
        return err
    }
    
    // 調用 Python 影像處理服務
    req := &pb.ImageProcessingRequest{
        FileId:   uint32(fileID),
        FilePath: file.FilePath,
        FileType: string(file.FileType),
        Options: &pb.ProcessingOptions{
            ExtractMetadata:   true,
            ConvertToNifti:    file.FileType == DicomFileType,
            GenerateThumbnail: true,
        },
    }
    
    result, err := s.imageClient.ProcessImage(ctx, req)
    if err != nil {
        return fmt.Errorf("image processing failed: %w", err)
    }
    
    // 更新檔案記錄
    if result.Success {
        file.Status = ProcessedStatus
        file.ProcessedAt = timePtr(time.Now())
        if result.Metadata != "" {
            file.Metadata = datatypes.JSON(result.Metadata)
        }
        return s.repo.Update(ctx, file)
    }
    
    return fmt.Errorf("image processing failed: %s", result.ErrorMessage)
}
```

## 🐍 Python 醫學影像處理服務

### 完整的 Python 影像處理服務
```python
# medical_imaging_service.py
import grpc
from concurrent import futures
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pydicom
from pathlib import Path
import json
import time
import logging

import medical_imaging_pb2_grpc
import medical_imaging_pb2 as pb

class MedicalImagingService(
    medical_imaging_pb2_grpc.NiftiProcessingServiceServicer,
    medical_imaging_pb2_grpc.SimpleITKProcessingServiceServicer,
    medical_imaging_pb2_grpc.AIInferenceServiceServicer
):
    """統一的醫學影像處理服務"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 載入 AI 模型
        self.ai_models = self.load_ai_models()
    
    # ===== NiBabel 處理方法 =====
    def LoadNifti(self, request, context):
        """載入 NIfTI 檔案"""
        try:
            file_path = request.file_path
            if not Path(file_path).exists():
                return pb.LoadNiftiResponse(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # 使用 NiBabel 載入
            nifti_obj = nib.load(file_path)
            
            # 建立回應
            response = pb.LoadNiftiResponse(success=True)
            
            # 填入影像資訊
            response.info.affine.extend(nifti_obj.affine.flatten().tolist())
            response.info.shape.extend(nifti_obj.shape)
            response.info.spacing.extend(nifti_obj.header.get_zooms())
            response.info.axcodes.extend(nib.aff2axcodes(nifti_obj.affine))
            
            # 標頭資訊
            header_dict = dict(nifti_obj.header)
            for key, value in header_dict.items():
                if isinstance(value, (str, int, float)):
                    response.info.header_info[key] = str(value)
            
            # 如果需要載入資料
            if request.load_data:
                image_data = nifti_obj.get_fdata()
                response.image_data = image_data.tobytes()
            
            return response
            
        except Exception as e:
            self.logger.error(f"LoadNifti failed: {e}")
            return pb.LoadNiftiResponse(
                success=False,
                error_message=str(e)
            )
    
    def SaveNifti(self, request, context):
        """儲存 NIfTI 檔案"""
        try:
            # 從 bytes 重建 numpy 陣列
            image_data = np.frombuffer(request.image_data, dtype=np.float32)
            image_data = image_data.reshape(request.shape)
            
            # 重建仿射矩陣
            affine = np.array(request.affine).reshape(4, 4)
            
            # 建立 NIfTI 影像
            nifti_img = nib.Nifti1Image(image_data, affine)
            
            # 儲存檔案
            nib.save(nifti_img, request.output_path)
            
            return pb.SaveNiftiResponse(
                success=True,
                output_path=request.output_path
            )
            
        except Exception as e:
            return pb.SaveNiftiResponse(
                success=False,
                error_message=str(e)
            )
    
    # ===== SimpleITK 處理方法 =====
    def ResampleImage(self, request, context):
        """影像重採樣"""
        try:
            # 載入影像
            image = sitk.ReadImage(request.input_path)
            
            # 設置重採樣參數
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            new_spacing = list(request.new_spacing)
            
            # 計算新尺寸
            new_size = [int(sz * spc / new_spc + 0.5) 
                       for sz, spc, new_spc in zip(original_size, original_spacing, new_spacing)]
            
            # 建立重採樣器
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            
            # 設置插值方法
            interpolation_map = {
                "linear": sitk.sitkLinear,
                "nearest": sitk.sitkNearestNeighbor,
                "cubic": sitk.sitkBSpline
            }
            interpolator = interpolation_map.get(request.interpolation, sitk.sitkLinear)
            resampler.SetInterpolator(interpolator)
            
            # 執行重採樣
            resampled_image = resampler.Execute(image)
            
            # 儲存結果
            sitk.WriteImage(resampled_image, request.output_path)
            
            return pb.ResampleResponse(
                success=True,
                output_path=request.output_path,
                info=pb.ResampleInfo(
                    original_size=original_size,
                    new_size=new_size,
                    original_spacing=original_spacing,
                    new_spacing=new_spacing
                )
            )
            
        except Exception as e:
            return pb.ResampleResponse(
                success=False,
                error_message=str(e)
            )
    
    # ===== AI 推理方法 (保留現有實作) =====
    def ProcessSynthSeg(self, request, context):
        """SynthSeg 腦部分割"""
        try:
            start_time = time.time()
            
            # 使用現有的 SynthSeg 實作
            image_path = request.image_path
            
            # 載入影像 (NiBabel)
            nifti_obj = nib.load(image_path)
            image_data = nifti_obj.get_fdata()
            
            # 執行 SynthSeg (保留現有算法)
            segmentation = self.ai_models['synthseg'].predict(image_data)
            
            # 儲存結果 (NiBabel)
            output_path = image_path.replace('.nii.gz', '_synthseg.nii.gz')
            out_nib = nib.Nifti1Image(segmentation, nifti_obj.affine, nifti_obj.header)
            nib.save(out_nib, output_path)
            
            processing_time = time.time() - start_time
            
            return pb.SynthSegResponse(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                segmentation_volume=float(np.sum(segmentation > 0))
            )
            
        except Exception as e:
            self.logger.error(f"SynthSeg processing failed: {e}")
            return pb.SynthSegResponse(
                success=False,
                error_message=str(e)
            )
    
    def load_ai_models(self):
        """載入所有 AI 模型"""
        models = {}
        
        # 載入 SynthSeg 模型 (保留現有實作)
        try:
            from code_ai.SynthSeg.predict import SynthSeg
            models['synthseg'] = SynthSeg()
        except Exception as e:
            self.logger.error(f"Failed to load SynthSeg: {e}")
        
        # 載入其他模型...
        
        return models

def serve():
    """啟動 gRPC 服務器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # 註冊服務
    service = MedicalImagingService()
    medical_imaging_pb2_grpc.add_NiftiProcessingServiceServicer_to_server(service, server)
    medical_imaging_pb2_grpc.add_SimpleITKProcessingServiceServicer_to_server(service, server)
    medical_imaging_pb2_grpc.add_AIInferenceServiceServicer_to_server(service, server)
    
    # 啟動服務器
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    print(f"Medical Imaging Service starting on {listen_addr}")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
```

## 🔄 任務調度服務 (Go)

### Go 任務調度器 - 調用 Python 處理
```go
// TaskService 只負責調度，實際處理委託給 Python
type TaskService struct {
    repo        TaskRepository
    queue       TaskQueue
    imageClient ImageProcessingServiceClient
    logger      *zap.Logger
}

func (s *TaskService) CreateProcessingTask(ctx context.Context, req CreateTaskRequest) (*ProcessingTask, error) {
    // 1. 驗證輸入 (Go)
    if len(req.InputFileIDs) == 0 {
        return nil, ErrNoInputFiles
    }
    
    // 2. 建立任務記錄 (Go)
    task := &ProcessingTask{
        TaskType:   req.TaskType,
        Status:     PendingStatus,
        InputFiles: datatypes.JSON(req.InputFileIDs),
        Parameters: datatypes.JSON(req.Parameters),
        CreatedBy:  req.CreatedBy,
    }
    
    if err := s.repo.Create(ctx, task); err != nil {
        return nil, err
    }
    
    // 3. 加入任務佇列 (Go)
    jobData := TaskJobData{
        TaskID:       task.ID,
        TaskType:     req.TaskType,
        InputFileIDs: req.InputFileIDs,
        Parameters:   req.Parameters,
    }
    
    if err := s.queue.EnqueueTask(jobData); err != nil {
        return nil, err
    }
    
    return task, nil
}

// 任務執行器 - 調用 Python 服務
func (s *TaskService) ExecuteTask(ctx context.Context, taskID uint) error {
    // 1. 取得任務資訊 (Go)
    task, err := s.repo.GetByID(ctx, taskID)
    if err != nil {
        return err
    }
    
    // 2. 更新狀態為執行中 (Go)
    task.Status = RunningStatus
    task.StartedAt = timePtr(time.Now())
    s.repo.Update(ctx, task)
    
    // 3. 根據任務類型調用對應的 Python 服務
    var result *ProcessingResult
    
    switch task.TaskType {
    case DicomToNiftiTask:
        result, err = s.executeDicomToNifti(ctx, task)
    case BrainSegmentationTask:
        result, err = s.executeBrainSegmentation(ctx, task)
    case WMHDetectionTask:
        result, err = s.executeWMHDetection(ctx, task)
    default:
        err = fmt.Errorf("unsupported task type: %s", task.TaskType)
    }
    
    // 4. 更新任務結果 (Go)
    task.CompletedAt = timePtr(time.Now())
    if err != nil {
        task.Status = FailedStatus
        task.ErrorMessage = stringPtr(err.Error())
    } else {
        task.Status = CompletedStatus
        task.ResultData = datatypes.JSON(result.Data)
        task.OutputFiles = datatypes.JSON(result.OutputFiles)
    }
    
    return s.repo.Update(ctx, task)
}

func (s *TaskService) executeDicomToNifti(ctx context.Context, task *ProcessingTask) (*ProcessingResult, error) {
    var inputFileIDs []uint
    json.Unmarshal(task.InputFiles, &inputFileIDs)
    
    var results []string
    for _, fileID := range inputFileIDs {
        // 取得檔案路徑
        file, err := s.fileRepo.GetByID(ctx, fileID)
        if err != nil {
            return nil, err
        }
        
        // 調用 Python DICOM 處理服務
        req := &pb.DicomToNiftiRequest{
            InputPath:  file.FilePath,
            OutputPath: s.generateOutputPath(file.FilePath, "nifti"),
        }
        
        resp, err := s.imageClient.ConvertDicomToNifti(ctx, req)
        if err != nil {
            return nil, err
        }
        
        if !resp.Success {
            return nil, fmt.Errorf("conversion failed: %s", resp.ErrorMessage)
        }
        
        results = append(results, resp.OutputPath)
    }
    
    return &ProcessingResult{
        Data: map[string]interface{}{
            "converted_files": results,
            "conversion_count": len(results),
        },
        OutputFiles: results,
    }, nil
}
```

## 📊 修訂的遷移策略

### 精確的職責分工

#### Go 負責 (高效能、並行、Web 服務)
1. **Web API 層**
   - HTTP 請求處理
   - 認證授權
   - 請求驗證
   - 回應格式化

2. **業務邏輯層**
   - 使用者管理
   - 檔案管理 (元資料)
   - 任務調度
   - 權限控制

3. **資料存取層**
   - 資料庫 CRUD 操作
   - 快取管理
   - 檔案 I/O 操作

4. **基礎服務**
   - 日誌記錄
   - 監控指標
   - 健康檢查
   - 配置管理

#### Python 負責 (醫學影像專業處理)
1. **影像檔案處理**
   - NiBabel: NIfTI 讀寫、座標轉換
   - SimpleITK: 重採樣、配準、濾波
   - PyDICOM: 複雜 DICOM 處理

2. **AI 推理**
   - SynthSeg 腦部分割
   - WMH 檢測
   - CMB 檢測
   - 動脈瘤檢測

3. **科學計算**
   - NumPy 數值計算
   - SciPy 科學算法
   - scikit-image 影像處理
   - matplotlib 視覺化

4. **醫學演算法**
   - 腦部分割算法
   - 影像配準算法
   - 特徵提取算法
   - 統計分析

### 通訊介面設計
```go
// Go 客戶端調用 Python 服務
type MedicalImagingClient struct {
    niftiClient    pb.NiftiProcessingServiceClient
    sitkClient     pb.SimpleITKProcessingServiceClient
    aiClient       pb.AIInferenceServiceClient
    conn           *grpc.ClientConn
}

func (c *MedicalImagingClient) ProcessDicomToNifti(ctx context.Context, 
                                                  dicomPath string) (*ConversionResult, error) {
    // 1. 先用 dcm2niix 進行基礎轉換 (Go 系統調用)
    niftiPath, err := c.runDcm2niix(dicomPath)
    if err != nil {
        return nil, err
    }
    
    // 2. 調用 Python 服務進行後處理
    req := &pb.LoadNiftiRequest{
        FilePath: niftiPath,
        LoadData: true,
        LoadHeader: true,
    }
    
    resp, err := c.niftiClient.LoadNifti(ctx, req)
    if err != nil {
        return nil, fmt.Errorf("failed to process NIfTI: %w", err)
    }
    
    return &ConversionResult{
        NiftiPath: niftiPath,
        ImageInfo: resp.Info,
        Success:   resp.Success,
    }, nil
}

func (c *MedicalImagingClient) runDcm2niix(dicomPath string) (string, error) {
    outputDir := filepath.Dir(dicomPath)
    outputName := strings.TrimSuffix(filepath.Base(dicomPath), filepath.Ext(dicomPath))
    
    cmd := exec.Command("dcm2niix", 
        "-z", "y",                    // 壓縮輸出
        "-f", outputName,             // 輸出檔名
        "-o", outputDir,              // 輸出目錄
        dicomPath)                    // 輸入路徑
    
    output, err := cmd.CombinedOutput()
    if err != nil {
        return "", fmt.Errorf("dcm2niix failed: %v, output: %s", err, output)
    }
    
    niftiPath := filepath.Join(outputDir, outputName+".nii.gz")
    return niftiPath, nil
}
```

## 📋 修訂的實施時程

### 第一階段：Go 基礎服務 (4週)
```go
// 週1-2: 基礎框架
- API Gateway (Gin)
- 使用者服務 (GORM + JWT)
- 檔案服務 (僅元資料)
- 基礎 DICOM 解析 (go-dicom)

// 週3-4: 核心功能
- 任務調度服務
- 快取服務 (go-redis)
- 監控和日誌
- 基本 API 端點
```

### 第二階段：Python 服務分離 (4週)
```python
# 週5-6: Python 服務重構
- 提取 NiBabel 處理邏輯到獨立服務
- 提取 SimpleITK 處理邏輯到獨立服務
- 建立 gRPC 服務介面

# 週7-8: AI 服務整合
- 保留所有現有 AI 模型
- 建立統一的 AI 推理服務
- 實作 gRPC 通訊
```

### 第三階段：整合測試 (2週)
```bash
# 週9-10: 整合和測試
- Go 服務與 Python 服務整合測試
- 效能基準測試
- 功能對等性驗證
- 部署腳本準備
```

## 🐳 容器化部署策略

### Docker Compose 配置
```yaml
# docker-compose-medical.yml
version: '3.8'

services:
  # Go 服務
  api-gateway:
    build:
      context: ./go-services
      dockerfile: Dockerfile.api-gateway
    ports:
      - "8080:8080"
    environment:
      - PYTHON_IMAGING_SERVICE_URL=python-imaging:50051
    depends_on:
      - postgres
      - redis
      - python-imaging
  
  user-service:
    build:
      context: ./go-services
      dockerfile: Dockerfile.user-service
    environment:
      - DATABASE_URL=postgres://user:pass@postgres:5432/medical_db
      - REDIS_URL=redis://redis:6379
  
  file-service:
    build:
      context: ./go-services
      dockerfile: Dockerfile.file-service
    volumes:
      - medical_data:/data
    environment:
      - STORAGE_PATH=/data
      - PYTHON_IMAGING_SERVICE_URL=python-imaging:50051
  
  # Python 醫學影像處理服務
  python-imaging:
    build:
      context: ./python-services
      dockerfile: Dockerfile.medical-imaging
    ports:
      - "50051:50051"
    volumes:
      - medical_data:/data
      - model_cache:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_CACHE_DIR=/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # 基礎設施
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: medical_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  medical_data:
  model_cache:
  postgres_data:
  redis_data:
```

### Python 醫學影像服務 Dockerfile
```dockerfile
# Dockerfile.medical-imaging
FROM python:3.11-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    dcm2niix \
    fsl-core \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 安裝 Python 依賴
COPY requirements-medical.txt .
RUN pip install --no-cache-dir -r requirements-medical.txt

# 複製醫學影像處理程式碼
COPY code_ai/ ./code_ai/
COPY medical_imaging_service.py .
COPY medical_imaging_pb2.py .
COPY medical_imaging_pb2_grpc.py .

# 設置環境變數
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; print('OK')" || exit 1

# 暴露 gRPC 端口
EXPOSE 50051

# 啟動服務
CMD ["python", "medical_imaging_service.py"]
```

## 📊 效能預期

### 分工後的效能預期
| 服務類型 | 語言 | 效能預期 | 說明 |
|---------|------|----------|------|
| **Web API** | Go | 5000 req/s | 5倍提升 |
| **使用者管理** | Go | 3000 req/s | 3倍提升 |
| **檔案上傳** | Go | 500MB/s | 2倍提升 |
| **任務調度** | Go | 1000 tasks/s | 10倍提升 |
| **DICOM 解析** | Go | 100 files/s | 2倍提升 |
| **NIfTI 處理** | Python | 維持現有 | 功能保持 |
| **SimpleITK 處理** | Python | 維持現有 | 功能保持 |
| **AI 推理** | Python | 維持現有 | 功能保持 |

### 資源使用預期
| 資源 | Go 服務 | Python 服務 | 總計 | 對比現有 |
|------|---------|-------------|------|----------|
| CPU | 2 cores | 4 cores | 6 cores | 相同 |
| 記憶體 | 200MB | 2GB | 2.2GB | -1GB |
| 磁碟 | 100MB | 5GB | 5.1GB | 相同 |
| GPU | 0 | 1 GPU | 1 GPU | 相同 |

## 🎯 最終建議

### 🟢 完美的混合架構策略

基於檔案內容分析，這個精確分離的混合架構是最佳選擇：

#### 優勢
1. **發揮 Go 優勢**：高效能 Web 服務、並行處理、檔案 I/O
2. **保留 Python 優勢**：完整的醫學影像處理生態系統
3. **零功能損失**：所有 NiBabel、SimpleITK 功能完全保留
4. **效能大幅提升**：Web 層 5倍效能提升，影像處理功能不變

#### 實施優先級
1. **P0**: Go Web 服務層建立
2. **P1**: Python 醫學影像服務分離
3. **P2**: gRPC 通訊整合
4. **P3**: 效能調優和部署

### 📋 立即行動項目

1. **本週**：建立 Go 專案結構，設置基礎 API Gateway
2. **下週**：實作使用者和檔案服務的 Go 版本
3. **第三週**：建立 Python 醫學影像 gRPC 服務
4. **第四週**：整合測試和效能驗證

這個策略完美平衡了效能提升和功能保留，是醫學影像處理系統現代化的最佳路徑。

---

**策略確認**: 保留所有 NiBabel、SimpleITK 功能在 Python，Go 負責高效能 Web 服務  
**預期成果**: Web 效能 5倍提升，醫學影像處理功能 100% 保留
