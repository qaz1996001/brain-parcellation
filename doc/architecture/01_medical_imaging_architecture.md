# 醫學影像處理專用架構

## 📋 文件資訊
- **版本**: 1.0.0
- **建立日期**: 2025年9月24日
- **專業領域**: 醫學影像處理 (DICOM/NIfTI)
- **合規標準**: HIPAA, DICOM 3.0, HL7 FHIR

## 🏥 醫學影像系統特殊需求

### 資料格式支援
```python
# 支援的醫學影像格式
SUPPORTED_FORMATS = {
    'input': [
        'DICOM (.dcm)',
        'DICOM Series',
        'NIFTI (.nii, .nii.gz)',
        'ANALYZE (.hdr/.img)'
    ],
    'output': [
        'NIFTI (.nii.gz)',
        'DICOM-SEG',
        'JSON Reports',
        'Statistical Maps'
    ]
}
```

### 影像序列類型
```python
# 醫學影像序列分類
class ImageSequenceType(Enum):
    """影像序列類型枚舉"""
    T1 = "T1"
    T1_REFORMATTED = "T1_REFORMATTED"
    T2 = "T2"
    T2_FLAIR = "T2_FLAIR"
    T2_REFORMATTED = "T2_REFORMATTED"
    DWI = "DWI"
    ADC = "ADC"
    SWI = "SWI"
    BOLD = "BOLD"
    UNKNOWN = "UNKNOWN"
```

## 🔧 DICOM 處理引擎架構

### 核心組件設計
```
┌─────────────────────────────────────────────────────────┐
│                DICOM 處理引擎                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   DICOM 解析器   │  │   標籤提取器     │  │   格式轉換器     │  │
│  │ (DICOM Parser)  │  │ (Tag Extractor) │  │(Format Converter)│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   序列檢測器     │  │   資料驗證器     │  │   匿名化處理器   │  │
│  │(Sequence Detector)│ │(Data Validator) │ │(Anonymizer)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### DICOM 標籤安全存取模式
```python
# dicom/safe_accessor.py
from typing import Optional, Any, Union
import pydicom
from pydicom import Dataset

class SafeDicomAccessor:
    """安全的 DICOM 標籤存取器"""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def get_tag_value(self, 
                     tag: Union[str, int, tuple], 
                     default: Any = None) -> Any:
        """
        安全取得 DICOM 標籤值
        
        Args:
            tag: DICOM 標籤 (可以是關鍵字、十六進位或元組)
            default: 預設值
            
        Returns:
            標籤值或預設值
        """
        try:
            if isinstance(tag, str):
                # 使用關鍵字存取
                return getattr(self.dataset, tag, default)
            elif isinstance(tag, (int, tuple)):
                # 使用標籤號碼存取
                return self.dataset.get(tag, default)
            else:
                return default
        except (AttributeError, KeyError):
            return default
    
    def get_patient_info(self) -> dict:
        """安全取得患者資訊"""
        return {
            'patient_id': self.get_tag_value('PatientID', 'UNKNOWN'),
            'patient_name': self.get_tag_value('PatientName', 'UNKNOWN'),
            'patient_birth_date': self.get_tag_value('PatientBirthDate'),
            'patient_sex': self.get_tag_value('PatientSex'),
            'patient_age': self.get_tag_value('PatientAge')
        }
    
    def get_study_info(self) -> dict:
        """安全取得檢查資訊"""
        return {
            'study_instance_uid': self.get_tag_value('StudyInstanceUID'),
            'study_date': self.get_tag_value('StudyDate'),
            'study_time': self.get_tag_value('StudyTime'),
            'study_description': self.get_tag_value('StudyDescription'),
            'modality': self.get_tag_value('Modality')
        }
    
    def get_series_info(self) -> dict:
        """安全取得序列資訊"""
        return {
            'series_instance_uid': self.get_tag_value('SeriesInstanceUID'),
            'series_number': self.get_tag_value('SeriesNumber'),
            'series_description': self.get_tag_value('SeriesDescription'),
            'protocol_name': self.get_tag_value('ProtocolName'),
            'sequence_name': self.get_tag_value('SequenceName')
        }
```

### REFORMATTED 檢測邏輯
```python
# dicom/reformatted_detector.py
import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ReformattedDetectionResult:
    """REFORMATTED 檢測結果"""
    is_reformatted: bool
    confidence: float
    detection_method: str
    evidence: List[str]

class ReformattedDetector:
    """REFORMATTED 序列檢測器"""
    
    # REFORMATTED 關鍵字模式
    REFORMATTED_PATTERNS = [
        r'REFORMATTED',
        r'REFORMAT',
        r'MPR',  # Multi-Planar Reconstruction
        r'CURVED',
        r'OBLIQUE',
        r'CORONAL.*RECON',
        r'SAGITTAL.*RECON',
        r'3D.*RECON'
    ]
    
    # 原始序列指標
    ORIGINAL_INDICATORS = [
        r'ORIGINAL',
        r'PRIMARY',
        r'AXIAL.*SOURCE',
        r'RAW.*DATA'
    ]
    
    def detect_reformatted(self, dicom_accessor: SafeDicomAccessor) -> ReformattedDetectionResult:
        """
        檢測是否為 REFORMATTED 序列
        
        Args:
            dicom_accessor: 安全的 DICOM 存取器
            
        Returns:
            檢測結果
        """
        evidence = []
        confidence = 0.0
        detection_methods = []
        
        # 方法 1: 檢查序列描述
        series_desc = dicom_accessor.get_tag_value('SeriesDescription', '').upper()
        if self._check_patterns(series_desc, self.REFORMATTED_PATTERNS):
            evidence.append(f"序列描述包含 REFORMATTED 關鍵字: {series_desc}")
            confidence += 0.8
            detection_methods.append("series_description")
        
        # 方法 2: 檢查協定名稱
        protocol_name = dicom_accessor.get_tag_value('ProtocolName', '').upper()
        if self._check_patterns(protocol_name, self.REFORMATTED_PATTERNS):
            evidence.append(f"協定名稱包含 REFORMATTED 關鍵字: {protocol_name}")
            confidence += 0.7
            detection_methods.append("protocol_name")
        
        # 方法 3: 檢查影像類型
        image_type = dicom_accessor.get_tag_value('ImageType', [])
        if isinstance(image_type, list):
            image_type_str = ' '.join(image_type).upper()
            if 'DERIVED' in image_type_str and 'SECONDARY' in image_type_str:
                evidence.append(f"影像類型顯示為衍生影像: {image_type_str}")
                confidence += 0.6
                detection_methods.append("image_type")
        
        # 方法 4: 檢查序列變體
        sequence_variant = dicom_accessor.get_tag_value('SequenceVariant', [])
        if isinstance(sequence_variant, list):
            variant_str = ' '.join(sequence_variant).upper()
            if any(pattern in variant_str for pattern in ['MP', 'MTC', 'OSP']):
                evidence.append(f"序列變體顯示為重組: {variant_str}")
                confidence += 0.5
                detection_methods.append("sequence_variant")
        
        # 方法 5: 檢查切片厚度和間距
        slice_thickness = dicom_accessor.get_tag_value('SliceThickness')
        spacing_between_slices = dicom_accessor.get_tag_value('SpacingBetweenSlices')
        
        if slice_thickness and spacing_between_slices:
            try:
                thickness = float(slice_thickness)
                spacing = float(spacing_between_slices)
                # REFORMATTED 通常有不同的切片間距
                if abs(thickness - spacing) > 0.1:
                    evidence.append(f"切片厚度與間距不一致: {thickness}mm vs {spacing}mm")
                    confidence += 0.3
                    detection_methods.append("slice_geometry")
            except (ValueError, TypeError):
                pass
        
        # 排除明確的原始序列
        if self._check_patterns(series_desc + ' ' + protocol_name, self.ORIGINAL_INDICATORS):
            confidence = max(0, confidence - 0.9)
            evidence.append("檢測到原始序列指標")
        
        is_reformatted = confidence > 0.5
        
        return ReformattedDetectionResult(
            is_reformatted=is_reformatted,
            confidence=min(confidence, 1.0),
            detection_method='+'.join(detection_methods),
            evidence=evidence
        )
    
    def _check_patterns(self, text: str, patterns: List[str]) -> bool:
        """檢查文字是否符合任何模式"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
```

## 🤖 AI 推理引擎架構

### AI 模型管理
```python
# ai/model_manager.py
from typing import Dict, Any, Optional
from enum import Enum
import torch
import onnxruntime as ort

class ModelType(Enum):
    """AI 模型類型"""
    SYNTHSEG = "synthseg"
    WMH_DETECTION = "wmh_detection"
    CMB_DETECTION = "cmb_detection"
    ANEURYSM_DETECTION = "aneurysm_detection"
    BRAIN_PARCELLATION = "brain_parcellation"

class ModelManager:
    """AI 模型管理器"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
    
    async def load_model(self, 
                        model_type: ModelType, 
                        model_path: str,
                        device: str = "cpu") -> bool:
        """載入 AI 模型"""
        try:
            model_key = f"{model_type.value}_{device}"
            
            if model_path.endswith('.onnx'):
                # ONNX 模型
                providers = ['CPUExecutionProvider']
                if device == "cuda" and ort.get_device() == 'GPU':
                    providers = ['CUDAExecutionProvider'] + providers
                
                session = ort.InferenceSession(model_path, providers=providers)
                self.loaded_models[model_key] = session
                
            elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                # PyTorch 模型
                model = torch.load(model_path, map_location=device)
                model.eval()
                self.loaded_models[model_key] = model
            
            else:
                raise ValueError(f"不支援的模型格式: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"模型載入失敗: {e}")
            return False
    
    async def predict(self, 
                     model_type: ModelType,
                     input_data: Any,
                     device: str = "cpu") -> Dict[str, Any]:
        """執行模型推理"""
        model_key = f"{model_type.value}_{device}"
        
        if model_key not in self.loaded_models:
            raise ValueError(f"模型未載入: {model_key}")
        
        model = self.loaded_models[model_key]
        
        if isinstance(model, ort.InferenceSession):
            # ONNX 推理
            input_name = model.get_inputs()[0].name
            result = model.run(None, {input_name: input_data})
            return {"prediction": result[0]}
        
        elif hasattr(model, 'forward'):
            # PyTorch 推理
            with torch.no_grad():
                result = model(input_data)
            return {"prediction": result.cpu().numpy()}
        
        else:
            raise ValueError(f"未知的模型類型: {type(model)}")
```

### 影像處理管線
```python
# pipeline/medical_pipeline.py
from typing import Dict, Any, List
from pathlib import Path
import asyncio

class MedicalImagingPipeline:
    """醫學影像處理管線"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.processing_stages = {
            'preprocessing': self._preprocess,
            'skull_stripping': self._skull_strip,
            'registration': self._register,
            'segmentation': self._segment,
            'analysis': self._analyze,
            'postprocessing': self._postprocess
        }
    
    async def process_image(self, 
                          image_path: Path,
                          processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """處理醫學影像"""
        
        results = {
            'input_path': str(image_path),
            'stages': {},
            'final_outputs': {},
            'metadata': {}
        }
        
        try:
            # 階段 1: DICOM 解析和驗證
            dicom_info = await self._parse_dicom(image_path)
            results['metadata']['dicom_info'] = dicom_info
            
            # 階段 2: REFORMATTED 檢測
            reformatted_result = await self._detect_reformatted(image_path)
            results['metadata']['reformatted'] = reformatted_result
            
            # 階段 3: 格式轉換
            nifti_path = await self._convert_to_nifti(image_path, dicom_info)
            results['stages']['conversion'] = {'nifti_path': str(nifti_path)}
            
            # 階段 4: 依據配置執行處理管線
            for stage_name in processing_config.get('stages', []):
                if stage_name in self.processing_stages:
                    stage_result = await self.processing_stages[stage_name](
                        nifti_path, processing_config.get(stage_name, {})
                    )
                    results['stages'][stage_name] = stage_result
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            return results
    
    async def _parse_dicom(self, image_path: Path) -> Dict[str, Any]:
        """解析 DICOM 檔案"""
        import pydicom
        
        try:
            ds = pydicom.dcmread(image_path)
            accessor = SafeDicomAccessor(ds)
            
            return {
                'patient_info': accessor.get_patient_info(),
                'study_info': accessor.get_study_info(),
                'series_info': accessor.get_series_info()
            }
        except Exception as e:
            raise ValueError(f"DICOM 解析失敗: {e}")
    
    async def _detect_reformatted(self, image_path: Path) -> Dict[str, Any]:
        """檢測 REFORMATTED 序列"""
        import pydicom
        
        ds = pydicom.dcmread(image_path)
        accessor = SafeDicomAccessor(ds)
        detector = ReformattedDetector()
        
        result = detector.detect_reformatted(accessor)
        
        return {
            'is_reformatted': result.is_reformatted,
            'confidence': result.confidence,
            'method': result.detection_method,
            'evidence': result.evidence
        }
    
    async def _convert_to_nifti(self, 
                              dicom_path: Path, 
                              dicom_info: Dict) -> Path:
        """轉換為 NIfTI 格式"""
        # 實作 DICOM 到 NIfTI 的轉換邏輯
        # 這裡應該使用如 dcm2niix 或 pydicom + nibabel 的組合
        pass
    
    async def _preprocess(self, 
                         nifti_path: Path, 
                         config: Dict) -> Dict[str, Any]:
        """前處理階段"""
        # 實作前處理邏輯：標準化、去噪等
        pass
    
    async def _segment(self, 
                      nifti_path: Path, 
                      config: Dict) -> Dict[str, Any]:
        """分割階段"""
        # 使用 SynthSeg 或其他分割模型
        result = await self.model_manager.predict(
            ModelType.SYNTHSEG,
            nifti_path,
            config.get('device', 'cpu')
        )
        return result
```

## 🔒 醫學資料合規架構

### HIPAA 合規設計
```python
# compliance/hipaa.py
from typing import Dict, Any, List
import hashlib
import uuid

class HIPAACompliantProcessor:
    """HIPAA 合規處理器"""
    
    def __init__(self):
        self.phi_fields = [
            'PatientName',
            'PatientID', 
            'PatientBirthDate',
            'InstitutionName',
            'PhysicianName',
            'OperatorName'
        ]
    
    def anonymize_dicom(self, dataset) -> Dict[str, Any]:
        """匿名化 DICOM 資料"""
        anonymization_log = {
            'original_patient_id': None,
            'anonymized_patient_id': None,
            'removed_fields': [],
            'modified_fields': []
        }
        
        # 記錄原始患者 ID
        if hasattr(dataset, 'PatientID'):
            anonymization_log['original_patient_id'] = dataset.PatientID
            # 生成匿名化 ID
            anonymous_id = self._generate_anonymous_id(dataset.PatientID)
            dataset.PatientID = anonymous_id
            anonymization_log['anonymized_patient_id'] = anonymous_id
            anonymization_log['modified_fields'].append('PatientID')
        
        # 移除或修改 PHI 欄位
        for field in self.phi_fields:
            if hasattr(dataset, field):
                if field == 'PatientID':
                    continue  # 已處理
                elif field == 'PatientBirthDate':
                    # 保留年份，移除具體日期
                    if dataset.PatientBirthDate:
                        year = dataset.PatientBirthDate[:4]
                        dataset.PatientBirthDate = f"{year}0101"
                        anonymization_log['modified_fields'].append(field)
                else:
                    # 移除其他 PHI 欄位
                    delattr(dataset, field)
                    anonymization_log['removed_fields'].append(field)
        
        return anonymization_log
    
    def _generate_anonymous_id(self, original_id: str) -> str:
        """生成匿名化 ID"""
        # 使用 SHA-256 哈希生成一致的匿名 ID
        hash_object = hashlib.sha256(original_id.encode())
        return f"ANON_{hash_object.hexdigest()[:8].upper()}"
    
    def audit_log(self, 
                 action: str,
                 user_id: str,
                 patient_id: str,
                 details: Dict[str, Any]) -> Dict[str, Any]:
        """建立稽核日誌"""
        return {
            'audit_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id,
            'patient_id': patient_id,
            'details': details,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
```

## 📊 效能監控架構

### 醫學影像特定指標
```python
# monitoring/medical_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 業務指標
dicom_files_processed = Counter(
    'dicom_files_processed_total',
    'Total DICOM files processed',
    ['sequence_type', 'status']
)

processing_duration = Histogram(
    'medical_processing_duration_seconds',
    'Medical image processing duration',
    ['pipeline_stage', 'image_type']
)

active_ai_models = Gauge(
    'active_ai_models',
    'Number of active AI models',
    ['model_type', 'device']
)

reformatted_detection_accuracy = Gauge(
    'reformatted_detection_accuracy',
    'REFORMATTED detection accuracy',
    ['detection_method']
)

# 系統指標
gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device_id']
)

file_storage_usage = Gauge(
    'medical_file_storage_bytes',
    'Medical file storage usage',
    ['storage_type']
)
```

## 🔄 工作流引擎設計

### 醫學影像工作流
```python
# workflow/medical_workflow.py
from typing import Dict, Any, List, Callable
from enum import Enum
import asyncio

class WorkflowStage(Enum):
    """工作流階段"""
    UPLOAD = "upload"
    VALIDATION = "validation"
    CONVERSION = "conversion"
    PREPROCESSING = "preprocessing"
    AI_INFERENCE = "ai_inference"
    POSTPROCESSING = "postprocessing"
    REPORT_GENERATION = "report_generation"
    STORAGE = "storage"

class MedicalWorkflow:
    """醫學影像工作流引擎"""
    
    def __init__(self):
        self.stages: Dict[WorkflowStage, Callable] = {}
        self.dependencies: Dict[WorkflowStage, List[WorkflowStage]] = {}
        self.retry_policies: Dict[WorkflowStage, Dict] = {}
    
    def register_stage(self, 
                      stage: WorkflowStage,
                      handler: Callable,
                      dependencies: List[WorkflowStage] = None,
                      retry_policy: Dict = None):
        """註冊工作流階段"""
        self.stages[stage] = handler
        self.dependencies[stage] = dependencies or []
        self.retry_policies[stage] = retry_policy or {
            'max_retries': 3,
            'backoff_factor': 2,
            'retry_exceptions': [Exception]
        }
    
    async def execute_workflow(self, 
                             workflow_id: str,
                             input_data: Dict[str, Any],
                             stages_to_run: List[WorkflowStage]) -> Dict[str, Any]:
        """執行工作流"""
        
        workflow_state = {
            'workflow_id': workflow_id,
            'status': 'running',
            'stages': {},
            'current_data': input_data,
            'errors': []
        }
        
        try:
            # 根據依賴關係排序階段
            ordered_stages = self._topological_sort(stages_to_run)
            
            for stage in ordered_stages:
                stage_result = await self._execute_stage_with_retry(
                    stage, workflow_state
                )
                workflow_state['stages'][stage.value] = stage_result
                
                if not stage_result['success']:
                    workflow_state['status'] = 'failed'
                    workflow_state['errors'].append(
                        f"Stage {stage.value} failed: {stage_result.get('error')}"
                    )
                    break
                
                # 更新工作流資料
                if 'output_data' in stage_result:
                    workflow_state['current_data'].update(stage_result['output_data'])
            
            if workflow_state['status'] != 'failed':
                workflow_state['status'] = 'completed'
            
        except Exception as e:
            workflow_state['status'] = 'error'
            workflow_state['errors'].append(f"Workflow execution error: {e}")
        
        return workflow_state
    
    async def _execute_stage_with_retry(self, 
                                       stage: WorkflowStage,
                                       workflow_state: Dict) -> Dict[str, Any]:
        """帶重試的階段執行"""
        retry_policy = self.retry_policies[stage]
        max_retries = retry_policy['max_retries']
        backoff_factor = retry_policy['backoff_factor']
        
        for attempt in range(max_retries + 1):
            try:
                handler = self.stages[stage]
                result = await handler(workflow_state['current_data'])
                return {'success': True, 'output_data': result, 'attempt': attempt + 1}
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
    
    def _topological_sort(self, stages: List[WorkflowStage]) -> List[WorkflowStage]:
        """拓撲排序工作流階段"""
        # 實作拓撲排序算法
        # 確保依賴的階段先執行
        pass
```

## 📋 部署考量

### 醫學影像專用容器
```dockerfile
# Dockerfile.medical-imaging
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 安裝醫學影像處理工具
RUN apt-get update && apt-get install -y \
    dcm2niix \
    fsl-core \
    ants \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
COPY requirements-medical.txt .
RUN pip install -r requirements-medical.txt

# 複製應用程式
COPY . /app
WORKDIR /app

# 設置環境變數
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

CMD ["python", "-m", "medical_imaging.main"]
```

### 資料卷管理
```yaml
# docker-compose-medical.yml
version: '3.8'
services:
  medical-processor:
    build:
      context: .
      dockerfile: Dockerfile.medical-imaging
    volumes:
      - dicom_storage:/data/dicom
      - nifti_storage:/data/nifti
      - model_cache:/app/models
      - temp_processing:/tmp/processing
    environment:
      - GPU_ENABLED=true
      - MODEL_CACHE_DIR=/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  dicom_storage:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/medical_data/dicom
  
  nifti_storage:
    driver: local
    driver_opts:
      type: none  
      o: bind
      device: /mnt/medical_data/nifti
```

## 🎯 下一步實作

### 立即實作（本週）
1. 實作 SafeDicomAccessor 類別
2. 建立 ReformattedDetector 
3. 設計 ModelManager 基礎架構
4. 建立 HIPAA 合規處理器

### 短期實作（本月）
1. 完整的醫學影像處理管線
2. AI 模型整合和管理
3. 工作流引擎實作
4. 監控指標收集

---

**注意**: 本架構文件專注於醫學影像處理的特殊需求，需要與整體系統架構協調一致。所有醫學資料處理都必須遵循相關法規和標準。

