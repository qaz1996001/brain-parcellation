# 服務設計模式

## 📋 文件資訊
- **版本**: 1.0.0
- **建立日期**: 2025年9月24日
- **設計模式**: 微服務架構 + 領域驅動設計
- **審查狀態**: Generator 提案

## 🎯 服務設計原則

### 微服務設計原則
1. **單一責任**：每個服務只負責一個業務能力
2. **自主性**：服務可以獨立開發、部署和擴展
3. **去中心化**：避免共享資料庫，每個服務管理自己的資料
4. **容錯性**：假設其他服務可能失敗，設計容錯機制
5. **可觀測性**：提供充分的監控、日誌和追蹤

### API 設計原則
```python
# API 設計標準
API_DESIGN_STANDARDS = {
    'versioning': 'URL 路徑版本控制 (/api/v1/)',
    'naming': 'RESTful 資源命名',
    'status_codes': 'HTTP 標準狀態碼',
    'error_format': '統一錯誤回應格式',
    'pagination': '基於偏移量的分頁',
    'filtering': '查詢參數過濾',
    'sorting': '多欄位排序支援'
}
```

## 🏗️ 核心服務架構

### 1. 使用者服務 (User Service)

#### 服務責任
```python
# user_service/domain/user.py
from typing import List, Optional
from datetime import datetime
from enum import Enum

class UserRole(Enum):
    """使用者角色"""
    ADMIN = "admin"
    DOCTOR = "doctor"
    TECHNICIAN = "technician"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class Permission(Enum):
    """權限枚舉"""
    READ_PATIENT_DATA = "read_patient_data"
    WRITE_PATIENT_DATA = "write_patient_data"
    PROCESS_IMAGES = "process_images"
    MANAGE_USERS = "manage_users"
    VIEW_REPORTS = "view_reports"
    EXPORT_DATA = "export_data"

@dataclass
class User:
    """使用者領域模型"""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    full_name: str = ""
    role: UserRole = UserRole.VIEWER
    permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """檢查使用者是否有特定權限"""
        return permission in self.permissions
    
    def can_access_patient_data(self) -> bool:
        """檢查是否可以存取患者資料"""
        return self.has_permission(Permission.READ_PATIENT_DATA)
```

#### API 設計
```python
# user_service/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional

router = APIRouter(prefix="/api/v1/users", tags=["Users"])

@router.post("/", 
             response_model=UserResponse,
             status_code=status.HTTP_201_CREATED,
             summary="建立新使用者")
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_admin_user),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    建立新使用者
    
    需要管理員權限
    """
    if not current_user.has_permission(Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="沒有權限建立使用者"
        )
    
    try:
        user = await user_service.create_user(user_data)
        return UserResponse.from_domain(user)
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )

@router.get("/{user_id}",
            response_model=UserResponse,
            summary="取得使用者資訊")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """取得使用者資訊"""
    # 只能查看自己的資訊或管理員可以查看所有人
    if user_id != current_user.id and not current_user.has_permission(Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="沒有權限查看此使用者資訊"
        )
    
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="使用者不存在"
        )
    
    return UserResponse.from_domain(user)

@router.get("/",
            response_model=PaginatedResponse[UserResponse],
            summary="取得使用者列表")
async def list_users(
    skip: int = 0,
    limit: int = 20,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> PaginatedResponse[UserResponse]:
    """取得使用者列表（僅管理員）"""
    if not current_user.has_permission(Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="沒有權限查看使用者列表"
        )
    
    users, total = await user_service.list_users(
        skip=skip, 
        limit=limit,
        role=role,
        is_active=is_active
    )
    
    return PaginatedResponse(
        items=[UserResponse.from_domain(user) for user in users],
        total=total,
        skip=skip,
        limit=limit
    )
```

### 2. 檔案服務 (File Service)

#### 服務責任
```python
# file_service/domain/file.py
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path

class FileType(Enum):
    """檔案類型"""
    DICOM = "dicom"
    NIFTI = "nifti"
    REPORT = "report"
    RESULT = "result"

class FileStatus(Enum):
    """檔案狀態"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    ERROR = "error"

@dataclass
class MedicalFile:
    """醫學檔案領域模型"""
    id: Optional[int] = None
    filename: str = ""
    original_filename: str = ""
    file_type: FileType = FileType.DICOM
    file_size: int = 0
    file_path: str = ""
    checksum: str = ""
    status: FileStatus = FileStatus.UPLOADING
    
    # 醫學影像特定屬性
    patient_id: Optional[str] = None
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    modality: Optional[str] = None
    sequence_type: Optional[str] = None
    is_reformatted: Optional[bool] = None
    
    # 元資料
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 時間戳記
    uploaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    created_by: Optional[int] = None
    
    def is_medical_image(self) -> bool:
        """檢查是否為醫學影像檔案"""
        return self.file_type in [FileType.DICOM, FileType.NIFTI]
    
    def get_storage_path(self) -> Path:
        """取得儲存路徑"""
        return Path(self.file_path)
```

#### 檔案處理管線
```python
# file_service/services/file_processor.py
from typing import Dict, Any, Optional
import asyncio
import hashlib
import aiofiles

class FileProcessor:
    """檔案處理器"""
    
    def __init__(self, 
                 storage_service: StorageService,
                 dicom_service: DicomService):
        self.storage = storage_service
        self.dicom = dicom_service
    
    async def process_uploaded_file(self, 
                                  file_data: bytes,
                                  filename: str,
                                  user_id: int) -> MedicalFile:
        """處理上傳的檔案"""
        
        # 1. 檔案驗證
        file_info = await self._validate_file(file_data, filename)
        
        # 2. 計算檔案檢查碼
        checksum = self._calculate_checksum(file_data)
        
        # 3. 檢查重複檔案
        existing_file = await self._check_duplicate(checksum)
        if existing_file:
            return existing_file
        
        # 4. 儲存檔案
        storage_path = await self.storage.save_file(
            file_data, filename, file_info.file_type
        )
        
        # 5. 建立檔案記錄
        medical_file = MedicalFile(
            filename=filename,
            original_filename=filename,
            file_type=file_info.file_type,
            file_size=len(file_data),
            file_path=str(storage_path),
            checksum=checksum,
            status=FileStatus.UPLOADED,
            uploaded_at=datetime.utcnow(),
            created_by=user_id
        )
        
        # 6. 如果是 DICOM，提取元資料
        if file_info.file_type == FileType.DICOM:
            metadata = await self.dicom.extract_metadata(storage_path)
            medical_file.metadata = metadata
            medical_file.patient_id = metadata.get('patient_id')
            medical_file.study_instance_uid = metadata.get('study_instance_uid')
            medical_file.series_instance_uid = metadata.get('series_instance_uid')
            medical_file.modality = metadata.get('modality')
            medical_file.sequence_type = metadata.get('sequence_type')
            medical_file.is_reformatted = metadata.get('is_reformatted')
        
        return medical_file
    
    async def _validate_file(self, 
                           file_data: bytes, 
                           filename: str) -> FileValidationResult:
        """驗證檔案"""
        
        # 檔案大小檢查
        if len(file_data) > 500 * 1024 * 1024:  # 500MB
            raise FileTooLargeError("檔案大小超過限制")
        
        # 檔案類型檢測
        file_type = self._detect_file_type(file_data, filename)
        
        # 檔案格式驗證
        if file_type == FileType.DICOM:
            await self._validate_dicom(file_data)
        elif file_type == FileType.NIFTI:
            await self._validate_nifti(file_data)
        
        return FileValidationResult(
            is_valid=True,
            file_type=file_type
        )
    
    def _detect_file_type(self, file_data: bytes, filename: str) -> FileType:
        """檢測檔案類型"""
        # 檢查檔案標頭
        if file_data[:4] == b'DICM' or file_data[128:132] == b'DICM':
            return FileType.DICOM
        elif filename.endswith('.nii') or filename.endswith('.nii.gz'):
            return FileType.NIFTI
        else:
            # 根據副檔名判斷
            suffix = Path(filename).suffix.lower()
            if suffix == '.dcm':
                return FileType.DICOM
            elif suffix in ['.nii', '.gz']:
                return FileType.NIFTI
            else:
                raise UnsupportedFileTypeError(f"不支援的檔案類型: {suffix}")
    
    def _calculate_checksum(self, file_data: bytes) -> str:
        """計算檔案檢查碼"""
        return hashlib.sha256(file_data).hexdigest()
```

### 3. 處理服務 (Processing Service)

#### 服務責任
```python
# processing_service/domain/processing_job.py
from typing import Dict, Any, List, Optional
from enum import Enum

class ProcessingType(Enum):
    """處理類型"""
    DICOM_TO_NIFTI = "dicom_to_nifti"
    BRAIN_SEGMENTATION = "brain_segmentation"
    WMH_DETECTION = "wmh_detection"
    CMB_DETECTION = "cmb_detection"
    ANEURYSM_DETECTION = "aneurysm_detection"

class JobStatus(Enum):
    """任務狀態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingJob:
    """處理任務領域模型"""
    id: Optional[int] = None
    job_type: ProcessingType = ProcessingType.DICOM_TO_NIFTI
    status: JobStatus = JobStatus.PENDING
    
    # 輸入檔案
    input_files: List[int] = field(default_factory=list)  # file IDs
    
    # 處理參數
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 輸出結果
    output_files: List[int] = field(default_factory=list)  # file IDs
    result_data: Dict[str, Any] = field(default_factory=dict)
    
    # 執行資訊
    progress: float = 0.0
    error_message: Optional[str] = None
    
    # 時間戳記
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[int] = None
    
    def is_finished(self) -> bool:
        """檢查任務是否已完成"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def duration(self) -> Optional[float]:
        """計算執行時間（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
```

#### 任務調度器
```python
# processing_service/services/job_scheduler.py
import asyncio
from typing import Dict, List, Optional, Callable
from queue import PriorityQueue

class JobScheduler:
    """任務調度器"""
    
    def __init__(self, max_concurrent_jobs: int = 5):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running_jobs: Dict[int, asyncio.Task] = {}
        self.job_queue = PriorityQueue()
        self.processors: Dict[ProcessingType, Callable] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
    
    def register_processor(self, 
                         job_type: ProcessingType,
                         processor: Callable):
        """註冊處理器"""
        self.processors[job_type] = processor
    
    async def submit_job(self, job: ProcessingJob) -> int:
        """提交任務"""
        # 儲存任務到資料庫
        job_id = await self._save_job(job)
        
        # 加入佇列
        priority = self._calculate_priority(job)
        self.job_queue.put((priority, job_id, job))
        
        return job_id
    
    async def start_scheduler(self):
        """啟動調度器"""
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """停止調度器"""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """調度器主循環"""
        while True:
            try:
                # 清理已完成的任務
                await self._cleanup_completed_jobs()
                
                # 檢查是否可以啟動新任務
                if len(self.running_jobs) < self.max_concurrent_jobs and not self.job_queue.empty():
                    priority, job_id, job = self.job_queue.get()
                    
                    # 啟動任務
                    task = asyncio.create_task(self._execute_job(job))
                    self.running_jobs[job_id] = task
                
                await asyncio.sleep(1)  # 1秒檢查一次
                
            except Exception as e:
                print(f"調度器錯誤: {e}")
                await asyncio.sleep(5)
    
    async def _execute_job(self, job: ProcessingJob) -> None:
        """執行任務"""
        try:
            # 更新狀態為執行中
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            await self._update_job(job)
            
            # 取得處理器
            processor = self.processors.get(job.job_type)
            if not processor:
                raise ValueError(f"找不到處理器: {job.job_type}")
            
            # 執行處理
            result = await processor(job)
            
            # 更新結果
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            job.result_data = result
            
        except Exception as e:
            # 處理失敗
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
        
        finally:
            await self._update_job(job)
    
    def _calculate_priority(self, job: ProcessingJob) -> int:
        """計算任務優先級（數字越小優先級越高）"""
        priority_map = {
            ProcessingType.DICOM_TO_NIFTI: 1,
            ProcessingType.BRAIN_SEGMENTATION: 2,
            ProcessingType.WMH_DETECTION: 3,
            ProcessingType.CMB_DETECTION: 3,
            ProcessingType.ANEURYSM_DETECTION: 4,
        }
        return priority_map.get(job.job_type, 5)
```

## 🔄 服務間通訊模式

### 同步通訊 (HTTP/REST)
```python
# shared/http_client.py
import httpx
from typing import Dict, Any, Optional
import asyncio

class ServiceClient:
    """服務間 HTTP 客戶端"""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
    
    async def get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """GET 請求"""
        try:
            response = await self.client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ServiceCommunicationError(f"GET {path} 失敗: {e}")
    
    async def post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST 請求"""
        try:
            response = await self.client.post(path, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ServiceCommunicationError(f"POST {path} 失敗: {e}")
    
    async def close(self):
        """關閉客戶端"""
        await self.client.aclose()

# 使用範例
class FileServiceClient:
    """檔案服務客戶端"""
    
    def __init__(self, base_url: str):
        self.client = ServiceClient(base_url)
    
    async def get_file_info(self, file_id: int) -> Dict[str, Any]:
        """取得檔案資訊"""
        return await self.client.get(f"/files/{file_id}")
    
    async def update_file_status(self, 
                               file_id: int, 
                               status: str) -> Dict[str, Any]:
        """更新檔案狀態"""
        return await self.client.post(
            f"/files/{file_id}/status",
            {"status": status}
        )
```

### 非同步通訊 (事件驅動)
```python
# shared/event_bus.py
from typing import Dict, Any, Callable, List
import asyncio
import json
from abc import ABC, abstractmethod

class Event:
    """事件基類"""
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.utcnow()
        self.event_id = str(uuid.uuid4())

class EventHandler(ABC):
    """事件處理器抽象基類"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """處理事件"""
        pass

class EventBus:
    """事件匯流排"""
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = {}
    
    def subscribe(self, event_type: str, handler: EventHandler):
        """訂閱事件"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def publish(self, event: Event):
        """發布事件"""
        handlers = self.handlers.get(event.event_type, [])
        
        # 並行處理所有處理器
        tasks = [handler.handle(event) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# 事件定義
class FileUploadedEvent(Event):
    """檔案上傳事件"""
    def __init__(self, file_id: int, file_type: str, user_id: int):
        super().__init__("file.uploaded", {
            "file_id": file_id,
            "file_type": file_type,
            "user_id": user_id
        })

class ProcessingCompletedEvent(Event):
    """處理完成事件"""
    def __init__(self, job_id: int, result: Dict[str, Any]):
        super().__init__("processing.completed", {
            "job_id": job_id,
            "result": result
        })

# 事件處理器範例
class FileProcessingHandler(EventHandler):
    """檔案處理事件處理器"""
    
    def __init__(self, processing_service):
        self.processing_service = processing_service
    
    async def handle(self, event: Event):
        """處理檔案上傳事件"""
        if event.event_type == "file.uploaded":
            file_id = event.data["file_id"]
            file_type = event.data["file_type"]
            
            # 如果是 DICOM 檔案，自動啟動轉換任務
            if file_type == "dicom":
                await self.processing_service.create_conversion_job(file_id)
```

## 🔒 服務安全模式

### JWT 令牌驗證
```python
# shared/auth.py
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class JWTManager:
    """JWT 令牌管理器"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, 
                          user_id: int,
                          permissions: List[str],
                          expires_delta: Optional[timedelta] = None) -> str:
        """建立存取令牌"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        payload = {
            "sub": str(user_id),
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """驗證令牌"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("令牌已過期")
        except jwt.JWTError:
            raise InvalidTokenError("無效的令牌")

# FastAPI 依賴注入
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()
jwt_manager = JWTManager(settings.SECRET_KEY)

async def get_current_user(token: str = Depends(security)) -> User:
    """取得當前使用者"""
    try:
        payload = jwt_manager.verify_token(token.credentials)
        user_id = int(payload["sub"])
        permissions = payload["permissions"]
        
        # 從使用者服務取得使用者資訊
        user_client = UserServiceClient(settings.USER_SERVICE_URL)
        user_data = await user_client.get_user(user_id)
        
        return User.from_dict(user_data)
        
    except (TokenExpiredError, InvalidTokenError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
```

### 服務間認證
```python
# shared/service_auth.py
import hmac
import hashlib
from datetime import datetime

class ServiceAuthenticator:
    """服務間認證"""
    
    def __init__(self, service_key: str):
        self.service_key = service_key
    
    def create_signature(self, 
                        method: str,
                        path: str,
                        body: str,
                        timestamp: str) -> str:
        """建立請求簽名"""
        message = f"{method}\n{path}\n{body}\n{timestamp}"
        signature = hmac.new(
            self.service_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self,
                        method: str,
                        path: str,
                        body: str,
                        timestamp: str,
                        signature: str) -> bool:
        """驗證請求簽名"""
        expected_signature = self.create_signature(method, path, body, timestamp)
        return hmac.compare_digest(signature, expected_signature)

# 中介軟體
from fastapi import Request, HTTPException

async def service_auth_middleware(request: Request, call_next):
    """服務間認證中介軟體"""
    
    # 檢查是否為服務間請求
    if request.headers.get("X-Service-Auth"):
        auth_header = request.headers.get("Authorization")
        timestamp = request.headers.get("X-Timestamp")
        signature = request.headers.get("X-Signature")
        
        if not all([auth_header, timestamp, signature]):
            raise HTTPException(
                status_code=401,
                detail="缺少必要的認證標頭"
            )
        
        # 讀取請求體
        body = await request.body()
        
        # 驗證簽名
        authenticator = ServiceAuthenticator(settings.SERVICE_KEY)
        if not authenticator.verify_signature(
            request.method,
            str(request.url.path),
            body.decode(),
            timestamp,
            signature
        ):
            raise HTTPException(
                status_code=401,
                detail="服務認證失敗"
            )
    
    response = await call_next(request)
    return response
```

## 📊 監控和可觀測性

### 分散式追蹤
```python
# shared/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class TracingManager:
    """分散式追蹤管理"""
    
    def __init__(self, service_name: str, jaeger_endpoint: str):
        self.service_name = service_name
        
        # 設置追蹤器
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # 設置 Jaeger 匯出器
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
    
    def trace_request(self, operation_name: str):
        """追蹤請求裝飾器"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(operation_name) as span:
                    span.set_attribute("service.name", self.service_name)
                    span.set_attribute("operation.name", operation_name)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error.message", str(e))
                        raise
            return wrapper
        return decorator
```

## 🎯 部署和擴展策略

### Docker 容器化
```dockerfile
# 服務基礎映像
FROM python:3.11-slim

# 安裝依賴
COPY requirements.txt .
RUN pip install -r requirements.txt

# 複製應用程式
COPY . /app
WORKDIR /app

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 執行應用程式
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes 部署
```yaml
# k8s/user-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
    spec:
      containers:
      - name: user-service
        image: medical-imaging/user-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

**注意**: 本文件定義了微服務架構的核心設計模式。每個服務都應該遵循這些模式來確保一致性和可維護性。

