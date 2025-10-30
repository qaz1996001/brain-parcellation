# 資料架構設計

## 📋 文件資訊
- **版本**: 1.0.0
- **建立日期**: 2025年9月24日
- **資料策略**: 多模型資料庫 + 分層儲存
- **合規標準**: HIPAA, GDPR, 醫學資料保護法

## 🎯 資料架構原則

### 核心原則
1. **資料主權**：每個服務擁有自己的資料
2. **ACID 合規**：關鍵業務資料保證事務完整性
3. **最終一致性**：跨服務資料同步採用最終一致性
4. **資料分層**：熱、溫、冷資料分層儲存
5. **隱私保護**：敏感資料加密和匿名化

### 資料分類
```python
# 資料分類標準
DATA_CLASSIFICATION = {
    'public': {
        'description': '公開資料',
        'examples': ['API 文檔', '系統狀態'],
        'encryption': False,
        'retention': '無限期'
    },
    'internal': {
        'description': '內部資料',
        'examples': ['系統日誌', '效能指標'],
        'encryption': False,
        'retention': '2年'
    },
    'confidential': {
        'description': '機密資料',
        'examples': ['使用者資料', '業務資料'],
        'encryption': True,
        'retention': '7年'
    },
    'restricted': {
        'description': '限制級資料',
        'examples': ['患者資料', '醫學影像'],
        'encryption': True,
        'retention': '依法規要求'
    }
}
```

## 🗄️ 資料庫架構

### 1. PostgreSQL - 主要業務資料

#### 資料庫分割策略
```sql
-- 使用者服務資料庫
CREATE DATABASE user_service_db;

-- 檔案服務資料庫  
CREATE DATABASE file_service_db;

-- 處理服務資料庫
CREATE DATABASE processing_service_db;

-- 稽核日誌資料庫
CREATE DATABASE audit_log_db;
```

#### 使用者服務資料模型
```sql
-- users 表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role user_role NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 索引
    CONSTRAINT users_username_check CHECK (length(username) >= 3),
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- 使用者角色枚舉
CREATE TYPE user_role AS ENUM ('admin', 'doctor', 'technician', 'researcher', 'viewer');

-- 權限表
CREATE TABLE permissions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    resource VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 使用者權限關聯表
CREATE TABLE user_permissions (
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    permission_id INTEGER REFERENCES permissions(id) ON DELETE CASCADE,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by INTEGER REFERENCES users(id),
    PRIMARY KEY (user_id, permission_id)
);

-- 會話表
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);
```

#### 檔案服務資料模型
```sql
-- 檔案類型枚舉
CREATE TYPE file_type AS ENUM ('dicom', 'nifti', 'report', 'result');
CREATE TYPE file_status AS ENUM ('uploading', 'uploaded', 'processing', 'processed', 'archived', 'error');

-- 檔案表
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type file_type NOT NULL,
    file_size BIGINT NOT NULL,
    file_path TEXT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    status file_status DEFAULT 'uploading',
    
    -- 醫學影像特定欄位
    patient_id VARCHAR(64),
    study_instance_uid VARCHAR(64),
    series_instance_uid VARCHAR(64),
    modality VARCHAR(16),
    sequence_type VARCHAR(50),
    is_reformatted BOOLEAN,
    
    -- 元資料 (JSONB 格式)
    metadata JSONB DEFAULT '{}',
    
    -- 時間戳記
    uploaded_at TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by INTEGER,
    
    -- 約束
    CONSTRAINT files_checksum_unique UNIQUE (checksum),
    CONSTRAINT files_size_positive CHECK (file_size > 0)
);

-- 檔案標籤表
CREATE TABLE file_tags (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    tag_name VARCHAR(100) NOT NULL,
    tag_value TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by INTEGER
);

-- 檔案存取日誌
CREATE TABLE file_access_logs (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    action VARCHAR(50) NOT NULL, -- 'view', 'download', 'delete'
    ip_address INET,
    user_agent TEXT,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_files_patient_id ON files(patient_id);
CREATE INDEX idx_files_study_uid ON files(study_instance_uid);
CREATE INDEX idx_files_series_uid ON files(series_instance_uid);
CREATE INDEX idx_files_type ON files(file_type);
CREATE INDEX idx_files_status ON files(status);
CREATE INDEX idx_files_created_by ON files(created_by);
CREATE INDEX idx_files_metadata ON files USING gin(metadata);
CREATE INDEX idx_file_tags_name ON file_tags(tag_name);
CREATE INDEX idx_file_access_logs_file_id ON file_access_logs(file_id);
CREATE INDEX idx_file_access_logs_user_id ON file_access_logs(user_id);
```

#### 處理服務資料模型
```sql
-- 處理類型枚舉
CREATE TYPE processing_type AS ENUM (
    'dicom_to_nifti',
    'brain_segmentation', 
    'wmh_detection',
    'cmb_detection',
    'aneurysm_detection'
);

CREATE TYPE job_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');

-- 處理任務表
CREATE TABLE processing_jobs (
    id SERIAL PRIMARY KEY,
    job_type processing_type NOT NULL,
    status job_status DEFAULT 'pending',
    
    -- 輸入檔案 (JSON 陣列存儲檔案 ID)
    input_files JSONB NOT NULL DEFAULT '[]',
    
    -- 處理參數
    parameters JSONB DEFAULT '{}',
    
    -- 輸出結果
    output_files JSONB DEFAULT '[]',
    result_data JSONB DEFAULT '{}',
    
    -- 執行資訊
    progress DECIMAL(5,4) DEFAULT 0.0,
    error_message TEXT,
    
    -- 資源使用情況
    cpu_usage DECIMAL(5,2),
    memory_usage BIGINT,
    gpu_usage DECIMAL(5,2),
    
    -- 時間戳記
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by INTEGER NOT NULL,
    
    -- 約束
    CONSTRAINT processing_jobs_progress_range CHECK (progress >= 0 AND progress <= 1)
);

-- 任務依賴關係表
CREATE TABLE job_dependencies (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES processing_jobs(id) ON DELETE CASCADE,
    depends_on_job_id INTEGER REFERENCES processing_jobs(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 防止循環依賴
    CONSTRAINT job_dependencies_no_self_ref CHECK (job_id != depends_on_job_id)
);

-- 任務執行日誌
CREATE TABLE job_execution_logs (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES processing_jobs(id) ON DELETE CASCADE,
    log_level VARCHAR(10) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    message TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 結構化日誌資料
    log_data JSONB DEFAULT '{}'
);

-- 索引
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_type ON processing_jobs(job_type);
CREATE INDEX idx_processing_jobs_created_by ON processing_jobs(created_by);
CREATE INDEX idx_processing_jobs_created_at ON processing_jobs(created_at);
CREATE INDEX idx_job_dependencies_job_id ON job_dependencies(job_id);
CREATE INDEX idx_job_execution_logs_job_id ON job_execution_logs(job_id);
CREATE INDEX idx_job_execution_logs_level ON job_execution_logs(log_level);
```

### 2. Redis - 快取和會話儲存

#### 快取策略
```python
# cache/strategies.py
from typing import Any, Optional, Dict
import redis.asyncio as redis
import json
import pickle
from datetime import timedelta

class CacheStrategy:
    """快取策略基類"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """取得快取值"""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """設定快取值"""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """刪除快取"""
        return await self.redis.delete(key) > 0

class JsonCacheStrategy(CacheStrategy):
    """JSON 快取策略"""
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except (json.JSONDecodeError, TypeError):
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            serialized = json.dumps(value, default=str)
            return await self.redis.setex(key, ttl, serialized)
        except (TypeError, ValueError):
            return False

class PickleCacheStrategy(CacheStrategy):
    """Pickle 快取策略（用於複雜物件）"""
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self.redis.get(key)
            return pickle.loads(value) if value else None
        except (pickle.PickleError, TypeError):
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            serialized = pickle.dumps(value)
            return await self.redis.setex(key, ttl, serialized)
        except pickle.PickleError:
            return False

class CacheManager:
    """快取管理器"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.strategies = {
            'json': JsonCacheStrategy(self.redis),
            'pickle': PickleCacheStrategy(self.redis)
        }
    
    async def get_user_info(self, user_id: int) -> Optional[Dict]:
        """取得使用者資訊快取"""
        key = f"user:{user_id}"
        return await self.strategies['json'].get(key)
    
    async def cache_user_info(self, user_id: int, user_data: Dict, ttl: int = 1800):
        """快取使用者資訊（30分鐘）"""
        key = f"user:{user_id}"
        return await self.strategies['json'].set(key, user_data, ttl)
    
    async def get_file_metadata(self, file_id: int) -> Optional[Dict]:
        """取得檔案元資料快取"""
        key = f"file:metadata:{file_id}"
        return await self.strategies['json'].get(key)
    
    async def cache_file_metadata(self, file_id: int, metadata: Dict, ttl: int = 3600):
        """快取檔案元資料（1小時）"""
        key = f"file:metadata:{file_id}"
        return await self.strategies['json'].set(key, metadata, ttl)
    
    async def get_processing_result(self, job_id: int) -> Optional[Any]:
        """取得處理結果快取"""
        key = f"processing:result:{job_id}"
        return await self.strategies['pickle'].get(key)
    
    async def cache_processing_result(self, job_id: int, result: Any, ttl: int = 7200):
        """快取處理結果（2小時）"""
        key = f"processing:result:{job_id}"
        return await self.strategies['pickle'].set(key, result, ttl)
```

#### 會話管理
```python
# session/manager.py
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class SessionManager:
    """會話管理器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_prefix = "session:"
        self.default_ttl = 1800  # 30分鐘
    
    async def create_session(self, 
                           user_id: int,
                           user_data: Dict[str, Any],
                           ttl: int = None) -> str:
        """建立會話"""
        session_id = str(uuid.uuid4())
        session_key = f"{self.session_prefix}{session_id}"
        
        session_data = {
            'user_id': user_id,
            'user_data': user_data,
            'created_at': datetime.utcnow().isoformat(),
            'last_accessed': datetime.utcnow().isoformat()
        }
        
        ttl = ttl or self.default_ttl
        await self.redis.setex(
            session_key,
            ttl,
            json.dumps(session_data, default=str)
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """取得會話資料"""
        session_key = f"{self.session_prefix}{session_id}"
        session_data = await self.redis.get(session_key)
        
        if session_data:
            data = json.loads(session_data)
            # 更新最後存取時間
            data['last_accessed'] = datetime.utcnow().isoformat()
            await self.redis.setex(
                session_key,
                self.default_ttl,
                json.dumps(data, default=str)
            )
            return data
        
        return None
    
    async def update_session(self, 
                           session_id: str,
                           update_data: Dict[str, Any]) -> bool:
        """更新會話資料"""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False
        
        session_data['user_data'].update(update_data)
        session_data['last_accessed'] = datetime.utcnow().isoformat()
        
        session_key = f"{self.session_prefix}{session_id}"
        await self.redis.setex(
            session_key,
            self.default_ttl,
            json.dumps(session_data, default=str)
        )
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """刪除會話"""
        session_key = f"{self.session_prefix}{session_id}"
        return await self.redis.delete(session_key) > 0
```

## 💾 檔案儲存架構

### 分層儲存策略
```python
# storage/tiered_storage.py
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path
import asyncio
import aiofiles

class StorageTier(Enum):
    """儲存層級"""
    HOT = "hot"      # SSD - 經常存取的檔案
    WARM = "warm"    # 混合儲存 - 偶爾存取的檔案  
    COLD = "cold"    # HDD - 很少存取的檔案
    ARCHIVE = "archive"  # 磁帶/雲端 - 歸檔檔案

class StorageManager:
    """分層儲存管理器"""
    
    def __init__(self, config: Dict[str, str]):
        self.storage_paths = {
            StorageTier.HOT: Path(config['hot_storage_path']),
            StorageTier.WARM: Path(config['warm_storage_path']),
            StorageTier.COLD: Path(config['cold_storage_path']),
            StorageTier.ARCHIVE: Path(config['archive_storage_path'])
        }
        
        # 確保目錄存在
        for path in self.storage_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def determine_storage_tier(self, 
                             file_type: str,
                             access_frequency: int,
                             file_age_days: int) -> StorageTier:
        """決定儲存層級"""
        
        # 新上傳的 DICOM 檔案放在 HOT 層
        if file_type == 'dicom' and file_age_days < 7:
            return StorageTier.HOT
        
        # 經常存取的檔案
        if access_frequency > 10:
            return StorageTier.HOT
        
        # 偶爾存取的檔案
        if access_frequency > 2 or file_age_days < 30:
            return StorageTier.WARM
        
        # 很少存取的檔案
        if file_age_days < 365:
            return StorageTier.COLD
        
        # 歸檔檔案
        return StorageTier.ARCHIVE
    
    async def store_file(self, 
                        file_data: bytes,
                        filename: str,
                        tier: StorageTier) -> str:
        """儲存檔案到指定層級"""
        
        storage_path = self.storage_paths[tier]
        file_path = storage_path / filename
        
        # 確保子目錄存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)
        
        return str(file_path)
    
    async def move_file(self, 
                       current_path: str,
                       target_tier: StorageTier) -> str:
        """移動檔案到不同儲存層級"""
        
        current_file = Path(current_path)
        if not current_file.exists():
            raise FileNotFoundError(f"檔案不存在: {current_path}")
        
        target_path = self.storage_paths[target_tier] / current_file.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 移動檔案
        import shutil
        shutil.move(str(current_file), str(target_path))
        
        return str(target_path)
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """取得檔案資訊"""
        path = Path(file_path)
        if not path.exists():
            return {}
        
        stat = path.stat()
        return {
            'size': stat.st_size,
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'tier': self._get_tier_from_path(path)
        }
    
    def _get_tier_from_path(self, file_path: Path) -> StorageTier:
        """從路徑判斷儲存層級"""
        for tier, tier_path in self.storage_paths.items():
            if tier_path in file_path.parents:
                return tier
        return StorageTier.COLD
```

### 檔案生命週期管理
```python
# storage/lifecycle.py
from typing import List, Dict
from datetime import datetime, timedelta

class FileLifecycleManager:
    """檔案生命週期管理器"""
    
    def __init__(self, 
                 storage_manager: StorageManager,
                 database_service):
        self.storage = storage_manager
        self.db = database_service
        
        # 生命週期規則
        self.lifecycle_rules = {
            'dicom_hot_to_warm': {
                'condition': lambda f: f['file_type'] == 'dicom' and f['age_days'] > 30,
                'action': 'move_to_warm'
            },
            'warm_to_cold': {
                'condition': lambda f: f['age_days'] > 90 and f['access_count'] < 5,
                'action': 'move_to_cold'
            },
            'cold_to_archive': {
                'condition': lambda f: f['age_days'] > 365,
                'action': 'move_to_archive'
            },
            'delete_temp_files': {
                'condition': lambda f: f['file_type'] == 'temp' and f['age_days'] > 7,
                'action': 'delete'
            }
        }
    
    async def apply_lifecycle_policies(self):
        """套用生命週期政策"""
        
        # 取得所有檔案
        files = await self.db.get_all_files_with_stats()
        
        for file_info in files:
            await self._apply_rules_to_file(file_info)
    
    async def _apply_rules_to_file(self, file_info: Dict):
        """對單一檔案套用規則"""
        
        for rule_name, rule in self.lifecycle_rules.items():
            if rule['condition'](file_info):
                await self._execute_action(file_info, rule['action'])
                
                # 記錄生命週期操作
                await self.db.log_lifecycle_action(
                    file_id=file_info['id'],
                    action=rule['action'],
                    rule_name=rule_name
                )
    
    async def _execute_action(self, file_info: Dict, action: str):
        """執行生命週期動作"""
        
        if action == 'move_to_warm':
            new_path = await self.storage.move_file(
                file_info['file_path'], 
                StorageTier.WARM
            )
            await self.db.update_file_path(file_info['id'], new_path)
            
        elif action == 'move_to_cold':
            new_path = await self.storage.move_file(
                file_info['file_path'], 
                StorageTier.COLD
            )
            await self.db.update_file_path(file_info['id'], new_path)
            
        elif action == 'move_to_archive':
            new_path = await self.storage.move_file(
                file_info['file_path'], 
                StorageTier.ARCHIVE
            )
            await self.db.update_file_path(file_info['id'], new_path)
            
        elif action == 'delete':
            await self.storage.delete_file(file_info['file_path'])
            await self.db.mark_file_deleted(file_info['id'])
```

## 🔐 資料安全和加密

### 資料加密策略
```python
# security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    """資料加密服務"""
    
    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self._setup_encryption()
    
    def _setup_encryption(self):
        """設置加密器"""
        # 使用 PBKDF2 派生密鑰
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher = Fernet(key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感資料"""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感資料"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def encrypt_file(self, file_path: str) -> str:
        """加密檔案"""
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        encrypted_path = f"{file_path}.encrypted"
        with open(encrypted_path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted_data)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_path: str, output_path: str) -> str:
        """解密檔案"""
        with open(encrypted_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as output_file:
            output_file.write(decrypted_data)
        
        return output_path

# 資料庫層級加密
class DatabaseEncryption:
    """資料庫加密服務"""
    
    def __init__(self, encryption_service: DataEncryption):
        self.encryption = encryption_service
        
        # 需要加密的欄位
        self.encrypted_fields = {
            'users': ['email', 'full_name'],
            'files': ['patient_id'],
            'processing_jobs': ['result_data']
        }
    
    def encrypt_row_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """加密資料列資料"""
        if table not in self.encrypted_fields:
            return data
        
        encrypted_data = data.copy()
        for field in self.encrypted_fields[table]:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encryption.encrypt_sensitive_data(
                    str(encrypted_data[field])
                )
        
        return encrypted_data
    
    def decrypt_row_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """解密資料列資料"""
        if table not in self.encrypted_fields:
            return data
        
        decrypted_data = data.copy()
        for field in self.encrypted_fields[table]:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = self.encryption.decrypt_sensitive_data(
                        decrypted_data[field]
                    )
                except Exception:
                    # 如果解密失敗，保持原值（可能是未加密的舊資料）
                    pass
        
        return decrypted_data
```

## 📊 資料監控和分析

### 資料品質監控
```python
# monitoring/data_quality.py
from typing import Dict, List, Any
from datetime import datetime, timedelta

class DataQualityMonitor:
    """資料品質監控器"""
    
    def __init__(self, database_service):
        self.db = database_service
        self.quality_rules = {
            'completeness': self._check_completeness,
            'uniqueness': self._check_uniqueness,
            'validity': self._check_validity,
            'consistency': self._check_consistency
        }
    
    async def run_quality_checks(self) -> Dict[str, Any]:
        """執行資料品質檢查"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {},
            'overall_score': 0.0
        }
        
        total_score = 0
        check_count = 0
        
        for check_name, check_function in self.quality_rules.items():
            try:
                check_result = await check_function()
                results['checks'][check_name] = check_result
                total_score += check_result['score']
                check_count += 1
            except Exception as e:
                results['checks'][check_name] = {
                    'score': 0.0,
                    'error': str(e)
                }
        
        if check_count > 0:
            results['overall_score'] = total_score / check_count
        
        return results
    
    async def _check_completeness(self) -> Dict[str, Any]:
        """檢查資料完整性"""
        
        # 檢查必要欄位的空值比例
        tables_to_check = {
            'users': ['username', 'email', 'full_name'],
            'files': ['filename', 'file_path', 'checksum'],
            'processing_jobs': ['job_type', 'status']
        }
        
        completeness_scores = []
        
        for table, required_fields in tables_to_check.items():
            for field in required_fields:
                null_count = await self.db.count_null_values(table, field)
                total_count = await self.db.count_total_records(table)
                
                if total_count > 0:
                    completeness = 1 - (null_count / total_count)
                    completeness_scores.append(completeness)
        
        average_completeness = sum(completeness_scores) / len(completeness_scores)
        
        return {
            'score': average_completeness,
            'details': {
                'checked_fields': len(completeness_scores),
                'average_completeness': average_completeness
            }
        }
    
    async def _check_uniqueness(self) -> Dict[str, Any]:
        """檢查資料唯一性"""
        
        # 檢查應該唯一的欄位
        uniqueness_checks = [
            ('users', 'username'),
            ('users', 'email'),
            ('files', 'checksum')
        ]
        
        uniqueness_scores = []
        
        for table, field in uniqueness_checks:
            duplicate_count = await self.db.count_duplicates(table, field)
            total_count = await self.db.count_total_records(table)
            
            if total_count > 0:
                uniqueness = 1 - (duplicate_count / total_count)
                uniqueness_scores.append(uniqueness)
        
        average_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores)
        
        return {
            'score': average_uniqueness,
            'details': {
                'checked_fields': len(uniqueness_scores),
                'average_uniqueness': average_uniqueness
            }
        }
```

## 🔄 資料備份和恢復

### 備份策略
```python
# backup/strategy.py
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List

class BackupManager:
    """備份管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_strategies = {
            'full': self._full_backup,
            'incremental': self._incremental_backup,
            'differential': self._differential_backup
        }
    
    async def create_backup(self, backup_type: str = 'incremental') -> Dict[str, Any]:
        """建立備份"""
        
        backup_info = {
            'backup_id': f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'backup_type': backup_type,
            'started_at': datetime.utcnow().isoformat(),
            'databases': [],
            'files': [],
            'status': 'running'
        }
        
        try:
            # 執行資料庫備份
            db_backup_result = await self._backup_databases()
            backup_info['databases'] = db_backup_result
            
            # 執行檔案備份
            file_backup_result = await self._backup_files(backup_type)
            backup_info['files'] = file_backup_result
            
            backup_info['status'] = 'completed'
            backup_info['completed_at'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            backup_info['status'] = 'failed'
            backup_info['error'] = str(e)
        
        return backup_info
    
    async def _backup_databases(self) -> List[Dict[str, Any]]:
        """備份所有資料庫"""
        
        databases = [
            'user_service_db',
            'file_service_db', 
            'processing_service_db',
            'audit_log_db'
        ]
        
        backup_results = []
        
        for db_name in databases:
            try:
                backup_file = f"/backups/db/{db_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql"
                
                # 使用 pg_dump 備份
                cmd = [
                    'pg_dump',
                    '-h', self.config['db_host'],
                    '-U', self.config['db_user'],
                    '-d', db_name,
                    '-f', backup_file,
                    '--verbose'
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    backup_results.append({
                        'database': db_name,
                        'backup_file': backup_file,
                        'status': 'success',
                        'size': os.path.getsize(backup_file)
                    })
                else:
                    backup_results.append({
                        'database': db_name,
                        'status': 'failed',
                        'error': stderr.decode()
                    })
                    
            except Exception as e:
                backup_results.append({
                    'database': db_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return backup_results
    
    async def _backup_files(self, backup_type: str) -> Dict[str, Any]:
        """備份檔案"""
        
        backup_paths = [
            self.config['hot_storage_path'],
            self.config['warm_storage_path']
        ]
        
        backup_destination = f"/backups/files/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if backup_type == 'full':
                # 完整備份
                cmd = ['rsync', '-av', '--progress'] + backup_paths + [backup_destination]
            else:
                # 增量備份
                cmd = ['rsync', '-av', '--progress', '--link-dest=/backups/files/latest'] + backup_paths + [backup_destination]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # 更新最新備份連結
                if os.path.exists('/backups/files/latest'):
                    os.remove('/backups/files/latest')
                os.symlink(backup_destination, '/backups/files/latest')
                
                return {
                    'status': 'success',
                    'backup_path': backup_destination,
                    'output': stdout.decode()
                }
            else:
                return {
                    'status': 'failed',
                    'error': stderr.decode()
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
```

---

**注意**: 本資料架構設計考慮了醫學影像處理的特殊需求，包括大檔案處理、資料合規性和長期保存要求。所有敏感資料都採用加密保護，並建立了完整的備份和恢復機制。

