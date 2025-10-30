# Django + Django Ninja 医疗影像AI系统重新设计方案

## 目录
- [技术栈选型理由](#技术栈选型理由)
- [Django项目架构设计](#django项目架构设计)
- [数据模型设计](#数据模型设计)
- [API设计 (Django Ninja)](#api设计-django-ninja)
- [异步任务处理](#异步任务处理)
- [缓存策略](#缓存策略)
- [认证授权](#认证授权)
- [部署配置](#部署配置)
- [迁移路径](#迁移路径)

---

## 技术栈选型理由

### 为什么选择Django + Django Ninja?

#### Django的优势
1. **成熟的ORM**: Django ORM功能强大,支持复杂查询和事务
2. **强大的Admin界面**: 自动生成管理后台,快速开发
3. **安全性内置**: CSRF保护、SQL注入防护、XSS过滤
4. **中间件系统**: 灵活的请求/响应处理链
5. **完善的生态**: 丰富的第三方包和社区支持

#### Django Ninja的优势
1. **FastAPI风格API**: Pydantic验证、自动文档生成
2. **类型安全**: 完整的类型提示支持
3. **高性能**: 基于Pydantic,性能接近FastAPI
4. **异步支持**: 原生支持async/await
5. **Django集成**: 无缝集成Django ORM和认证系统

#### 技术栈对比

| 特性 | FastAPI (当前) | Django + Django Ninja | 优势方 |
|-----|---------------|---------------------|--------|
| 开发速度 | 中等 | 快速 (Admin+ORM) | Django |
| 性能 | 极高 | 高 | FastAPI |
| 类型安全 | 优秀 | 优秀 | 平手 |
| 生态成熟度 | 良好 | 极佳 | Django |
| 学习曲线 | 平缓 | 中等 | FastAPI |
| Admin界面 | 无 | 自动生成 | Django |
| 异步支持 | 原生 | 需配置 | FastAPI |
| 医疗合规 | 需自建 | 成熟方案 | Django |

**结论**: 对于医疗AI系统,Django的成熟度和安全性更适合,Django Ninja提供现代API体验。

---

## Django项目架构设计

### 1. 项目结构

```
brain_parcellation/                    # Django项目根目录
├── manage.py                          # Django管理脚本
├── pyproject.toml                     # 依赖管理
├── docker-compose.yml                 # Docker编排
├── Dockerfile                         # 容器构建
│
├── config/                            # 项目配置
│   ├── __init__.py
│   ├── settings/                      # 分环境配置
│   │   ├── __init__.py
│   │   ├── base.py                    # 基础配置
│   │   ├── development.py             # 开发环境
│   │   ├── staging.py                 # 测试环境
│   │   └── production.py              # 生产环境
│   ├── urls.py                        # 根URL配置
│   ├── wsgi.py                        # WSGI入口
│   └── asgi.py                        # ASGI入口(异步)
│
├── apps/                              # Django应用目录
│   │
│   ├── core/                          # 核心应用
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py                  # 基础模型
│   │   ├── admin.py                   # Admin配置
│   │   ├── middleware.py              # 自定义中间件
│   │   └── management/                # 管理命令
│   │       └── commands/
│   │
│   ├── users/                         # 用户管理
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py                  # User, Role, Permission
│   │   ├── admin.py
│   │   ├── api.py                     # Django Ninja API
│   │   ├── schemas.py                 # Pydantic schemas
│   │   ├── services.py                # 业务逻辑层
│   │   ├── selectors.py               # 查询逻辑层
│   │   ├── tests/
│   │   └── migrations/
│   │
│   ├── studies/                       # 研究/病例管理
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py                  # Study, Series, Instance
│   │   ├── admin.py
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   ├── selectors.py
│   │   ├── tasks.py                   # Celery任务
│   │   ├── tests/
│   │   └── migrations/
│   │
│   ├── dicom/                         # DICOM处理
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py                  # DicomFile, DicomSeries
│   │   ├── admin.py
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   ├── converters/                # DICOM转换器
│   │   │   ├── __init__.py
│   │   │   ├── dicom_to_nifti.py
│   │   │   └── nifti_to_dicom.py
│   │   ├── tasks.py
│   │   ├── tests/
│   │   └── migrations/
│   │
│   ├── ai_processing/                 # AI处理引擎
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py                  # ProcessingJob, Result
│   │   ├── admin.py
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   ├── pipelines/                 # 处理管道
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── wmh_pipeline.py
│   │   │   ├── cmb_pipeline.py
│   │   │   └── dwi_pipeline.py
│   │   ├── models_ai/                 # AI模型管理
│   │   │   ├── __init__.py
│   │   │   ├── synthseg.py
│   │   │   └── loader.py
│   │   ├── tasks.py
│   │   ├── tests/
│   │   └── migrations/
│   │
│   ├── results/                       # 结果管理
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py                  # ProcessingResult, Metrics
│   │   ├── admin.py
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   ├── exporters/                 # 结果导出
│   │   │   ├── __init__.py
│   │   │   ├── pdf_exporter.py
│   │   │   └── dicom_seg_exporter.py
│   │   ├── tests/
│   │   └── migrations/
│   │
│   └── audit/                         # 审计日志
│       ├── __init__.py
│       ├── apps.py
│       ├── models.py                  # AuditLog
│       ├── admin.py
│       ├── api.py
│       ├── middleware.py              # 审计中间件
│       ├── tests/
│       └── migrations/
│
├── libs/                              # 共享库
│   ├── __init__.py
│   ├── cache.py                       # 缓存工具
│   ├── exceptions.py                  # 自定义异常
│   ├── permissions.py                 # 权限检查
│   ├── pagination.py                  # 分页工具
│   └── utils.py                       # 通用工具
│
├── static/                            # 静态文件
│   ├── css/
│   ├── js/
│   └── images/
│
├── media/                             # 上传文件
│   ├── dicom/
│   ├── nifti/
│   └── results/
│
├── templates/                         # Django模板
│   ├── base.html
│   └── admin/
│
├── locale/                            # 国际化
│   ├── en/
│   └── zh_TW/
│
└── tests/                             # 集成测试
    ├── __init__.py
    ├── conftest.py
    └── integration/
```

### 2. 设计模式和最佳实践

#### Service-Selector模式

```python
# apps/studies/selectors.py
from typing import List, Optional
from django.db.models import QuerySet, Prefetch
from .models import Study, Series

class StudySelector:
    """
    查询逻辑层 - 负责所有数据库查询
    遵循单一职责原则,与业务逻辑分离
    """

    @staticmethod
    def get_by_id(study_id: int) -> Optional[Study]:
        """获取单个研究"""
        return Study.objects.select_related(
            'patient',
            'institution'
        ).prefetch_related(
            Prefetch(
                'series_set',
                queryset=Series.objects.select_related('modality')
            )
        ).filter(id=study_id).first()

    @staticmethod
    def list_by_patient(
        patient_id: int,
        status: Optional[str] = None
    ) -> QuerySet[Study]:
        """按患者列出研究"""
        qs = Study.objects.filter(patient_id=patient_id)
        if status:
            qs = qs.filter(processing_status=status)
        return qs.select_related('patient').order_by('-created_at')

    @staticmethod
    def list_pending_processing() -> QuerySet[Study]:
        """获取待处理的研究"""
        return Study.objects.filter(
            processing_status='pending'
        ).select_related('patient').order_by('created_at')


# apps/studies/services.py
from typing import Dict, Any
from django.db import transaction
from .models import Study, Series
from .selectors import StudySelector
from apps.audit.services import AuditService

class StudyService:
    """
    业务逻辑层 - 负责业务规则和数据修改
    """

    def __init__(self):
        self.selector = StudySelector()
        self.audit = AuditService()

    @transaction.atomic
    def create_study(
        self,
        patient_id: int,
        study_data: Dict[str, Any],
        user_id: int
    ) -> Study:
        """创建新研究"""
        # 业务验证
        if not self._validate_patient(patient_id):
            raise ValueError("Invalid patient")

        # 创建研究
        study = Study.objects.create(
            patient_id=patient_id,
            **study_data
        )

        # 记录审计日志
        self.audit.log_create(
            model='Study',
            object_id=study.id,
            user_id=user_id,
            data=study_data
        )

        return study

    @transaction.atomic
    def update_processing_status(
        self,
        study_id: int,
        new_status: str,
        user_id: int
    ) -> Study:
        """更新处理状态"""
        study = self.selector.get_by_id(study_id)
        if not study:
            raise ValueError("Study not found")

        old_status = study.processing_status
        study.processing_status = new_status
        study.save(update_fields=['processing_status', 'updated_at'])

        # 审计日志
        self.audit.log_update(
            model='Study',
            object_id=study_id,
            user_id=user_id,
            changes={
                'processing_status': {
                    'old': old_status,
                    'new': new_status
                }
            }
        )

        return study

    def _validate_patient(self, patient_id: int) -> bool:
        """验证患者是否存在"""
        from apps.users.models import Patient
        return Patient.objects.filter(id=patient_id).exists()


# apps/studies/api.py
from ninja import Router, Query
from typing import List
from django.shortcuts import get_object_or_404
from .schemas import StudySchema, StudyCreateSchema, StudyUpdateSchema
from .services import StudyService
from .selectors import StudySelector
from libs.permissions import require_permission

router = Router(tags=['Studies'])
study_service = StudyService()
study_selector = StudySelector()


@router.get('/', response=List[StudySchema])
@require_permission('studies.view_study')
def list_studies(
    request,
    patient_id: int = Query(None),
    status: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """列出研究"""
    if patient_id:
        qs = study_selector.list_by_patient(patient_id, status)
    else:
        qs = Study.objects.all()

    return qs[offset:offset + limit]


@router.get('/{study_id}', response=StudySchema)
@require_permission('studies.view_study')
def get_study(request, study_id: int):
    """获取单个研究"""
    study = study_selector.get_by_id(study_id)
    if not study:
        return {'error': 'Study not found'}, 404
    return study


@router.post('/', response=StudySchema)
@require_permission('studies.add_study')
def create_study(request, payload: StudyCreateSchema):
    """创建研究"""
    study = study_service.create_study(
        patient_id=payload.patient_id,
        study_data=payload.dict(exclude={'patient_id'}),
        user_id=request.user.id
    )
    return study


@router.patch('/{study_id}', response=StudySchema)
@require_permission('studies.change_study')
def update_study(request, study_id: int, payload: StudyUpdateSchema):
    """更新研究"""
    study = study_service.update_study(
        study_id=study_id,
        update_data=payload.dict(exclude_unset=True),
        user_id=request.user.id
    )
    return study
```

---

## 数据模型设计

### 1. 核心模型

```python
# apps/core/models.py
from django.db import models
from django.utils import timezone
import uuid

class TimeStampedModel(models.Model):
    """抽象基类 - 时间戳"""
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UUIDModel(models.Model):
    """抽象基类 - UUID主键"""
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )

    class Meta:
        abstract = True


class SoftDeleteManager(models.Manager):
    """软删除管理器"""
    def get_queryset(self):
        return super().get_queryset().filter(deleted_at__isnull=True)


class SoftDeleteModel(models.Model):
    """抽象基类 - 软删除"""
    deleted_at = models.DateTimeField(null=True, blank=True, db_index=True)

    objects = SoftDeleteManager()
    all_objects = models.Manager()  # 包含已删除

    def soft_delete(self):
        """软删除"""
        self.deleted_at = timezone.now()
        self.save(update_fields=['deleted_at'])

    def restore(self):
        """恢复"""
        self.deleted_at = None
        self.save(update_fields=['deleted_at'])

    class Meta:
        abstract = True


# apps/users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from apps.core.models import TimeStampedModel, UUIDModel

class User(AbstractUser, UUIDModel, TimeStampedModel):
    """用户模型"""
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    organization = models.ForeignKey(
        'Organization',
        on_delete=models.CASCADE,
        related_name='users'
    )
    role = models.ForeignKey(
        'Role',
        on_delete=models.PROTECT,
        related_name='users'
    )
    is_verified = models.BooleanField(default=False)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['organization', 'role']),
        ]


class Organization(UUIDModel, TimeStampedModel):
    """机构模型"""
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=50, unique=True)
    type = models.CharField(
        max_length=50,
        choices=[
            ('hospital', 'Hospital'),
            ('clinic', 'Clinic'),
            ('research', 'Research Institute'),
        ]
    )
    contact_email = models.EmailField()
    contact_phone = models.CharField(max_length=20)
    address = models.TextField()
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'organizations'
        verbose_name = 'Organization'
        verbose_name_plural = 'Organizations'


class Role(UUIDModel, TimeStampedModel):
    """角色模型"""
    name = models.CharField(max_length=100, unique=True)
    code = models.SlugField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='roles'
    )
    is_system = models.BooleanField(default=False)  # 系统角色不可删除

    class Meta:
        db_table = 'roles'
        verbose_name = 'Role'
        verbose_name_plural = 'Roles'


# apps/studies/models.py
from django.db import models
from apps.core.models import TimeStampedModel, UUIDModel, SoftDeleteModel

class Patient(UUIDModel, TimeStampedModel, SoftDeleteModel):
    """患者模型"""
    patient_id = models.CharField(max_length=100, unique=True, db_index=True)
    name = models.CharField(max_length=255)  # 加密存储
    date_of_birth = models.DateField()
    gender = models.CharField(
        max_length=10,
        choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')]
    )
    organization = models.ForeignKey(
        'users.Organization',
        on_delete=models.CASCADE,
        related_name='patients'
    )

    class Meta:
        db_table = 'patients'
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'
        indexes = [
            models.Index(fields=['patient_id', 'organization']),
            models.Index(fields=['date_of_birth']),
        ]


class Study(UUIDModel, TimeStampedModel, SoftDeleteModel):
    """研究/检查模型"""
    study_uid = models.CharField(max_length=255, unique=True, db_index=True)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name='studies'
    )
    study_date = models.DateField(db_index=True)
    study_time = models.TimeField(null=True, blank=True)
    study_description = models.TextField(blank=True)
    modality = models.CharField(max_length=20)  # MR, CT, etc.
    institution = models.ForeignKey(
        'users.Organization',
        on_delete=models.CASCADE,
        related_name='studies'
    )
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
        ],
        default='pending',
        db_index=True
    )
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_completed_at = models.DateTimeField(null=True, blank=True)
    priority = models.IntegerField(
        default=1,
        choices=[(1, 'Low'), (2, 'Normal'), (3, 'High'), (4, 'Urgent')]
    )

    class Meta:
        db_table = 'studies'
        verbose_name = 'Study'
        verbose_name_plural = 'Studies'
        indexes = [
            models.Index(fields=['study_uid']),
            models.Index(fields=['patient', 'study_date']),
            models.Index(fields=['processing_status', 'priority']),
            models.Index(fields=['institution', 'study_date']),
        ]
        ordering = ['-study_date', '-created_at']


class Series(UUIDModel, TimeStampedModel):
    """序列模型"""
    series_uid = models.CharField(max_length=255, unique=True, db_index=True)
    study = models.ForeignKey(
        Study,
        on_delete=models.CASCADE,
        related_name='series'
    )
    series_number = models.IntegerField()
    series_description = models.TextField(blank=True)
    modality = models.CharField(max_length=20)
    series_type = models.CharField(
        max_length=50,
        choices=[
            ('T1', 'T1-weighted'),
            ('T2', 'T2-weighted'),
            ('FLAIR', 'FLAIR'),
            ('DWI', 'Diffusion Weighted'),
            ('SWI', 'Susceptibility Weighted'),
            ('MRA', 'MR Angiography'),
        ],
        null=True,
        blank=True
    )
    orientation = models.CharField(
        max_length=20,
        choices=[
            ('axial', 'Axial'),
            ('sagittal', 'Sagittal'),
            ('coronal', 'Coronal'),
        ],
        null=True,
        blank=True
    )
    instance_count = models.IntegerField(default=0)

    class Meta:
        db_table = 'series'
        verbose_name = 'Series'
        verbose_name_plural = 'Series'
        indexes = [
            models.Index(fields=['series_uid']),
            models.Index(fields=['study', 'series_number']),
            models.Index(fields=['series_type']),
        ]
        unique_together = [['study', 'series_number']]


# apps/ai_processing/models.py
from django.db import models
from apps.core.models import TimeStampedModel, UUIDModel

class ProcessingJob(UUIDModel, TimeStampedModel):
    """AI处理任务模型"""
    study = models.ForeignKey(
        'studies.Study',
        on_delete=models.CASCADE,
        related_name='processing_jobs'
    )
    pipeline_type = models.CharField(
        max_length=50,
        choices=[
            ('wmh', 'White Matter Hyperintensities'),
            ('cmb', 'Cerebral Microbleeds'),
            ('dwi', 'Diffusion Weighted Imaging'),
            ('aneurysm', 'Aneurysm Detection'),
            ('synthseg', 'Brain Segmentation'),
        ]
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ('queued', 'Queued'),
            ('running', 'Running'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('cancelled', 'Cancelled'),
        ],
        default='queued',
        db_index=True
    )
    priority = models.IntegerField(default=1)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)

    class Meta:
        db_table = 'processing_jobs'
        verbose_name = 'Processing Job'
        verbose_name_plural = 'Processing Jobs'
        indexes = [
            models.Index(fields=['status', 'priority']),
            models.Index(fields=['study', 'pipeline_type']),
            models.Index(fields=['celery_task_id']),
        ]


class ProcessingResult(UUIDModel, TimeStampedModel):
    """处理结果模型"""
    job = models.ForeignKey(
        ProcessingJob,
        on_delete=models.CASCADE,
        related_name='results'
    )
    result_type = models.CharField(max_length=50)  # segmentation, detection, metrics
    file_path = models.CharField(max_length=500)
    file_format = models.CharField(max_length=20)  # nifti, dicom, json
    file_size = models.BigIntegerField()  # bytes
    metadata = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)

    class Meta:
        db_table = 'processing_results'
        verbose_name = 'Processing Result'
        verbose_name_plural = 'Processing Results'
        indexes = [
            models.Index(fields=['job', 'result_type']),
        ]


# apps/audit/models.py
from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from apps.core.models import TimeStampedModel, UUIDModel

class AuditLog(UUIDModel, TimeStampedModel):
    """审计日志模型"""
    user = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        related_name='audit_logs'
    )
    action = models.CharField(
        max_length=20,
        choices=[
            ('create', 'Create'),
            ('read', 'Read'),
            ('update', 'Update'),
            ('delete', 'Delete'),
        ],
        db_index=True
    )
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.CharField(max_length=255)
    content_object = GenericForeignKey('content_type', 'object_id')
    changes = models.JSONField(default=dict)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)

    class Meta:
        db_table = 'audit_logs'
        verbose_name = 'Audit Log'
        verbose_name_plural = 'Audit Logs'
        indexes = [
            models.Index(fields=['user', 'action', 'created_at']),
            models.Index(fields=['content_type', 'object_id']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
```

### 2. 数据库优化配置

```python
# config/settings/base.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'medical_ai'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
        'ATOMIC_REQUESTS': True,  # 自动事务
        'CONN_MAX_AGE': 600,  # 连接复用
        'OPTIONS': {
            'connect_timeout': 10,
            'options': '-c statement_timeout=30000',  # 30秒查询超时
        }
    },
    # 只读副本
    'replica': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_REPLICA_NAME', 'medical_ai'),
        'USER': os.getenv('DB_REPLICA_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_REPLICA_PASSWORD'),
        'HOST': os.getenv('DB_REPLICA_HOST', 'localhost'),
        'PORT': os.getenv('DB_REPLICA_PORT', '5433'),
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}

# 数据库路由 - 读写分离
class ReplicaRouter:
    """读写分离路由器"""

    def db_for_read(self, model, **hints):
        """读操作使用副本"""
        return 'replica'

    def db_for_write(self, model, **hints):
        """写操作使用主库"""
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        """允许关系"""
        return True

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """只在主库执行迁移"""
        return db == 'default'


DATABASE_ROUTERS = ['config.db_routers.ReplicaRouter']

# 连接池配置 (使用django-db-geventpool)
DATABASES['default']['ENGINE'] = 'django_db_geventpool.backends.postgresql_psycopg2'
DATABASES['default']['CONN_MAX_AGE'] = None
DATABASES['default']['OPTIONS'] = {
    'MAX_CONNS': 20,
    'REUSE_CONNS': True,
}
```

---

## API设计 (Django Ninja)

### 1. 主API配置

```python
# config/api.py
from ninja import NinjaAPI
from ninja.security import HttpBearer
from apps.users.api import router as users_router
from apps.studies.api import router as studies_router
from apps.dicom.api import router as dicom_router
from apps.ai_processing.api import router as ai_router
from apps.results.api import router as results_router

class AuthBearer(HttpBearer):
    """JWT认证"""
    def authenticate(self, request, token):
        from apps.users.services import AuthService
        user = AuthService.verify_token(token)
        return user


api = NinjaAPI(
    title='Medical AI API',
    version='2.0.0',
    description='Brain Parcellation Medical Imaging AI System',
    auth=AuthBearer(),
    docs_url='/api/docs',
)

# 注册路由
api.add_router('/users', users_router)
api.add_router('/studies', studies_router)
api.add_router('/dicom', dicom_router)
api.add_router('/ai-processing', ai_router)
api.add_router('/results', results_router)

# 健康检查端点 (无需认证)
@api.get('/health', auth=None)
def health_check(request):
    """健康检查"""
    return {'status': 'healthy', 'version': '2.0.0'}

@api.get('/ready', auth=None)
def readiness_check(request):
    """就绪检查"""
    from django.db import connections
    from django.core.cache import cache

    # 检查数据库
    try:
        connections['default'].cursor()
        db_ok = True
    except Exception:
        db_ok = False

    # 检查Redis
    try:
        cache.set('healthcheck', 'ok', 1)
        cache_ok = cache.get('healthcheck') == 'ok'
    except Exception:
        cache_ok = False

    ready = db_ok and cache_ok
    return {
        'ready': ready,
        'checks': {
            'database': db_ok,
            'cache': cache_ok,
        }
    }


# config/urls.py
from django.contrib import admin
from django.urls import path
from .api import api

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', api.urls),
]
```

### 2. Pydantic Schemas

```python
# apps/studies/schemas.py
from ninja import Schema, Field
from typing import Optional, List
from datetime import datetime, date
from pydantic import validator

class PatientSchema(Schema):
    """患者响应Schema"""
    id: str
    patient_id: str
    name: str
    date_of_birth: date
    gender: str
    created_at: datetime


class StudyCreateSchema(Schema):
    """创建研究Schema"""
    patient_id: int
    study_uid: str = Field(..., max_length=255)
    study_date: date
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    modality: str = Field(..., max_length=20)
    priority: int = Field(default=1, ge=1, le=4)

    @validator('study_uid')
    def validate_study_uid(cls, v):
        """验证Study UID格式"""
        if not v or len(v) < 10:
            raise ValueError('Invalid Study UID format')
        return v


class StudyUpdateSchema(Schema):
    """更新研究Schema"""
    study_description: Optional[str] = None
    processing_status: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=4)


class SeriesSchema(Schema):
    """序列响应Schema"""
    id: str
    series_uid: str
    series_number: int
    series_description: str
    modality: str
    series_type: Optional[str]
    orientation: Optional[str]
    instance_count: int


class StudySchema(Schema):
    """研究响应Schema"""
    id: str
    study_uid: str
    patient: PatientSchema
    study_date: date
    study_time: Optional[str]
    study_description: str
    modality: str
    processing_status: str
    priority: int
    series: List[SeriesSchema]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class StudyListSchema(Schema):
    """研究列表Schema (分页)"""
    count: int
    next: Optional[str]
    previous: Optional[str]
    results: List[StudySchema]
```

---

## 异步任务处理

### 1. Celery配置

```python
# config/celery.py
import os
from celery import Celery
from celery.schedules import crontab

os.setenv('DJANGO_SETTINGS_MODULE', 'config.settings.production')

app = Celery('brain_parcellation')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# 定时任务
app.conf.beat_schedule = {
    'check-pending-jobs': {
        'task': 'apps.ai_processing.tasks.check_pending_jobs',
        'schedule': crontab(minute='*/5'),  # 每5分钟
    },
    'cleanup-old-files': {
        'task': 'apps.core.tasks.cleanup_old_files',
        'schedule': crontab(hour=2, minute=0),  # 每天凌晨2点
    },
    'generate-daily-report': {
        'task': 'apps.results.tasks.generate_daily_report',
        'schedule': crontab(hour=8, minute=0),  # 每天早上8点
    },
}

# Celery配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Taipei',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1小时硬限制
    task_soft_time_limit=3000,  # 50分钟软限制
    worker_prefetch_multiplier=1,  # 预取1个任务
    worker_max_tasks_per_child=100,  # 100个任务后重启worker
)


# config/settings/base.py
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/2')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/3')
CELERY_CACHE_BACKEND = 'django-cache'
CELERY_TASK_ALWAYS_EAGER = False  # 生产环境必须False
```

### 2. AI处理任务

```python
# apps/ai_processing/tasks.py
from celery import shared_task
from celery.utils.log import get_task_logger
from django.utils import timezone
from .models import ProcessingJob
from .services import AIProcessingService
from apps.studies.selectors import StudySelector

logger = get_task_logger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=300  # 5分钟后重试
)
def process_study(self, job_id: str):
    """
    处理研究的Celery任务

    Args:
        job_id: ProcessingJob的UUID
    """
    try:
        # 获取任务
        job = ProcessingJob.objects.get(id=job_id)

        # 更新状态
        job.status = 'running'
        job.started_at = timezone.now()
        job.celery_task_id = self.request.id
        job.save(update_fields=['status', 'started_at', 'celery_task_id'])

        # 获取研究数据
        study = StudySelector.get_by_id(job.study_id)
        if not study:
            raise ValueError(f"Study {job.study_id} not found")

        # 执行AI处理
        service = AIProcessingService()
        result = service.process_pipeline(
            study=study,
            pipeline_type=job.pipeline_type
        )

        # 更新任务状态
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.save(update_fields=['status', 'completed_at'])

        logger.info(f"Job {job_id} completed successfully")
        return {'success': True, 'result_id': str(result.id)}

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {str(exc)}", exc_info=True)

        # 更新失败状态
        job.status = 'failed'
        job.error_message = str(exc)
        job.retry_count += 1
        job.save(update_fields=['status', 'error_message', 'retry_count'])

        # 重试
        if job.retry_count < job.max_retries:
            raise self.retry(exc=exc)

        return {'success': False, 'error': str(exc)}


@shared_task
def check_pending_jobs():
    """检查并启动待处理任务"""
    pending_jobs = ProcessingJob.objects.filter(
        status='queued'
    ).order_by('priority', 'created_at')[:10]

    for job in pending_jobs:
        # 异步启动处理任务
        process_study.apply_async(args=[str(job.id)])

    return {'processed': len(pending_jobs)}


@shared_task
def batch_process_studies(study_ids: List[str], pipeline_type: str):
    """批量处理研究"""
    from .services import AIProcessingService

    service = AIProcessingService()
    results = []

    for study_id in study_ids:
        try:
            # 创建处理任务
            job = service.create_processing_job(
                study_id=study_id,
                pipeline_type=pipeline_type
            )
            # 启动任务
            process_study.apply_async(args=[str(job.id)])
            results.append({'study_id': study_id, 'success': True})
        except Exception as e:
            results.append({'study_id': study_id, 'success': False, 'error': str(e)})

    return results
```

---

## 缓存策略

### 1. Redis缓存配置

```python
# config/settings/base.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_CACHE_URL', 'redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
        },
        'KEY_PREFIX': 'medical_ai',
        'TIMEOUT': 300,  # 5分钟默认过期
    },
    # 长期缓存
    'long_term': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_CACHE_URL', 'redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'medical_ai_long',
        'TIMEOUT': 86400,  # 24小时
    },
}

# Session使用Redis
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
```

### 2. 缓存工具类

```python
# libs/cache.py
from django.core.cache import caches
from django.core.cache.utils import make_template_fragment_key
from functools import wraps
from typing import Any, Callable, Optional
import hashlib
import json

def cache_result(
    timeout: int = 300,
    key_prefix: str = '',
    cache_alias: str = 'default'
):
    """
    方法结果缓存装饰器

    Usage:
        @cache_result(timeout=600, key_prefix='study')
        def get_study_details(study_id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存key
            cache_key = _generate_cache_key(
                func.__name__,
                args,
                kwargs,
                prefix=key_prefix
            )

            # 尝试从缓存获取
            cache = caches[cache_alias]
            result = cache.get(cache_key)

            if result is not None:
                return result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result, timeout)

            return result
        return wrapper
    return decorator


def _generate_cache_key(
    func_name: str,
    args: tuple,
    kwargs: dict,
    prefix: str = ''
) -> str:
    """生成缓存key"""
    # 序列化参数
    serialized = json.dumps({
        'args': args,
        'kwargs': kwargs
    }, sort_keys=True, default=str)

    # 生成hash
    key_hash = hashlib.md5(serialized.encode()).hexdigest()[:16]

    return f"{prefix}:{func_name}:{key_hash}"


def invalidate_cache_pattern(pattern: str, cache_alias: str = 'default'):
    """
    清除匹配模式的所有缓存

    Args:
        pattern: 缓存key模式,如 'study:*'
    """
    cache = caches[cache_alias]
    cache.delete_pattern(f"*{pattern}*")


class CachedProperty:
    """带缓存的property装饰器"""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def __call__(self, func):
        @wraps(func)
        def wrapper(instance):
            cache_key = f"{instance.__class__.__name__}:{instance.pk}:{func.__name__}"
            cache = caches['default']

            result = cache.get(cache_key)
            if result is None:
                result = func(instance)
                cache.set(cache_key, result, self.timeout)

            return result
        return property(wrapper)


# 使用示例
from apps.studies.models import Study

class Study(models.Model):
    # ...

    @CachedProperty(timeout=600)
    def total_series_count(self):
        """缓存的计算属性"""
        return self.series.count()

    def invalidate_cache(self):
        """清除该研究的所有缓存"""
        pattern = f"Study:{self.pk}:*"
        invalidate_cache_pattern(pattern)
```

---

## 认证授权

### 1. JWT认证实现

```python
# apps/users/services/auth_service.py
from datetime import datetime, timedelta
from typing import Optional
import jwt
from django.conf import settings
from django.contrib.auth import authenticate
from apps.users.models import User

class AuthService:
    """认证服务"""

    @staticmethod
    def generate_token(user: User) -> dict:
        """生成JWT token"""
        now = datetime.utcnow()
        payload = {
            'user_id': str(user.id),
            'email': user.email,
            'role': user.role.code,
            'iat': now,
            'exp': now + timedelta(seconds=settings.JWT_EXPIRATION_DELTA),
        }

        access_token = jwt.encode(
            payload,
            settings.SECRET_KEY,
            algorithm='HS256'
        )

        refresh_payload = {
            'user_id': str(user.id),
            'iat': now,
            'exp': now + timedelta(days=7),
        }

        refresh_token = jwt.encode(
            refresh_payload,
            settings.SECRET_KEY,
            algorithm='HS256'
        )

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': settings.JWT_EXPIRATION_DELTA,
        }

    @staticmethod
    def verify_token(token: str) -> Optional[User]:
        """验证JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=['HS256']
            )
            user_id = payload.get('user_id')
            return User.objects.get(id=user_id)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, User.DoesNotExist):
            return None

    @staticmethod
    def login(email: str, password: str) -> dict:
        """登录"""
        user = authenticate(email=email, password=password)
        if not user:
            raise ValueError("Invalid credentials")

        if not user.is_active:
            raise ValueError("User is not active")

        # 更新最后登录
        user.last_login = datetime.now()
        user.save(update_fields=['last_login'])

        return AuthService.generate_token(user)


# apps/users/api.py
from ninja import Router
from .schemas import LoginSchema, TokenSchema
from .services import AuthService

router = Router(tags=['Authentication'])


@router.post('/login', response=TokenSchema, auth=None)
def login(request, payload: LoginSchema):
    """用户登录"""
    try:
        tokens = AuthService.login(
            email=payload.email,
            password=payload.password
        )
        return tokens
    except ValueError as e:
        return {'error': str(e)}, 401


@router.post('/refresh', response=TokenSchema, auth=None)
def refresh_token(request, refresh_token: str):
    """刷新token"""
    # 实现refresh逻辑
    pass
```

### 2. 权限系统

```python
# libs/permissions.py
from functools import wraps
from django.core.exceptions import PermissionDenied

def require_permission(permission: str):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                raise PermissionDenied("Authentication required")

            # 检查权限
            if not request.user.has_perm(permission):
                raise PermissionDenied(f"Permission required: {permission}")

            return func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_role(role_code: str):
    """角色检查装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                raise PermissionDenied("Authentication required")

            if request.user.role.code != role_code:
                raise PermissionDenied(f"Role required: {role_code}")

            return func(request, *args, **kwargs)
        return wrapper
    return decorator


# apps/studies/api.py
from libs.permissions import require_permission, require_role

@router.get('/{study_id}')
@require_permission('studies.view_study')
def get_study(request, study_id: int):
    """获取研究详情"""
    pass


@router.delete('/{study_id}')
@require_role('admin')
def delete_study(request, study_id: int):
    """删除研究 (仅管理员)"""
    pass
```

---

## 部署配置

### 1. Docker配置

```dockerfile
# Dockerfile
FROM python:3.11-slim as base

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN pip install uv

WORKDIR /app

# 复制依赖文件
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

# 复制应用代码
COPY . .

# 收集静态文件
RUN python manage.py collectstatic --noinput

# 运行服务
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.9'

services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: medical_ai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  web:
    build: .
    command: gunicorn config.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
      - static_volume:/app/static
      - media_volume:/app/media
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    environment:
      - DJANGO_SETTINGS_MODULE=config.settings.production
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/medical_ai
      - REDIS_URL=redis://redis:6379/0

  celery_worker:
    build: .
    command: celery -A config worker -l info
    volumes:
      - .:/app
    depends_on:
      - db
      - redis

  celery_beat:
    build: .
    command: celery -A config beat -l info
    volumes:
      - .:/app
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  static_volume:
  media_volume:
```

---

## 迁移路径

### 从FastAPI到Django的迁移策略

#### Phase 1: 并行运行 (1个月)
1. Django项目搭建完成
2. 核心模型迁移
3. 数据迁移脚本
4. 两套系统并行运行

#### Phase 2: 功能迁移 (2个月)
1. API逐个迁移到Django Ninja
2. Celery任务迁移
3. 前端调整API端点
4. 测试验证

#### Phase 3: 切换 (2周)
1. 流量切换到Django
2. 监控稳定性
3. FastAPI系统下线

---

**文档版本**: v1.0
**最后更新**: 2025-10-15