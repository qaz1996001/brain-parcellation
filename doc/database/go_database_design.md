# Go 資料庫設計文件

## 📋 文件資訊
- **版本**: 1.0.0
- **建立日期**: 2025年9月24日
- **資料庫**: PostgreSQL + Redis
- **ORM**: GORM v2
- **遷移策略**: 從 Python SQLAlchemy 到 Go GORM

## 🎯 資料庫設計原則

### Go 特定原則
1. **結構體驅動**：使用 Go 結構體定義資料模型
2. **編譯時安全**：利用 Go 的類型系統避免運行時錯誤
3. **並行友好**：設計支援 Go 的並行處理模式
4. **零值友好**：合理使用 Go 的零值語義
5. **介面導向**：定義清晰的資料存取介面

### 效能原則
```go
// 連接池配置 - 符合效能最佳化規則
type DatabaseConfig struct {
    MaxOpenConns    int           `yaml:"max_open_conns" default:"25"`
    MaxIdleConns    int           `yaml:"max_idle_conns" default:"5"`
    ConnMaxLifetime time.Duration `yaml:"conn_max_lifetime" default:"1h"`
    ConnMaxIdleTime time.Duration `yaml:"conn_max_idle_time" default:"30m"`
}
```

## 🗄️ 核心資料模型

### 1. 使用者模型
```go
// User 使用者模型
type User struct {
    ID        uint      `json:"id" gorm:"primaryKey"`
    Username  string    `json:"username" gorm:"unique;not null;size:50" validate:"required,min=3,max=50"`
    Email     string    `json:"email" gorm:"unique;not null;size:255" validate:"required,email"`
    FullName  string    `json:"full_name" gorm:"not null;size:255" validate:"required,min=2,max=100"`
    Role      UserRole  `json:"role" gorm:"not null;type:varchar(20)"`
    IsActive  bool      `json:"is_active" gorm:"default:true"`
    
    // 密碼相關（不在 JSON 中返回）
    PasswordHash string `json:"-" gorm:"not null;size:255"`
    
    // 時間戳記
    LastLogin *time.Time `json:"last_login,omitempty"`
    CreatedAt time.Time  `json:"created_at"`
    UpdatedAt time.Time  `json:"updated_at"`
    
    // 關聯關係
    Files           []File           `json:"files,omitempty" gorm:"foreignKey:CreatedBy"`
    ProcessingTasks []ProcessingTask `json:"tasks,omitempty" gorm:"foreignKey:CreatedBy"`
    Sessions        []UserSession    `json:"-" gorm:"foreignKey:UserID"`
}

// UserRole 使用者角色枚舉
type UserRole string

const (
    AdminRole      UserRole = "admin"
    DoctorRole     UserRole = "doctor"
    TechnicianRole UserRole = "technician"
    ResearcherRole UserRole = "researcher"
    ViewerRole     UserRole = "viewer"
)

// 實作 Valuer 和 Scanner 介面支援資料庫存儲
func (r UserRole) Value() (driver.Value, error) {
    return string(r), nil
}

func (r *UserRole) Scan(value interface{}) error {
    if value == nil {
        *r = ViewerRole
        return nil
    }
    if str, ok := value.(string); ok {
        *r = UserRole(str)
        return nil
    }
    return fmt.Errorf("cannot scan %T into UserRole", value)
}

// 業務方法
func (u *User) HasPermission(permission Permission) bool {
    rolePermissions := map[UserRole][]Permission{
        AdminRole:      {ReadUsers, WriteUsers, ManageSystem, ReadPatientData, WritePatientData},
        DoctorRole:     {ReadPatientData, WritePatientData, ProcessImages},
        TechnicianRole: {ReadPatientData, ProcessImages},
        ResearcherRole: {ReadPatientData, ViewReports},
        ViewerRole:     {ViewReports},
    }
    
    permissions, exists := rolePermissions[u.Role]
    if !exists {
        return false
    }
    
    for _, p := range permissions {
        if p == permission {
            return true
        }
    }
    return false
}

func (u *User) CanAccessPatientData() bool {
    return u.HasPermission(ReadPatientData)
}
```

### 2. 檔案模型
```go
// File 檔案模型
type File struct {
    ID               uint       `json:"id" gorm:"primaryKey"`
    Filename         string     `json:"filename" gorm:"not null;size:255"`
    OriginalFilename string     `json:"original_filename" gorm:"not null;size:255"`
    FileType         FileType   `json:"file_type" gorm:"not null;type:varchar(20)"`
    FileSize         int64      `json:"file_size" gorm:"not null" validate:"min=1"`
    FilePath         string     `json:"file_path" gorm:"not null;size:500"`
    Checksum         string     `json:"checksum" gorm:"unique;not null;size:64"`
    Status           FileStatus `json:"status" gorm:"default:'uploaded';type:varchar(20)"`
    
    // 醫學影像特定欄位
    PatientID         *string `json:"patient_id,omitempty" gorm:"size:64;index"`
    StudyInstanceUID  *string `json:"study_instance_uid,omitempty" gorm:"size:64;index"`
    SeriesInstanceUID *string `json:"series_instance_uid,omitempty" gorm:"size:64;index"`
    Modality          *string `json:"modality,omitempty" gorm:"size:16;index"`
    SequenceType      *string `json:"sequence_type,omitempty" gorm:"size:50;index"`
    IsReformatted     *bool   `json:"is_reformatted,omitempty"`
    
    // 元資料（JSONB）
    Metadata datatypes.JSON `json:"metadata" gorm:"type:jsonb"`
    
    // 時間戳記
    UploadedAt  *time.Time `json:"uploaded_at,omitempty"`
    ProcessedAt *time.Time `json:"processed_at,omitempty"`
    CreatedAt   time.Time  `json:"created_at"`
    UpdatedAt   time.Time  `json:"updated_at"`
    CreatedBy   uint       `json:"created_by" gorm:"not null;index"`
    
    // 關聯關係
    Creator User               `json:"creator,omitempty" gorm:"foreignKey:CreatedBy"`
    Tags    []FileTag          `json:"tags,omitempty" gorm:"foreignKey:FileID"`
    Tasks   []ProcessingTask   `json:"tasks,omitempty" gorm:"many2many:task_files;"`
    AccessLogs []FileAccessLog `json:"-" gorm:"foreignKey:FileID"`
}

// FileType 檔案類型枚舉
type FileType string

const (
    DicomFileType  FileType = "dicom"
    NiftiFileType  FileType = "nifti"
    ReportFileType FileType = "report"
    ResultFileType FileType = "result"
)

// FileStatus 檔案狀態枚舉
type FileStatus string

const (
    UploadingStatus  FileStatus = "uploading"
    UploadedStatus   FileStatus = "uploaded"
    ProcessingStatus FileStatus = "processing"
    ProcessedStatus  FileStatus = "processed"
    ArchivedStatus   FileStatus = "archived"
    ErrorStatus      FileStatus = "error"
)

// 業務方法
func (f *File) IsMedicalImage() bool {
    return f.FileType == DicomFileType || f.FileType == NiftiFileType
}

func (f *File) IsProcessed() bool {
    return f.Status == ProcessedStatus
}

func (f *File) GetStorageTier() StorageTier {
    // 根據檔案年齡和存取頻率決定儲存層級
    age := time.Since(f.CreatedAt)
    
    if age < 7*24*time.Hour {
        return HotTier
    } else if age < 30*24*time.Hour {
        return WarmTier
    } else if age < 365*24*time.Hour {
        return ColdTier
    }
    
    return ArchiveTier
}
```

### 3. 處理任務模型
```go
// ProcessingTask 處理任務模型
type ProcessingTask struct {
    ID          uint            `json:"id" gorm:"primaryKey"`
    TaskType    ProcessingType  `json:"task_type" gorm:"not null;type:varchar(50)"`
    Status      TaskStatus      `json:"status" gorm:"default:'pending';type:varchar(20)"`
    Priority    int             `json:"priority" gorm:"default:5" validate:"min=1,max=10"`
    
    // 輸入輸出
    InputFiles  datatypes.JSON `json:"input_files" gorm:"type:jsonb"`
    OutputFiles datatypes.JSON `json:"output_files" gorm:"type:jsonb"`
    Parameters  datatypes.JSON `json:"parameters" gorm:"type:jsonb"`
    ResultData  datatypes.JSON `json:"result_data" gorm:"type:jsonb"`
    
    // 執行資訊
    Progress     float32 `json:"progress" gorm:"default:0" validate:"min=0,max=1"`
    ErrorMessage *string `json:"error_message,omitempty" gorm:"type:text"`
    
    // 資源使用
    CPUUsage    *float32 `json:"cpu_usage,omitempty" validate:"omitempty,min=0,max=100"`
    MemoryUsage *int64   `json:"memory_usage,omitempty" validate:"omitempty,min=0"`
    GPUUsage    *float32 `json:"gpu_usage,omitempty" validate:"omitempty,min=0,max=100"`
    
    // 時間戳記
    CreatedAt   time.Time  `json:"created_at"`
    StartedAt   *time.Time `json:"started_at,omitempty"`
    CompletedAt *time.Time `json:"completed_at,omitempty"`
    CreatedBy   uint       `json:"created_by" gorm:"not null;index"`
    
    // 關聯關係
    Creator      User                `json:"creator,omitempty" gorm:"foreignKey:CreatedBy"`
    Files        []File              `json:"files,omitempty" gorm:"many2many:task_files;"`
    Dependencies []TaskDependency    `json:"dependencies,omitempty" gorm:"foreignKey:TaskID"`
    ExecutionLogs []TaskExecutionLog `json:"logs,omitempty" gorm:"foreignKey:TaskID"`
}

// ProcessingType 處理類型枚舉
type ProcessingType string

const (
    DicomToNiftiTask     ProcessingType = "dicom_to_nifti"
    BrainSegmentationTask ProcessingType = "brain_segmentation"
    WMHDetectionTask     ProcessingType = "wmh_detection"
    CMBDetectionTask     ProcessingType = "cmb_detection"
    AneurysmDetectionTask ProcessingType = "aneurysm_detection"
)

// TaskStatus 任務狀態枚舉
type TaskStatus string

const (
    PendingStatus   TaskStatus = "pending"
    RunningStatus   TaskStatus = "running"
    CompletedStatus TaskStatus = "completed"
    FailedStatus    TaskStatus = "failed"
    CancelledStatus TaskStatus = "cancelled"
)

// 業務方法
func (t *ProcessingTask) IsFinished() bool {
    return t.Status == CompletedStatus || t.Status == FailedStatus || t.Status == CancelledStatus
}

func (t *ProcessingTask) Duration() *time.Duration {
    if t.StartedAt != nil && t.CompletedAt != nil {
        duration := t.CompletedAt.Sub(*t.StartedAt)
        return &duration
    }
    return nil
}

func (t *ProcessingTask) CanRetry() bool {
    return t.Status == FailedStatus && t.ErrorMessage != nil
}
```

## 🔧 Repository 模式實作

### 泛型 Repository
```go
// Repository 泛型資料存取介面
type Repository[T any] interface {
    Create(ctx context.Context, entity *T) error
    GetByID(ctx context.Context, id uint) (*T, error)
    Update(ctx context.Context, entity *T) error
    Delete(ctx context.Context, id uint) error
    List(ctx context.Context, params ListParams) ([]T, int64, error)
}

// BaseRepository 基礎 Repository 實作
type BaseRepository[T any] struct {
    db *gorm.DB
}

func NewBaseRepository[T any](db *gorm.DB) *BaseRepository[T] {
    return &BaseRepository[T]{db: db}
}

func (r *BaseRepository[T]) Create(ctx context.Context, entity *T) error {
    return r.db.WithContext(ctx).Create(entity).Error
}

func (r *BaseRepository[T]) GetByID(ctx context.Context, id uint) (*T, error) {
    var entity T
    err := r.db.WithContext(ctx).First(&entity, id).Error
    if err != nil {
        if errors.Is(err, gorm.ErrRecordNotFound) {
            return nil, ErrNotFound
        }
        return nil, err
    }
    return &entity, nil
}

func (r *BaseRepository[T]) Update(ctx context.Context, entity *T) error {
    return r.db.WithContext(ctx).Save(entity).Error
}

func (r *BaseRepository[T]) Delete(ctx context.Context, id uint) error {
    var entity T
    return r.db.WithContext(ctx).Delete(&entity, id).Error
}

func (r *BaseRepository[T]) List(ctx context.Context, params ListParams) ([]T, int64, error) {
    var entities []T
    var total int64
    
    query := r.db.WithContext(ctx).Model(new(T))
    
    // 應用過濾條件
    for key, value := range params.Filters {
        query = query.Where(fmt.Sprintf("%s = ?", key), value)
    }
    
    // 計算總數
    if err := query.Count(&total).Error; err != nil {
        return nil, 0, err
    }
    
    // 應用排序和分頁
    if params.OrderBy != "" {
        query = query.Order(params.OrderBy)
    }
    
    err := query.Offset(params.Offset).Limit(params.Limit).Find(&entities).Error
    return entities, total, err
}

// ListParams 查詢參數
type ListParams struct {
    Offset  int
    Limit   int
    OrderBy string
    Filters map[string]interface{}
}
```

### 專用 Repository
```go
// UserRepository 使用者資料存取
type UserRepository struct {
    *BaseRepository[User]
    redis *redis.Client
}

func NewUserRepository(db *gorm.DB, redis *redis.Client) *UserRepository {
    return &UserRepository{
        BaseRepository: NewBaseRepository[User](db),
        redis:         redis,
    }
}

// 特定查詢方法
func (r *UserRepository) GetByUsername(ctx context.Context, username string) (*User, error) {
    var user User
    err := r.db.WithContext(ctx).Where("username = ?", username).First(&user).Error
    if err != nil {
        if errors.Is(err, gorm.ErrRecordNotFound) {
            return nil, ErrUserNotFound
        }
        return nil, err
    }
    return &user, nil
}

func (r *UserRepository) GetByEmail(ctx context.Context, email string) (*User, error) {
    var user User
    err := r.db.WithContext(ctx).Where("email = ?", email).First(&user).Error
    if err != nil {
        if errors.Is(err, gorm.ErrRecordNotFound) {
            return nil, ErrUserNotFound
        }
        return nil, err
    }
    return &user, nil
}

// 帶快取的查詢
func (r *UserRepository) GetByIDWithCache(ctx context.Context, id uint) (*User, error) {
    cacheKey := fmt.Sprintf("user:%d", id)
    
    // 檢查 Redis 快取
    cached := r.redis.Get(ctx, cacheKey)
    if cached.Err() == nil {
        var user User
        if err := json.Unmarshal([]byte(cached.Val()), &user); err == nil {
            return &user, nil
        }
    }
    
    // 查詢資料庫
    user, err := r.GetByID(ctx, id)
    if err != nil {
        return nil, err
    }
    
    // 快取結果
    if userJSON, err := json.Marshal(user); err == nil {
        r.redis.Set(ctx, cacheKey, userJSON, 30*time.Minute)
    }
    
    return user, nil
}

// 批次操作
func (r *UserRepository) BulkCreate(ctx context.Context, users []User) error {
    return r.db.WithContext(ctx).CreateInBatches(users, 100).Error
}
```

### 檔案 Repository
```go
// FileRepository 檔案資料存取
type FileRepository struct {
    *BaseRepository[File]
    redis *redis.Client
}

func NewFileRepository(db *gorm.DB, redis *redis.Client) *FileRepository {
    return &FileRepository{
        BaseRepository: NewBaseRepository[File](db),
        redis:         redis,
    }
}

// 醫學影像特定查詢
func (r *FileRepository) GetByPatientID(ctx context.Context, patientID string) ([]File, error) {
    var files []File
    err := r.db.WithContext(ctx).
        Where("patient_id = ?", patientID).
        Order("created_at DESC").
        Find(&files).Error
    return files, err
}

func (r *FileRepository) GetByStudyUID(ctx context.Context, studyUID string) ([]File, error) {
    var files []File
    err := r.db.WithContext(ctx).
        Where("study_instance_uid = ?", studyUID).
        Order("series_instance_uid, created_at").
        Find(&files).Error
    return files, err
}

func (r *FileRepository) GetReformattedFiles(ctx context.Context, limit int) ([]File, error) {
    var files []File
    err := r.db.WithContext(ctx).
        Where("is_reformatted = ?", true).
        Limit(limit).
        Find(&files).Error
    return files, err
}

// 檔案統計
func (r *FileRepository) GetFileStats(ctx context.Context) (*FileStats, error) {
    var stats FileStats
    
    // 使用原生 SQL 進行統計查詢
    err := r.db.WithContext(ctx).Raw(`
        SELECT 
            COUNT(*) as total_files,
            COUNT(*) FILTER (WHERE file_type = 'dicom') as dicom_count,
            COUNT(*) FILTER (WHERE file_type = 'nifti') as nifti_count,
            COUNT(*) FILTER (WHERE status = 'processed') as processed_count,
            SUM(file_size) as total_size,
            AVG(file_size) as average_size
        FROM files
        WHERE created_at > NOW() - INTERVAL '30 days'
    `).Scan(&stats).Error
    
    return &stats, err
}

type FileStats struct {
    TotalFiles     int64   `json:"total_files"`
    DicomCount     int64   `json:"dicom_count"`
    NiftiCount     int64   `json:"nifti_count"`
    ProcessedCount int64   `json:"processed_count"`
    TotalSize      int64   `json:"total_size"`
    AverageSize    float64 `json:"average_size"`
}
```

## 🔄 資料庫遷移

### 遷移管理
```go
// Migration 遷移結構
type Migration struct {
    ID        uint      `gorm:"primaryKey"`
    Version   string    `gorm:"unique;not null"`
    Name      string    `gorm:"not null"`
    AppliedAt time.Time `gorm:"autoCreateTime"`
}

// MigrationManager 遷移管理器
type MigrationManager struct {
    db *gorm.DB
}

func NewMigrationManager(db *gorm.DB) *MigrationManager {
    return &MigrationManager{db: db}
}

func (m *MigrationManager) RunMigrations() error {
    // 確保 migrations 表存在
    if err := m.db.AutoMigrate(&Migration{}); err != nil {
        return fmt.Errorf("failed to create migrations table: %w", err)
    }
    
    // 執行所有遷移
    migrations := []MigrationStep{
        {"001", "create_users_table", m.createUsersTable},
        {"002", "create_files_table", m.createFilesTable},
        {"003", "create_tasks_table", m.createTasksTable},
        {"004", "add_medical_fields", m.addMedicalFields},
        {"005", "create_indexes", m.createIndexes},
    }
    
    for _, migration := range migrations {
        if err := m.runMigration(migration); err != nil {
            return fmt.Errorf("migration %s failed: %w", migration.Version, err)
        }
    }
    
    return nil
}

func (m *MigrationManager) runMigration(migration MigrationStep) error {
    // 檢查是否已經執行
    var existing Migration
    err := m.db.Where("version = ?", migration.Version).First(&existing).Error
    if err == nil {
        // 已執行，跳過
        return nil
    }
    if !errors.Is(err, gorm.ErrRecordNotFound) {
        return err
    }
    
    // 執行遷移
    if err := migration.Function(); err != nil {
        return err
    }
    
    // 記錄遷移
    migrationRecord := Migration{
        Version: migration.Version,
        Name:    migration.Name,
    }
    return m.db.Create(&migrationRecord).Error
}

type MigrationStep struct {
    Version  string
    Name     string
    Function func() error
}

// 具體遷移函數
func (m *MigrationManager) createUsersTable() error {
    return m.db.AutoMigrate(&User{})
}

func (m *MigrationManager) createFilesTable() error {
    return m.db.AutoMigrate(&File{})
}

func (m *MigrationManager) createTasksTable() error {
    return m.db.AutoMigrate(&ProcessingTask{})
}

func (m *MigrationManager) createIndexes() error {
    // 建立複合索引
    return m.db.Exec(`
        CREATE INDEX IF NOT EXISTS idx_files_patient_study 
        ON files(patient_id, study_instance_uid);
        
        CREATE INDEX IF NOT EXISTS idx_tasks_status_created 
        ON processing_tasks(status, created_at);
        
        CREATE INDEX IF NOT EXISTS idx_files_type_status 
        ON files(file_type, status);
    `).Error
}
```

## 🚀 事務管理

### 事務管理器
```go
// TransactionManager 事務管理器
type TransactionManager struct {
    db *gorm.DB
}

func NewTransactionManager(db *gorm.DB) *TransactionManager {
    return &TransactionManager{db: db}
}

// WithTransaction 事務包裝器
func (tm *TransactionManager) WithTransaction(ctx context.Context, fn func(*gorm.DB) error) error {
    return tm.db.WithContext(ctx).Transaction(fn)
}

// 使用範例：複雜業務邏輯
func (s *FileService) ProcessFileWithMetadata(ctx context.Context, fileID uint, metadata map[string]interface{}) error {
    return s.txManager.WithTransaction(ctx, func(tx *gorm.DB) error {
        // 1. 更新檔案狀態
        if err := tx.Model(&File{}).Where("id = ?", fileID).Update("status", ProcessingStatus).Error; err != nil {
            return err
        }
        
        // 2. 建立處理任務
        task := ProcessingTask{
            TaskType:   DicomToNiftiTask,
            InputFiles: datatypes.JSON(fmt.Sprintf(`[%d]`, fileID)),
            Parameters: datatypes.JSON(metadata),
            CreatedBy:  s.getCurrentUserID(ctx),
        }
        if err := tx.Create(&task).Error; err != nil {
            return err
        }
        
        // 3. 記錄操作日誌
        log := FileAccessLog{
            FileID:     fileID,
            UserID:     s.getCurrentUserID(ctx),
            Action:     "process_started",
            IPAddress:  s.getClientIP(ctx),
            AccessedAt: time.Now(),
        }
        if err := tx.Create(&log).Error; err != nil {
            return err
        }
        
        return nil
    })
}
```

## 📊 查詢最佳化

### 預載入策略
```go
// 預載入關聯資料避免 N+1 問題
func (r *UserRepository) GetUsersWithFiles(ctx context.Context, limit int) ([]User, error) {
    var users []User
    err := r.db.WithContext(ctx).
        Preload("Files", "status = ?", ProcessedStatus).  // 只載入已處理的檔案
        Preload("Files.Tags").                            // 預載入檔案標籤
        Limit(limit).
        Find(&users).Error
    return users, err
}

func (r *FileRepository) GetFileWithDetails(ctx context.Context, id uint) (*File, error) {
    var file File
    err := r.db.WithContext(ctx).
        Preload("Creator").
        Preload("Tags").
        Preload("Tasks", "status IN ?", []TaskStatus{RunningStatus, CompletedStatus}).
        First(&file, id).Error
    if err != nil {
        if errors.Is(err, gorm.ErrRecordNotFound) {
            return nil, ErrFileNotFound
        }
        return nil, err
    }
    return &file, nil
}
```

### 複雜查詢
```go
// 複雜統計查詢
func (r *FileRepository) GetProcessingStatistics(ctx context.Context, days int) (*ProcessingStats, error) {
    var stats ProcessingStats
    
    query := `
        WITH daily_stats AS (
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as files_processed,
                COUNT(*) FILTER (WHERE file_type = 'dicom') as dicom_processed,
                COUNT(*) FILTER (WHERE status = 'processed') as successful_processing,
                AVG(EXTRACT(EPOCH FROM (processed_at - created_at))) as avg_processing_time
            FROM files 
            WHERE created_at > NOW() - INTERVAL '%d days'
            GROUP BY DATE(created_at)
        )
        SELECT 
            COUNT(*) as total_days,
            SUM(files_processed) as total_files,
            SUM(dicom_processed) as total_dicom,
            SUM(successful_processing) as total_successful,
            AVG(avg_processing_time) as overall_avg_time
        FROM daily_stats
    `
    
    err := r.db.WithContext(ctx).Raw(fmt.Sprintf(query, days)).Scan(&stats).Error
    return &stats, err
}

type ProcessingStats struct {
    TotalDays       int     `json:"total_days"`
    TotalFiles      int64   `json:"total_files"`
    TotalDicom      int64   `json:"total_dicom"`
    TotalSuccessful int64   `json:"total_successful"`
    OverallAvgTime  float64 `json:"overall_avg_time"`
}
```

## 🔒 資料安全

### 敏感資料加密
```go
// EncryptedField 加密欄位類型
type EncryptedField string

func (e EncryptedField) Value() (driver.Value, error) {
    if e == "" {
        return nil, nil
    }
    
    encrypted, err := encrypt(string(e))
    if err != nil {
        return nil, err
    }
    return encrypted, nil
}

func (e *EncryptedField) Scan(value interface{}) error {
    if value == nil {
        *e = ""
        return nil
    }
    
    switch v := value.(type) {
    case string:
        decrypted, err := decrypt(v)
        if err != nil {
            return err
        }
        *e = EncryptedField(decrypted)
    case []byte:
        decrypted, err := decrypt(string(v))
        if err != nil {
            return err
        }
        *e = EncryptedField(decrypted)
    default:
        return fmt.Errorf("cannot scan %T into EncryptedField", value)
    }
    
    return nil
}

// 敏感資料模型
type SensitivePatientData struct {
    ID              uint           `gorm:"primaryKey"`
    PatientName     EncryptedField `gorm:"type:text"` // 加密存儲
    PatientBirthDate EncryptedField `gorm:"type:text"` // 加密存儲
    MedicalRecord   EncryptedField `gorm:"type:text"` // 加密存儲
    CreatedAt       time.Time
    UpdatedAt       time.Time
}
```

### 稽核日誌
```go
// AuditLog 稽核日誌模型
type AuditLog struct {
    ID        uint      `gorm:"primaryKey"`
    UserID    uint      `gorm:"not null;index"`
    Action    string    `gorm:"not null;size:100"`
    Resource  string    `gorm:"not null;size:100"`
    ResourceID uint     `gorm:"index"`
    IPAddress string    `gorm:"size:45"`
    UserAgent string    `gorm:"type:text"`
    Details   datatypes.JSON `gorm:"type:jsonb"`
    CreatedAt time.Time `gorm:"autoCreateTime"`
    
    // 關聯關係
    User User `gorm:"foreignKey:UserID"`
}

// AuditLogger 稽核日誌記錄器
type AuditLogger struct {
    db *gorm.DB
}

func (al *AuditLogger) LogAction(ctx context.Context, audit AuditLog) error {
    return al.db.WithContext(ctx).Create(&audit).Error
}

// 自動稽核中介軟體
func AuditMiddleware(auditLogger *AuditLogger) gin.HandlerFunc {
    return func(c *gin.Context) {
        // 記錄請求開始
        start := time.Now()
        
        c.Next()
        
        // 只記錄修改操作
        if c.Request.Method != "GET" && c.Request.Method != "HEAD" {
            user := getCurrentUser(c)
            if user != nil {
                audit := AuditLog{
                    UserID:    user.ID,
                    Action:    c.Request.Method,
                    Resource:  c.FullPath(),
                    IPAddress: c.ClientIP(),
                    UserAgent: c.Request.UserAgent(),
                    Details: datatypes.JSON(fmt.Sprintf(`{
                        "status_code": %d,
                        "duration_ms": %.2f,
                        "path": "%s"
                    }`, c.Writer.Status(), float64(time.Since(start).Nanoseconds())/1e6, c.Request.URL.Path)),
                }
                
                // 非同步記錄稽核日誌
                go func() {
                    ctx := context.Background()
                    auditLogger.LogAction(ctx, audit)
                }()
            }
        }
    }
}
```

## 🔧 資料庫初始化

### 應用程式初始化
```go
// DatabaseManager 資料庫管理器
type DatabaseManager struct {
    db              *gorm.DB
    migrationManager *MigrationManager
    config          DatabaseConfig
}

func NewDatabaseManager(config DatabaseConfig) (*DatabaseManager, error) {
    // 建立資料庫連接
    dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable TimeZone=Asia/Taipei",
        config.Host, config.Port, config.Username, config.Password, config.Database)
    
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),
        NamingStrategy: schema.NamingStrategy{
            TablePrefix:   "",
            SingularTable: false,
        },
    })
    if err != nil {
        return nil, fmt.Errorf("failed to connect to database: %w", err)
    }
    
    // 配置連接池
    sqlDB, err := db.DB()
    if err != nil {
        return nil, err
    }
    
    sqlDB.SetMaxOpenConns(config.MaxOpenConns)
    sqlDB.SetMaxIdleConns(config.MaxIdleConns)
    sqlDB.SetConnMaxLifetime(config.ConnMaxLifetime)
    sqlDB.SetConnMaxIdleTime(config.ConnMaxIdleTime)
    
    return &DatabaseManager{
        db:              db,
        migrationManager: NewMigrationManager(db),
        config:          config,
    }, nil
}

func (dm *DatabaseManager) Initialize() error {
    // 執行遷移
    if err := dm.migrationManager.RunMigrations(); err != nil {
        return fmt.Errorf("migration failed: %w", err)
    }
    
    // 建立預設資料
    if err := dm.seedDefaultData(); err != nil {
        return fmt.Errorf("seeding failed: %w", err)
    }
    
    return nil
}

func (dm *DatabaseManager) seedDefaultData() error {
    // 建立預設管理員使用者
    var adminCount int64
    dm.db.Model(&User{}).Where("role = ?", AdminRole).Count(&adminCount)
    
    if adminCount == 0 {
        admin := User{
            Username:     "admin",
            Email:        "admin@system.local",
            FullName:     "系統管理員",
            Role:         AdminRole,
            PasswordHash: hashPassword("admin123"), // 預設密碼，首次登入需修改
            IsActive:     true,
        }
        
        if err := dm.db.Create(&admin).Error; err != nil {
            return fmt.Errorf("failed to create admin user: %w", err)
        }
    }
    
    return nil
}

// 健康檢查
func (dm *DatabaseManager) HealthCheck(ctx context.Context) error {
    sqlDB, err := dm.db.DB()
    if err != nil {
        return err
    }
    
    return sqlDB.PingContext(ctx)
}

// 取得連接池狀態
func (dm *DatabaseManager) GetPoolStats() map[string]interface{} {
    sqlDB, _ := dm.db.DB()
    stats := sqlDB.Stats()
    
    return map[string]interface{}{
        "max_open_connections":     stats.MaxOpenConnections,
        "open_connections":         stats.OpenConnections,
        "in_use":                  stats.InUse,
        "idle":                    stats.Idle,
        "wait_count":              stats.WaitCount,
        "wait_duration":           stats.WaitDuration.String(),
        "max_idle_closed":         stats.MaxIdleClosed,
        "max_idle_time_closed":    stats.MaxIdleTimeClosed,
        "max_lifetime_closed":     stats.MaxLifetimeClosed,
    }
}
```

## 📋 與 Python 版本的對應

### 資料模型對應表
| Python SQLAlchemy | Go GORM | 對應度 | 備註 |
|------------------|---------|--------|------|
| `Column(Integer, primary_key=True)` | `uint \`gorm:"primaryKey"\`` | 100% | 完全對應 |
| `Column(String(50), nullable=False)` | `string \`gorm:"not null;size:50"\`` | 100% | 完全對應 |
| `Column(JSON)` | `datatypes.JSON \`gorm:"type:jsonb"\`` | 100% | 完全對應 |
| `relationship()` | `gorm:"foreignKey:ID"` | 95% | 功能相同，語法不同 |
| `Index()` | `gorm:"index"` | 95% | 功能相同 |

### 查詢語法對應
| Python SQLAlchemy | Go GORM | 範例 |
|------------------|---------|------|
| `session.query(User).filter(User.id == 1)` | `db.Where("id = ?", 1).First(&user)` | 基本查詢 |
| `session.query(User).join(File)` | `db.Joins("Files").Find(&users)` | 關聯查詢 |
| `session.query(User).options(selectinload(User.files))` | `db.Preload("Files").Find(&users)` | 預載入 |

---

**資料庫設計版本**: 2.0  
**遷移完成目標**: 2025年11月  
**效能提升預期**: 資料庫操作效能提升 3-5倍
