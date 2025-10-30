# Go API 規範文件

## 📋 文件資訊
- **版本**: 1.0.0
- **建立日期**: 2025年9月24日
- **框架**: Go Gin + Swaggo
- **API 版本**: v2.0 (Go 重構版本)

## 🎯 API 設計原則

### RESTful 設計標準
```go
// API 路徑設計標準
const (
    // 資源集合
    UsersPath    = "/api/v2/users"
    FilesPath    = "/api/v2/files"
    TasksPath    = "/api/v2/tasks"
    
    // 具體資源
    UserPath     = "/api/v2/users/:id"
    FilePath     = "/api/v2/files/:id"
    TaskPath     = "/api/v2/tasks/:id"
    
    // 子資源
    UserFilesPath = "/api/v2/users/:id/files"
    FileTasksPath = "/api/v2/files/:id/tasks"
)
```

### 統一回應格式
```go
// 標準 API 回應結構
type APIResponse struct {
    Success   bool        `json:"success" example:"true"`
    Message   string      `json:"message,omitempty" example:"操作成功"`
    Data      interface{} `json:"data,omitempty"`
    Errors    []string    `json:"errors,omitempty"`
    Timestamp string      `json:"timestamp" example:"2025-09-24T10:00:00Z"`
}

type PaginatedResponse struct {
    APIResponse
    Pagination PaginationInfo `json:"pagination"`
}

type PaginationInfo struct {
    Page      int  `json:"page" example:"1"`
    Size      int  `json:"size" example:"20"`
    Total     int  `json:"total" example:"100"`
    HasNext   bool `json:"has_next" example:"true"`
    HasPrev   bool `json:"has_prev" example:"false"`
}
```

## 👥 使用者管理 API

### 使用者模型定義
```go
// 使用者相關結構體
type User struct {
    ID        uint      `json:"id" gorm:"primaryKey"`
    Username  string    `json:"username" gorm:"unique;not null" validate:"required,min=3,max=50"`
    Email     string    `json:"email" gorm:"unique;not null" validate:"required,email"`
    FullName  string    `json:"full_name" gorm:"not null" validate:"required,min=2,max=100"`
    Role      UserRole  `json:"role" gorm:"not null"`
    IsActive  bool      `json:"is_active" gorm:"default:true"`
    LastLogin *time.Time `json:"last_login,omitempty"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

type UserRole string

const (
    AdminRole      UserRole = "admin"
    DoctorRole     UserRole = "doctor"
    TechnicianRole UserRole = "technician"
    ResearcherRole UserRole = "researcher"
    ViewerRole     UserRole = "viewer"
)

// 請求/回應模型
type CreateUserRequest struct {
    Username string   `json:"username" validate:"required,min=3,max=50" example:"doctor_chen"`
    Email    string   `json:"email" validate:"required,email" example:"chen@hospital.com"`
    FullName string   `json:"full_name" validate:"required,min=2,max=100" example:"陳醫師"`
    Password string   `json:"password" validate:"required,min=8" example:"SecurePass123"`
    Role     UserRole `json:"role" validate:"required" example:"doctor"`
}

type UpdateUserRequest struct {
    FullName *string   `json:"full_name,omitempty" validate:"omitempty,min=2,max=100"`
    Role     *UserRole `json:"role,omitempty"`
    IsActive *bool     `json:"is_active,omitempty"`
}

type UserResponse struct {
    ID        uint      `json:"id" example:"1"`
    Username  string    `json:"username" example:"doctor_chen"`
    Email     string    `json:"email" example:"chen@hospital.com"`
    FullName  string    `json:"full_name" example:"陳醫師"`
    Role      UserRole  `json:"role" example:"doctor"`
    IsActive  bool      `json:"is_active" example:"true"`
    LastLogin *time.Time `json:"last_login,omitempty"`
    CreatedAt time.Time `json:"created_at"`
}
```

### 使用者 API 端點
```go
// @title Medical Imaging API - User Management
// @version 2.0
// @description Go-based 使用者管理 API

// @Summary 建立新使用者
// @Description 建立新的系統使用者（需要管理員權限）
// @Tags users
// @Accept json
// @Produce json
// @Param user body CreateUserRequest true "使用者資料"
// @Success 201 {object} APIResponse{data=UserResponse}
// @Failure 400 {object} APIResponse
// @Failure 401 {object} APIResponse
// @Failure 403 {object} APIResponse
// @Failure 409 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/users [post]
func (h *UserHandler) CreateUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "請求資料格式錯誤",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 驗證請求資料
    if err := h.validator.Struct(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "資料驗證失敗",
            Errors:  extractValidationErrors(err),
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 檢查權限
    currentUser := getCurrentUser(c)
    if currentUser.Role != AdminRole {
        c.JSON(403, APIResponse{
            Success: false,
            Message: "權限不足",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 建立使用者
    user, err := h.userService.CreateUser(c.Request.Context(), req)
    if err != nil {
        if errors.Is(err, ErrUserExists) {
            c.JSON(409, APIResponse{
                Success: false,
                Message: "使用者已存在",
                Timestamp: time.Now().Format(time.RFC3339),
            })
            return
        }
        
        c.JSON(500, APIResponse{
            Success: false,
            Message: "建立使用者失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(201, APIResponse{
        Success: true,
        Message: "使用者建立成功",
        Data:    convertToUserResponse(user),
        Timestamp: time.Now().Format(time.RFC3339),
    })
}

// @Summary 取得使用者資訊
// @Description 根據 ID 取得使用者詳細資訊
// @Tags users
// @Accept json
// @Produce json
// @Param id path int true "使用者 ID"
// @Success 200 {object} APIResponse{data=UserResponse}
// @Failure 404 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/users/{id} [get]
func (h *UserHandler) GetUser(c *gin.Context) {
    var req GetUserRequest
    if err := c.ShouldBindUri(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "無效的使用者 ID",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    user, err := h.userService.GetUser(c.Request.Context(), req.ID)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            c.JSON(404, APIResponse{
                Success: false,
                Message: "使用者不存在",
                Timestamp: time.Now().Format(time.RFC3339),
            })
            return
        }
        
        c.JSON(500, APIResponse{
            Success: false,
            Message: "取得使用者失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(200, APIResponse{
        Success: true,
        Data:    convertToUserResponse(user),
        Timestamp: time.Now().Format(time.RFC3339),
    })
}
```

## 📁 檔案管理 API

### 檔案模型定義
```go
type File struct {
    ID               uint      `json:"id" gorm:"primaryKey"`
    Filename         string    `json:"filename" gorm:"not null"`
    OriginalFilename string    `json:"original_filename" gorm:"not null"`
    FileType         FileType  `json:"file_type" gorm:"not null"`
    FileSize         int64     `json:"file_size" gorm:"not null"`
    FilePath         string    `json:"file_path" gorm:"not null"`
    Checksum         string    `json:"checksum" gorm:"unique;not null"`
    Status           FileStatus `json:"status" gorm:"default:'uploaded'"`
    
    // 醫學影像特定欄位
    PatientID         *string `json:"patient_id,omitempty"`
    StudyInstanceUID  *string `json:"study_instance_uid,omitempty"`
    SeriesInstanceUID *string `json:"series_instance_uid,omitempty"`
    Modality          *string `json:"modality,omitempty"`
    SequenceType      *string `json:"sequence_type,omitempty"`
    IsReformatted     *bool   `json:"is_reformatted,omitempty"`
    
    // 元資料
    Metadata  datatypes.JSON `json:"metadata" gorm:"type:jsonb"`
    
    // 時間戳記
    UploadedAt  *time.Time `json:"uploaded_at,omitempty"`
    ProcessedAt *time.Time `json:"processed_at,omitempty"`
    CreatedAt   time.Time  `json:"created_at"`
    UpdatedAt   time.Time  `json:"updated_at"`
    CreatedBy   uint       `json:"created_by"`
}

type FileType string

const (
    DicomFileType  FileType = "dicom"
    NiftiFileType  FileType = "nifti"
    ReportFileType FileType = "report"
    ResultFileType FileType = "result"
)

type FileStatus string

const (
    UploadingStatus  FileStatus = "uploading"
    UploadedStatus   FileStatus = "uploaded"
    ProcessingStatus FileStatus = "processing"
    ProcessedStatus  FileStatus = "processed"
    ArchivedStatus   FileStatus = "archived"
    ErrorStatus      FileStatus = "error"
)
```

### 檔案 API 端點
```go
// @Summary 上傳檔案
// @Description 上傳醫學影像檔案（支援 DICOM、NIfTI 格式）
// @Tags files
// @Accept multipart/form-data
// @Produce json
// @Param file formData file true "檔案"
// @Param description formData string false "檔案描述"
// @Success 201 {object} APIResponse{data=FileResponse}
// @Failure 400 {object} APIResponse
// @Failure 413 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/files [post]
func (h *FileHandler) UploadFile(c *gin.Context) {
    // 檔案大小限制
    c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 500<<20) // 500MB
    
    file, header, err := c.Request.FormFile("file")
    if err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "檔案上傳失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    defer file.Close()
    
    // 檔案類型驗證
    fileType, err := h.detectFileType(header.Filename, file)
    if err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "不支援的檔案格式",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    currentUser := getCurrentUser(c)
    
    // 處理檔案上傳
    fileInfo, err := h.fileService.ProcessUpload(c.Request.Context(), ProcessUploadRequest{
        File:        file,
        Header:      header,
        FileType:    fileType,
        Description: c.PostForm("description"),
        UploadedBy:  currentUser.ID,
    })
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "檔案處理失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(201, APIResponse{
        Success: true,
        Message: "檔案上傳成功",
        Data:    convertToFileResponse(fileInfo),
        Timestamp: time.Now().Format(time.RFC3339),
    })
}

// @Summary 取得檔案列表
// @Description 取得使用者的檔案列表，支援分頁和過濾
// @Tags files
// @Accept json
// @Produce json
// @Param page query int false "頁數" default(1)
// @Param size query int false "每頁大小" default(20)
// @Param file_type query string false "檔案類型過濾" Enums(dicom,nifti,report,result)
// @Param status query string false "狀態過濾" Enums(uploaded,processing,processed,archived)
// @Success 200 {object} PaginatedResponse{data=[]FileResponse}
// @Failure 400 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/files [get]
func (h *FileHandler) ListFiles(c *gin.Context) {
    var req ListFilesRequest
    if err := c.ShouldBindQuery(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "查詢參數錯誤",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 設定預設值
    if req.Page <= 0 {
        req.Page = 1
    }
    if req.Size <= 0 || req.Size > 100 {
        req.Size = 20
    }
    
    currentUser := getCurrentUser(c)
    
    files, total, err := h.fileService.ListFiles(c.Request.Context(), ListFilesParams{
        UserID:   currentUser.ID,
        Page:     req.Page,
        Size:     req.Size,
        FileType: req.FileType,
        Status:   req.Status,
    })
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "取得檔案列表失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    fileResponses := make([]FileResponse, len(files))
    for i, file := range files {
        fileResponses[i] = convertToFileResponse(&file)
    }
    
    c.JSON(200, PaginatedResponse{
        APIResponse: APIResponse{
            Success:   true,
            Message:   "取得檔案列表成功",
            Data:      fileResponses,
            Timestamp: time.Now().Format(time.RFC3339),
        },
        Pagination: PaginationInfo{
            Page:    req.Page,
            Size:    req.Size,
            Total:   total,
            HasNext: req.Page*req.Size < total,
            HasPrev: req.Page > 1,
        },
    })
}
```

## 🔄 任務處理 API

### 任務模型定義
```go
type ProcessingTask struct {
    ID          uint            `json:"id" gorm:"primaryKey"`
    TaskType    ProcessingType  `json:"task_type" gorm:"not null"`
    Status      TaskStatus      `json:"status" gorm:"default:'pending'"`
    InputFiles  datatypes.JSON  `json:"input_files" gorm:"type:jsonb"`
    Parameters  datatypes.JSON  `json:"parameters" gorm:"type:jsonb"`
    OutputFiles datatypes.JSON  `json:"output_files" gorm:"type:jsonb"`
    ResultData  datatypes.JSON  `json:"result_data" gorm:"type:jsonb"`
    Progress    float32         `json:"progress" gorm:"default:0"`
    ErrorMessage *string        `json:"error_message,omitempty"`
    CreatedAt   time.Time       `json:"created_at"`
    StartedAt   *time.Time      `json:"started_at,omitempty"`
    CompletedAt *time.Time      `json:"completed_at,omitempty"`
    CreatedBy   uint            `json:"created_by"`
}

type ProcessingType string

const (
    DicomToNiftiTask     ProcessingType = "dicom_to_nifti"
    BrainSegmentationTask ProcessingType = "brain_segmentation"
    WMHDetectionTask     ProcessingType = "wmh_detection"
    CMBDetectionTask     ProcessingType = "cmb_detection"
    AneurysmDetectionTask ProcessingType = "aneurysm_detection"
)

type TaskStatus string

const (
    PendingStatus   TaskStatus = "pending"
    RunningStatus   TaskStatus = "running"
    CompletedStatus TaskStatus = "completed"
    FailedStatus    TaskStatus = "failed"
    CancelledStatus TaskStatus = "cancelled"
)
```

### 任務 API 端點
```go
// @Summary 建立處理任務
// @Description 建立新的影像處理任務
// @Tags tasks
// @Accept json
// @Produce json
// @Param task body CreateTaskRequest true "任務資料"
// @Success 201 {object} APIResponse{data=TaskResponse}
// @Failure 400 {object} APIResponse
// @Failure 404 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/tasks [post]
func (h *TaskHandler) CreateTask(c *gin.Context) {
    var req CreateTaskRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "請求資料格式錯誤",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 驗證檔案存在
    for _, fileID := range req.InputFileIDs {
        exists, err := h.fileService.FileExists(c.Request.Context(), fileID)
        if err != nil || !exists {
            c.JSON(404, APIResponse{
                Success: false,
                Message: fmt.Sprintf("檔案 %d 不存在", fileID),
                Timestamp: time.Now().Format(time.RFC3339),
            })
            return
        }
    }
    
    currentUser := getCurrentUser(c)
    
    // 建立任務
    task, err := h.taskService.CreateTask(c.Request.Context(), CreateTaskParams{
        TaskType:     req.TaskType,
        InputFileIDs: req.InputFileIDs,
        Parameters:   req.Parameters,
        CreatedBy:    currentUser.ID,
    })
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "建立任務失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(201, APIResponse{
        Success: true,
        Message: "任務建立成功",
        Data:    convertToTaskResponse(task),
        Timestamp: time.Now().Format(time.RFC3339),
    })
}

// @Summary 取得任務狀態
// @Description 取得處理任務的執行狀態和進度
// @Tags tasks
// @Accept json
// @Produce json
// @Param id path int true "任務 ID"
// @Success 200 {object} APIResponse{data=TaskStatusResponse}
// @Failure 404 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/tasks/{id}/status [get]
func (h *TaskHandler) GetTaskStatus(c *gin.Context) {
    var req GetTaskRequest
    if err := c.ShouldBindUri(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "無效的任務 ID",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    task, err := h.taskService.GetTask(c.Request.Context(), req.ID)
    if err != nil {
        if errors.Is(err, ErrTaskNotFound) {
            c.JSON(404, APIResponse{
                Success: false,
                Message: "任務不存在",
                Timestamp: time.Now().Format(time.RFC3339),
            })
            return
        }
        
        c.JSON(500, APIResponse{
            Success: false,
            Message: "取得任務狀態失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(200, APIResponse{
        Success: true,
        Data:    convertToTaskStatusResponse(task),
        Timestamp: time.Now().Format(time.RFC3339),
    })
}
```

## 🤖 AI 服務整合 API

### AI 處理請求
```go
// @Summary 執行腦部分割
// @Description 使用 SynthSeg 模型執行腦部分割
// @Tags ai
// @Accept json
// @Produce json
// @Param request body SynthSegRequest true "分割請求"
// @Success 200 {object} APIResponse{data=AIProcessingResponse}
// @Failure 400 {object} APIResponse
// @Failure 500 {object} APIResponse
// @Security BearerAuth
// @Router /api/v2/ai/synthseg [post]
func (h *AIHandler) ProcessSynthSeg(c *gin.Context) {
    var req SynthSegRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "請求資料格式錯誤",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 驗證檔案存在
    fileExists, err := h.fileService.FileExists(c.Request.Context(), req.FileID)
    if err != nil || !fileExists {
        c.JSON(404, APIResponse{
            Success: false,
            Message: "檔案不存在",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 取得檔案路徑
    filePath, err := h.fileService.GetFilePath(c.Request.Context(), req.FileID)
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "無法取得檔案路徑",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 調用 Python AI 服務
    result, err := h.aiClient.ProcessSynthSeg(c.Request.Context(), AIProcessingRequest{
        ImagePath: filePath,
        Options:   req.Options,
    })
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "AI 處理失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(200, APIResponse{
        Success: true,
        Message: "腦部分割完成",
        Data:    convertToAIResponse(result),
        Timestamp: time.Now().Format(time.RFC3339),
    })
}
```

## 🔒 認證和授權 API

### JWT 認證
```go
// @Summary 使用者登入
// @Description 使用者登入並取得 JWT 令牌
// @Tags auth
// @Accept json
// @Produce json
// @Param credentials body LoginRequest true "登入憑證"
// @Success 200 {object} APIResponse{data=LoginResponse}
// @Failure 401 {object} APIResponse
// @Router /api/v2/auth/login [post]
func (h *AuthHandler) Login(c *gin.Context) {
    var req LoginRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, APIResponse{
            Success: false,
            Message: "請求資料格式錯誤",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 驗證使用者憑證
    user, err := h.authService.ValidateCredentials(c.Request.Context(), req.Username, req.Password)
    if err != nil {
        c.JSON(401, APIResponse{
            Success: false,
            Message: "使用者名稱或密碼錯誤",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 生成 JWT 令牌
    token, err := h.authService.GenerateToken(user)
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "令牌生成失敗",
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    // 更新最後登入時間
    h.userService.UpdateLastLogin(c.Request.Context(), user.ID)
    
    c.JSON(200, APIResponse{
        Success: true,
        Message: "登入成功",
        Data: LoginResponse{
            Token:     token,
            ExpiresIn: 3600, // 1小時
            User:      convertToUserResponse(user),
        },
        Timestamp: time.Now().Format(time.RFC3339),
    })
}
```

## 📊 監控和健康檢查 API

### 系統監控端點
```go
// @Summary 系統健康檢查
// @Description 檢查系統各組件的健康狀態
// @Tags monitoring
// @Accept json
// @Produce json
// @Success 200 {object} HealthCheckResponse
// @Failure 503 {object} HealthCheckResponse
// @Router /api/v2/health [get]
func (h *MonitoringHandler) HealthCheck(c *gin.Context) {
    health := &HealthCheckResponse{
        Status:    "healthy",
        Timestamp: time.Now().Format(time.RFC3339),
        Services:  make(map[string]ServiceHealth),
    }
    
    // 檢查資料庫
    dbHealth := h.checkDatabase(c.Request.Context())
    health.Services["database"] = dbHealth
    
    // 檢查 Redis
    redisHealth := h.checkRedis(c.Request.Context())
    health.Services["redis"] = redisHealth
    
    // 檢查 AI 服務
    aiHealth := h.checkAIService(c.Request.Context())
    health.Services["ai_service"] = aiHealth
    
    // 檢查檔案儲存
    storageHealth := h.checkStorage()
    health.Services["storage"] = storageHealth
    
    // 判斷整體狀態
    allHealthy := true
    for _, service := range health.Services {
        if service.Status != "healthy" {
            allHealthy = false
            break
        }
    }
    
    if !allHealthy {
        health.Status = "unhealthy"
        c.JSON(503, health)
        return
    }
    
    c.JSON(200, health)
}

// @Summary 取得系統指標
// @Description 取得系統效能指標和統計資料
// @Tags monitoring
// @Accept json
// @Produce json
// @Success 200 {object} APIResponse{data=SystemMetrics}
// @Security BearerAuth
// @Router /api/v2/metrics [get]
func (h *MonitoringHandler) GetMetrics(c *gin.Context) {
    metrics, err := h.metricsService.GetSystemMetrics(c.Request.Context())
    if err != nil {
        c.JSON(500, APIResponse{
            Success: false,
            Message: "取得系統指標失敗",
            Errors:  []string{err.Error()},
            Timestamp: time.Now().Format(time.RFC3339),
        })
        return
    }
    
    c.JSON(200, APIResponse{
        Success: true,
        Data:    metrics,
        Timestamp: time.Now().Format(time.RFC3339),
    })
}
```

## 🔧 中介軟體配置

### 標準中介軟體堆疊
```go
func SetupMiddleware(r *gin.Engine) {
    // 1. 恢復中介軟體 - 防止 panic
    r.Use(gin.Recovery())
    
    // 2. CORS 中介軟體
    r.Use(CORSMiddleware())
    
    // 3. 日誌中介軟體
    r.Use(LoggingMiddleware())
    
    // 4. 認證中介軟體
    r.Use(AuthMiddleware())
    
    // 5. 限流中介軟體
    r.Use(RateLimitMiddleware())
    
    // 6. 監控中介軟體
    r.Use(PrometheusMiddleware())
    
    // 7. 錯誤處理中介軟體
    r.Use(ErrorHandlingMiddleware())
}

// CORS 配置 - 符合安全規則
func CORSMiddleware() gin.HandlerFunc {
    config := cors.Config{
        AllowOrigins:     []string{"http://localhost:3000", "https://yourdomain.com"},
        AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        AllowHeaders:     []string{"Content-Type", "Authorization"},
        AllowCredentials: true,
        MaxAge:           12 * time.Hour,
    }
    return cors.New(config)
}

// 限流中介軟體
func RateLimitMiddleware() gin.HandlerFunc {
    limiter := rate.NewLimiter(rate.Limit(100), 200) // 100 req/s, burst 200
    
    return func(c *gin.Context) {
        if !limiter.Allow() {
            c.JSON(429, APIResponse{
                Success: false,
                Message: "請求過於頻繁，請稍後再試",
                Timestamp: time.Now().Format(time.RFC3339),
            })
            c.Abort()
            return
        }
        c.Next()
    }
}
```

## 📋 API 文檔生成

### Swaggo 配置
```go
// main.go
//go:generate swag init

// @title Medical Imaging API
// @version 2.0
// @description Go-based 醫學影像處理系統 API
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.url http://www.swagger.io/support
// @contact.email support@swagger.io

// @license.name MIT
// @license.url https://opensource.org/licenses/MIT

// @host localhost:8080
// @BasePath /api/v2

// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description JWT Bearer token

func main() {
    r := gin.Default()
    
    // Swagger 文檔
    r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
    
    setupRoutes(r)
    r.Run(":8080")
}
```

## 🎯 與 Python 版本的對應關係

### API 端點對應表
| Python FastAPI | Go Gin | 功能 | 狀態 |
|----------------|--------|------|------|
| `POST /api/v1/users/` | `POST /api/v2/users` | 建立使用者 | ✅ 完全對應 |
| `GET /api/v1/users/{id}` | `GET /api/v2/users/{id}` | 取得使用者 | ✅ 完全對應 |
| `POST /api/v1/files/upload` | `POST /api/v2/files` | 檔案上傳 | ✅ 功能增強 |
| `GET /api/v1/files/` | `GET /api/v2/files` | 檔案列表 | ✅ 效能提升 |
| `POST /api/v1/tasks/` | `POST /api/v2/tasks` | 建立任務 | ✅ 完全對應 |
| `GET /api/v1/tasks/{id}` | `GET /api/v2/tasks/{id}` | 任務狀態 | ✅ 完全對應 |

### 效能對比預期
| API 端點 | Python 回應時間 | Go 預期回應時間 | 改善幅度 |
|---------|-----------------|----------------|----------|
| 使用者登入 | 150ms | 30ms | 80% 改善 |
| 檔案上傳 | 2000ms | 400ms | 80% 改善 |
| 檔案列表 | 200ms | 40ms | 80% 改善 |
| 任務建立 | 300ms | 60ms | 80% 改善 |
| 任務狀態 | 100ms | 20ms | 80% 改善 |

---

**API 規範版本**: 2.0  
**目標上線時間**: 2025年12月  
**向後相容性**: 支援 v1 API 6個月過渡期
