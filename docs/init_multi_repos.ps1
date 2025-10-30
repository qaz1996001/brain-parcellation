# Task04_git 多倉庫拆分初始化腳本
# 用法: .\init_multi_repos.ps1
# 運行環境: PowerShell 5.0+ (Windows)

$RootPath = "D:\00_Chen\Task04_git_rdx"
$repos = @("rdxai", "brain-aneurysm", "brain-cmb", "brain-parcellation", "dicom2nii")

Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Task04_git 多倉庫拆分初始化                         ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 檢查根目錄
if (-not (Test-Path $RootPath)) {
    Write-Host "錯誤：根目錄不存在: $RootPath" -ForegroundColor Red
    exit 1
}

Write-Host "根目錄: $RootPath" -ForegroundColor Yellow
Write-Host ""

# 為每個倉庫設置 Git
Write-Host "初始化倉庫..." -ForegroundColor Cyan
foreach ($repo in $repos) {
    $path = Join-Path $RootPath $repo
    $gitPath = Join-Path $path ".git"
    
    if (Test-Path $gitPath) {
        Write-Host "✓ $repo: 已存在" -ForegroundColor Green
    } else {
        Write-Host "設置 $repo..." -ForegroundColor Yellow
        Push-Location $path
        
        # 初始化 Git
        git init 2>&1 | Out-Null
        
        # 創建初始文件
        if (-not (Test-Path "README.md")) {
            "$repo project" | Out-File -FilePath "README.md" -Encoding UTF8
        }
        
        # 初始提交
        git add . 2>&1 | Out-Null
        git commit -m "Initial commit: $(Get-Date -Format 'yyyy-MM-dd')" 2>&1 | Out-Null
        
        Pop-Location
        Write-Host "✓ $repo: 初始化完成" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   初始化完成！                                         ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "  1. 複製代碼到各倉庫"
Write-Host "  2. 更新導入路徑"
Write-Host "  3. 安裝依賴: pip install -e rdxai && pip install -e brain-aneurysm 等"
Write-Host "  4. 運行測試驗證"
Write-Host ""
