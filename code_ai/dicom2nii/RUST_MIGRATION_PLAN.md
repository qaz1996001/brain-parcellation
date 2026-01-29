# DICOM2NII Rust 遷移規劃

## 1. 專案結構設計

```
dicom2nii-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # 庫入口 (給 pyo3 和內部使用)
│   ├── main.rs                   # CLI 入口
│   │
│   ├── config/
│   │   ├── mod.rs
│   │   ├── enums.rs              # 所有枚舉定義
│   │   └── constants.rs          # 常量定義
│   │
│   ├── dicom/
│   │   ├── mod.rs
│   │   ├── parser.rs             # DICOM 標籤解析
│   │   ├── reader.rs             # DICOM 文件讀取
│   │   └── metadata.rs           # 元數據提取
│   │
│   ├── nifti/
│   │   ├── mod.rs
│   │   ├── converter.rs          # DICOM → NIfTI 轉換
│   │   ├── writer.rs             # NIfTI 文件寫入
│   │   └── header.rs             # NIfTI 頭部處理
│   │
│   ├── strategies/
│   │   ├── mod.rs
│   │   ├── traits.rs             # 處理策略 trait 定義
│   │   ├── mr/
│   │   │   ├── mod.rs
│   │   │   ├── t1.rs             # T1ProcessingStrategy
│   │   │   ├── t2.rs             # T2ProcessingStrategy
│   │   │   ├── dwi.rs            # DwiProcessingStrategy
│   │   │   ├── adc.rs            # ADCProcessingStrategy
│   │   │   ├── swan.rs           # SWANProcessingStrategy
│   │   │   ├── mra.rs            # MRA 系列策略
│   │   │   ├── asl.rs            # ASLProcessingStrategy
│   │   │   ├── dsc.rs            # DSCProcessingStrategy
│   │   │   ├── dti.rs            # DTIProcessingStrategy
│   │   │   └── resting.rs        # RestingProcessingStrategy
│   │   └── ct/
│   │       ├── mod.rs
│   │       └── basic.rs          # CT 處理策略
│   │
│   ├── postprocess/
│   │   ├── mod.rs
│   │   ├── dicom_postprocess.rs  # DICOM 後處理
│   │   └── nifti_postprocess.rs  # NIfTI 後處理
│   │
│   ├── manager/
│   │   ├── mod.rs
│   │   ├── convert_manager.rs    # 主轉換管理器
│   │   └── pipeline.rs           # 完整流程管道
│   │
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── fs.rs                 # 文件系統操作
│   │   ├── parallel.rs           # 並行處理
│   │   └── regex_patterns.rs     # 正則表達式模式
│   │
│   ├── cli/
│   │   ├── mod.rs
│   │   ├── args.rs               # 命令行參數定義
│   │   └── commands.rs           # 子命令實現
│   │
│   ├── ffi/
│   │   ├── mod.rs
│   │   └── c_api.rs              # C FFI 接口
│   │
│   └── python/
│       ├── mod.rs
│       └── bindings.rs           # PyO3 Python 綁定
│
├── python/
│   └── dicom2nii/
│       ├── __init__.py           # Python 包入口
│       └── py.typed              # PEP 561 類型標記
│
└── tests/
    ├── integration/
    └── fixtures/
```

---

## 2. Cargo.toml 配置

```toml
[package]
name = "dicom2nii"
version = "0.1.0"
edition = "2021"
description = "DICOM to NIfTI converter with Python bindings"
license = "MIT"

[lib]
name = "dicom2nii"
crate-type = ["cdylib", "rlib", "staticlib"]  # 支持多種輸出

[[bin]]
name = "dicom2nii-cli"
path = "src/main.rs"

[dependencies]
# DICOM 處理
dicom = "0.7"                      # DICOM 解析
dicom-core = "0.7"
dicom-object = "0.7"
dicom-dictionary-std = "0.7"

# NIfTI 處理
nifti = "0.16"                     # NIfTI 讀寫

# 數據處理
ndarray = "0.15"                   # N維數組 (類似 numpy)
num-traits = "0.2"

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# 並行處理
rayon = "1.10"                     # 數據並行
crossbeam = "0.8"                  # 並發工具

# CLI
clap = { version = "4.5", features = ["derive"] }

# 文件系統
walkdir = "2.5"                    # 目錄遍歷
glob = "0.3"                       # Glob 匹配

# 正則表達式
regex = "1.10"
lazy_static = "1.5"

# 錯誤處理
thiserror = "1.0"
anyhow = "1.0"

# 日誌
tracing = "0.1"
tracing-subscriber = "0.3"

# 進度條
indicatif = "0.17"                 # 類似 tqdm

# Python 綁定 (可選)
[dependencies.pyo3]
version = "0.22"
features = ["extension-module", "abi3-py38"]
optional = true

[features]
default = []
python = ["pyo3"]
ffi = []  # C FFI 接口

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = true
```

---

## 3. 核心 Trait 設計

### 3.1 處理策略 Trait

```rust
// src/strategies/traits.rs

use crate::config::enums::{MRSeriesRenameEnum, NullEnum};
use crate::dicom::metadata::DicomDataset;
use anyhow::Result;

/// 所有處理策略的基礎 trait
pub trait ProcessingStrategy: Send + Sync {
    /// 策略名稱
    fn name(&self) -> &'static str;

    /// 處理 DICOM 數據集，返回識別結果
    fn process(&self, dicom_ds: &DicomDataset) -> Result<ProcessingResult>;

    /// 是否支持該模態
    fn supports_modality(&self, modality: &str) -> bool;
}

/// MR 系列重命名策略
pub trait MRRenameStrategy: ProcessingStrategy {
    /// 處理 MR DICOM，返回序列名稱
    fn process_mr(&self, dicom_ds: &DicomDataset) -> Result<Option<MRSeriesRenameEnum>>;

    /// 獲取匹配的正則表達式
    fn get_pattern(&self) -> &regex::Regex;
}

/// NIfTI 後處理策略
pub trait NiftiPostProcessStrategy: Send + Sync {
    /// 策略名稱
    fn name(&self) -> &'static str;

    /// 處理 NIfTI 文件
    fn process(&self, nifti_path: &Path, study_path: &Path) -> Result<()>;

    /// 最小有效文件大小 (KB)
    fn min_file_size_kb(&self) -> u64;
}

/// 處理結果
pub enum ProcessingResult {
    Matched(String),       // 匹配成功，返回序列名
    NotMatched,            // 不匹配
    Error(String),         // 處理錯誤
}
```

### 3.2 DICOM 數據集抽象

```rust
// src/dicom/metadata.rs

use dicom_object::DefaultDicomObject;
use std::collections::HashMap;

/// DICOM 數據集包裝
pub struct DicomDataset {
    inner: DefaultDicomObject,
    cache: HashMap<(u16, u16), Option<String>>,  // 標籤緩存
}

impl DicomDataset {
    pub fn from_file(path: &Path) -> Result<Self> { ... }

    /// 獲取 DICOM 標籤值
    pub fn get_tag(&self, group: u16, element: u16) -> Option<&str> { ... }

    /// 獲取模態 (0008,0060)
    pub fn modality(&self) -> Option<&str> {
        self.get_tag(0x0008, 0x0060)
    }

    /// 獲取序列描述 (0008,103E)
    pub fn series_description(&self) -> Option<&str> {
        self.get_tag(0x0008, 0x103E)
    }

    /// 獲取圖像類型 (0008,0008)
    pub fn image_type(&self) -> Option<Vec<String>> { ... }

    /// 獲取圖像方向 (0020,0037)
    pub fn image_orientation(&self) -> Option<[f64; 6]> { ... }

    /// 獲取 b 值 (0043,1039) - GE specific
    pub fn b_value(&self) -> Option<i32> { ... }

    /// 獲取脈衝序列名 (0019,109C)
    pub fn pulse_sequence_name(&self) -> Option<&str> { ... }

    /// 獲取 TR (0018,0080)
    pub fn repetition_time(&self) -> Option<f64> { ... }

    /// 獲取 TE (0018,0081)
    pub fn echo_time(&self) -> Option<f64> { ... }
}
```

---

## 4. 枚舉定義遷移

```rust
// src/config/enums.rs

use serde::{Deserialize, Serialize};
use std::fmt;

/// 模態枚舉
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    CT,
    MR,
}

/// 圖像方向枚舉
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageOrientation {
    AXI,      // 軸向
    SAG,      // 矢狀
    COR,      // 冠狀
    AXIr,     // 軸向重格式化
    SAGr,     // 矢狀重格式化
    CORr,     // 冠狀重格式化
}

/// 對比劑枚舉
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Contrast {
    CE,   // 有對比劑
    NE,   // 無對比劑
}

/// MR 採集類型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MRAcquisitionType {
    Type2D,
    Type3D,
}

/// MR 序列重命名枚舉 (完整列表)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MRSeriesRename {
    // DWI 系列
    DWI0,
    DWI1000,
    ADC,
    EADC,

    // T1 系列
    T1_AXI,
    T1_SAG,
    T1_COR,
    T1CE_AXI,
    T1CE_SAG,
    T1CE_COR,
    T1FLAIR_AXI,
    T1FLAIR_SAG,
    T1CUBE_AXI,
    T1CUBE_SAG,
    T1CUBECE_AXI,
    T1BRAVO_AXI,
    T1BRAVO_SAG,
    T1BRAVOCE_AXI,
    T1BRAVOCE_SAG,
    // ... 更多 T1 變體

    // T2 系列
    T2_AXI,
    T2_SAG,
    T2_COR,
    T2FLAIR_AXI,
    T2FLAIR_SAG,
    T2FLAIR_COR,
    T2CUBE_AXI,
    T2CUBE_SAG,
    T2CUBEFLAIR_AXI,
    // ... 更多 T2 變體

    // SWAN 系列
    SWAN,
    ESWAN,

    // MRA 系列
    MRABrain,
    MRANeck,
    MRAVRBrain,
    MRAVRNeck,

    // 功能性成像
    ASL,
    ASLSEQ,
    DSC,
    Resting,
    CVR,
    DTI32D,
    DTI64D,
}

impl fmt::Display for MRSeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
```

---

## 5. 具體策略實現示例

### 5.1 DWI 處理策略

```rust
// src/strategies/mr/dwi.rs

use crate::strategies::traits::{MRRenameStrategy, ProcessingResult, ProcessingStrategy};
use crate::dicom::metadata::DicomDataset;
use crate::config::enums::MRSeriesRename;
use regex::Regex;
use lazy_static::lazy_static;
use anyhow::Result;

lazy_static! {
    static ref DWI_PATTERN: Regex = Regex::new(r"(?i)(DWI|AUTODIFF)").unwrap();
}

pub struct DwiProcessingStrategy;

impl ProcessingStrategy for DwiProcessingStrategy {
    fn name(&self) -> &'static str {
        "DWI"
    }

    fn process(&self, dicom_ds: &DicomDataset) -> Result<ProcessingResult> {
        match self.process_mr(dicom_ds)? {
            Some(series) => Ok(ProcessingResult::Matched(series.to_string())),
            None => Ok(ProcessingResult::NotMatched),
        }
    }

    fn supports_modality(&self, modality: &str) -> bool {
        modality.eq_ignore_ascii_case("MR")
    }
}

impl MRRenameStrategy for DwiProcessingStrategy {
    fn process_mr(&self, dicom_ds: &DicomDataset) -> Result<Option<MRSeriesRename>> {
        // 檢查序列描述
        let series_desc = match dicom_ds.series_description() {
            Some(desc) => desc,
            None => return Ok(None),
        };

        // 匹配 DWI 模式
        if !self.get_pattern().is_match(series_desc) {
            return Ok(None);
        }

        // 獲取 b 值
        let b_value = dicom_ds.b_value().unwrap_or(0);

        // 根據 b 值返回對應的序列
        let result = match b_value {
            0 => Some(MRSeriesRename::DWI0),
            1000 => Some(MRSeriesRename::DWI1000),
            _ => None,
        };

        Ok(result)
    }

    fn get_pattern(&self) -> &Regex {
        &DWI_PATTERN
    }
}
```

### 5.2 T1 處理策略

```rust
// src/strategies/mr/t1.rs

use crate::strategies::traits::{MRRenameStrategy, ProcessingResult, ProcessingStrategy};
use crate::dicom::metadata::DicomDataset;
use crate::config::enums::{Contrast, ImageOrientation, MRAcquisitionType, MRSeriesRename};
use regex::Regex;
use lazy_static::lazy_static;
use anyhow::Result;
use std::collections::HashMap;

lazy_static! {
    static ref T1_PATTERN: Regex = Regex::new(r"(?i)T1").unwrap();

    // 2D T1 組合映射
    static ref T1_2D_MAP: HashMap<(ImageOrientation, Contrast, bool, bool), MRSeriesRename> = {
        let mut m = HashMap::new();
        // (方向, 對比劑, FLAIR, CUBE) -> 序列名
        m.insert((ImageOrientation::AXI, Contrast::NE, false, false), MRSeriesRename::T1_AXI);
        m.insert((ImageOrientation::SAG, Contrast::NE, false, false), MRSeriesRename::T1_SAG);
        m.insert((ImageOrientation::COR, Contrast::NE, false, false), MRSeriesRename::T1_COR);
        m.insert((ImageOrientation::AXI, Contrast::CE, false, false), MRSeriesRename::T1CE_AXI);
        m.insert((ImageOrientation::SAG, Contrast::CE, false, false), MRSeriesRename::T1CE_SAG);
        // ... 更多組合
        m
    };

    // 3D T1 組合映射
    static ref T1_3D_MAP: HashMap<(ImageOrientation, Contrast, bool, bool), MRSeriesRename> = {
        let mut m = HashMap::new();
        m.insert((ImageOrientation::AXI, Contrast::NE, false, true), MRSeriesRename::T1CUBE_AXI);
        m.insert((ImageOrientation::SAG, Contrast::NE, false, true), MRSeriesRename::T1CUBE_SAG);
        // ... 更多組合
        m
    };
}

pub struct T1ProcessingStrategy;

impl ProcessingStrategy for T1ProcessingStrategy {
    fn name(&self) -> &'static str {
        "T1"
    }

    fn process(&self, dicom_ds: &DicomDataset) -> Result<ProcessingResult> {
        match self.process_mr(dicom_ds)? {
            Some(series) => Ok(ProcessingResult::Matched(series.to_string())),
            None => Ok(ProcessingResult::NotMatched),
        }
    }

    fn supports_modality(&self, modality: &str) -> bool {
        modality.eq_ignore_ascii_case("MR")
    }
}

impl MRRenameStrategy for T1ProcessingStrategy {
    fn process_mr(&self, dicom_ds: &DicomDataset) -> Result<Option<MRSeriesRename>> {
        let series_desc = match dicom_ds.series_description() {
            Some(desc) => desc,
            None => return Ok(None),
        };

        if !self.get_pattern().is_match(series_desc) {
            return Ok(None);
        }

        // 提取特徵
        let orientation = get_image_orientation(dicom_ds)?;
        let contrast = get_contrast(dicom_ds)?;
        let is_flair = check_flair(dicom_ds)?;
        let is_cube = check_cube(dicom_ds)?;
        let is_bravo = check_bravo(dicom_ds)?;
        let acq_type = get_acquisition_type(dicom_ds)?;

        // 根據採集類型選擇映射表
        let key = (orientation, contrast, is_flair, is_cube || is_bravo);
        let result = match acq_type {
            MRAcquisitionType::Type2D => T1_2D_MAP.get(&key).copied(),
            MRAcquisitionType::Type3D => T1_3D_MAP.get(&key).copied(),
        };

        Ok(result)
    }

    fn get_pattern(&self) -> &Regex {
        &T1_PATTERN
    }
}

// 輔助函數
fn get_image_orientation(ds: &DicomDataset) -> Result<ImageOrientation> {
    let orientation = ds.image_orientation().unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    // 判斷主方向
    let (row_x, row_y, row_z) = (orientation[0], orientation[1], orientation[2]);
    let (col_x, col_y, col_z) = (orientation[3], orientation[4], orientation[5]);

    // 簡化判斷邏輯
    if row_z.abs() < 0.5 && col_z.abs() < 0.5 {
        Ok(ImageOrientation::AXI)
    } else if row_x.abs() < 0.5 && col_x.abs() < 0.5 {
        Ok(ImageOrientation::SAG)
    } else {
        Ok(ImageOrientation::COR)
    }
}

fn get_contrast(ds: &DicomDataset) -> Result<Contrast> {
    // 檢查對比劑相關標籤
    if ds.get_tag(0x0018, 0x0010).is_some() {
        Ok(Contrast::CE)
    } else {
        Ok(Contrast::NE)
    }
}

fn check_flair(ds: &DicomDataset) -> Result<bool> {
    let tr = ds.repetition_time().unwrap_or(0.0);
    let te = ds.echo_time().unwrap_or(0.0);
    Ok(tr >= 800.0 && tr <= 3000.0 && te <= 30.0)
}

fn check_cube(ds: &DicomDataset) -> Result<bool> {
    let pulse_seq = ds.pulse_sequence_name().unwrap_or("");
    Ok(pulse_seq.to_uppercase().contains("CUBE"))
}

fn check_bravo(ds: &DicomDataset) -> Result<bool> {
    let pulse_seq = ds.pulse_sequence_name().unwrap_or("");
    let upper = pulse_seq.to_uppercase();
    Ok(upper.contains("BRAVO") || upper.contains("FSPGR"))
}
```

---

## 6. 轉換管理器

```rust
// src/manager/convert_manager.rs

use crate::strategies::traits::ProcessingStrategy;
use crate::strategies::mr::*;
use crate::dicom::metadata::DicomDataset;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn};

/// 轉換管理器
pub struct ConvertManager {
    input_path: PathBuf,
    output_path: PathBuf,
    strategies: Vec<Arc<dyn ProcessingStrategy>>,
}

impl ConvertManager {
    pub fn new(input_path: impl AsRef<Path>, output_path: impl AsRef<Path>) -> Self {
        Self {
            input_path: input_path.as_ref().to_path_buf(),
            output_path: output_path.as_ref().to_path_buf(),
            strategies: Self::default_strategies(),
        }
    }

    fn default_strategies() -> Vec<Arc<dyn ProcessingStrategy>> {
        vec![
            Arc::new(DwiProcessingStrategy),
            Arc::new(AdcProcessingStrategy),
            Arc::new(EadcProcessingStrategy),
            Arc::new(SwanProcessingStrategy),
            Arc::new(EswanProcessingStrategy),
            Arc::new(MraBrainProcessingStrategy),
            Arc::new(MraNeckProcessingStrategy),
            Arc::new(T1ProcessingStrategy),
            Arc::new(T2ProcessingStrategy),
            Arc::new(AslProcessingStrategy),
            Arc::new(DscProcessingStrategy),
            Arc::new(RestingProcessingStrategy),
            Arc::new(CvrProcessingStrategy),
            Arc::new(DtiProcessingStrategy),
        ]
    }

    /// 運行轉換
    pub fn run(&self, num_workers: usize) -> Result<Vec<PathBuf>> {
        info!("Starting conversion with {} workers", num_workers);

        // 收集所有 DICOM 文件
        let dicom_files = self.collect_dicom_files()?;
        info!("Found {} DICOM files", dicom_files.len());

        // 並行處理
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()?;

        let results: Vec<_> = pool.install(|| {
            dicom_files
                .par_iter()
                .map(|path| self.process_dicom(path))
                .collect()
        });

        // 收集成功的輸出路徑
        let output_paths: Vec<_> = results
            .into_iter()
            .filter_map(|r| r.ok())
            .flatten()
            .collect();

        info!("Conversion complete. {} studies processed", output_paths.len());
        Ok(output_paths)
    }

    fn collect_dicom_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for entry in walkdir::WalkDir::new(&self.input_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() && Self::is_dicom_file(path) {
                files.push(path.to_path_buf());
            }
        }
        Ok(files)
    }

    fn is_dicom_file(path: &Path) -> bool {
        path.extension().map_or(false, |ext| ext == "dcm")
            || path.extension().is_none()  // DICOM 文件可能沒有擴展名
    }

    fn process_dicom(&self, path: &Path) -> Result<Option<PathBuf>> {
        let dicom_ds = DicomDataset::from_file(path)?;

        // 嘗試所有策略
        for strategy in &self.strategies {
            let modality = dicom_ds.modality().unwrap_or("");
            if !strategy.supports_modality(modality) {
                continue;
            }

            match strategy.process(&dicom_ds)? {
                ProcessingResult::Matched(series_name) => {
                    return self.rename_and_copy(path, &dicom_ds, &series_name);
                }
                ProcessingResult::NotMatched => continue,
                ProcessingResult::Error(e) => {
                    warn!("Strategy {} error: {}", strategy.name(), e);
                    continue;
                }
            }
        }

        Ok(None)
    }

    fn rename_and_copy(
        &self,
        src: &Path,
        dicom_ds: &DicomDataset,
        series_name: &str,
    ) -> Result<Option<PathBuf>> {
        // 生成輸出路徑: {patient_id}_{date}_{modality}_{accession}/{series_name}/
        let output_study = self.get_output_study_path(dicom_ds)?;
        let output_series = output_study.join(series_name);

        std::fs::create_dir_all(&output_series)?;

        // 複製文件
        let filename = src.file_name().unwrap_or_default();
        let dest = output_series.join(filename);
        std::fs::copy(src, &dest)?;

        Ok(Some(output_study))
    }

    fn get_output_study_path(&self, dicom_ds: &DicomDataset) -> Result<PathBuf> {
        let patient_id = dicom_ds.get_tag(0x0010, 0x0020).unwrap_or("UNKNOWN");
        let study_date = dicom_ds.get_tag(0x0008, 0x0020).unwrap_or("19700101");
        let modality = dicom_ds.modality().unwrap_or("XX");
        let accession = dicom_ds.get_tag(0x0008, 0x0050).unwrap_or("0");

        let folder_name = format!("{}_{}_{}_{}",
            patient_id, study_date, modality, accession);

        Ok(self.output_path.join(folder_name))
    }
}
```

---

## 7. CLI 實現

```rust
// src/cli/args.rs

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dicom2nii")]
#[command(author = "Your Name")]
#[command(version = "0.1.0")]
#[command(about = "DICOM to NIfTI converter", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Number of worker threads
    #[arg(short = 'j', long, default_value = "4", global = true)]
    pub jobs: usize,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Rename and organize DICOM files
    Rename {
        /// Input directory containing raw DICOM files
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for organized DICOM files
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Convert DICOM to NIfTI format
    Convert {
        /// Input directory containing organized DICOM files
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for NIfTI files
        #[arg(short, long)]
        output: PathBuf,

        /// Skip DICOM rename step (input is already organized)
        #[arg(long)]
        skip_rename: bool,
    },

    /// Run full pipeline (rename + convert + postprocess)
    Pipeline {
        /// Input directory containing raw DICOM files
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for final NIfTI files
        #[arg(short, long)]
        output: PathBuf,

        /// Keep intermediate DICOM files
        #[arg(long)]
        keep_intermediate: bool,
    },

    /// Generate statistics report
    Stats {
        /// Directory to analyze
        #[arg(short, long)]
        path: PathBuf,

        /// Output format (csv, json, excel)
        #[arg(short, long, default_value = "csv")]
        format: String,
    },

    /// Convert NIfTI back to DICOM
    Nii2dcm {
        /// Input NIfTI file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for DICOM files
        #[arg(short, long)]
        output: PathBuf,

        /// Reference DICOM for metadata
        #[arg(long)]
        reference: Option<PathBuf>,
    },
}
```

```rust
// src/main.rs

use clap::Parser;
use dicom2nii::cli::{args::Cli, commands};
use tracing_subscriber;
use anyhow::Result;

fn main() -> Result<()> {
    // 初始化日誌
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    // 設置工作線程數
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.jobs)
        .build_global()?;

    match cli.command {
        Commands::Rename { input, output } => {
            commands::run_rename(&input, &output, cli.jobs)?;
        }
        Commands::Convert { input, output, skip_rename } => {
            commands::run_convert(&input, &output, skip_rename, cli.jobs)?;
        }
        Commands::Pipeline { input, output, keep_intermediate } => {
            commands::run_pipeline(&input, &output, keep_intermediate, cli.jobs)?;
        }
        Commands::Stats { path, format } => {
            commands::run_stats(&path, &format)?;
        }
        Commands::Nii2dcm { input, output, reference } => {
            commands::run_nii2dcm(&input, &output, reference.as_deref())?;
        }
    }

    Ok(())
}
```

---

## 8. PyO3 Python 綁定

```rust
// src/python/bindings.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::path::PathBuf;
use crate::manager::convert_manager::ConvertManager;
use crate::manager::pipeline::Pipeline;

/// Python 模塊
#[pymodule]
fn dicom2nii(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConvertManager>()?;
    m.add_class::<PyPipeline>()?;
    m.add_function(wrap_pyfunction!(convert_dicom_to_nifti, m)?)?;
    m.add_function(wrap_pyfunction!(rename_dicom, m)?)?;
    Ok(())
}

/// 轉換管理器的 Python 包裝
#[pyclass(name = "ConvertManager")]
pub struct PyConvertManager {
    inner: ConvertManager,
}

#[pymethods]
impl PyConvertManager {
    #[new]
    fn new(input_path: &str, output_path: &str) -> Self {
        Self {
            inner: ConvertManager::new(input_path, output_path),
        }
    }

    /// 運行轉換
    fn run(&self, workers: Option<usize>) -> PyResult<Vec<String>> {
        let workers = workers.unwrap_or(4);
        self.inner
            .run(workers)
            .map(|paths| paths.iter().map(|p| p.display().to_string()).collect())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// 完整管道的 Python 包裝
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    input_path: PathBuf,
    output_path: PathBuf,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new(input_path: &str, output_path: &str) -> Self {
        Self {
            input_path: PathBuf::from(input_path),
            output_path: PathBuf::from(output_path),
        }
    }

    /// 運行完整管道
    fn run(
        &self,
        workers: Option<usize>,
        keep_intermediate: Option<bool>,
    ) -> PyResult<Vec<String>> {
        let workers = workers.unwrap_or(4);
        let keep = keep_intermediate.unwrap_or(false);

        let pipeline = Pipeline::new(&self.input_path, &self.output_path);
        pipeline
            .run(workers, keep)
            .map(|paths| paths.iter().map(|p| p.display().to_string()).collect())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// 簡化的轉換函數
#[pyfunction]
#[pyo3(signature = (input_path, output_path, workers=4))]
fn convert_dicom_to_nifti(
    input_path: &str,
    output_path: &str,
    workers: usize,
) -> PyResult<Vec<String>> {
    let manager = ConvertManager::new(input_path, output_path);
    manager
        .run(workers)
        .map(|paths| paths.iter().map(|p| p.display().to_string()).collect())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// 簡化的重命名函數
#[pyfunction]
#[pyo3(signature = (input_path, output_path, workers=4))]
fn rename_dicom(
    input_path: &str,
    output_path: &str,
    workers: usize,
) -> PyResult<Vec<String>> {
    let manager = ConvertManager::new(input_path, output_path);
    manager
        .rename_only(workers)
        .map(|paths| paths.iter().map(|p| p.display().to_string()).collect())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
```

---

## 9. C FFI 接口 (可選)

```rust
// src/ffi/c_api.rs

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// C 接口：轉換 DICOM 到 NIfTI
///
/// # Safety
/// 調用者必須確保 input_path 和 output_path 是有效的 UTF-8 字符串
#[no_mangle]
pub unsafe extern "C" fn dicom2nii_convert(
    input_path: *const c_char,
    output_path: *const c_char,
    workers: usize,
) -> i32 {
    let input = match CStr::from_ptr(input_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let output = match CStr::from_ptr(output_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let manager = crate::manager::convert_manager::ConvertManager::new(input, output);
    match manager.run(workers) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// 獲取版本字符串
#[no_mangle]
pub extern "C" fn dicom2nii_version() -> *const c_char {
    static VERSION: &str = "0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// 釋放由本庫分配的字符串
///
/// # Safety
/// ptr 必須是由本庫分配的有效指針
#[no_mangle]
pub unsafe extern "C" fn dicom2nii_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}
```

---

## 10. 構建和分發

### 10.1 構建命令

```bash
# 構建靜態 CLI
cargo build --release

# 構建 Python 擴展 (需要 maturin)
pip install maturin
maturin build --release

# 構建帶 C FFI 的靜態庫
cargo build --release --features ffi

# 交叉編譯到其他平台
cargo build --release --target x86_64-unknown-linux-musl  # Linux 靜態鏈接
cargo build --release --target x86_64-pc-windows-gnu      # Windows
cargo build --release --target x86_64-apple-darwin        # macOS
```

### 10.2 Python 包結構

```
dicom2nii/
├── pyproject.toml
├── Cargo.toml
├── src/
│   └── ...
└── python/
    └── dicom2nii/
        ├── __init__.py
        └── py.typed
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "dicom2nii"
version = "0.1.0"
description = "Fast DICOM to NIfTI converter"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
features = ["python"]
python-source = "python"
module-name = "dicom2nii._native"
```

```python
# python/dicom2nii/__init__.py
from dicom2nii._native import (
    ConvertManager,
    Pipeline,
    convert_dicom_to_nifti,
    rename_dicom,
)

__version__ = "0.1.0"
__all__ = [
    "ConvertManager",
    "Pipeline",
    "convert_dicom_to_nifti",
    "rename_dicom",
]
```

---

## 11. 遷移優先級

### 第一階段：核心功能 (MVP)
1. ✅ DICOM 解析 (dicom-rs)
2. ✅ 枚舉和配置遷移
3. ✅ 基礎策略 trait
4. ✅ DWI/ADC 策略實現
5. ✅ T1/T2 策略實現
6. ✅ CLI 基本功能

### 第二階段：完整功能
1. ⬜ 所有 15 種 MR 策略
2. ⬜ CT 策略
3. ⬜ NIfTI 轉換 (dcm2niix 包裝或原生實現)
4. ⬜ 後處理策略
5. ⬜ 統計報告生成

### 第三階段：集成和優化
1. ⬜ PyO3 Python 綁定
2. ⬜ C FFI 接口
3. ⬜ 性能優化
4. ⬜ 完整測試套件
5. ⬜ 文檔和示例

---

## 12. 預期優勢

| 方面 | Python 現狀 | Rust 預期 |
|------|-------------|-----------|
| 啟動時間 | ~500ms | <10ms |
| 內存使用 | 高 (GC) | 低 (無 GC) |
| 並行處理 | GIL 限制 | 真正並行 |
| 分發 | 需要 Python 環境 | 單一二進制 |
| 類型安全 | 運行時檢查 | 編譯時保證 |
| 錯誤處理 | 異常 | Result 類型 |

---

## 13. 風險和挑戰

1. **dicom-rs 功能差異**
   - 某些 GE 私有標籤可能需要手動處理
   - 需要驗證所有 DICOM 標籤的讀取

2. **NIfTI 轉換**
   - 可能需要繼續依賴 dcm2niix
   - 或使用 nifti-rs 自行實現 (工作量大)

3. **正則表達式兼容性**
   - Python re 和 Rust regex 語法略有不同
   - 需要測試所有模式

4. **測試數據**
   - 需要準備各種 DICOM 測試數據
   - 確保與 Python 版本結果一致
