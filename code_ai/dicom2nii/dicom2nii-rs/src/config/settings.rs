//! Configuration settings management
//!
//! This module provides structures and functions for loading and managing
//! application configuration from TOML files.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Global configuration instance
static CONFIG: OnceLock<Settings> = OnceLock::new();

/// Get the global configuration instance
pub fn get_config() -> &'static Settings {
    CONFIG.get_or_init(|| Settings::default())
}

/// Initialize configuration from a file
pub fn init_config(path: impl AsRef<Path>) -> Result<&'static Settings> {
    let settings = Settings::load(path)?;
    CONFIG
        .set(settings)
        .map_err(|_| anyhow::anyhow!("Configuration already initialized"))?;
    Ok(CONFIG.get().unwrap())
}

/// Initialize configuration with defaults
pub fn init_default_config() -> &'static Settings {
    CONFIG.get_or_init(|| Settings::default())
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// General application settings
    #[serde(default)]
    pub general: GeneralSettings,

    /// File system settings
    #[serde(default)]
    pub filesystem: FilesystemSettings,

    /// dcm2niix settings
    #[serde(default)]
    pub dcm2niix: Dcm2niixSettings,

    /// Threshold settings
    #[serde(default)]
    pub thresholds: ThresholdSettings,

    /// DWI/DTI settings
    #[serde(default)]
    pub dwi: DwiSettings,

    /// Orientation settings
    #[serde(default)]
    pub orientation: OrientationSettings,

    /// DICOM tag definitions
    #[serde(default)]
    pub dicom_tags: DicomTagSettings,

    /// Regex patterns
    #[serde(default)]
    pub patterns: PatternSettings,

    /// Exclusion lists
    #[serde(default)]
    pub exclusions: ExclusionSettings,

    /// Output settings
    #[serde(default)]
    pub output: OutputSettings,

    /// Post-processing settings
    #[serde(default)]
    pub postprocess: PostprocessSettings,

    /// NIfTI to DICOM settings
    #[serde(default)]
    pub nii2dcm: Nii2dcmSettings,

    /// Modality settings
    #[serde(default)]
    pub modalities: ModalitySettings,

    /// Series type mappings
    #[serde(default)]
    pub series_types: SeriesTypeSettings,
}

impl Settings {
    /// Load configuration from a TOML file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;
        let settings: Settings = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path))?;
        Ok(settings)
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Find config file in standard locations
    pub fn find_config_file() -> Option<PathBuf> {
        let locations = [
            PathBuf::from("config.toml"),
            PathBuf::from("dicom2nii.toml"),
            dirs::config_dir()
                .map(|p| p.join("dicom2nii").join("config.toml"))
                .unwrap_or_default(),
        ];

        locations.into_iter().find(|p| p.exists())
    }

    /// Load from default locations or use defaults
    pub fn load_or_default() -> Self {
        Self::find_config_file()
            .and_then(|p| Self::load(&p).ok())
            .unwrap_or_default()
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            general: GeneralSettings::default(),
            filesystem: FilesystemSettings::default(),
            dcm2niix: Dcm2niixSettings::default(),
            thresholds: ThresholdSettings::default(),
            dwi: DwiSettings::default(),
            orientation: OrientationSettings::default(),
            dicom_tags: DicomTagSettings::default(),
            patterns: PatternSettings::default(),
            exclusions: ExclusionSettings::default(),
            output: OutputSettings::default(),
            postprocess: PostprocessSettings::default(),
            nii2dcm: Nii2dcmSettings::default(),
            modalities: ModalitySettings::default(),
            series_types: SeriesTypeSettings::default(),
        }
    }
}

// ============================================================================
// General Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralSettings {
    pub name: String,
    pub version: String,
    pub default_workers: usize,
    pub max_workers: usize,
    pub min_workers: usize,
    pub log_level: String,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            name: "dicom2nii".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            default_workers: 4,
            max_workers: 16,
            min_workers: 1,
            log_level: "info".to_string(),
        }
    }
}

// ============================================================================
// Filesystem Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemSettings {
    pub nifti_extension: String,
    pub json_extension: String,
    pub dicom_extension: String,
    pub jsonlines_extension: String,
    pub meta_directory: String,
    pub intermediate_directory: String,
    pub dicom_magic_offset: usize,
}

impl Default for FilesystemSettings {
    fn default() -> Self {
        Self {
            nifti_extension: ".nii.gz".to_string(),
            json_extension: ".json".to_string(),
            dicom_extension: ".dcm".to_string(),
            jsonlines_extension: ".jsonlines".to_string(),
            meta_directory: ".meta".to_string(),
            intermediate_directory: ".intermediate_dicom".to_string(),
            dicom_magic_offset: 128,
        }
    }
}

// ============================================================================
// DCM2NIIX Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dcm2niixSettings {
    pub executable: String,
    pub compression: String,
    pub filename_format: String,
    pub extra_args: Vec<String>,
    pub output_format: String,
}

impl Default for Dcm2niixSettings {
    fn default() -> Self {
        Self {
            executable: "dcm2niix".to_string(),
            compression: "y".to_string(),
            filename_format: "%f".to_string(),
            extra_args: vec![],
            output_format: "nii.gz".to_string(),
        }
    }
}

impl Dcm2niixSettings {
    /// Build the dcm2niix command arguments
    pub fn build_args(&self, output_dir: &Path, input_dir: &Path, filename: &str) -> Vec<String> {
        let mut args = vec![
            "-z".to_string(),
            self.compression.clone(),
            "-f".to_string(),
            filename.to_string(),
            "-o".to_string(),
            output_dir.to_string_lossy().to_string(),
        ];
        args.extend(self.extra_args.clone());
        args.push(input_dir.to_string_lossy().to_string());
        args
    }
}

// ============================================================================
// Threshold Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSettings {
    pub file_size: FileSizeThresholds,
    pub mr_parameters: MrParameterThresholds,
}

impl Default for ThresholdSettings {
    fn default() -> Self {
        Self {
            file_size: FileSizeThresholds::default(),
            mr_parameters: MrParameterThresholds::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSizeThresholds {
    /// General minimum file size in KB
    pub general: u64,
    /// ADC minimum file size in KB
    pub adc: u64,
    /// DWI minimum file size in KB
    pub dwi: u64,
    /// SWAN minimum file size in KB
    pub swan: u64,
    /// T1 minimum file size in KB
    pub t1: u64,
    /// T2 minimum file size in KB
    pub t2: u64,
    /// eSWAN minimum file size in KB
    pub eswan: u64,
    /// eADC minimum file size in KB
    pub eadc: u64,
}

impl Default for FileSizeThresholds {
    fn default() -> Self {
        Self {
            general: 1024,  // 1 MB
            adc: 100,
            dwi: 100,
            swan: 800,
            t1: 800,
            t2: 800,
            eswan: 800,
            eadc: 100,
        }
    }
}

impl FileSizeThresholds {
    /// Get threshold for a series type in bytes
    pub fn get_threshold_bytes(&self, series_type: &str) -> u64 {
        let kb = match series_type.to_uppercase().as_str() {
            s if s.contains("ADC") && !s.contains("EADC") => self.adc,
            s if s.contains("EADC") => self.eadc,
            s if s.contains("DWI") => self.dwi,
            s if s.contains("SWAN") && !s.contains("ESWAN") => self.swan,
            s if s.contains("ESWAN") => self.eswan,
            s if s.starts_with("T1") => self.t1,
            s if s.starts_with("T2") => self.t2,
            _ => self.general,
        };
        kb * 1024
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MrParameterThresholds {
    pub t1_flair: TrTeThreshold,
    pub t1_standard: TrTeThreshold,
    pub t2_flair: TrTeThreshold,
    pub t2_standard: TrTeThreshold,
}

impl Default for MrParameterThresholds {
    fn default() -> Self {
        Self {
            t1_flair: TrTeThreshold {
                tr_min: Some(800.0),
                tr_max: Some(3000.0),
                te_min: None,
                te_max: Some(30.0),
            },
            t1_standard: TrTeThreshold {
                tr_min: None,
                tr_max: Some(800.0),
                te_min: None,
                te_max: Some(30.0),
            },
            t2_flair: TrTeThreshold {
                tr_min: Some(5990.0),
                tr_max: Some(10000.0),
                te_min: Some(80.0),
                te_max: None,
            },
            t2_standard: TrTeThreshold {
                tr_min: Some(1000.0),
                tr_max: Some(5990.0),
                te_min: Some(80.0),
                te_max: None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrTeThreshold {
    pub tr_min: Option<f64>,
    pub tr_max: Option<f64>,
    pub te_min: Option<f64>,
    pub te_max: Option<f64>,
}

impl TrTeThreshold {
    /// Check if TR/TE values match this threshold
    pub fn matches(&self, tr: f64, te: f64) -> bool {
        let tr_ok = self.tr_min.map_or(true, |min| tr >= min)
            && self.tr_max.map_or(true, |max| tr <= max);
        let te_ok = self.te_min.map_or(true, |min| te >= min)
            && self.te_max.map_or(true, |max| te <= max);
        tr_ok && te_ok
    }
}

// ============================================================================
// DWI Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwiSettings {
    pub b_value_0: i32,
    pub b_value_1000: i32,
    pub dti_directions_32: u32,
    pub dti_directions_64: u32,
}

impl Default for DwiSettings {
    fn default() -> Self {
        Self {
            b_value_0: 0,
            b_value_1000: 1000,
            dti_directions_32: 32,
            dti_directions_64: 64,
        }
    }
}

// ============================================================================
// Orientation Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrientationSettings {
    pub target_axcodes: Vec<String>,
    pub source_axcodes_ras: Vec<String>,
    pub source_axcodes_las: Vec<String>,
    pub axial: String,
    pub sagittal: String,
    pub coronal: String,
    pub axial_reformatted: String,
    pub sagittal_reformatted: String,
    pub coronal_reformatted: String,
}

impl Default for OrientationSettings {
    fn default() -> Self {
        Self {
            target_axcodes: vec!["S".into(), "P".into(), "L".into()],
            source_axcodes_ras: vec!["R".into(), "A".into(), "S".into()],
            source_axcodes_las: vec!["L".into(), "A".into(), "S".into()],
            axial: "AXI".to_string(),
            sagittal: "SAG".to_string(),
            coronal: "COR".to_string(),
            axial_reformatted: "AXIr".to_string(),
            sagittal_reformatted: "SAGr".to_string(),
            coronal_reformatted: "CORr".to_string(),
        }
    }
}

// ============================================================================
// DICOM Tag Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DicomTagSettings {
    // Patient Information
    pub patient_id: [u16; 2],
    pub patient_birth_date: [u16; 2],
    pub patient_sex: [u16; 2],
    pub patient_age: [u16; 2],

    // Study Information
    pub study_date: [u16; 2],
    pub study_time: [u16; 2],
    pub accession_number: [u16; 2],
    pub modality: [u16; 2],
    pub study_description: [u16; 2],

    // Series Information
    pub series_date: [u16; 2],
    pub series_time: [u16; 2],
    pub series_description: [u16; 2],

    // Image Information
    pub image_type: [u16; 2],
    pub instance_creation_time: [u16; 2],
    pub image_orientation: [u16; 2],
    pub image_position: [u16; 2],

    // MR Parameters
    pub mr_acquisition_type: [u16; 2],
    pub repetition_time: [u16; 2],
    pub echo_time: [u16; 2],
    pub inversion_time: [u16; 2],
    pub contrast_agent: [u16; 2],
    pub pulse_sequence_name: [u16; 2],

    // Diffusion Parameters
    pub b_value: [u16; 2],
    pub dti_diffusion_directions: [u16; 2],

    // Vendor-specific
    pub magnitude_phase_info: [u16; 2],
    pub asl_technique: [u16; 2],
    pub functional_processing: [u16; 2],
    pub manufacturer_model: [u16; 2],
    pub conversion_type: [u16; 2],
}

impl Default for DicomTagSettings {
    fn default() -> Self {
        Self {
            // Patient Information
            patient_id: [0x0010, 0x0020],
            patient_birth_date: [0x0010, 0x0030],
            patient_sex: [0x0010, 0x0040],
            patient_age: [0x0010, 0x1010],

            // Study Information
            study_date: [0x0008, 0x0020],
            study_time: [0x0008, 0x0030],
            accession_number: [0x0008, 0x0050],
            modality: [0x0008, 0x0060],
            study_description: [0x0008, 0x1030],

            // Series Information
            series_date: [0x0008, 0x0021],
            series_time: [0x0008, 0x0031],
            series_description: [0x0008, 0x103E],

            // Image Information
            image_type: [0x0008, 0x0008],
            instance_creation_time: [0x0008, 0x0013],
            image_orientation: [0x0020, 0x0037],
            image_position: [0x0020, 0x0032],

            // MR Parameters
            mr_acquisition_type: [0x0018, 0x0023],
            repetition_time: [0x0018, 0x0080],
            echo_time: [0x0018, 0x0081],
            inversion_time: [0x0018, 0x0082],
            contrast_agent: [0x0018, 0x0010],
            pulse_sequence_name: [0x0019, 0x109C],

            // Diffusion Parameters (GE-specific)
            b_value: [0x0043, 0x1039],
            dti_diffusion_directions: [0x0019, 0x10E0],

            // Vendor-specific
            magnitude_phase_info: [0x0043, 0x102F],
            asl_technique: [0x0043, 0x10A4],
            functional_processing: [0x0051, 0x1002],
            manufacturer_model: [0x0008, 0x1090],
            conversion_type: [0x0008, 0x0064],
        }
    }
}

// ============================================================================
// Pattern Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternSettings {
    // DWI/ADC
    pub dwi: String,
    pub adc: String,
    pub eadc: String,

    // SWAN
    pub swan: String,
    pub eswan: String,

    // MRA
    pub mra_brain: String,
    pub mra_neck: String,
    pub mravr_brain: String,
    pub mravr_neck: String,

    // T1/T2
    pub t1: String,
    pub t1_with_orientation: String,
    pub t2: String,
    pub flair: String,
    pub cube: String,
    pub bravo: String,

    // Functional
    pub cvr: String,
    pub cvr_1000: String,
    pub cvr_2000: String,
    pub cvr_2000_ear: String,
    pub cvr_2000_eye: String,
    pub resting: String,
    pub dti: String,

    // ASL
    pub asl_seq: String,
    pub asl_att: String,
    pub asl_att_color: String,
    pub asl_cbf: String,
    pub asl_cbf_color: String,
    pub asl_pw: String,
    pub asl_prod: String,
    pub asl_prod_cbf: String,
    pub asl_prod_cbf_color: String,

    // DSC
    pub dsc: String,
    pub dsc_cbf: String,
    pub dsc_cbv: String,
    pub dsc_mtt: String,

    // Misc
    pub contrast: String,

    // File patterns
    pub nifti_adc: String,
    pub nifti_adc_suffix: String,
    pub nifti_dwi: String,
    pub nifti_swan: String,
    pub nifti_swan_suffix: String,
    pub nifti_t1: String,
    pub nifti_t1_suffix: String,
    pub nifti_t2: String,
    pub nifti_t2_suffix: String,
    pub nifti_dwi_bvalue: String,
    pub nifti_dwi_bvalue_suffix: String,

    // Utility
    pub dcm2niix_output: String,
    pub study_folder: String,
    pub age_format: String,
    pub date_format: String,
    pub time_format: String,
}

// ============================================================================
// Exclusion Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExclusionSettings {
    pub convert_exclude: Vec<String>,
    pub dicom_postprocess_exclude: Vec<String>,
    pub nii2dcm_exclude: Vec<String>,
    pub dicom_tags: DicomTagExclusions,
}

impl Default for ExclusionSettings {
    fn default() -> Self {
        Self {
            convert_exclude: vec!["MRAVR_BRAIN".into(), "MRAVR_NECK".into()],
            dicom_postprocess_exclude: vec![
                "RESTING".into(), "RESTING2000".into(),
                "CVR".into(), "CVR1000".into(), "CVR2000".into(),
                "CVR2000_EAR".into(), "CVR2000_EYE".into(),
                "eADC".into(), "eSWAN".into(),
                "DTI32D".into(), "DTI64D".into(),
                "MRAVR_NECK".into(), "MRAVR_BRAIN".into(),
                "DSC".into(), "rCBV".into(), "rCBF".into(), "MTT".into(),
                "ASLSEQ".into(),
            ],
            nii2dcm_exclude: vec![
                "MRAVR_BRAIN".into(), "MRAVR_NECK".into(),
                "DTI32D".into(), "DTI64D".into(),
                "RESTING".into(), "RESTING2000".into(),
            ],
            dicom_tags: DicomTagExclusions::default(),
        }
    }
}

impl ExclusionSettings {
    /// Check if a series should be excluded from conversion
    pub fn is_convert_excluded(&self, series: &str) -> bool {
        self.convert_exclude.iter().any(|s| s == series)
    }

    /// Check if a series should be excluded from DICOM post-processing
    pub fn is_dicom_postprocess_excluded(&self, series: &str) -> bool {
        self.dicom_postprocess_exclude.iter().any(|s| s == series)
    }

    /// Check if a series should be excluded from NIfTI to DICOM conversion
    pub fn is_nii2dcm_excluded(&self, series: &str) -> bool {
        self.nii2dcm_exclude.iter().any(|s| s == series)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DicomTagExclusions {
    pub metadata_exclude: Vec<String>,
}

impl Default for DicomTagExclusions {
    fn default() -> Self {
        Self {
            metadata_exclude: vec![
                "00020012".into(), "00020013".into(),
                "00080005".into(), "00080008".into(),
                "00080016".into(), "00080018".into(),
                "00080020".into(), "00080021".into(),
                "00080022".into(), "00080023".into(),
                "00080030".into(), "00080031".into(),
                "00080032".into(), "00080033".into(),
                "00080050".into(),
            ],
        }
    }
}

// ============================================================================
// Output Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSettings {
    pub study_folder_format: String,
    pub char_offset: u8,
    pub stats_formats: Vec<String>,
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            study_folder_format: "{patient_id}_{study_date}_{modality}_{accession}".to_string(),
            char_offset: 95,
            stats_formats: vec!["csv".into(), "json".into(), "xlsx".into()],
        }
    }
}

// ============================================================================
// Post-process Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessSettings {
    pub delete_json: bool,
    pub cleanup_small_files: bool,
    pub adc_header_correction: bool,
}

impl Default for PostprocessSettings {
    fn default() -> Self {
        Self {
            delete_json: true,
            cleanup_small_files: true,
            adc_header_correction: true,
        }
    }
}

// ============================================================================
// Nii2dcm Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nii2dcmSettings {
    pub data_type: String,
    pub max_workers: usize,
}

impl Default for Nii2dcmSettings {
    fn default() -> Self {
        Self {
            data_type: "int16".to_string(),
            max_workers: 2,
        }
    }
}

// ============================================================================
// Modality Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalitySettings {
    pub supported: Vec<String>,
}

impl Default for ModalitySettings {
    fn default() -> Self {
        Self {
            supported: vec!["CT".into(), "MR".into()],
        }
    }
}

// ============================================================================
// Series Type Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SeriesTypeSettings {
    pub mr_series: Vec<String>,
    pub t1_series: Vec<String>,
    pub t2_series: Vec<String>,
    pub ct_series: Vec<String>,
    pub asl_series: Vec<String>,
    pub dsc_series: Vec<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.general.default_workers, 4);
        assert_eq!(settings.filesystem.nifti_extension, ".nii.gz");
    }

    #[test]
    fn test_save_and_load() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("test_config.toml");

        let settings = Settings::default();
        settings.save(&path).unwrap();

        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded.general.default_workers, settings.general.default_workers);
    }

    #[test]
    fn test_file_size_threshold() {
        let thresholds = FileSizeThresholds::default();
        assert_eq!(thresholds.get_threshold_bytes("ADC"), 100 * 1024);
        assert_eq!(thresholds.get_threshold_bytes("T1_AXI"), 800 * 1024);
        assert_eq!(thresholds.get_threshold_bytes("UNKNOWN"), 1024 * 1024);
    }

    #[test]
    fn test_tr_te_threshold() {
        let threshold = TrTeThreshold {
            tr_min: Some(800.0),
            tr_max: Some(3000.0),
            te_min: None,
            te_max: Some(30.0),
        };

        assert!(threshold.matches(1000.0, 20.0));
        assert!(!threshold.matches(500.0, 20.0));  // TR too low
        assert!(!threshold.matches(1000.0, 40.0)); // TE too high
    }

    #[test]
    fn test_dcm2niix_args() {
        let settings = Dcm2niixSettings::default();
        let args = settings.build_args(
            Path::new("/output"),
            Path::new("/input"),
            "T1_BRAVO",
        );

        assert!(args.contains(&"-z".to_string()));
        assert!(args.contains(&"y".to_string()));
        assert!(args.contains(&"-f".to_string()));
        assert!(args.contains(&"T1_BRAVO".to_string()));
    }
}
