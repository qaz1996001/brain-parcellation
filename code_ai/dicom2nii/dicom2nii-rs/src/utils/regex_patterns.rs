//! Pre-compiled regex patterns for DICOM series recognition
//!
//! This module provides all regex patterns used for identifying
//! different MRI and CT series types from DICOM metadata.

use lazy_static::lazy_static;
use regex::Regex;

/// Collection of all pre-compiled regex patterns
pub struct Patterns {
    // ========== DWI/ADC Patterns ==========
    /// DWI or AUTODIFF series
    pub dwi: Regex,
    /// ADC (Apparent Diffusion Coefficient) - excludes eADC
    pub adc: Regex,
    /// eADC (exponential ADC)
    pub eadc: Regex,

    // ========== SWAN Patterns ==========
    /// SWAN series - excludes eSWAN
    pub swan: Regex,
    /// eSWAN (enhanced SWAN)
    pub eswan: Regex,

    // ========== MRA Patterns ==========
    /// MRA Brain (TOF, not Neck)
    pub mra_brain: Regex,
    /// MRA Neck (TOF with Neck)
    pub mra_neck: Regex,
    /// MRAVR Brain (MRA VR, not TOF, not Neck)
    pub mravr_brain: Regex,
    /// MRAVR Neck (MRA with Neck)
    pub mravr_neck: Regex,

    // ========== T1 Patterns ==========
    /// T1 weighted series
    pub t1: Regex,
    /// T1 with orientation keywords (AX, COR, SAG)
    pub t1_orientation: Regex,
    /// FLAIR sequence
    pub flair: Regex,
    /// CUBE sequence
    pub cube: Regex,
    /// BRAVO or FSPGR sequence
    pub bravo: Regex,

    // ========== T2 Patterns ==========
    /// T2 weighted series
    pub t2: Regex,

    // ========== ASL Patterns ==========
    /// Multi-Delay ASL SEQ
    pub asl_seq: Regex,
    /// ASL Transit delay
    pub asl_att: Regex,
    /// ASL Color Transit delay
    pub asl_att_color: Regex,
    /// ASL Transit corrected CBF
    pub asl_cbf: Regex,
    /// ASL Color Transit corrected CBF
    pub asl_cbf_color: Regex,
    /// ASL per del, mean PW, REF
    pub asl_pw: Regex,
    /// General ASL
    pub asl_prod: Regex,
    /// ASL CBF or Cerebral Blood Flow
    pub asl_prod_cbf: Regex,
    /// ASL CBF Color or SCREENSAVE
    pub asl_prod_cbf_color: Regex,

    // ========== DSC Patterns ==========
    /// DSC AUTOPWI or Perfusion
    pub dsc: Regex,
    /// DSC CBF
    pub dsc_cbf: Regex,
    /// DSC CBV
    pub dsc_cbv: Regex,
    /// DSC MTT
    pub dsc_mtt: Regex,

    // ========== Functional Patterns ==========
    /// CVR (Cerebrovascular Reactivity)
    pub cvr: Regex,
    /// CVR with 1000
    pub cvr_1000: Regex,
    /// CVR with 2000
    pub cvr_2000: Regex,
    /// CVR with 2000 and EAR
    pub cvr_2000_ear: Regex,
    /// CVR with 2000 and EYE
    pub cvr_2000_eye: Regex,
    /// Resting state fMRI
    pub resting: Regex,
    /// DTI (Diffusion Tensor Imaging)
    pub dti: Regex,

    // ========== Contrast Patterns ==========
    /// Contrast enhanced (+C or C+)
    pub contrast: Regex,

    // ========== File Patterns ==========
    /// NIfTI ADC file pattern
    pub nifti_adc: Regex,
    /// NIfTI ADC with suffix
    pub nifti_adc_suffix: Regex,
    /// NIfTI DWI pattern
    pub nifti_dwi: Regex,
    /// NIfTI SWAN pattern
    pub nifti_swan: Regex,
    /// NIfTI SWAN with suffix
    pub nifti_swan_suffix: Regex,
    /// NIfTI T1 pattern
    pub nifti_t1: Regex,
    /// NIfTI T1 with orientation suffix
    pub nifti_t1_suffix: Regex,
    /// NIfTI T2 pattern
    pub nifti_t2: Regex,
    /// NIfTI T2 with orientation suffix
    pub nifti_t2_suffix: Regex,
    /// NIfTI DWI0 or DWI1000 pattern
    pub nifti_dwi_bvalue: Regex,
    /// NIfTI DWI0 or DWI1000 with suffix
    pub nifti_dwi_bvalue_suffix: Regex,

    // ========== Misc Patterns ==========
    /// dcm2niix output pattern
    pub dcm2niix_output: Regex,
    /// Study folder name pattern
    pub study_folder: Regex,
    /// Age format (e.g., 057Y)
    pub age_format: Regex,
    /// Date format (YYYYMMDD)
    pub date_format: Regex,
    /// Time format (HHMMSS)
    pub time_format: Regex,
}

impl Patterns {
    /// Create all patterns
    fn new() -> Self {
        Self {
            // DWI/ADC
            dwi: Regex::new(r"(?i).*(DWI|AUTODIFF).*").unwrap(),
            adc: Regex::new(r"(?i).*(?<!e)(ADC|Apparent Diffusion Coefficient).*").unwrap(),
            eadc: Regex::new(r"(?i).*(eADC).*").unwrap(),

            // SWAN
            swan: Regex::new(r"(?i).*(?<!e)(SWAN).*").unwrap(),
            eswan: Regex::new(r"(?i).*(eSWAN).*").unwrap(),

            // MRA
            mra_brain: Regex::new(r"(?i).+(TOF)(((?!Neck).)*)$").unwrap(),
            mra_neck: Regex::new(r"(?i).*(TOF).*((Neck+).*)$").unwrap(),
            mravr_brain: Regex::new(r"(?i)((?!TOF|Neck).)*(MRA)((?!Neck).)*$").unwrap(),
            mravr_neck: Regex::new(r"(?i)((?!TOF).)*(Neck.*MRA)|(MRA.*Neck).*$").unwrap(),

            // T1
            t1: Regex::new(r"(?i).*(T1).*").unwrap(),
            t1_orientation: Regex::new(r"(?i).*(T1|AX|COR|SAG).*").unwrap(),
            flair: Regex::new(r"(?i)(FLAIR)").unwrap(),
            cube: Regex::new(r"(?i).*(CUBE).*").unwrap(),
            bravo: Regex::new(r"(?i).*(BRAVO|FSPGR).*").unwrap(),

            // T2
            t2: Regex::new(r"(?i).*(T2).*").unwrap(),

            // ASL
            asl_seq: Regex::new(r"(?i)(multi-Delay ASL SEQ)").unwrap(),
            asl_att: Regex::new(r"(?i)\(Transit delay\)").unwrap(),
            asl_att_color: Regex::new(r"(?i)\(Color Transit delay\)").unwrap(),
            asl_cbf: Regex::new(r"(?i)\(Transit corrected CBF\)").unwrap(),
            asl_cbf_color: Regex::new(r"(?i)\(Color Transit corrected CBF\)").unwrap(),
            asl_pw: Regex::new(r"(?i)\(per del, mean PW, REF\)").unwrap(),
            asl_prod: Regex::new(r"(?i).*(ASL).*").unwrap(),
            asl_prod_cbf: Regex::new(r"(?i).*((?<!r)CBF|Cerebral Blood Flow).*").unwrap(),
            asl_prod_cbf_color: Regex::new(r"(?i).*((?<!r)CBF|SCREENSAVE).*").unwrap(),

            // DSC
            dsc: Regex::new(r"(?i).*(AUTOPWI|Perfusion).*").unwrap(),
            dsc_cbf: Regex::new(r"(?i).*(CBF).*").unwrap(),
            dsc_cbv: Regex::new(r"(?i).*(CBV).*").unwrap(),
            dsc_mtt: Regex::new(r"(?i).*(MTT).*").unwrap(),

            // Functional
            cvr: Regex::new(r"(?i).*(CVR).*$").unwrap(),
            cvr_1000: Regex::new(r"(?i).*(CVR).*(1000).*$").unwrap(),
            cvr_2000: Regex::new(r"(?i).*(CVR).*(2000).*$").unwrap(),
            cvr_2000_ear: Regex::new(r"(?i).*(CVR).*(2000).*(ear).*$").unwrap(),
            cvr_2000_eye: Regex::new(r"(?i).*(CVR).*(2000).*(eye).*$").unwrap(),
            resting: Regex::new(r"(?i).*(resting).*$").unwrap(),
            dti: Regex::new(r"(?i).*(DTI).*").unwrap(),

            // Contrast
            contrast: Regex::new(r"(?i)(\+C|C\+)").unwrap(),

            // NIfTI file patterns
            nifti_adc: Regex::new(r"(?<!e)(ADC[a-z]{0,1}?)(\.nii\.gz)$").unwrap(),
            nifti_adc_suffix: Regex::new(r"(?<!e)(ADC)([a-z]{0,1}?)(\.nii\.gz)$").unwrap(),
            nifti_dwi: Regex::new(r"(?i)(DWI0)([a-z]{0,1}?)(\.nii\.gz)$").unwrap(),
            nifti_swan: Regex::new(r"(?i)(?<!e)(SWAN[a-z]{0,2}?)(\.nii\.gz)$").unwrap(),
            nifti_swan_suffix: Regex::new(r"(?i)(?<!e)(SWAN)([a-z]{0,2}?)(\.nii\.gz)$").unwrap(),
            nifti_t1: Regex::new(r"(T1.*)(\.nii\.gz)$").unwrap(),
            nifti_t1_suffix: Regex::new(r"(T1.*)(AXIr?|CORr?|SAGr?)([a-z]{0,1})(\.nii\.gz)$").unwrap(),
            nifti_t2: Regex::new(r"(T2.*)(\.nii\.gz)$").unwrap(),
            nifti_t2_suffix: Regex::new(r"(T2.*)(AXIr?|CORr?|SAGr?)([a-z]{0,1})(\.nii\.gz)$").unwrap(),
            nifti_dwi_bvalue: Regex::new(r"(DWI0|DWI1000)(\.nii\.gz)$").unwrap(),
            nifti_dwi_bvalue_suffix: Regex::new(r"(DWI0|DWI1000)([a-z]{0,2}?)(\.nii\.gz)$").unwrap(),

            // Misc
            dcm2niix_output: Regex::new(r"DICOM as (.*)\s\(").unwrap(),
            study_folder: Regex::new(r"(?i)^(\d{8})_(\d{8})_(MR|CT)_(.*)$").unwrap(),
            age_format: Regex::new(r"^(\d{3})Y$").unwrap(),
            date_format: Regex::new(r"^(\d{8})$").unwrap(),
            time_format: Regex::new(r"^(\d{6})$").unwrap(),
        }
    }
}

lazy_static! {
    /// Global pre-compiled regex patterns
    pub static ref PATTERNS: Patterns = Patterns::new();
}

/// Check if a string matches a DWI pattern
pub fn is_dwi(s: &str) -> bool {
    PATTERNS.dwi.is_match(s)
}

/// Check if a string matches an ADC pattern (excluding eADC)
pub fn is_adc(s: &str) -> bool {
    PATTERNS.adc.is_match(s) && !PATTERNS.eadc.is_match(s)
}

/// Check if a string matches an eADC pattern
pub fn is_eadc(s: &str) -> bool {
    PATTERNS.eadc.is_match(s)
}

/// Check if a string matches a SWAN pattern (excluding eSWAN)
pub fn is_swan(s: &str) -> bool {
    PATTERNS.swan.is_match(s) && !PATTERNS.eswan.is_match(s)
}

/// Check if a string matches an eSWAN pattern
pub fn is_eswan(s: &str) -> bool {
    PATTERNS.eswan.is_match(s)
}

/// Check if a string matches a T1 pattern
pub fn is_t1(s: &str) -> bool {
    PATTERNS.t1.is_match(s)
}

/// Check if a string matches a T2 pattern
pub fn is_t2(s: &str) -> bool {
    PATTERNS.t2.is_match(s)
}

/// Check if a string matches a FLAIR pattern
pub fn is_flair(s: &str) -> bool {
    PATTERNS.flair.is_match(s)
}

/// Check if a string matches a CUBE pattern
pub fn is_cube(s: &str) -> bool {
    PATTERNS.cube.is_match(s)
}

/// Check if a string matches a BRAVO/FSPGR pattern
pub fn is_bravo(s: &str) -> bool {
    PATTERNS.bravo.is_match(s)
}

/// Check if a string matches an MRA Brain pattern
pub fn is_mra_brain(s: &str) -> bool {
    PATTERNS.mra_brain.is_match(s)
}

/// Check if a string matches an MRA Neck pattern
pub fn is_mra_neck(s: &str) -> bool {
    PATTERNS.mra_neck.is_match(s)
}

/// Check if a string indicates contrast enhancement
pub fn is_contrast_enhanced(s: &str) -> bool {
    PATTERNS.contrast.is_match(s)
}

/// Check if a string matches a DTI pattern
pub fn is_dti(s: &str) -> bool {
    PATTERNS.dti.is_match(s)
}

/// Check if a string matches a resting state pattern
pub fn is_resting(s: &str) -> bool {
    PATTERNS.resting.is_match(s)
}

/// Check if a string matches a CVR pattern
pub fn is_cvr(s: &str) -> bool {
    PATTERNS.cvr.is_match(s)
}

/// Check if a string matches an ASL pattern
pub fn is_asl(s: &str) -> bool {
    PATTERNS.asl_prod.is_match(s)
}

/// Check if a string matches a DSC pattern
pub fn is_dsc(s: &str) -> bool {
    PATTERNS.dsc.is_match(s)
}

/// Extract output path from dcm2niix output
pub fn extract_dcm2niix_output(output: &str) -> Option<String> {
    PATTERNS
        .dcm2niix_output
        .captures(output)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
}

/// Parse study folder name into components
pub fn parse_study_folder(name: &str) -> Option<StudyFolderInfo> {
    PATTERNS.study_folder.captures(name).map(|caps| {
        StudyFolderInfo {
            patient_id: caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
            study_date: caps.get(2).map(|m| m.as_str().to_string()).unwrap_or_default(),
            modality: caps.get(3).map(|m| m.as_str().to_string()).unwrap_or_default(),
            accession_number: caps.get(4).map(|m| m.as_str().to_string()).unwrap_or_default(),
        }
    })
}

/// Information extracted from study folder name
#[derive(Debug, Clone)]
pub struct StudyFolderInfo {
    pub patient_id: String,
    pub study_date: String,
    pub modality: String,
    pub accession_number: String,
}

/// Validate age format (e.g., "057Y")
pub fn is_valid_age_format(s: &str) -> bool {
    PATTERNS.age_format.is_match(s)
}

/// Validate date format (YYYYMMDD)
pub fn is_valid_date_format(s: &str) -> bool {
    PATTERNS.date_format.is_match(s)
}

/// Validate time format (HHMMSS)
pub fn is_valid_time_format(s: &str) -> bool {
    PATTERNS.time_format.is_match(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwi_pattern() {
        assert!(is_dwi("AX DWI b1000"));
        assert!(is_dwi("AUTODIFF b0"));
        assert!(!is_dwi("T1 BRAVO"));
    }

    #[test]
    fn test_adc_pattern() {
        assert!(is_adc("ADC Map"));
        assert!(is_adc("Apparent Diffusion Coefficient"));
        assert!(!is_adc("eADC")); // Should exclude eADC
    }

    #[test]
    fn test_eadc_pattern() {
        assert!(is_eadc("eADC Map"));
        assert!(!is_eadc("ADC Map"));
    }

    #[test]
    fn test_swan_pattern() {
        assert!(is_swan("SWAN"));
        assert!(is_swan("AX SWAN"));
        assert!(!is_swan("eSWAN")); // Should exclude eSWAN
    }

    #[test]
    fn test_t1_pattern() {
        assert!(is_t1("T1 BRAVO AXI"));
        assert!(is_t1("SAG T1 FLAIR"));
        assert!(!is_t1("T2 FLAIR"));
    }

    #[test]
    fn test_t2_pattern() {
        assert!(is_t2("T2 FLAIR AXI"));
        assert!(is_t2("AX T2 CUBE"));
        assert!(!is_t2("T1 BRAVO"));
    }

    #[test]
    fn test_mra_brain_pattern() {
        assert!(is_mra_brain("3D TOF Brain"));
        assert!(!is_mra_brain("3D TOF Neck"));
    }

    #[test]
    fn test_contrast_pattern() {
        assert!(is_contrast_enhanced("+C"));
        assert!(is_contrast_enhanced("C+"));
        assert!(is_contrast_enhanced("T1 +C BRAVO"));
        assert!(!is_contrast_enhanced("T1 BRAVO"));
    }

    #[test]
    fn test_age_format() {
        assert!(is_valid_age_format("057Y"));
        assert!(is_valid_age_format("001Y"));
        assert!(!is_valid_age_format("57Y"));
        assert!(!is_valid_age_format("057"));
    }

    #[test]
    fn test_date_format() {
        assert!(is_valid_date_format("20240101"));
        assert!(!is_valid_date_format("2024-01-01"));
        assert!(!is_valid_date_format("240101"));
    }

    #[test]
    fn test_time_format() {
        assert!(is_valid_time_format("143052"));
        assert!(!is_valid_time_format("14:30:52"));
        assert!(!is_valid_time_format("1430"));
    }

    #[test]
    fn test_dcm2niix_output_extraction() {
        let output = "DICOM as /output/path/T1_BRAVO (Compression ignored)\n";
        let path = extract_dcm2niix_output(output);
        assert_eq!(path, Some("/output/path/T1_BRAVO".to_string()));
    }

    #[test]
    fn test_study_folder_parsing() {
        let info = parse_study_folder("12345678_20240101_MR_ACC001");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.patient_id, "12345678");
        assert_eq!(info.study_date, "20240101");
        assert_eq!(info.modality, "MR");
        assert_eq!(info.accession_number, "ACC001");
    }
}
