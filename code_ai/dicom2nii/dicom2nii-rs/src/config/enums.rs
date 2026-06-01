//! Enum definitions for DICOM to NIfTI conversion
//!
//! This module contains all enumerations used for:
//! - Series naming conventions
//! - Image orientations
//! - Modalities
//! - Contrast types
//! - Acquisition parameters

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

// ============================================================================
// Trait for enum utilities
// ============================================================================

/// Trait providing utility methods for enums
pub trait EnumExt: Sized {
    /// Returns all variants as a vector
    fn to_list() -> Vec<Self>;

    /// Returns the string representation
    fn as_str(&self) -> &'static str;
}

// ============================================================================
// Null/Empty enum
// ============================================================================

/// Represents a null or unmatched value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum NullEnum {
    #[default]
    Null,
}

impl fmt::Display for NullEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl EnumExt for NullEnum {
    fn to_list() -> Vec<Self> {
        vec![NullEnum::Null]
    }

    fn as_str(&self) -> &'static str {
        ""
    }
}

// ============================================================================
// Modality enums
// ============================================================================

/// Medical imaging modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    CT,
    MR,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::CT => write!(f, "CT"),
            Modality::MR => write!(f, "MR"),
        }
    }
}

impl FromStr for Modality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "CT" => Ok(Modality::CT),
            "MR" | "MRI" => Ok(Modality::MR),
            _ => Err(format!("Unknown modality: {}", s)),
        }
    }
}

impl EnumExt for Modality {
    fn to_list() -> Vec<Self> {
        vec![Modality::CT, Modality::MR]
    }

    fn as_str(&self) -> &'static str {
        match self {
            Modality::CT => "CT",
            Modality::MR => "MR",
        }
    }
}

// ============================================================================
// MR Acquisition Type
// ============================================================================

/// MR acquisition type (2D or 3D)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MRAcquisitionType {
    Type2D,
    Type3D,
}

impl fmt::Display for MRAcquisitionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MRAcquisitionType::Type2D => write!(f, "2D"),
            MRAcquisitionType::Type3D => write!(f, "3D"),
        }
    }
}

impl FromStr for MRAcquisitionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "2D" => Ok(MRAcquisitionType::Type2D),
            "3D" => Ok(MRAcquisitionType::Type3D),
            _ => Err(format!("Unknown acquisition type: {}", s)),
        }
    }
}

impl EnumExt for MRAcquisitionType {
    fn to_list() -> Vec<Self> {
        vec![MRAcquisitionType::Type2D, MRAcquisitionType::Type3D]
    }

    fn as_str(&self) -> &'static str {
        match self {
            MRAcquisitionType::Type2D => "2D",
            MRAcquisitionType::Type3D => "3D",
        }
    }
}

// ============================================================================
// Image Orientation
// ============================================================================

/// Image orientation/plane
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageOrientation {
    /// Axial (transverse) - ORIGINAL/PRIMARY/OTHER
    AXI,
    /// Sagittal - ORIGINAL/PRIMARY/OTHER
    SAG,
    /// Coronal - ORIGINAL/PRIMARY/OTHER
    COR,
    /// Axial reformatted - DERIVED/SECONDARY/REFORMATTED
    AXIr,
    /// Sagittal reformatted - DERIVED/SECONDARY/REFORMATTED
    SAGr,
    /// Coronal reformatted - DERIVED/SECONDARY/REFORMATTED
    CORr,
}

impl fmt::Display for ImageOrientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ImageOrientation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "AXI" => Ok(ImageOrientation::AXI),
            "SAG" => Ok(ImageOrientation::SAG),
            "COR" => Ok(ImageOrientation::COR),
            "AXIr" => Ok(ImageOrientation::AXIr),
            "SAGr" => Ok(ImageOrientation::SAGr),
            "CORr" => Ok(ImageOrientation::CORr),
            _ => Err(format!("Unknown orientation: {}", s)),
        }
    }
}

impl EnumExt for ImageOrientation {
    fn to_list() -> Vec<Self> {
        vec![
            ImageOrientation::AXI,
            ImageOrientation::SAG,
            ImageOrientation::COR,
            ImageOrientation::AXIr,
            ImageOrientation::SAGr,
            ImageOrientation::CORr,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            ImageOrientation::AXI => "AXI",
            ImageOrientation::SAG => "SAG",
            ImageOrientation::COR => "COR",
            ImageOrientation::AXIr => "AXIr",
            ImageOrientation::SAGr => "SAGr",
            ImageOrientation::CORr => "CORr",
        }
    }
}

impl ImageOrientation {
    /// Check if this orientation is a reformatted (derived) image
    pub fn is_reformatted(&self) -> bool {
        matches!(
            self,
            ImageOrientation::AXIr | ImageOrientation::SAGr | ImageOrientation::CORr
        )
    }

    /// Get the base orientation (without reformatted suffix)
    pub fn base_orientation(&self) -> ImageOrientation {
        match self {
            ImageOrientation::AXIr => ImageOrientation::AXI,
            ImageOrientation::SAGr => ImageOrientation::SAG,
            ImageOrientation::CORr => ImageOrientation::COR,
            other => *other,
        }
    }
}

// ============================================================================
// Contrast
// ============================================================================

/// Contrast agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Contrast {
    /// Contrast Enhanced
    CE,
    /// Non-Enhanced (no contrast)
    NE,
}

impl fmt::Display for Contrast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Contrast::CE => write!(f, "CE"),
            Contrast::NE => write!(f, "NE"),
        }
    }
}

impl EnumExt for Contrast {
    fn to_list() -> Vec<Self> {
        vec![Contrast::CE, Contrast::NE]
    }

    fn as_str(&self) -> &'static str {
        match self {
            Contrast::CE => "CE",
            Contrast::NE => "NE",
        }
    }
}

// ============================================================================
// Series Types
// ============================================================================

/// Common MR series types/pulse sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeriesType {
    FLAIR,
    CUBE,
    BRAVO,
    /// FSPGR (Fast Spoiled Gradient Echo) - efgre3d
    FSPGR,
    SWAN,
    ESWAN,
    /// Minimum Intensity Projection
    MIP,
    /// Magnitude
    MAG,
    ORIGINAL,
    SWANPHASE,
}

impl fmt::Display for SeriesType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for SeriesType {
    fn to_list() -> Vec<Self> {
        vec![
            SeriesType::FLAIR,
            SeriesType::CUBE,
            SeriesType::BRAVO,
            SeriesType::FSPGR,
            SeriesType::SWAN,
            SeriesType::ESWAN,
            SeriesType::MIP,
            SeriesType::MAG,
            SeriesType::ORIGINAL,
            SeriesType::SWANPHASE,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            SeriesType::FLAIR => "FLAIR",
            SeriesType::CUBE => "CUBE",
            SeriesType::BRAVO => "BRAVO",
            SeriesType::FSPGR => "efgre3d",
            SeriesType::SWAN => "SWAN",
            SeriesType::ESWAN => "eSWAN",
            SeriesType::MIP => "mIP",
            SeriesType::MAG => "mAG",
            SeriesType::ORIGINAL => "ORIGINAL",
            SeriesType::SWANPHASE => "SWANPHASE",
        }
    }
}

// ============================================================================
// CT Series Rename
// ============================================================================

/// CT series naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CTSeriesRename {
    /// Non-Contrast CT 5mm
    NCCT5mm,
    /// Non-Contrast CT Coronal
    NCCT_COR,
    /// Non-Contrast CT Bone Window
    NCCTBONE,
    /// Contrast-Enhanced CT 5mm
    CECT5mm,
    /// CT Angiography
    CTA,
}

impl fmt::Display for CTSeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for CTSeriesRename {
    fn to_list() -> Vec<Self> {
        vec![
            CTSeriesRename::NCCT5mm,
            CTSeriesRename::NCCT_COR,
            CTSeriesRename::NCCTBONE,
            CTSeriesRename::CECT5mm,
            CTSeriesRename::CTA,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            CTSeriesRename::NCCT5mm => "NCCT5mm",
            CTSeriesRename::NCCT_COR => "NCCT_COR",
            CTSeriesRename::NCCTBONE => "NCCTBONE",
            CTSeriesRename::CECT5mm => "CECT5mm",
            CTSeriesRename::CTA => "CTA",
        }
    }
}

// ============================================================================
// MR Series Rename - General
// ============================================================================

/// General MR series naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MRSeriesRename {
    // CVR (Cerebrovascular Reactivity)
    CVR,
    CVR1000,
    CVR2000,
    CVR2000_EAR,
    CVR2000_EYE,

    // Resting state
    RESTING,
    RESTING2000,

    // DSC (Dynamic Susceptibility Contrast)
    DSC_RAW,

    // DTI (Diffusion Tensor Imaging)
    DTI32D,
    DTI64D,

    // DWI (Diffusion Weighted Imaging)
    ADC,
    DWI,
    DWI0,
    SynthsegDWI0OriginalDWI,
    DWI1000,

    // SWAN (Susceptibility Weighted Angiography)
    SWAN,
    SWANmIP,
    SWANPHASE,

    // Enhanced/Extended versions
    EADC,
    ESWAN,
    ESWANmag,
    ESWANmIP,

    // MRA (MR Angiography)
    MRA_BRAIN,
    MRA_NECK,
    MRAVR_BRAIN,
    MRAVR_NECK,

    // MRV (MR Venography)
    MRV,
    MRV_SAG,
}

impl fmt::Display for MRSeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for MRSeriesRename {
    fn to_list() -> Vec<Self> {
        vec![
            MRSeriesRename::CVR,
            MRSeriesRename::CVR1000,
            MRSeriesRename::CVR2000,
            MRSeriesRename::CVR2000_EAR,
            MRSeriesRename::CVR2000_EYE,
            MRSeriesRename::RESTING,
            MRSeriesRename::RESTING2000,
            MRSeriesRename::DSC_RAW,
            MRSeriesRename::DTI32D,
            MRSeriesRename::DTI64D,
            MRSeriesRename::ADC,
            MRSeriesRename::DWI,
            MRSeriesRename::DWI0,
            MRSeriesRename::SynthsegDWI0OriginalDWI,
            MRSeriesRename::DWI1000,
            MRSeriesRename::SWAN,
            MRSeriesRename::SWANmIP,
            MRSeriesRename::SWANPHASE,
            MRSeriesRename::EADC,
            MRSeriesRename::ESWAN,
            MRSeriesRename::ESWANmag,
            MRSeriesRename::ESWANmIP,
            MRSeriesRename::MRA_BRAIN,
            MRSeriesRename::MRA_NECK,
            MRSeriesRename::MRAVR_BRAIN,
            MRSeriesRename::MRAVR_NECK,
            MRSeriesRename::MRV,
            MRSeriesRename::MRV_SAG,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            MRSeriesRename::CVR => "CVR",
            MRSeriesRename::CVR1000 => "CVR1000",
            MRSeriesRename::CVR2000 => "CVR2000",
            MRSeriesRename::CVR2000_EAR => "CVR2000_EAR",
            MRSeriesRename::CVR2000_EYE => "CVR2000_EYE",
            MRSeriesRename::RESTING => "RESTING",
            MRSeriesRename::RESTING2000 => "RESTING2000",
            MRSeriesRename::DSC_RAW => "DSC_RAW",
            MRSeriesRename::DTI32D => "DTI32D",
            MRSeriesRename::DTI64D => "DTI64D",
            MRSeriesRename::ADC => "ADC",
            MRSeriesRename::DWI => "DWI",
            MRSeriesRename::DWI0 => "DWI0",
            MRSeriesRename::SynthsegDWI0OriginalDWI => "synthseg_DWI0_original_DWI",
            MRSeriesRename::DWI1000 => "DWI1000",
            MRSeriesRename::SWAN => "SWAN",
            MRSeriesRename::SWANmIP => "SWANmIP",
            MRSeriesRename::SWANPHASE => "SWANPHASE",
            MRSeriesRename::EADC => "eADC",
            MRSeriesRename::ESWAN => "eSWAN",
            MRSeriesRename::ESWANmag => "eSWANmag",
            MRSeriesRename::ESWANmIP => "eSWANmIP",
            MRSeriesRename::MRA_BRAIN => "MRA_BRAIN",
            MRSeriesRename::MRA_NECK => "MRA_NECK",
            MRSeriesRename::MRAVR_BRAIN => "MRAVR_BRAIN",
            MRSeriesRename::MRAVR_NECK => "MRAVR_NECK",
            MRSeriesRename::MRV => "MRV",
            MRSeriesRename::MRV_SAG => "MRV_SAG",
        }
    }
}

// ============================================================================
// T1 Series Rename
// ============================================================================

/// T1-weighted series naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum T1SeriesRename {
    // Base
    T1,

    // 2D Basic
    T1_AXI,
    T1_COR,
    T1_SAG,

    // 2D Contrast Enhanced
    T1CE,
    T1CE_AXI,
    T1CE_COR,
    T1CE_SAG,

    // 2D FLAIR
    T1FLAIR,
    T1FLAIR_AXI,
    T1FLAIR_COR,
    T1FLAIR_SAG,

    // 2D FLAIR CE
    T1FLAIRCE,
    T1FLAIRCE_AXI,
    T1FLAIRCE_COR,
    T1FLAIRCE_SAG,

    // 3D CUBE
    T1CUBE,
    T1CUBE_AXI,
    T1CUBE_COR,
    T1CUBE_SAG,

    // 3D CUBE CE
    T1CUBECE,
    T1CUBECE_AXI,
    T1CUBECE_COR,
    T1CUBECE_SAG,

    // 3D FLAIR CUBE
    T1FLAIRCUBE,
    T1FLAIRCUBE_AXI,
    T1FLAIRCUBE_COR,
    T1FLAIRCUBE_SAG,

    // 3D FLAIR CUBE CE
    T1FLAIRCUBECE,
    T1FLAIRCUBECE_AXI,
    T1FLAIRCUBECE_COR,
    T1FLAIRCUBECE_SAG,

    // 3D BRAVO
    T1BRAVO,
    T1BRAVO_AXI,
    T1BRAVOCE_AXI,
    T1BRAVO_SAG,
    T1BRAVOCE_SAG,
    T1BRAVO_COR,
    T1BRAVOCE_COR,

    // 3D CUBE Reformatted
    T1CUBE_AXIr,
    T1CUBE_CORr,
    T1CUBE_SAGr,

    // 3D CUBE CE Reformatted
    T1CUBECE_AXIr,
    T1CUBECE_CORr,
    T1CUBECE_SAGr,

    // 3D FLAIR CUBE Reformatted
    T1FLAIRCUBE_AXIr,
    T1FLAIRCUBE_CORr,
    T1FLAIRCUBE_SAGr,

    // 3D FLAIR CUBE CE Reformatted
    T1FLAIRCUBECE_AXIr,
    T1FLAIRCUBECE_CORr,
    T1FLAIRCUBECE_SAGr,

    // 3D BRAVO Reformatted
    T1BRAVO_AXIr,
    T1BRAVOCE_AXIr,
    T1BRAVO_SAGr,
    T1BRAVOCE_SAGr,
    T1BRAVO_CORr,
    T1BRAVOCE_CORr,
}

impl fmt::Display for T1SeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for T1SeriesRename {
    fn to_list() -> Vec<Self> {
        vec![
            T1SeriesRename::T1,
            T1SeriesRename::T1_AXI,
            T1SeriesRename::T1_COR,
            T1SeriesRename::T1_SAG,
            T1SeriesRename::T1CE,
            T1SeriesRename::T1CE_AXI,
            T1SeriesRename::T1CE_COR,
            T1SeriesRename::T1CE_SAG,
            T1SeriesRename::T1FLAIR,
            T1SeriesRename::T1FLAIR_AXI,
            T1SeriesRename::T1FLAIR_COR,
            T1SeriesRename::T1FLAIR_SAG,
            T1SeriesRename::T1FLAIRCE,
            T1SeriesRename::T1FLAIRCE_AXI,
            T1SeriesRename::T1FLAIRCE_COR,
            T1SeriesRename::T1FLAIRCE_SAG,
            T1SeriesRename::T1CUBE,
            T1SeriesRename::T1CUBE_AXI,
            T1SeriesRename::T1CUBE_COR,
            T1SeriesRename::T1CUBE_SAG,
            T1SeriesRename::T1CUBECE,
            T1SeriesRename::T1CUBECE_AXI,
            T1SeriesRename::T1CUBECE_COR,
            T1SeriesRename::T1CUBECE_SAG,
            T1SeriesRename::T1FLAIRCUBE,
            T1SeriesRename::T1FLAIRCUBE_AXI,
            T1SeriesRename::T1FLAIRCUBE_COR,
            T1SeriesRename::T1FLAIRCUBE_SAG,
            T1SeriesRename::T1FLAIRCUBECE,
            T1SeriesRename::T1FLAIRCUBECE_AXI,
            T1SeriesRename::T1FLAIRCUBECE_COR,
            T1SeriesRename::T1FLAIRCUBECE_SAG,
            T1SeriesRename::T1BRAVO,
            T1SeriesRename::T1BRAVO_AXI,
            T1SeriesRename::T1BRAVOCE_AXI,
            T1SeriesRename::T1BRAVO_SAG,
            T1SeriesRename::T1BRAVOCE_SAG,
            T1SeriesRename::T1BRAVO_COR,
            T1SeriesRename::T1BRAVOCE_COR,
            T1SeriesRename::T1CUBE_AXIr,
            T1SeriesRename::T1CUBE_CORr,
            T1SeriesRename::T1CUBE_SAGr,
            T1SeriesRename::T1CUBECE_AXIr,
            T1SeriesRename::T1CUBECE_CORr,
            T1SeriesRename::T1CUBECE_SAGr,
            T1SeriesRename::T1FLAIRCUBE_AXIr,
            T1SeriesRename::T1FLAIRCUBE_CORr,
            T1SeriesRename::T1FLAIRCUBE_SAGr,
            T1SeriesRename::T1FLAIRCUBECE_AXIr,
            T1SeriesRename::T1FLAIRCUBECE_CORr,
            T1SeriesRename::T1FLAIRCUBECE_SAGr,
            T1SeriesRename::T1BRAVO_AXIr,
            T1SeriesRename::T1BRAVOCE_AXIr,
            T1SeriesRename::T1BRAVO_SAGr,
            T1SeriesRename::T1BRAVOCE_SAGr,
            T1SeriesRename::T1BRAVO_CORr,
            T1SeriesRename::T1BRAVOCE_CORr,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            T1SeriesRename::T1 => "T1",
            T1SeriesRename::T1_AXI => "T1_AXI",
            T1SeriesRename::T1_COR => "T1_COR",
            T1SeriesRename::T1_SAG => "T1_SAG",
            T1SeriesRename::T1CE => "T1CE",
            T1SeriesRename::T1CE_AXI => "T1CE_AXI",
            T1SeriesRename::T1CE_COR => "T1CE_COR",
            T1SeriesRename::T1CE_SAG => "T1CE_SAG",
            T1SeriesRename::T1FLAIR => "T1FLAIR",
            T1SeriesRename::T1FLAIR_AXI => "T1FLAIR_AXI",
            T1SeriesRename::T1FLAIR_COR => "T1FLAIR_COR",
            T1SeriesRename::T1FLAIR_SAG => "T1FLAIR_SAG",
            T1SeriesRename::T1FLAIRCE => "T1FLAIRCE",
            T1SeriesRename::T1FLAIRCE_AXI => "T1FLAIRCE_AXI",
            T1SeriesRename::T1FLAIRCE_COR => "T1FLAIRCE_COR",
            T1SeriesRename::T1FLAIRCE_SAG => "T1FLAIRCE_SAG",
            T1SeriesRename::T1CUBE => "T1CUBE",
            T1SeriesRename::T1CUBE_AXI => "T1CUBE_AXI",
            T1SeriesRename::T1CUBE_COR => "T1CUBE_COR",
            T1SeriesRename::T1CUBE_SAG => "T1CUBE_SAG",
            T1SeriesRename::T1CUBECE => "T1CUBECE",
            T1SeriesRename::T1CUBECE_AXI => "T1CUBECE_AXI",
            T1SeriesRename::T1CUBECE_COR => "T1CUBECE_COR",
            T1SeriesRename::T1CUBECE_SAG => "T1CUBECE_SAG",
            T1SeriesRename::T1FLAIRCUBE => "T1FLAIRCUBE",
            T1SeriesRename::T1FLAIRCUBE_AXI => "T1FLAIRCUBE_AXI",
            T1SeriesRename::T1FLAIRCUBE_COR => "T1FLAIRCUBE_COR",
            T1SeriesRename::T1FLAIRCUBE_SAG => "T1FLAIRCUBE_SAG",
            T1SeriesRename::T1FLAIRCUBECE => "T1FLAIRCUBECE",
            T1SeriesRename::T1FLAIRCUBECE_AXI => "T1FLAIRCUBECE_AXI",
            T1SeriesRename::T1FLAIRCUBECE_COR => "T1FLAIRCUBECE_COR",
            T1SeriesRename::T1FLAIRCUBECE_SAG => "T1FLAIRCUBECE_SAG",
            T1SeriesRename::T1BRAVO => "T1BRAVO",
            T1SeriesRename::T1BRAVO_AXI => "T1BRAVO_AXI",
            T1SeriesRename::T1BRAVOCE_AXI => "T1BRAVOCE_AXI",
            T1SeriesRename::T1BRAVO_SAG => "T1BRAVO_SAG",
            T1SeriesRename::T1BRAVOCE_SAG => "T1BRAVOCE_SAG",
            T1SeriesRename::T1BRAVO_COR => "T1BRAVO_COR",
            T1SeriesRename::T1BRAVOCE_COR => "T1BRAVOCE_COR",
            T1SeriesRename::T1CUBE_AXIr => "T1CUBE_AXIr",
            T1SeriesRename::T1CUBE_CORr => "T1CUBE_CORr",
            T1SeriesRename::T1CUBE_SAGr => "T1CUBE_SAGr",
            T1SeriesRename::T1CUBECE_AXIr => "T1CUBECE_AXIr",
            T1SeriesRename::T1CUBECE_CORr => "T1CUBECE_CORr",
            T1SeriesRename::T1CUBECE_SAGr => "T1CUBECE_SAGr",
            T1SeriesRename::T1FLAIRCUBE_AXIr => "T1FLAIRCUBE_AXIr",
            T1SeriesRename::T1FLAIRCUBE_CORr => "T1FLAIRCUBE_CORr",
            T1SeriesRename::T1FLAIRCUBE_SAGr => "T1FLAIRCUBE_SAGr",
            T1SeriesRename::T1FLAIRCUBECE_AXIr => "T1FLAIRCUBECE_AXIr",
            T1SeriesRename::T1FLAIRCUBECE_CORr => "T1FLAIRCUBECE_CORr",
            T1SeriesRename::T1FLAIRCUBECE_SAGr => "T1FLAIRCUBECE_SAGr",
            T1SeriesRename::T1BRAVO_AXIr => "T1BRAVO_AXIr",
            T1SeriesRename::T1BRAVOCE_AXIr => "T1BRAVOCE_AXIr",
            T1SeriesRename::T1BRAVO_SAGr => "T1BRAVO_SAGr",
            T1SeriesRename::T1BRAVOCE_SAGr => "T1BRAVOCE_SAGr",
            T1SeriesRename::T1BRAVO_CORr => "T1BRAVO_CORr",
            T1SeriesRename::T1BRAVOCE_CORr => "T1BRAVOCE_CORr",
        }
    }
}

// ============================================================================
// T2 Series Rename
// ============================================================================

/// T2-weighted series naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum T2SeriesRename {
    // Base
    T2,

    // 2D Basic
    T2_AXI,
    T2_COR,
    T2_SAG,

    // 2D Contrast Enhanced
    T2CE,
    T2CE_AXI,
    T2CE_COR,
    T2CE_SAG,

    // 2D FLAIR
    T2FLAIR,
    T2FLAIR_AXI,
    T2FLAIR_COR,
    T2FLAIR_SAG,

    // 2D FLAIR CE
    T2FLAIRCE,
    T2FLAIRCE_AXI,
    T2FLAIRCE_COR,
    T2FLAIRCE_SAG,

    // 3D CUBE
    T2CUBE,
    T2CUBE_AXI,
    T2CUBE_COR,
    T2CUBE_SAG,

    // 3D CUBE CE
    T2CUBECE,
    T2CUBECE_AXI,
    T2CUBECE_COR,
    T2CUBECE_SAG,

    // 3D FLAIR CUBE
    T2FLAIRCUBE,
    T2FLAIRCUBE_AXI,
    T2FLAIRCUBE_COR,
    T2FLAIRCUBE_SAG,

    // 3D FLAIR CUBE CE
    T2FLAIRCUBECE,
    T2FLAIRCUBECE_AXI,
    T2FLAIRCUBECE_COR,
    T2FLAIRCUBECE_SAG,

    // 3D CUBE Reformatted
    T2CUBE_AXIr,
    T2CUBE_CORr,
    T2CUBE_SAGr,

    // 3D CUBE CE Reformatted
    T2CUBECE_AXIr,
    T2CUBECE_CORr,
    T2CUBECE_SAGr,

    // 3D FLAIR CUBE Reformatted
    T2FLAIRCUBE_AXIr,
    T2FLAIRCUBE_CORr,
    T2FLAIRCUBE_SAGr,

    // 3D FLAIR CUBE CE Reformatted
    T2FLAIRCUBECE_AXIr,
    T2FLAIRCUBECE_CORr,
    T2FLAIRCUBECE_SAGr,
}

impl fmt::Display for T2SeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for T2SeriesRename {
    fn to_list() -> Vec<Self> {
        vec![
            T2SeriesRename::T2,
            T2SeriesRename::T2_AXI,
            T2SeriesRename::T2_COR,
            T2SeriesRename::T2_SAG,
            T2SeriesRename::T2CE,
            T2SeriesRename::T2CE_AXI,
            T2SeriesRename::T2CE_COR,
            T2SeriesRename::T2CE_SAG,
            T2SeriesRename::T2FLAIR,
            T2SeriesRename::T2FLAIR_AXI,
            T2SeriesRename::T2FLAIR_COR,
            T2SeriesRename::T2FLAIR_SAG,
            T2SeriesRename::T2FLAIRCE,
            T2SeriesRename::T2FLAIRCE_AXI,
            T2SeriesRename::T2FLAIRCE_COR,
            T2SeriesRename::T2FLAIRCE_SAG,
            T2SeriesRename::T2CUBE,
            T2SeriesRename::T2CUBE_AXI,
            T2SeriesRename::T2CUBE_COR,
            T2SeriesRename::T2CUBE_SAG,
            T2SeriesRename::T2CUBECE,
            T2SeriesRename::T2CUBECE_AXI,
            T2SeriesRename::T2CUBECE_COR,
            T2SeriesRename::T2CUBECE_SAG,
            T2SeriesRename::T2FLAIRCUBE,
            T2SeriesRename::T2FLAIRCUBE_AXI,
            T2SeriesRename::T2FLAIRCUBE_COR,
            T2SeriesRename::T2FLAIRCUBE_SAG,
            T2SeriesRename::T2FLAIRCUBECE,
            T2SeriesRename::T2FLAIRCUBECE_AXI,
            T2SeriesRename::T2FLAIRCUBECE_COR,
            T2SeriesRename::T2FLAIRCUBECE_SAG,
            T2SeriesRename::T2CUBE_AXIr,
            T2SeriesRename::T2CUBE_CORr,
            T2SeriesRename::T2CUBE_SAGr,
            T2SeriesRename::T2CUBECE_AXIr,
            T2SeriesRename::T2CUBECE_CORr,
            T2SeriesRename::T2CUBECE_SAGr,
            T2SeriesRename::T2FLAIRCUBE_AXIr,
            T2SeriesRename::T2FLAIRCUBE_CORr,
            T2SeriesRename::T2FLAIRCUBE_SAGr,
            T2SeriesRename::T2FLAIRCUBECE_AXIr,
            T2SeriesRename::T2FLAIRCUBECE_CORr,
            T2SeriesRename::T2FLAIRCUBECE_SAGr,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            T2SeriesRename::T2 => "T2",
            T2SeriesRename::T2_AXI => "T2_AXI",
            T2SeriesRename::T2_COR => "T2_COR",
            T2SeriesRename::T2_SAG => "T2_SAG",
            T2SeriesRename::T2CE => "T2CE",
            T2SeriesRename::T2CE_AXI => "T2CE_AXI",
            T2SeriesRename::T2CE_COR => "T2CE_COR",
            T2SeriesRename::T2CE_SAG => "T2CE_SAG",
            T2SeriesRename::T2FLAIR => "T2FLAIR",
            T2SeriesRename::T2FLAIR_AXI => "T2FLAIR_AXI",
            T2SeriesRename::T2FLAIR_COR => "T2FLAIR_COR",
            T2SeriesRename::T2FLAIR_SAG => "T2FLAIR_SAG",
            T2SeriesRename::T2FLAIRCE => "T2FLAIRCE",
            T2SeriesRename::T2FLAIRCE_AXI => "T2FLAIRCE_AXI",
            T2SeriesRename::T2FLAIRCE_COR => "T2FLAIRCE_COR",
            T2SeriesRename::T2FLAIRCE_SAG => "T2FLAIRCE_SAG",
            T2SeriesRename::T2CUBE => "T2CUBE",
            T2SeriesRename::T2CUBE_AXI => "T2CUBE_AXI",
            T2SeriesRename::T2CUBE_COR => "T2CUBE_COR",
            T2SeriesRename::T2CUBE_SAG => "T2CUBE_SAG",
            T2SeriesRename::T2CUBECE => "T2CUBECE",
            T2SeriesRename::T2CUBECE_AXI => "T2CUBECE_AXI",
            T2SeriesRename::T2CUBECE_COR => "T2CUBECE_COR",
            T2SeriesRename::T2CUBECE_SAG => "T2CUBECE_SAG",
            T2SeriesRename::T2FLAIRCUBE => "T2FLAIRCUBE",
            T2SeriesRename::T2FLAIRCUBE_AXI => "T2FLAIRCUBE_AXI",
            T2SeriesRename::T2FLAIRCUBE_COR => "T2FLAIRCUBE_COR",
            T2SeriesRename::T2FLAIRCUBE_SAG => "T2FLAIRCUBE_SAG",
            T2SeriesRename::T2FLAIRCUBECE => "T2FLAIRCUBECE",
            T2SeriesRename::T2FLAIRCUBECE_AXI => "T2FLAIRCUBECE_AXI",
            T2SeriesRename::T2FLAIRCUBECE_COR => "T2FLAIRCUBECE_COR",
            T2SeriesRename::T2FLAIRCUBECE_SAG => "T2FLAIRCUBECE_SAG",
            T2SeriesRename::T2CUBE_AXIr => "T2CUBE_AXIr",
            T2SeriesRename::T2CUBE_CORr => "T2CUBE_CORr",
            T2SeriesRename::T2CUBE_SAGr => "T2CUBE_SAGr",
            T2SeriesRename::T2CUBECE_AXIr => "T2CUBECE_AXIr",
            T2SeriesRename::T2CUBECE_CORr => "T2CUBECE_CORr",
            T2SeriesRename::T2CUBECE_SAGr => "T2CUBECE_SAGr",
            T2SeriesRename::T2FLAIRCUBE_AXIr => "T2FLAIRCUBE_AXIr",
            T2SeriesRename::T2FLAIRCUBE_CORr => "T2FLAIRCUBE_CORr",
            T2SeriesRename::T2FLAIRCUBE_SAGr => "T2FLAIRCUBE_SAGr",
            T2SeriesRename::T2FLAIRCUBECE_AXIr => "T2FLAIRCUBECE_AXIr",
            T2SeriesRename::T2FLAIRCUBECE_CORr => "T2FLAIRCUBECE_CORr",
            T2SeriesRename::T2FLAIRCUBECE_SAGr => "T2FLAIRCUBECE_SAGr",
        }
    }
}

// ============================================================================
// ASL Series Rename
// ============================================================================

/// ASL (Arterial Spin Labeling) series naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ASLSeriesRename {
    /// Base ASL sequence
    ASLSEQ,
    /// ASL Arterial Transit Time
    ASLSEQATT,
    /// ASL ATT Color map
    ASLSEQATT_COLOR,
    /// ASL Cerebral Blood Flow
    ASLSEQCBF,
    /// ASL CBF Color map
    ASLSEQCBF_COLOR,
    /// ASL Product
    ASLPROD,
    /// ASL Product CBF
    ASLPRODCBF,
    /// ASL Product CBF Color map
    ASLPRODCBF_COLOR,
    /// ASL Perfusion Weighted
    ASLSEQPW,
    /// General ASL
    ASL,
    /// Cerebral Blood Flow
    CBF,
    /// Color map
    COLOR,
    /// Cerebral Blood Flow (full name)
    CerebralBloodFlow,
}

impl fmt::Display for ASLSeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for ASLSeriesRename {
    fn to_list() -> Vec<Self> {
        vec![
            ASLSeriesRename::ASLSEQ,
            ASLSeriesRename::ASLSEQATT,
            ASLSeriesRename::ASLSEQATT_COLOR,
            ASLSeriesRename::ASLSEQCBF,
            ASLSeriesRename::ASLSEQCBF_COLOR,
            ASLSeriesRename::ASLPROD,
            ASLSeriesRename::ASLPRODCBF,
            ASLSeriesRename::ASLPRODCBF_COLOR,
            ASLSeriesRename::ASLSEQPW,
            ASLSeriesRename::ASL,
            ASLSeriesRename::CBF,
            ASLSeriesRename::COLOR,
            ASLSeriesRename::CerebralBloodFlow,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            ASLSeriesRename::ASLSEQ => "ASLSEQ",
            ASLSeriesRename::ASLSEQATT => "ASLSEQATT",
            ASLSeriesRename::ASLSEQATT_COLOR => "ASLSEQATT_COLOR",
            ASLSeriesRename::ASLSEQCBF => "ASLSEQCBF",
            ASLSeriesRename::ASLSEQCBF_COLOR => "ASLSEQCBF_COLOR",
            ASLSeriesRename::ASLPROD => "ASLPROD",
            ASLSeriesRename::ASLPRODCBF => "ASLPRODCBF",
            ASLSeriesRename::ASLPRODCBF_COLOR => "ASLPRODCBF_COLOR",
            ASLSeriesRename::ASLSEQPW => "ASLSEQPW",
            ASLSeriesRename::ASL => "ASL",
            ASLSeriesRename::CBF => "CBF",
            ASLSeriesRename::COLOR => "COLOR",
            ASLSeriesRename::CerebralBloodFlow => "Cerebral Blood Flow",
        }
    }
}

// ============================================================================
// DSC Series Rename
// ============================================================================

/// DSC (Dynamic Susceptibility Contrast) series naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DSCSeriesRename {
    /// Base DSC
    DSC,
    /// Relative CBF Color map
    RCBF,
    /// Relative CBV Color map
    RCBV,
    /// Mean Transit Time
    MTT,
}

impl fmt::Display for DSCSeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for DSCSeriesRename {
    fn to_list() -> Vec<Self> {
        vec![
            DSCSeriesRename::DSC,
            DSCSeriesRename::RCBF,
            DSCSeriesRename::RCBV,
            DSCSeriesRename::MTT,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            DSCSeriesRename::DSC => "DSC",
            DSCSeriesRename::RCBF => "DSCCBF_COLOR",
            DSCSeriesRename::RCBV => "DSCCBV_COLOR",
            DSCSeriesRename::MTT => "DSCMTT_COLOR",
        }
    }
}

// ============================================================================
// DTI Series
// ============================================================================

/// DTI (Diffusion Tensor Imaging) series
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DTISeries {
    /// 32 directions
    DTI32D,
    /// 64 directions
    DTI64D,
}

impl DTISeries {
    /// Get the number of diffusion directions
    pub fn directions(&self) -> u32 {
        match self {
            DTISeries::DTI32D => 32,
            DTISeries::DTI64D => 64,
        }
    }
}

impl fmt::Display for DTISeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for DTISeries {
    fn to_list() -> Vec<Self> {
        vec![DTISeries::DTI32D, DTISeries::DTI64D]
    }

    fn as_str(&self) -> &'static str {
        match self {
            DTISeries::DTI32D => "DTI32D",
            DTISeries::DTI64D => "DTI64D",
        }
    }
}

// ============================================================================
// Repetition Time
// ============================================================================

/// Common repetition times (TR) in milliseconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepetitionTime {
    TR1000,
    TR2000,
}

impl RepetitionTime {
    /// Get the TR value in milliseconds
    pub fn value_ms(&self) -> u32 {
        match self {
            RepetitionTime::TR1000 => 1000,
            RepetitionTime::TR2000 => 2000,
        }
    }
}

impl fmt::Display for RepetitionTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value_ms())
    }
}

impl EnumExt for RepetitionTime {
    fn to_list() -> Vec<Self> {
        vec![RepetitionTime::TR1000, RepetitionTime::TR2000]
    }

    fn as_str(&self) -> &'static str {
        match self {
            RepetitionTime::TR1000 => "1000",
            RepetitionTime::TR2000 => "2000",
        }
    }
}

// ============================================================================
// Echo Time
// ============================================================================

/// Common echo times (TE) in milliseconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EchoTime {
    TE30,
}

impl EchoTime {
    /// Get the TE value in milliseconds
    pub fn value_ms(&self) -> u32 {
        match self {
            EchoTime::TE30 => 30,
        }
    }
}

impl fmt::Display for EchoTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value_ms())
    }
}

impl EnumExt for EchoTime {
    fn to_list() -> Vec<Self> {
        vec![EchoTime::TE30]
    }

    fn as_str(&self) -> &'static str {
        match self {
            EchoTime::TE30 => "30",
        }
    }
}

// ============================================================================
// Body Part
// ============================================================================

/// Body part examined
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyPart {
    EYE,
    EAR,
}

impl fmt::Display for BodyPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl EnumExt for BodyPart {
    fn to_list() -> Vec<Self> {
        vec![BodyPart::EYE, BodyPart::EAR]
    }

    fn as_str(&self) -> &'static str {
        match self {
            BodyPart::EYE => "EYE",
            BodyPart::EAR => "EAR",
        }
    }
}

// ============================================================================
// B-Values for DWI
// ============================================================================

/// Common b-values for DWI sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BValue {
    B0,
    B1000,
}

impl BValue {
    /// Get the b-value
    pub fn value(&self) -> u32 {
        match self {
            BValue::B0 => 0,
            BValue::B1000 => 1000,
        }
    }
}

impl fmt::Display for BValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl EnumExt for BValue {
    fn to_list() -> Vec<Self> {
        vec![BValue::B0, BValue::B1000]
    }

    fn as_str(&self) -> &'static str {
        match self {
            BValue::B0 => "0",
            BValue::B1000 => "1000",
        }
    }
}

// ============================================================================
// Unified Series Rename (combining all series types)
// ============================================================================

/// Unified series rename enum that can hold any series type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeriesRename {
    CT(CTSeriesRename),
    MR(MRSeriesRename),
    T1(T1SeriesRename),
    T2(T2SeriesRename),
    ASL(ASLSeriesRename),
    DSC(DSCSeriesRename),
    DTI(DTISeries),
    Null,
}

impl fmt::Display for SeriesRename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesRename::CT(s) => write!(f, "{}", s),
            SeriesRename::MR(s) => write!(f, "{}", s),
            SeriesRename::T1(s) => write!(f, "{}", s),
            SeriesRename::T2(s) => write!(f, "{}", s),
            SeriesRename::ASL(s) => write!(f, "{}", s),
            SeriesRename::DSC(s) => write!(f, "{}", s),
            SeriesRename::DTI(s) => write!(f, "{}", s),
            SeriesRename::Null => write!(f, ""),
        }
    }
}

impl Default for SeriesRename {
    fn default() -> Self {
        SeriesRename::Null
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_display() {
        assert_eq!(Modality::CT.to_string(), "CT");
        assert_eq!(Modality::MR.to_string(), "MR");
    }

    #[test]
    fn test_modality_from_str() {
        assert_eq!(Modality::from_str("CT").unwrap(), Modality::CT);
        assert_eq!(Modality::from_str("MR").unwrap(), Modality::MR);
        assert_eq!(Modality::from_str("mr").unwrap(), Modality::MR);
        assert!(Modality::from_str("UNKNOWN").is_err());
    }

    #[test]
    fn test_image_orientation_reformatted() {
        assert!(!ImageOrientation::AXI.is_reformatted());
        assert!(ImageOrientation::AXIr.is_reformatted());
        assert_eq!(
            ImageOrientation::AXIr.base_orientation(),
            ImageOrientation::AXI
        );
    }

    #[test]
    fn test_dti_directions() {
        assert_eq!(DTISeries::DTI32D.directions(), 32);
        assert_eq!(DTISeries::DTI64D.directions(), 64);
    }

    #[test]
    fn test_b_value() {
        assert_eq!(BValue::B0.value(), 0);
        assert_eq!(BValue::B1000.value(), 1000);
    }

    #[test]
    fn test_enum_to_list() {
        let modalities = Modality::to_list();
        assert_eq!(modalities.len(), 2);
        assert!(modalities.contains(&Modality::CT));
        assert!(modalities.contains(&Modality::MR));
    }

    #[test]
    fn test_series_rename_display() {
        let ct = SeriesRename::CT(CTSeriesRename::CTA);
        assert_eq!(ct.to_string(), "CTA");

        let t1 = SeriesRename::T1(T1SeriesRename::T1BRAVO_AXI);
        assert_eq!(t1.to_string(), "T1BRAVO_AXI");
    }
}
