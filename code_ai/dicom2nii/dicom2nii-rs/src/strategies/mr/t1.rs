//! T1-weighted series processing strategy
//!
//! This module implements the strategy for identifying T1-weighted MRI series
//! including variants: T1, T1CE, T1FLAIR, T1CUBE, T1BRAVO, etc.

use crate::config::{
    get_config, Contrast, ImageOrientation, MRAcquisitionType, T1SeriesRename,
};
use crate::dicom::DicomMetadata;
use crate::strategies::traits::{ProcessingResult, ProcessingStrategy};
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    /// Pattern for T1 series description
    static ref T1_PATTERN: Regex = Regex::new(r"(?i).*(T1).*").unwrap();

    /// Pattern for T1 with orientation keywords
    static ref T1_ORIENTATION_PATTERN: Regex = Regex::new(r"(?i).*(T1|AX|COR|SAG).*").unwrap();

    /// 2D T1 series mapping
    static ref T1_2D_MAP: HashMap<T1Key, T1SeriesRename> = {
        let mut m = HashMap::new();
        // Basic T1
        m.insert(T1Key::new(false, false, false, ImageOrientation::AXI, Contrast::NE), T1SeriesRename::T1_AXI);
        m.insert(T1Key::new(false, false, false, ImageOrientation::SAG, Contrast::NE), T1SeriesRename::T1_SAG);
        m.insert(T1Key::new(false, false, false, ImageOrientation::COR, Contrast::NE), T1SeriesRename::T1_COR);

        // T1 CE
        m.insert(T1Key::new(false, false, false, ImageOrientation::AXI, Contrast::CE), T1SeriesRename::T1CE_AXI);
        m.insert(T1Key::new(false, false, false, ImageOrientation::SAG, Contrast::CE), T1SeriesRename::T1CE_SAG);
        m.insert(T1Key::new(false, false, false, ImageOrientation::COR, Contrast::CE), T1SeriesRename::T1CE_COR);

        // T1 FLAIR
        m.insert(T1Key::new(true, false, false, ImageOrientation::AXI, Contrast::NE), T1SeriesRename::T1FLAIR_AXI);
        m.insert(T1Key::new(true, false, false, ImageOrientation::SAG, Contrast::NE), T1SeriesRename::T1FLAIR_SAG);
        m.insert(T1Key::new(true, false, false, ImageOrientation::COR, Contrast::NE), T1SeriesRename::T1FLAIR_COR);

        // T1 FLAIR CE
        m.insert(T1Key::new(true, false, false, ImageOrientation::AXI, Contrast::CE), T1SeriesRename::T1FLAIRCE_AXI);
        m.insert(T1Key::new(true, false, false, ImageOrientation::SAG, Contrast::CE), T1SeriesRename::T1FLAIRCE_SAG);
        m.insert(T1Key::new(true, false, false, ImageOrientation::COR, Contrast::CE), T1SeriesRename::T1FLAIRCE_COR);
        m
    };

    /// 3D T1 series mapping (original)
    static ref T1_3D_ORIGINAL_MAP: HashMap<T1Key, T1SeriesRename> = {
        let mut m = HashMap::new();
        // T1 CUBE
        m.insert(T1Key::new(false, true, false, ImageOrientation::AXI, Contrast::NE), T1SeriesRename::T1CUBE_AXI);
        m.insert(T1Key::new(false, true, false, ImageOrientation::SAG, Contrast::NE), T1SeriesRename::T1CUBE_SAG);
        m.insert(T1Key::new(false, true, false, ImageOrientation::COR, Contrast::NE), T1SeriesRename::T1CUBE_COR);

        // T1 CUBE CE
        m.insert(T1Key::new(false, true, false, ImageOrientation::AXI, Contrast::CE), T1SeriesRename::T1CUBECE_AXI);
        m.insert(T1Key::new(false, true, false, ImageOrientation::SAG, Contrast::CE), T1SeriesRename::T1CUBECE_SAG);
        m.insert(T1Key::new(false, true, false, ImageOrientation::COR, Contrast::CE), T1SeriesRename::T1CUBECE_COR);

        // T1 FLAIR CUBE
        m.insert(T1Key::new(true, true, false, ImageOrientation::AXI, Contrast::NE), T1SeriesRename::T1FLAIRCUBE_AXI);
        m.insert(T1Key::new(true, true, false, ImageOrientation::SAG, Contrast::NE), T1SeriesRename::T1FLAIRCUBE_SAG);
        m.insert(T1Key::new(true, true, false, ImageOrientation::COR, Contrast::NE), T1SeriesRename::T1FLAIRCUBE_COR);

        // T1 FLAIR CUBE CE
        m.insert(T1Key::new(true, true, false, ImageOrientation::AXI, Contrast::CE), T1SeriesRename::T1FLAIRCUBECE_AXI);
        m.insert(T1Key::new(true, true, false, ImageOrientation::SAG, Contrast::CE), T1SeriesRename::T1FLAIRCUBECE_SAG);
        m.insert(T1Key::new(true, true, false, ImageOrientation::COR, Contrast::CE), T1SeriesRename::T1FLAIRCUBECE_COR);

        // T1 BRAVO
        m.insert(T1Key::new(false, false, true, ImageOrientation::AXI, Contrast::NE), T1SeriesRename::T1BRAVO_AXI);
        m.insert(T1Key::new(false, false, true, ImageOrientation::SAG, Contrast::NE), T1SeriesRename::T1BRAVO_SAG);
        m.insert(T1Key::new(false, false, true, ImageOrientation::COR, Contrast::NE), T1SeriesRename::T1BRAVO_COR);

        // T1 BRAVO CE
        m.insert(T1Key::new(false, false, true, ImageOrientation::AXI, Contrast::CE), T1SeriesRename::T1BRAVOCE_AXI);
        m.insert(T1Key::new(false, false, true, ImageOrientation::SAG, Contrast::CE), T1SeriesRename::T1BRAVOCE_SAG);
        m.insert(T1Key::new(false, false, true, ImageOrientation::COR, Contrast::CE), T1SeriesRename::T1BRAVOCE_COR);
        m
    };

    /// 3D T1 series mapping (reformatted)
    static ref T1_3D_REFORMATTED_MAP: HashMap<T1Key, T1SeriesRename> = {
        let mut m = HashMap::new();
        // T1 CUBE reformatted
        m.insert(T1Key::new(false, true, false, ImageOrientation::AXIr, Contrast::NE), T1SeriesRename::T1CUBE_AXIr);
        m.insert(T1Key::new(false, true, false, ImageOrientation::SAGr, Contrast::NE), T1SeriesRename::T1CUBE_SAGr);
        m.insert(T1Key::new(false, true, false, ImageOrientation::CORr, Contrast::NE), T1SeriesRename::T1CUBE_CORr);

        // T1 CUBE CE reformatted
        m.insert(T1Key::new(false, true, false, ImageOrientation::AXIr, Contrast::CE), T1SeriesRename::T1CUBECE_AXIr);
        m.insert(T1Key::new(false, true, false, ImageOrientation::SAGr, Contrast::CE), T1SeriesRename::T1CUBECE_SAGr);
        m.insert(T1Key::new(false, true, false, ImageOrientation::CORr, Contrast::CE), T1SeriesRename::T1CUBECE_CORr);

        // T1 FLAIR CUBE reformatted
        m.insert(T1Key::new(true, true, false, ImageOrientation::AXIr, Contrast::NE), T1SeriesRename::T1FLAIRCUBE_AXIr);
        m.insert(T1Key::new(true, true, false, ImageOrientation::SAGr, Contrast::NE), T1SeriesRename::T1FLAIRCUBE_SAGr);
        m.insert(T1Key::new(true, true, false, ImageOrientation::CORr, Contrast::NE), T1SeriesRename::T1FLAIRCUBE_CORr);

        // T1 FLAIR CUBE CE reformatted
        m.insert(T1Key::new(true, true, false, ImageOrientation::AXIr, Contrast::CE), T1SeriesRename::T1FLAIRCUBECE_AXIr);
        m.insert(T1Key::new(true, true, false, ImageOrientation::SAGr, Contrast::CE), T1SeriesRename::T1FLAIRCUBECE_SAGr);
        m.insert(T1Key::new(true, true, false, ImageOrientation::CORr, Contrast::CE), T1SeriesRename::T1FLAIRCUBECE_CORr);

        // T1 BRAVO reformatted
        m.insert(T1Key::new(false, false, true, ImageOrientation::AXIr, Contrast::NE), T1SeriesRename::T1BRAVO_AXIr);
        m.insert(T1Key::new(false, false, true, ImageOrientation::SAGr, Contrast::NE), T1SeriesRename::T1BRAVO_SAGr);
        m.insert(T1Key::new(false, false, true, ImageOrientation::CORr, Contrast::NE), T1SeriesRename::T1BRAVO_CORr);

        // T1 BRAVO CE reformatted
        m.insert(T1Key::new(false, false, true, ImageOrientation::AXIr, Contrast::CE), T1SeriesRename::T1BRAVOCE_AXIr);
        m.insert(T1Key::new(false, false, true, ImageOrientation::SAGr, Contrast::CE), T1SeriesRename::T1BRAVOCE_SAGr);
        m.insert(T1Key::new(false, false, true, ImageOrientation::CORr, Contrast::CE), T1SeriesRename::T1BRAVOCE_CORr);
        m
    };
}

/// Key for T1 series lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct T1Key {
    is_flair: bool,
    is_cube: bool,
    is_bravo: bool,
    orientation: ImageOrientation,
    contrast: Contrast,
}

impl T1Key {
    fn new(
        is_flair: bool,
        is_cube: bool,
        is_bravo: bool,
        orientation: ImageOrientation,
        contrast: Contrast,
    ) -> Self {
        Self {
            is_flair,
            is_cube,
            is_bravo,
            orientation,
            contrast,
        }
    }
}

/// T1-weighted series processing strategy
pub struct T1ProcessingStrategy {
    pattern: Regex,
}

impl T1ProcessingStrategy {
    /// Create a new T1 processing strategy
    pub fn new() -> Self {
        Self {
            pattern: T1_PATTERN.clone(),
        }
    }
}

impl Default for T1ProcessingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessingStrategy for T1ProcessingStrategy {
    fn name(&self) -> &'static str {
        "T1"
    }

    fn priority(&self) -> i32 {
        50 // Medium priority
    }

    fn supports_modality(&self, modality: &str) -> bool {
        modality.eq_ignore_ascii_case("MR")
    }

    fn process(&self, metadata: &DicomMetadata) -> Result<ProcessingResult> {
        // Check series description
        let series_desc = match &metadata.series_description {
            Some(desc) => desc,
            None => return Ok(ProcessingResult::NotMatched),
        };

        // Check T1 pattern
        if !self.pattern.is_match(series_desc) {
            return Ok(ProcessingResult::NotMatched);
        }

        // Get image orientation
        let orientation = metadata.get_orientation().unwrap_or(ImageOrientation::AXI);

        // Get contrast status
        let contrast = metadata.get_contrast();

        // Check sequence type
        let is_flair = metadata.is_t1_flair();
        let is_cube = metadata.is_cube();
        let is_bravo = metadata.is_bravo();

        // Get acquisition type
        let acq_type = metadata.get_acquisition_type();

        // Build lookup key
        let key = T1Key::new(is_flair, is_cube, is_bravo, orientation, contrast);

        // Look up in appropriate map
        let series = match acq_type {
            Some(MRAcquisitionType::Type2D) => T1_2D_MAP.get(&key),
            Some(MRAcquisitionType::Type3D) => {
                if orientation.is_reformatted() {
                    T1_3D_REFORMATTED_MAP.get(&key)
                } else {
                    T1_3D_ORIGINAL_MAP.get(&key)
                }
            }
            None => {
                // Try 3D first, then 2D
                T1_3D_ORIGINAL_MAP.get(&key)
                    .or_else(|| T1_3D_REFORMATTED_MAP.get(&key))
                    .or_else(|| T1_2D_MAP.get(&key))
            }
        };

        match series {
            Some(s) => Ok(ProcessingResult::matched(s.to_string())),
            None => {
                // Fallback to basic T1 with orientation
                let fallback = match orientation {
                    ImageOrientation::AXI | ImageOrientation::AXIr => T1SeriesRename::T1_AXI,
                    ImageOrientation::SAG | ImageOrientation::SAGr => T1SeriesRename::T1_SAG,
                    ImageOrientation::COR | ImageOrientation::CORr => T1SeriesRename::T1_COR,
                };
                Ok(ProcessingResult::matched(fallback.to_string()))
            }
        }
    }

    fn get_pattern(&self) -> Option<&Regex> {
        Some(&self.pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t1_key_hash() {
        let key1 = T1Key::new(false, false, false, ImageOrientation::AXI, Contrast::NE);
        let key2 = T1Key::new(false, false, false, ImageOrientation::AXI, Contrast::NE);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_t1_pattern() {
        assert!(T1_PATTERN.is_match("AX T1"));
        assert!(T1_PATTERN.is_match("T1 BRAVO"));
        assert!(T1_PATTERN.is_match("SAG T1 FLAIR"));
        assert!(!T1_PATTERN.is_match("T2 FLAIR"));
    }
}
