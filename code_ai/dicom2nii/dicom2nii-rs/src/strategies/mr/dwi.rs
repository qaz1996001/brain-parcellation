//! DWI (Diffusion Weighted Imaging) series processing strategy
//!
//! This module implements the strategy for identifying DWI MRI series
//! including b-value based classification (DWI0, DWI1000, etc.)

use crate::config::{get_config, MRSeriesRename};
use crate::dicom::DicomMetadata;
use crate::strategies::traits::{ProcessingResult, ProcessingStrategy};
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Pattern for DWI series description
    static ref DWI_PATTERN: Regex = Regex::new(r"(?i).*(DWI|DIFFUSION|DW_|DTI).*").unwrap();

    /// Pattern to exclude from DWI (ADC, eADC, etc.)
    static ref DWI_EXCLUDE_PATTERN: Regex = Regex::new(r"(?i).*(ADC|EADC|APPARENT).*").unwrap();
}

/// DWI series processing strategy
pub struct DwiProcessingStrategy {
    pattern: Regex,
}

impl DwiProcessingStrategy {
    /// Create a new DWI processing strategy
    pub fn new() -> Self {
        Self {
            pattern: DWI_PATTERN.clone(),
        }
    }

    /// Determine the DWI series name based on b-value
    fn get_dwi_series_name(&self, metadata: &DicomMetadata) -> MRSeriesRename {
        let config = get_config();

        // Check b-value
        if let Some(b_value) = metadata.b_value {
            if b_value <= config.dwi.b_value_0 {
                return MRSeriesRename::DWI0;
            } else if b_value >= config.dwi.b_value_1000 {
                return MRSeriesRename::DWI1000;
            }
        }

        // Default to generic DWI
        MRSeriesRename::DWI
    }
}

impl Default for DwiProcessingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessingStrategy for DwiProcessingStrategy {
    fn name(&self) -> &'static str {
        "DWI"
    }

    fn priority(&self) -> i32 {
        30 // Higher priority than T1 (lower number = higher priority)
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

        // Check if this matches DWI pattern
        if !self.pattern.is_match(series_desc) {
            return Ok(ProcessingResult::NotMatched);
        }

        // Exclude ADC/eADC series
        if DWI_EXCLUDE_PATTERN.is_match(series_desc) {
            return Ok(ProcessingResult::NotMatched);
        }

        // Get the series name based on b-value
        let series = self.get_dwi_series_name(metadata);

        // Check if this is a synthseg-eligible DWI0
        if series == MRSeriesRename::DWI0 && metadata.is_original() {
            return Ok(ProcessingResult::matched_with_metadata(
                MRSeriesRename::SynthsegDWI0OriginalDWI.to_string(),
                "synthseg_eligible",
            ));
        }

        Ok(ProcessingResult::matched(series.to_string()))
    }

    fn get_pattern(&self) -> Option<&Regex> {
        Some(&self.pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwi_pattern() {
        assert!(DWI_PATTERN.is_match("AX DWI"));
        assert!(DWI_PATTERN.is_match("DIFFUSION"));
        assert!(DWI_PATTERN.is_match("DW_TRACE"));
        assert!(DWI_PATTERN.is_match("DTI"));
        assert!(!DWI_PATTERN.is_match("T1 BRAVO"));
    }

    #[test]
    fn test_dwi_exclude_pattern() {
        assert!(DWI_EXCLUDE_PATTERN.is_match("ADC"));
        assert!(DWI_EXCLUDE_PATTERN.is_match("eADC"));
        assert!(DWI_EXCLUDE_PATTERN.is_match("APPARENT DIFFUSION"));
        assert!(!DWI_EXCLUDE_PATTERN.is_match("DWI"));
    }
}
