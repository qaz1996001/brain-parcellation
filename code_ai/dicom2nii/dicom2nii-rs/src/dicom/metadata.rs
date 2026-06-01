//! DICOM metadata extraction
//!
//! This module provides structures and functions for extracting
//! relevant metadata from DICOM files.

use crate::config::{
    get_config, ImageOrientation, MRAcquisitionType, Modality, Contrast,
};
use anyhow::{Context, Result};
use dicom::object::{FileDicomObject, InMemDicomObject, Tag};
use dicom::dictionary_std::tags;
use std::path::Path;

/// DICOM metadata container
#[derive(Debug, Clone)]
pub struct DicomMetadata {
    // Patient Information
    pub patient_id: Option<String>,
    pub patient_birth_date: Option<String>,
    pub patient_sex: Option<String>,
    pub patient_age: Option<String>,

    // Study Information
    pub study_date: Option<String>,
    pub study_time: Option<String>,
    pub accession_number: Option<String>,
    pub modality: Option<String>,
    pub study_description: Option<String>,

    // Series Information
    pub series_date: Option<String>,
    pub series_time: Option<String>,
    pub series_description: Option<String>,
    pub series_number: Option<i32>,

    // Image Information
    pub image_type: Vec<String>,
    pub instance_creation_time: Option<String>,
    pub image_orientation: Option<[f64; 6]>,
    pub image_position: Option<[f64; 3]>,

    // MR-specific Parameters
    pub mr_acquisition_type: Option<String>,
    pub repetition_time: Option<f64>,
    pub echo_time: Option<f64>,
    pub inversion_time: Option<f64>,
    pub contrast_agent: Option<String>,
    pub pulse_sequence_name: Option<String>,

    // Diffusion Parameters
    pub b_value: Option<i32>,
    pub dti_diffusion_directions: Option<u32>,

    // Vendor-specific
    pub manufacturer_model: Option<String>,
}

impl DicomMetadata {
    /// Extract metadata from a DICOM object
    pub fn from_dicom(obj: &FileDicomObject<InMemDicomObject>) -> Result<Self> {
        Ok(Self {
            // Patient Information
            patient_id: get_string(obj, tags::PATIENT_ID),
            patient_birth_date: get_string(obj, tags::PATIENT_BIRTH_DATE),
            patient_sex: get_string(obj, tags::PATIENT_SEX),
            patient_age: get_string(obj, tags::PATIENT_AGE),

            // Study Information
            study_date: get_string(obj, tags::STUDY_DATE),
            study_time: get_string(obj, tags::STUDY_TIME),
            accession_number: get_string(obj, tags::ACCESSION_NUMBER),
            modality: get_string(obj, tags::MODALITY),
            study_description: get_string(obj, tags::STUDY_DESCRIPTION),

            // Series Information
            series_date: get_string(obj, tags::SERIES_DATE),
            series_time: get_string(obj, tags::SERIES_TIME),
            series_description: get_string(obj, tags::SERIES_DESCRIPTION),
            series_number: get_i32(obj, tags::SERIES_NUMBER),

            // Image Information
            image_type: get_string_list(obj, tags::IMAGE_TYPE),
            instance_creation_time: get_string(obj, tags::INSTANCE_CREATION_TIME),
            image_orientation: get_f64_array_6(obj, tags::IMAGE_ORIENTATION_PATIENT),
            image_position: get_f64_array_3(obj, tags::IMAGE_POSITION_PATIENT),

            // MR-specific Parameters
            mr_acquisition_type: get_string(obj, tags::MR_ACQUISITION_TYPE),
            repetition_time: get_f64(obj, tags::REPETITION_TIME),
            echo_time: get_f64(obj, tags::ECHO_TIME),
            inversion_time: get_f64(obj, tags::INVERSION_TIME),
            contrast_agent: get_string(obj, tags::CONTRAST_BOLUS_AGENT),
            pulse_sequence_name: get_string_by_tag(obj, 0x0019, 0x109C),

            // Diffusion Parameters (GE-specific)
            b_value: get_i32_by_tag(obj, 0x0043, 0x1039),
            dti_diffusion_directions: get_u32_by_tag(obj, 0x0019, 0x10E0),

            // Vendor-specific
            manufacturer_model: get_string(obj, tags::MANUFACTURER_MODEL_NAME),
        })
    }

    /// Get the parsed modality
    pub fn get_modality(&self) -> Option<Modality> {
        self.modality.as_ref().and_then(|m| m.parse().ok())
    }

    /// Get the MR acquisition type (2D or 3D)
    pub fn get_acquisition_type(&self) -> Option<MRAcquisitionType> {
        self.mr_acquisition_type.as_ref().and_then(|t| t.parse().ok())
    }

    /// Determine image orientation from direction cosines
    pub fn get_orientation(&self) -> Option<ImageOrientation> {
        let orientation = self.image_orientation?;
        let is_reformatted = self.is_reformatted();

        // Calculate absolute values and determine primary axes
        let abs_vals: Vec<f64> = orientation.iter().map(|v| v.abs()).collect();

        // Sort indices by absolute value
        let mut indices: Vec<usize> = (0..6).collect();
        indices.sort_by(|&a, &b| abs_vals[b].partial_cmp(&abs_vals[a]).unwrap());

        let (i1, i2) = (indices[0], indices[1]);

        // Determine orientation based on dominant axes
        let base_orientation = match (i1, i2) {
            (0, 4) | (4, 0) | (0, 1) | (1, 0) => ImageOrientation::AXI,
            (1, 5) | (5, 1) | (1, 2) | (2, 1) => ImageOrientation::SAG,
            (0, 5) | (5, 0) | (2, 3) | (3, 2) => ImageOrientation::COR,
            _ => ImageOrientation::AXI, // Default to axial
        };

        // Apply reformatted suffix if needed
        Some(if is_reformatted {
            match base_orientation {
                ImageOrientation::AXI => ImageOrientation::AXIr,
                ImageOrientation::SAG => ImageOrientation::SAGr,
                ImageOrientation::COR => ImageOrientation::CORr,
                other => other,
            }
        } else {
            base_orientation
        })
    }

    /// Check if this is a reformatted image
    pub fn is_reformatted(&self) -> bool {
        self.image_type.iter().any(|t| t.to_uppercase() == "REFORMATTED")
    }

    /// Check if this is a derived image
    pub fn is_derived(&self) -> bool {
        self.image_type.first()
            .map(|t| t.to_uppercase() == "DERIVED")
            .unwrap_or(false)
    }

    /// Check if this is an original/primary image
    pub fn is_original(&self) -> bool {
        self.image_type.first()
            .map(|t| t.to_uppercase() == "ORIGINAL")
            .unwrap_or(false)
    }

    /// Get contrast status
    pub fn get_contrast(&self) -> Contrast {
        // Check contrast agent tag
        if self.contrast_agent.is_some() {
            return Contrast::CE;
        }

        // Check series description for +C or C+
        if let Some(ref desc) = self.series_description {
            let upper = desc.to_uppercase();
            if upper.contains("+C") || upper.contains("C+") {
                return Contrast::CE;
            }
        }

        Contrast::NE
    }

    /// Check if T1 FLAIR based on TR/TE thresholds
    pub fn is_t1_flair(&self) -> bool {
        let config = get_config();
        let thresholds = &config.thresholds.mr_parameters.t1_flair;

        match (self.repetition_time, self.echo_time) {
            (Some(tr), Some(te)) => thresholds.matches(tr, te),
            _ => false,
        }
    }

    /// Check if T2 FLAIR based on TR/TE and inversion time
    pub fn is_t2_flair(&self) -> bool {
        let config = get_config();
        let thresholds = &config.thresholds.mr_parameters.t2_flair;

        match (self.repetition_time, self.echo_time) {
            (Some(tr), Some(te)) => {
                thresholds.matches(tr, te) && self.inversion_time.is_some()
            }
            _ => false,
        }
    }

    /// Check if CUBE sequence
    pub fn is_cube(&self) -> bool {
        self.pulse_sequence_name
            .as_ref()
            .map(|s| s.to_uppercase().contains("CUBE"))
            .unwrap_or(false)
    }

    /// Check if BRAVO/FSPGR sequence
    pub fn is_bravo(&self) -> bool {
        self.pulse_sequence_name
            .as_ref()
            .map(|s| {
                let upper = s.to_uppercase();
                upper.contains("BRAVO") || upper.contains("FSPGR") || upper.contains("EFGRE3D")
            })
            .unwrap_or(false)
    }

    /// Generate study folder name
    pub fn generate_study_folder(&self) -> String {
        let patient_id = self.patient_id.as_deref().unwrap_or("UNKNOWN");
        let study_date = self.study_date.as_deref().unwrap_or("19700101");
        let modality = self.modality.as_deref().unwrap_or("XX");
        let accession = self.accession_number.as_deref().unwrap_or("0");

        format!("{}_{}_{}_{}",
            sanitize_filename(patient_id),
            sanitize_filename(study_date),
            sanitize_filename(modality),
            sanitize_filename(accession)
        )
    }
}

// Helper functions for extracting DICOM values
fn get_string(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<String> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn get_string_by_tag(obj: &FileDicomObject<InMemDicomObject>, group: u16, element: u16) -> Option<String> {
    let tag = Tag(group, element);
    get_string(obj, tag)
}

fn get_string_list(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Vec<String> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.split('\\').map(|p| p.trim().to_string()).collect())
        .unwrap_or_default()
}

fn get_f64(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<f64> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_float64().ok())
}

fn get_i32(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<i32> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
}

fn get_i32_by_tag(obj: &FileDicomObject<InMemDicomObject>, group: u16, element: u16) -> Option<i32> {
    let tag = Tag(group, element);
    get_i32(obj, tag)
}

fn get_u32_by_tag(obj: &FileDicomObject<InMemDicomObject>, group: u16, element: u16) -> Option<u32> {
    let tag = Tag(group, element);
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_int::<u32>().ok())
}

fn get_f64_array_6(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<[f64; 6]> {
    let elem = obj.element(tag).ok()?;
    let values: Vec<f64> = elem.to_multi_float64().ok()?;
    if values.len() >= 6 {
        Some([values[0], values[1], values[2], values[3], values[4], values[5]])
    } else {
        None
    }
}

fn get_f64_array_3(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<[f64; 3]> {
    let elem = obj.element(tag).ok()?;
    let values: Vec<f64> = elem.to_multi_float64().ok()?;
    if values.len() >= 3 {
        Some([values[0], values[1], values[2]])
    } else {
        None
    }
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("test123"), "test123");
        assert_eq!(sanitize_filename("test/file"), "test_file");
        assert_eq!(sanitize_filename("test:file"), "test_file");
    }
}
