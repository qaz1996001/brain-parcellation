//! NIfTI processing module
//!
//! This module provides NIfTI file handling and conversion using dcm2niix.

pub mod converter;

pub use converter::NiftiConverter;
