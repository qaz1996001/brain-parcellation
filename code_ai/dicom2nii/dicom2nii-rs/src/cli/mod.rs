//! Command-line interface module
//!
//! This module provides the CLI functionality for dicom2nii:
//! - [`args`]: Command-line argument definitions using clap
//! - [`commands`]: Implementation of CLI commands

pub mod args;
pub mod commands;

pub use args::{Cli, Commands};
pub use commands::run;
