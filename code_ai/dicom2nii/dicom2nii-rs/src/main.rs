//! DICOM to NIfTI CLI tool
//!
//! This binary provides command-line access to the dicom2nii library.
//!
//! # Usage
//!
//! ```bash
//! # Rename and organize DICOM files
//! dicom2nii-cli rename -i ./raw_dicom -o ./organized_dicom
//!
//! # Convert DICOM to NIfTI
//! dicom2nii-cli convert -i ./organized_dicom -o ./nifti
//!
//! # Run full pipeline
//! dicom2nii-cli pipeline -i ./raw_dicom -o ./nifti
//!
//! # Generate statistics
//! dicom2nii-cli stats -p ./nifti -f csv
//! ```

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// DICOM to NIfTI converter CLI
#[derive(Parser)]
#[command(name = "dicom2nii-cli")]
#[command(author = "Brain Parcellation Team")]
#[command(version = dicom2nii::VERSION)]
#[command(about = "High-performance DICOM to NIfTI converter", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Number of worker threads
    #[arg(short = 'j', long, default_value = "4", global = true)]
    jobs: usize,
}

#[derive(Subcommand)]
enum Commands {
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

    /// Show version information
    Version,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Set up thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.jobs)
        .build_global()?;

    match cli.command {
        Commands::Rename { input, output } => {
            info!("Renaming DICOM files from {:?} to {:?}", input, output);
            info!("Using {} worker threads", cli.jobs);
            // TODO: Implement rename command
            println!("DICOM rename not yet implemented");
        }

        Commands::Convert {
            input,
            output,
            skip_rename,
        } => {
            info!(
                "Converting DICOM to NIfTI from {:?} to {:?}",
                input, output
            );
            info!("Skip rename: {}", skip_rename);
            info!("Using {} worker threads", cli.jobs);
            // TODO: Implement convert command
            println!("DICOM to NIfTI conversion not yet implemented");
        }

        Commands::Pipeline {
            input,
            output,
            keep_intermediate,
        } => {
            info!("Running full pipeline from {:?} to {:?}", input, output);
            info!("Keep intermediate: {}", keep_intermediate);
            info!("Using {} worker threads", cli.jobs);
            // TODO: Implement pipeline command
            println!("Full pipeline not yet implemented");
        }

        Commands::Stats { path, format } => {
            info!("Generating statistics for {:?} in {} format", path, format);
            // TODO: Implement stats command
            println!("Statistics generation not yet implemented");
        }

        Commands::Nii2dcm {
            input,
            output,
            reference,
        } => {
            info!("Converting NIfTI to DICOM from {:?} to {:?}", input, output);
            if let Some(ref_path) = &reference {
                info!("Using reference DICOM: {:?}", ref_path);
            }
            // TODO: Implement nii2dcm command
            println!("NIfTI to DICOM conversion not yet implemented");
        }

        Commands::Version => {
            println!("dicom2nii-cli version {}", dicom2nii::VERSION);
            println!("Rust edition 2021");
        }
    }

    Ok(())
}
