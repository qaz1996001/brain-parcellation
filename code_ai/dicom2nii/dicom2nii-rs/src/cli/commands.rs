//! Command implementations
//!
//! This module implements the actual logic for each CLI command.

use crate::cli::args::*;
use crate::config::{EnumExt, Modality, MRSeriesRename, T1SeriesRename, T2SeriesRename, CTSeriesRename};
use crate::utils::fs::{self, collect_dicom_files, ensure_dir, list_nifti_files};
use crate::utils::parallel::{ParallelConfig, ParallelExecutor, ProcessingStats, ProcessingTimer};

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, error, info, warn};

/// Run the CLI with the given arguments
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Rename(args) => run_rename(args, &cli),
        Commands::Convert(args) => run_convert(args, &cli),
        Commands::Pipeline(args) => run_pipeline(args, &cli),
        Commands::Postprocess(args) => run_postprocess(args, &cli),
        Commands::Stats(args) => run_stats(args, &cli),
        Commands::Nii2dcm(args) => run_nii2dcm(args, &cli),
        Commands::ListSeries => run_list_series(),
        Commands::Validate(args) => run_validate(args, &cli),
        Commands::Info => run_info(),
    }
}

/// Run the rename command
fn run_rename(args: RenameArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("DICOM rename");

    info!("Starting DICOM rename");
    info!("  Input:  {:?}", args.input);
    info!("  Output: {:?}", args.output);
    info!("  Workers: {}", cli.effective_workers());

    // Validate input directory
    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", args.input);
    }

    if !args.input.is_dir() {
        anyhow::bail!("Input path is not a directory: {:?}", args.input);
    }

    // Create output directory
    ensure_dir(&args.output)?;

    // Collect DICOM files
    info!("Scanning for DICOM files...");
    let dicom_files = collect_dicom_files(&args.input, true)?;
    info!("Found {} DICOM files", dicom_files.len());

    if dicom_files.is_empty() {
        warn!("No DICOM files found in input directory");
        return Ok(());
    }

    if args.dry_run {
        info!("Dry run mode - no files will be modified");
        println!("\nDry run summary:");
        println!("  DICOM files found: {}", dicom_files.len());
        println!("  Would process: {:?} -> {:?}", args.input, args.output);
        return Ok(());
    }

    // TODO: Implement actual rename logic
    // This requires the ConvertManager to be implemented
    warn!("DICOM rename not fully implemented yet");
    println!("\nRename command structure ready.");
    println!("Actual rename logic requires ConvertManager implementation.");

    timer.stop();
    Ok(())
}

/// Run the convert command
fn run_convert(args: ConvertArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("DICOM to NIfTI conversion");

    info!("Starting DICOM to NIfTI conversion");
    info!("  Input:  {:?}", args.input);
    info!("  Output: {:?}", args.output);
    info!("  Workers: {}", cli.effective_workers());

    // Validate input directory
    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", args.input);
    }

    // Create output directory
    ensure_dir(&args.output)?;

    // TODO: Implement actual conversion logic
    // This requires the Dicm2NiixConverter to be implemented
    warn!("DICOM to NIfTI conversion not fully implemented yet");
    println!("\nConvert command structure ready.");
    println!("Actual conversion requires dcm2niix integration.");

    timer.stop();
    Ok(())
}

/// Run the full pipeline command
fn run_pipeline(args: PipelineArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("Full pipeline");

    info!("Starting full pipeline");
    info!("  Input:  {:?}", args.input);
    info!("  Output: {:?}", args.output);
    info!("  Workers: {}", cli.effective_workers());

    // Validate input directory
    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", args.input);
    }

    // Create output directory
    ensure_dir(&args.output)?;

    // Determine intermediate directory
    let intermediate_dir = args.intermediate.clone().unwrap_or_else(|| {
        args.output.join(".intermediate_dicom")
    });

    info!("Intermediate directory: {:?}", intermediate_dir);
    ensure_dir(&intermediate_dir)?;

    // Step 1: Rename DICOM files
    info!("\n=== Step 1: DICOM Rename ===");
    let rename_args = RenameArgs {
        input: args.input.clone(),
        output: intermediate_dir.clone(),
        overwrite: args.overwrite,
        copy: true,
        skip_validation: false,
        modality: None,
        exclude: None,
        dry_run: false,
    };
    run_rename(rename_args, cli)?;

    // Step 2: Convert to NIfTI
    info!("\n=== Step 2: DICOM to NIfTI ===");
    let convert_args = ConvertArgs {
        input: intermediate_dir.clone(),
        output: args.output.clone(),
        skip_rename: true,
        skip_postprocess: false,
        overwrite: args.overwrite,
        keep_json: false,
        compression: 6,
        filename_format: "%f".to_string(),
    };
    run_convert(convert_args, cli)?;

    // Step 3: Post-process NIfTI
    info!("\n=== Step 3: Post-processing ===");
    let postprocess_args = PostprocessArgs {
        input: args.output.clone(),
        mode: PostprocessMode::All,
        cleanup: true,
        min_size_kb: 100,
    };
    run_postprocess(postprocess_args, cli)?;

    // Step 4: Generate stats if requested
    if args.stats {
        info!("\n=== Step 4: Statistics ===");
        let stats_args = StatsArgs {
            path: args.output.clone(),
            format: args.stats_format,
            output: None,
            recursive: true,
            include_hash: false,
        };
        run_stats(stats_args, cli)?;
    }

    // Cleanup intermediate files if not keeping
    if !args.keep_intermediate {
        info!("Cleaning up intermediate files...");
        if let Err(e) = fs::remove_dir_if_exists(&intermediate_dir) {
            warn!("Failed to remove intermediate directory: {}", e);
        }
    }

    timer.stop();
    Ok(())
}

/// Run the postprocess command
fn run_postprocess(args: PostprocessArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("NIfTI post-processing");

    info!("Starting NIfTI post-processing");
    info!("  Input:  {:?}", args.input);
    info!("  Mode: {:?}", args.mode);

    // Validate input directory
    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", args.input);
    }

    // List NIfTI files
    let nifti_files = list_nifti_files(&args.input, true)?;
    info!("Found {} NIfTI files", nifti_files.len());

    if nifti_files.is_empty() {
        warn!("No NIfTI files found in input directory");
        return Ok(());
    }

    // Cleanup small files if requested
    if args.cleanup {
        let mut removed = 0;
        for file in &nifti_files {
            if fs::is_file_too_small(file, args.min_size_kb) {
                debug!("Removing small file: {:?}", file);
                if let Err(e) = fs::remove_file_if_exists(file) {
                    warn!("Failed to remove file {:?}: {}", file, e);
                } else {
                    removed += 1;
                }
            }
        }
        if removed > 0 {
            info!("Removed {} small files (< {} KB)", removed, args.min_size_kb);
        }
    }

    // TODO: Implement actual post-processing logic
    warn!("NIfTI post-processing not fully implemented yet");
    println!("\nPost-process command structure ready.");
    println!("Actual post-processing requires strategy implementations.");

    timer.stop();
    Ok(())
}

/// Run the stats command
fn run_stats(args: StatsArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("Statistics generation");

    info!("Generating statistics");
    info!("  Path:   {:?}", args.path);
    info!("  Format: {:?}", args.format);

    // Validate path
    if !args.path.exists() {
        anyhow::bail!("Path does not exist: {:?}", args.path);
    }

    // Collect files
    let nifti_files = list_nifti_files(&args.path, args.recursive)?;
    let dicom_files = collect_dicom_files(&args.path, args.recursive).unwrap_or_default();

    println!("\n=== Statistics ===");
    println!("Path: {:?}", args.path);
    println!("NIfTI files: {}", nifti_files.len());
    println!("DICOM files: {}", dicom_files.len());

    // Calculate total size
    let total_nifti_size: u64 = nifti_files
        .iter()
        .filter_map(|p| fs::file_size(p).ok())
        .sum();

    let total_dicom_size: u64 = dicom_files
        .iter()
        .filter_map(|f| Some(f.size))
        .sum();

    println!("Total NIfTI size: {:.2} MB", total_nifti_size as f64 / 1024.0 / 1024.0);
    println!("Total DICOM size: {:.2} MB", total_dicom_size as f64 / 1024.0 / 1024.0);

    // Determine output file
    let output_path = args.output.unwrap_or_else(|| {
        let ext = match args.format {
            OutputFormat::Csv => "csv",
            OutputFormat::Json => "json",
            OutputFormat::Excel => "xlsx",
            OutputFormat::Table => "txt",
        };
        args.path.join(format!("stats.{}", ext))
    });

    // TODO: Generate actual report file
    info!("Statistics report would be saved to: {:?}", output_path);

    timer.stop();
    Ok(())
}

/// Run the nii2dcm command
fn run_nii2dcm(args: Nii2dcmArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("NIfTI to DICOM conversion");

    info!("Starting NIfTI to DICOM conversion");
    info!("  Input:  {:?}", args.input);
    info!("  Output: {:?}", args.output);

    // Validate input file
    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {:?}", args.input);
    }

    if !args.input.is_file() {
        anyhow::bail!("Input path is not a file: {:?}", args.input);
    }

    // Create output directory
    ensure_dir(&args.output)?;

    // TODO: Implement NIfTI to DICOM conversion
    warn!("NIfTI to DICOM conversion not implemented yet");
    println!("\nNii2dcm command structure ready.");
    println!("Actual conversion requires DICOM writer implementation.");

    timer.stop();
    Ok(())
}

/// Run the list-series command
fn run_list_series() -> Result<()> {
    println!("\n=== Supported MR Series Types ===\n");

    println!("General MR Series:");
    for series in MRSeriesRename::to_list() {
        println!("  - {}", series);
    }

    println!("\nT1 Weighted Series:");
    for series in T1SeriesRename::to_list() {
        println!("  - {}", series);
    }

    println!("\nT2 Weighted Series:");
    for series in T2SeriesRename::to_list() {
        println!("  - {}", series);
    }

    println!("\n=== Supported CT Series Types ===\n");
    for series in CTSeriesRename::to_list() {
        println!("  - {}", series);
    }

    println!("\n=== Modalities ===\n");
    for modality in Modality::to_list() {
        println!("  - {}", modality);
    }

    Ok(())
}

/// Run the validate command
fn run_validate(args: ValidateArgs, cli: &Cli) -> Result<()> {
    let timer = ProcessingTimer::start("DICOM validation");

    info!("Validating DICOM files");
    info!("  Path: {:?}", args.path);

    // Validate path
    if !args.path.exists() {
        anyhow::bail!("Path does not exist: {:?}", args.path);
    }

    // Collect DICOM files
    let dicom_files = collect_dicom_files(&args.path, args.recursive)?;

    println!("\n=== Validation Results ===");
    println!("Path: {:?}", args.path);
    println!("DICOM files found: {}", dicom_files.len());

    let mut valid_count = 0;
    let mut invalid_count = 0;

    for file in &dicom_files {
        // Basic validation - check file exists and has size
        if file.size > 0 {
            valid_count += 1;
            if args.detailed {
                println!("  ✓ {:?} ({} bytes)", file.path, file.size);
            }
        } else {
            invalid_count += 1;
            if args.detailed {
                println!("  ✗ {:?} (empty file)", file.path);
            }
        }
    }

    println!("\nSummary:");
    println!("  Valid:   {}", valid_count);
    println!("  Invalid: {}", invalid_count);

    // Write report if output specified
    if let Some(output) = args.output {
        info!("Validation report would be saved to: {:?}", output);
    }

    timer.stop();
    Ok(())
}

/// Run the info command
fn run_info() -> Result<()> {
    println!("\n=== dicom2nii Info ===\n");
    println!("Version:  {}", env!("CARGO_PKG_VERSION"));
    println!("Authors:  {}", env!("CARGO_PKG_AUTHORS"));

    #[cfg(debug_assertions)]
    println!("Build:    Debug");
    #[cfg(not(debug_assertions))]
    println!("Build:    Release");

    println!("\nFeatures:");
    #[cfg(feature = "python")]
    println!("  - Python bindings (pyo3)");
    #[cfg(feature = "ffi")]
    println!("  - C FFI interface");

    println!("\nSystem:");
    println!("  - CPUs: {}", num_cpus::get());
    println!("  - OS:   {}", std::env::consts::OS);
    println!("  - Arch: {}", std::env::consts::ARCH);

    println!("\nDependencies:");
    println!("  - dicom-rs for DICOM parsing");
    println!("  - nifti-rs for NIfTI handling");
    println!("  - rayon for parallel processing");
    println!("  - clap for CLI");

    println!("\nUsage:");
    println!("  dicom2nii pipeline -i /input -o /output");
    println!("  dicom2nii rename -i /raw -o /organized");
    println!("  dicom2nii convert -i /organized -o /nifti");
    println!("  dicom2nii stats -p /nifti");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_run_info() {
        assert!(run_info().is_ok());
    }

    #[test]
    fn test_run_list_series() {
        assert!(run_list_series().is_ok());
    }
}
