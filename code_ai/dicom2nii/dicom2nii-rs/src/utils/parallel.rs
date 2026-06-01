//! Parallel processing utilities
//!
//! This module provides utilities for parallel processing of DICOM files,
//! including progress tracking and error handling.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Whether to show progress bar
    pub show_progress: bool,
    /// Progress bar message prefix
    pub progress_prefix: String,
    /// Chunk size for batch processing (0 = auto)
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get().min(4),
            show_progress: true,
            progress_prefix: "Processing".to_string(),
            chunk_size: 0,
        }
    }
}

impl ParallelConfig {
    /// Create a new configuration with specified number of workers
    pub fn with_workers(num_workers: usize) -> Self {
        Self {
            num_workers: num_workers.max(1),
            ..Default::default()
        }
    }

    /// Set whether to show progress bar
    pub fn show_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Set progress bar prefix
    pub fn progress_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.progress_prefix = prefix.into();
        self
    }

    /// Set chunk size for batch processing
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
}

/// Parallel executor for batch processing
pub struct ParallelExecutor {
    config: ParallelConfig,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_workers(num_workers: usize) -> Self {
        Self::new(ParallelConfig::with_workers(num_workers))
    }

    /// Execute a function on each item in parallel
    ///
    /// # Arguments
    /// * `items` - Items to process
    /// * `f` - Function to apply to each item
    ///
    /// # Returns
    /// * `Vec<R>` - Results from processing each item
    pub fn execute<T, R, F>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(T) -> R + Send + Sync,
    {
        let total = items.len();

        if total == 0 {
            return Vec::new();
        }

        // Create progress bar if enabled
        let progress = if self.config.show_progress {
            let pb = ProgressBar::new(total as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{prefix} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_prefix(self.config.progress_prefix.clone());
            Some(pb)
        } else {
            None
        };

        // Build thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_workers)
            .build()
            .expect("Failed to build thread pool");

        // Process items
        let results: Vec<R> = pool.install(|| {
            items
                .into_par_iter()
                .map(|item| {
                    let result = f(item);
                    if let Some(ref pb) = progress {
                        pb.inc(1);
                    }
                    result
                })
                .collect()
        });

        // Finish progress bar
        if let Some(pb) = progress {
            pb.finish_with_message("Done");
        }

        results
    }

    /// Execute a fallible function on each item in parallel
    ///
    /// # Arguments
    /// * `items` - Items to process
    /// * `f` - Fallible function to apply to each item
    ///
    /// # Returns
    /// * `(Vec<R>, Vec<E>)` - Tuple of successful results and errors
    pub fn execute_fallible<T, R, E, F>(&self, items: Vec<T>, f: F) -> (Vec<R>, Vec<E>)
    where
        T: Send + Sync,
        R: Send,
        E: Send,
        F: Fn(T) -> Result<R, E> + Send + Sync,
    {
        let results = self.execute(items, f);

        let mut successes = Vec::new();
        let mut errors = Vec::new();

        for result in results {
            match result {
                Ok(r) => successes.push(r),
                Err(e) => errors.push(e),
            }
        }

        (successes, errors)
    }

    /// Execute with batching - processes items in chunks
    ///
    /// Useful when processing many small items to reduce overhead.
    ///
    /// # Arguments
    /// * `items` - Items to process
    /// * `f` - Function to apply to a batch of items
    ///
    /// # Returns
    /// * `Vec<R>` - Flattened results from all batches
    pub fn execute_batched<T, R, F>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send + Sync + Clone,
        R: Send,
        F: Fn(Vec<T>) -> Vec<R> + Send + Sync,
    {
        let total = items.len();

        if total == 0 {
            return Vec::new();
        }

        // Determine chunk size
        let chunk_size = if self.config.chunk_size > 0 {
            self.config.chunk_size
        } else {
            // Auto-calculate: aim for ~10 chunks per worker
            (total / (self.config.num_workers * 10)).max(1)
        };

        // Split into chunks
        let chunks: Vec<Vec<T>> = items
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        let num_chunks = chunks.len();

        // Create progress bar
        let progress = if self.config.show_progress {
            let pb = ProgressBar::new(num_chunks as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{prefix} [{bar:40.cyan/blue}] {pos}/{len} batches ({eta})")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_prefix(self.config.progress_prefix.clone());
            Some(pb)
        } else {
            None
        };

        // Build thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_workers)
            .build()
            .expect("Failed to build thread pool");

        // Process chunks
        let results: Vec<Vec<R>> = pool.install(|| {
            chunks
                .into_par_iter()
                .map(|chunk| {
                    let result = f(chunk);
                    if let Some(ref pb) = progress {
                        pb.inc(1);
                    }
                    result
                })
                .collect()
        });

        // Finish progress bar
        if let Some(pb) = progress {
            pb.finish_with_message("Done");
        }

        // Flatten results
        results.into_iter().flatten().collect()
    }
}

/// Statistics from parallel processing
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// Total items processed
    pub total: usize,
    /// Successfully processed items
    pub success: usize,
    /// Failed items
    pub failed: usize,
    /// Skipped items
    pub skipped: usize,
    /// Processing duration
    pub duration: Duration,
}

impl ProcessingStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.success as f64 / self.total as f64) * 100.0
        }
    }

    /// Get items processed per second
    pub fn items_per_second(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.total as f64 / secs
        }
    }
}

impl std::fmt::Display for ProcessingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Processed {} items in {:.2}s ({:.1}/s) - {} success, {} failed, {} skipped ({:.1}% success rate)",
            self.total,
            self.duration.as_secs_f64(),
            self.items_per_second(),
            self.success,
            self.failed,
            self.skipped,
            self.success_rate()
        )
    }
}

/// Atomic counter for thread-safe progress tracking
pub struct AtomicCounter {
    count: AtomicUsize,
}

impl AtomicCounter {
    /// Create a new counter starting at 0
    pub fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }

    /// Increment the counter and return the new value
    pub fn increment(&self) -> usize {
        self.count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get the current count
    pub fn get(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    /// Reset the counter to 0
    pub fn reset(&self) {
        self.count.store(0, Ordering::SeqCst);
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring processing duration
pub struct ProcessingTimer {
    start: Instant,
    label: String,
}

impl ProcessingTimer {
    /// Start a new timer
    pub fn start(label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            label: label.into(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop and print elapsed time
    pub fn stop(self) -> Duration {
        let elapsed = self.elapsed();
        tracing::info!("{} completed in {:.2}s", self.label, elapsed.as_secs_f64());
        elapsed
    }
}

/// Simple rate limiter for controlling processing speed
pub struct RateLimiter {
    interval: Duration,
    last_time: std::sync::Mutex<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `items_per_second` - Maximum items to process per second
    pub fn new(items_per_second: f64) -> Self {
        let interval = Duration::from_secs_f64(1.0 / items_per_second);
        Self {
            interval,
            last_time: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Wait if necessary to maintain the rate limit
    pub fn wait(&self) {
        let mut last = self.last_time.lock().unwrap();
        let elapsed = last.elapsed();

        if elapsed < self.interval {
            std::thread::sleep(self.interval - elapsed);
        }

        *last = Instant::now();
    }
}

/// Get the number of CPUs available
pub fn num_cpus() -> usize {
    num_cpus::get()
}

/// Get the recommended number of workers for I/O-bound tasks
pub fn recommended_io_workers() -> usize {
    num_cpus().min(8)
}

/// Get the recommended number of workers for CPU-bound tasks
pub fn recommended_cpu_workers() -> usize {
    num_cpus()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_executor() {
        let executor = ParallelExecutor::with_workers(2);
        let items: Vec<i32> = (0..100).collect();

        let results = executor.execute(items, |x| x * 2);

        assert_eq!(results.len(), 100);
        assert_eq!(results[0], 0);
        assert_eq!(results[50], 100);
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new();
        assert_eq!(counter.get(), 0);

        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.increment(), 2);
        assert_eq!(counter.get(), 2);

        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_processing_stats() {
        let stats = ProcessingStats {
            total: 100,
            success: 95,
            failed: 3,
            skipped: 2,
            duration: Duration::from_secs(10),
        };

        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(stats.items_per_second(), 10.0);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::with_workers(8)
            .show_progress(false)
            .progress_prefix("Test");

        assert_eq!(config.num_workers, 8);
        assert!(!config.show_progress);
        assert_eq!(config.progress_prefix, "Test");
    }
}
