//! Benchmarks for DICOM to NIfTI conversion
//!
//! Run with: cargo bench

use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // TODO: Add actual benchmark code
            let _x = 1 + 1;
        })
    });
}

criterion_group!(benches, benchmark_placeholder);
criterion_main!(benches);
