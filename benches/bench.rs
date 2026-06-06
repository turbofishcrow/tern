use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tern::words::{brightest_mos_mode_and_gener_bjorklund, brightest_mos_mode_and_gener_bresenham};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Bjorklund (small)", |b| {
        b.iter(|| brightest_mos_mode_and_gener_bjorklund(black_box(5), black_box(2)))
    });
    c.bench_function("Bresenham (small)", |b| {
        b.iter(|| brightest_mos_mode_and_gener_bresenham(black_box(5), black_box(2)))
    });
    c.bench_function("Bjorklund (large)", |b| {
        b.iter(|| brightest_mos_mode_and_gener_bjorklund(black_box(21), black_box(13)))
    });
    c.bench_function("Bresenham (large)", |b| {
        b.iter(|| brightest_mos_mode_and_gener_bresenham(black_box(21), black_box(13)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
