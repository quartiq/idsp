use core::f32::consts::PI;
use core::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};

use idsp::{atan2, cossin, iir, Filter, Lowpass, PLL, RPLL};

#[library_benchmark]
#[bench::some(-0x7304_2531_i32)]
fn bench_cossin(zi: i32) {
    black_box(cossin(zi));
}

library_benchmark_group!(
    name = bench_cossin_group;
    benchmarks = bench_cossin
);

main!(library_benchmark_groups = bench_cossin_group);
