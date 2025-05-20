use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn write_cossin_table() {
    const DEPTH: usize = 7;

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("cossin_table.rs");
    let mut file = File::create(dest_path).unwrap();

    writeln!(file, "pub(crate) const COSSIN_DEPTH: usize = {};", DEPTH).unwrap();
    write!(
        file,
        "pub(crate) const COSSIN: [u32; 1 << COSSIN_DEPTH] = ["
    )
    .unwrap();

    // Treat sin and cos as unsigned values since the sign will always be
    // positive in the range [0, pi/4).
    // No headroom for interpolation rounding error (this is needed for
    // DEPTH = 6 for example).
    const AMPLITUDE: f64 = u16::MAX as f64;

    for i in 0..(1 << DEPTH) {
        if i % 4 == 0 {
            write!(file, "\n   ").unwrap();
        }
        // Use midpoint samples to save one entry in the LUT
        let (sin, cos) = (PI / 4. * ((i as f64 + 0.5) / (1 << DEPTH) as f64)).sin_cos();
        // Add one bit accuracy to cos due to 0.5 < cos(z) <= 1 for |z| < pi/4
        // The -1 LSB is cancelled when unscaling with the biased half amplitude
        let cos = ((cos * 2. - 1.) * AMPLITUDE - 1.).round() as u32;
        let sin = (sin * AMPLITUDE).round() as u32;
        write!(file, " {},", cos + (sin << 16)).unwrap();
    }
    writeln!(file, "\n];").unwrap();
}

fn write_cordic_tables() {
    const DEPTH: i32 = 30;

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("cordic_tables.rs");
    let mut file = File::create(dest_path).unwrap();

    const Q31: f64 = (1i64 << 31) as _;
    writeln!(
        file,
        "/// Gain of cordic in circular mode.\npub const CORDIC_CIRCULAR_GAIN: f64 = {};",
        (0..DEPTH).fold(1.0, |f, i| f * (1.0 + 0.25f64.powi(i)).sqrt())
    )
    .unwrap();
    writeln!(
        file,
        "pub(crate) const CORDIC_CIRCULAR: [i32; {DEPTH}] = {:?};",
        (0..DEPTH)
            .map(|i| (0.5f64.powi(i).atan() / PI * Q31).round() as i64 as _)
            .collect::<Vec<i32>>()
    )
    .unwrap();

    let mut f = 1.0f64;
    let mut k = 4;
    for i in 1..DEPTH {
        let r = if i == k {
            k = 3 * i + 1;
            2
        } else {
            1
        };
        for _ in 0..r {
            f *= (1.0 - 0.25f64.powi(i)).sqrt();
        }
    }
    writeln!(
        file,
        "/// Gain of cordic in hyperbolic mode.\npub const CORDIC_HYPERBOLIC_GAIN: f64 = {};",
        f
    )
    .unwrap();
    writeln!(
        file,
        "pub(crate) const CORDIC_HYPERBOLIC: [i32; {DEPTH}] = {:?};",
        (0..DEPTH)
            .map(|i| (0.5f64.powi(i + 1).atanh() * Q31).round() as i64 as _)
            .collect::<Vec<i32>>()
    )
    .unwrap();
}

fn main() {
    write_cossin_table();
    write_cordic_tables();
    println!("cargo:rerun-if-changed=build.rs");
}
