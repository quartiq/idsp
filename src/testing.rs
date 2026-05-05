//! Tools to test and benchmark algorithms
#![allow(dead_code)]
use super::Complex;
use num_traits::Float;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Maximum acceptable error between a computed and actual value given fixed and relative
/// tolerances.
///
/// # Args
/// * `a` - First input.
/// * `b` - Second input. The relative tolerance is computed with respect to the maximum of the
///   absolute values of the first and second inputs.
/// * `rtol` - Relative tolerance.
/// * `atol` - Fixed tolerance.
///
/// # Returns
/// Maximum acceptable error.
pub(crate) fn max_error<T: Float>(a: T, b: T, rtol: T, atol: T) -> T {
    rtol * a.abs().max(b.abs()) + atol
}

/// Return whether two numbers are within absolute plus relative tolerance
pub(crate) fn isclose<T: Float>(a: T, b: T, rtol: T, atol: T) -> bool {
    (a - b).abs() <= max_error(a, b, rtol, atol)
}

/// Return whether all values are close
pub(crate) fn allclose<T: Float>(a: &[T], b: &[T], rtol: T, atol: T) -> bool {
    a.iter().zip(b).all(|(a, b)| isclose(*a, *b, rtol, atol))
}

/// Return whether both real and imaginary component are close
pub(crate) fn complex_isclose<T: Float>(a: Complex<T>, b: Complex<T>, rtol: T, atol: T) -> bool {
    isclose(a.re(), b.re(), rtol, atol) && isclose(a.im(), b.im(), rtol, atol)
}

/// Return whether all complex values are close
pub(crate) fn complex_allclose<T: Float>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    rtol: T,
    atol: T,
) -> bool {
    a.iter()
        .zip(b)
        .all(|(a, b)| complex_isclose(*a, *b, rtol, atol))
}

/// Spectrum metrics for a coherent single-tone DDS test.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct DdsMetrics {
    /// Carrier FFT bin.
    pub carrier_bin: usize,
    /// Strongest non-carrier FFT bin.
    pub strongest_spur_bin: usize,
    /// Spurious-free dynamic range in dBc.
    pub sfdr_db: f64,
    /// Signal-to-noise ratio in dBc, excluding harmonics.
    pub snr_db: f64,
    /// Total harmonic distortion in dBc.
    pub thd_db: f64,
    /// Total harmonic distortion plus noise in dBc.
    pub thdn_db: f64,
}

/// Convert a power ratio to dB.
pub(crate) fn db(ratio: f64) -> f64 {
    10.0 * ratio.log10()
}

/// One-sided power spectrum of a real sequence.
pub(crate) fn real_fft_power(x: &[f64]) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(x.len());
    let mut x: Vec<_> = x.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut x);
    x[..=x.len() / 2].iter().map(|x| x.norm_sqr()).collect()
}

fn alias_real_bin(bin: usize, n: usize) -> usize {
    let bin = bin % n;
    bin.min(n - bin)
}

/// Coherent single-tone DDS metrics from a one-sided FFT.
///
/// `carrier_bin` must be the fundamental bin of the tone in `x`. Harmonics are
/// folded into the real half-spectrum.
pub(crate) fn dds_metrics(x: &[f64], carrier_bin: usize, harmonics: usize) -> DdsMetrics {
    let n = x.len();
    let power = real_fft_power(x);
    let carrier = power[carrier_bin];

    let harmonic_bins: std::collections::BTreeSet<_> = (2..=harmonics)
        .map(|h| alias_real_bin(h * carrier_bin, n))
        .filter(|&bin| bin != 0 && bin != carrier_bin)
        .collect();

    let mut strongest_spur_bin = 0;
    let mut strongest_spur = 0.0;
    let mut noise = 0.0;
    let mut thd = 0.0;
    let mut thdn = 0.0;

    for (bin, &p) in power.iter().enumerate() {
        if bin == carrier_bin {
            continue;
        }
        if p > strongest_spur {
            strongest_spur = p;
            strongest_spur_bin = bin;
        }
        thdn += p;
        if harmonic_bins.contains(&bin) {
            thd += p;
        } else {
            noise += p;
        }
    }

    DdsMetrics {
        carrier_bin,
        strongest_spur_bin,
        sfdr_db: db(carrier / strongest_spur),
        snr_db: db(carrier / noise),
        thd_db: db(carrier / thd),
        thdn_db: db(carrier / thdn),
    }
}
