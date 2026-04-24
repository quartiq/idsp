//! Real-input DDC / lock-in.
//!
//! Showcases:
//! - `idsp::iir::coefficients::Filter` and `idsp::iir::Biquad` for the shared post-mix lowpass
//! - `f32` throughout: this example is about graph shape and lane duplication,
//!   not fixed-point arithmetic policy
//! - `[f32; 2]` as an explicit I/Q lane type so `dsp_process` can see the lane structure
//! - `dsp_process::Split::lanes()` to share one lowpass configuration across I/Q state
//! - `dsp_process::Minor` to keep the graph visible as `mix -> lowpass`
//!
//! The graph is:
//! `x[n] -> x * exp(-j w n) -> LPF_I/Q`

use dsp_process::{Process, Split, SplitProcess};
use idsp::iir::{Biquad, DirectForm1, coefficients::Filter};
use std::f32::consts::TAU;

// --- DSP graph ---

type Iq = [f32; 2];

#[derive(Debug, Clone, Copy)]
struct QuadratureMix {
    step: f32,
}

impl SplitProcess<f32, Iq, f32> for QuadratureMix {
    fn process(&self, phase: &mut f32, x: f32) -> Iq {
        let (s, c) = phase.sin_cos();
        *phase = (*phase + TAU * self.step).rem_euclid(TAU);
        [x * c, -x * s]
    }
}

fn ddc(lo_freq: f32, cutoff: f32) -> impl Process<f32, Iq> {
    let mix = Split::new(QuadratureMix { step: lo_freq }, 0.0);
    let mut filter = Filter::default();
    filter.critical_frequency(cutoff);
    let lowpass: Biquad<f32> = filter.lowpass().into();
    let post = Split::new(lowpass, DirectForm1::default()).lanes();
    mix * post
}

#[derive(Debug, Clone, Copy)]
struct DdcResult {
    expected: Iq,
    mean: Iq,
    rms: f32,
}

fn run_graph(lo_freq: f32, cutoff: f32, x: &[f32]) -> Vec<Iq> {
    let mut ddc = ddc(lo_freq, cutoff);
    x.iter().copied().map(|x| ddc.process(x)).collect()
}

// --- Fixture ---

fn tone(freq: f32, phase: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (TAU * freq * i as f32 + phase).cos())
        .collect()
}

// --- Metrics / checks ---

fn measure_ddc(y: &[Iq], expected: Iq, skip: usize) -> DdcResult {
    let mut sum = [0.0; 2];
    let mut err2 = 0.0;
    let mut used = 0.0;
    for y in &y[skip..] {
        sum[0] += y[0];
        sum[1] += y[1];
        err2 += (y[0] - expected[0]).powi(2) + (y[1] - expected[1]).powi(2);
        used += 1.0;
    }
    DdcResult {
        expected,
        mean: [sum[0] / used, sum[1] / used],
        rms: (err2 / used).sqrt(),
    }
}

fn run_ddc() -> DdcResult {
    let lo_freq = 0.173;
    let phi: f32 = 0.37;
    let expected = [0.5 * phi.cos(), 0.5 * phi.sin()];
    let x = tone(lo_freq, phi, 16384);
    let y = run_graph(lo_freq, 0.002, &x);
    measure_ddc(&y, expected, 12288)
}

fn main() {
    let r = run_ddc();
    println!(
        "ddc mean=({:.5}, {:.5}) expected=({:.5}, {:.5}) rms={:.5}",
        r.mean[0], r.mean[1], r.expected[0], r.expected[1], r.rms
    );
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn recovers_dc_iq() {
        let r = run_ddc();
        assert!((r.mean[0] - r.expected[0]).abs() < 3e-3);
        assert!((r.mean[1] - r.expected[1]).abs() < 3e-3);
        assert!(r.rms < 6e-3);
    }
}
