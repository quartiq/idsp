//! FM discriminator receiver core.
//!
//! Showcases:
//! - `idsp::cossin()` as a fixed-point complex oscillator for the synthetic RF input
//! - `idsp::Complex<Q32<32>>::into_bits()` plus mixed `Complex` multiply for
//!   one late-quantized conjugate products in the discriminator
//! - `idsp::iir::Biquad<Q32<30>>` built from the same filter builder, keeping
//!   deemphasis in the fixed-point graph
//! - a deliberately fixed-point DSP core; float is used only in the fixture and
//!   error measurement
//!
//! The DSP identity is:
//! `arg(x[n] * conj(x[n-1])) ~= dphi[n]`

use core::num::Wrapping as W;
use dsp_fixedpoint::Q32;
use dsp_process::{Process, Split, SplitProcess};
use idsp::{
    Complex, cossin,
    iir::{Biquad, DirectForm1, coefficients::Filter},
};
use std::f32::consts::TAU;

// --- DSP graph ---

#[derive(Debug, Clone, Copy)]
struct FmDiscriminator {
    carrier: i32,
}

impl SplitProcess<Complex<Q32<32>>, i32, Option<Complex<Q32<32>>>> for FmDiscriminator {
    fn process(&self, prev: &mut Option<Complex<Q32<32>>>, x: Complex<Q32<32>>) -> i32 {
        let Some(p) = prev.replace(x) else {
            return 0;
        };
        let z = x * p.into_bits().conj();
        (z.arg() - W(self.carrier)).0
    }
}

fn fm_core(carrier: i32, cutoff: f32) -> impl Process<Complex<Q32<32>>, i32> {
    let disc = Split::new(FmDiscriminator { carrier }, None);
    let mut filter = Filter::default();
    filter.critical_frequency(cutoff);
    let deemph: Biquad<Q32<30>> = filter.lowpass().into();
    let deemph = Split::new(deemph, DirectForm1::default());
    (disc * deemph).minor()
}

fn run_graph(carrier: i32, cutoff: f32, x: &[Complex<Q32<32>>]) -> Vec<i32> {
    let mut rx = fm_core(carrier, cutoff);
    x.iter().map(|&x| rx.process(x)).collect()
}

// --- Fixture ---

fn fm_signal(
    carrier: u32,
    deviation: i32,
    message_freq: f32,
    n: usize,
) -> (Vec<Complex<Q32<32>>>, Vec<f32>) {
    let mut phase = W(0i32);
    let mut x = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    for i in 0..n {
        let msg = (TAU * message_freq * i as f32).sin();
        phase += W(carrier as i32 + (deviation as f32 * msg) as i32);
        let (re, im) = cossin(phase.0);
        x.push(Complex::new(Q32::new(re), Q32::new(im)));
        m.push(msg);
    }
    (x, m)
}

// --- Metrics / checks ---

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let ab = a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    let aa = a.iter().map(|a| a * a).sum::<f32>().sqrt();
    let bb = b.iter().map(|b| b * b).sum::<f32>().sqrt();
    ab / (aa * bb)
}

#[derive(Debug)]
struct FmResult {
    corr: f32,
    gain: f32,
    rms: f32,
}

fn lowpass_reference(cutoff: f32, x: &[f32]) -> Vec<f32> {
    let mut filter = Filter::default();
    filter.critical_frequency(cutoff);
    let mut lpf = Split::new(
        Biquad::<f32>::from(filter.lowpass()),
        DirectForm1::default(),
    );
    x.iter().map(|&x| lpf.process(x)).collect()
}

fn bits_to_f32(x: &[i32]) -> Vec<f32> {
    let scale = TAU / (u32::MAX as f32 + 1.0);
    x.iter().map(|&x| x as f32 * scale).collect()
}

fn measure_fm(y: &[f32], m: &[f32], skip: usize) -> FmResult {
    let y = &y[skip..];
    let m = &m[skip..];
    let gain =
        y.iter().zip(m).map(|(y, m)| y * m).sum::<f32>() / m.iter().map(|m| m * m).sum::<f32>();
    let rms = y
        .iter()
        .zip(m)
        .map(|(y, m)| (y - gain * m).powi(2))
        .sum::<f32>()
        .sqrt()
        / y.len() as f32;
    FmResult {
        corr: corr(y, m),
        gain,
        rms,
    }
}

fn run_fm_disc() -> FmResult {
    let carrier = 0x1934_1234u32;
    let deviation = 0x0450_0000i32;
    let message_freq = 0.004;
    let scale = TAU / (u32::MAX as f32 + 1.0);
    let (x, msg) = fm_signal(carrier, deviation, message_freq, 4096);
    let y = bits_to_f32(&run_graph(carrier as i32, 0.02, &x));
    let m = lowpass_reference(
        0.02,
        &msg.iter()
            .map(|&m| deviation as f32 * scale * m)
            .collect::<Vec<_>>(),
    );
    measure_fm(&y, &m, 1024)
}

fn main() {
    let r = run_fm_disc();
    println!("fm corr={:.5} gain={:.5} rms={:.5}", r.corr, r.gain, r.rms);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tracks_known_modulation() {
        let r = run_fm_disc();
        assert!(r.corr > 0.999);
        assert!(r.gain > 0.95);
        assert!(r.gain < 1.05);
        assert!(r.rms < 5e-4);
    }
}
